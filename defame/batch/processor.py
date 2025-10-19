"""
Batch processing system for handling multiple claims
"""
import asyncio
import uuid
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

from defame.core.models import Claim, PipelineState
from defame.core.pipeline import get_pipeline_controller
from defame.core.database import DatabaseSession, ClaimModel
from defame.utils.logger import get_logger, PerformanceLogger
from config.globals import ClaimType, Priority

logger = get_logger(__name__)


class BatchStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchJob:
    """Batch processing job"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    claims: List[Claim] = field(default_factory=list)
    status: BatchStatus = BatchStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    total_claims: int = 0
    processed_claims: int = 0
    successful_claims: int = 0
    failed_claims: int = 0
    results: List[PipelineState] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def processing_time(self) -> float:
        """Get processing time in seconds"""
        if self.started_at:
            end_time = self.completed_at or datetime.utcnow()
            return (end_time - self.started_at).total_seconds()
        return 0.0
    
    @property
    def estimated_completion(self) -> Optional[datetime]:
        """Estimate completion time based on current progress"""
        if self.progress > 0 and self.started_at and self.status == BatchStatus.PROCESSING:
            elapsed = datetime.utcnow() - self.started_at
            total_estimated = elapsed / self.progress
            return self.started_at + total_estimated
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "total_claims": self.total_claims,
            "processed_claims": self.processed_claims,
            "successful_claims": self.successful_claims,
            "failed_claims": self.failed_claims,
            "processing_time": self.processing_time,
            "estimated_completion": self.estimated_completion.isoformat() if self.estimated_completion else None,
            "errors": self.errors,
            "metadata": self.metadata
        }


class BatchProcessor:
    """Batch processing manager"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_concurrent_jobs = config.get('max_concurrent_jobs', 3)
        self.max_concurrent_claims = config.get('max_concurrent_claims', 10)
        self.job_timeout_hours = config.get('job_timeout_hours', 24)
        
        # Active jobs
        self.active_jobs: Dict[str, BatchJob] = {}
        self.job_semaphore = asyncio.Semaphore(self.max_concurrent_jobs)
        
        # Job history (in production, store in database)
        self.job_history: Dict[str, BatchJob] = {}
    
    async def create_batch_job(
        self,
        name: str,
        claims_data: List[Dict[str, Any]],
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> BatchJob:
        """Create a new batch job"""
        try:
            # Convert claims data to Claim objects
            claims = []
            for claim_data in claims_data:
                claim = Claim(
                    content=claim_data.get('content', ''),
                    claim_type=ClaimType(claim_data.get('claim_type', 'text')),
                    priority=Priority(claim_data.get('priority', 'normal')),
                    source=claim_data.get('source'),
                    metadata=claim_data.get('metadata', {})
                )
                claims.append(claim)
            
            # Create batch job
            job = BatchJob(
                name=name,
                description=description,
                claims=claims,
                total_claims=len(claims),
                metadata=metadata or {}
            )
            
            # Store job
            self.active_jobs[job.id] = job
            
            logger.info(
                f"Created batch job {job.id}",
                job_name=name,
                total_claims=len(claims)
            )
            
            return job
            
        except Exception as e:
            logger.error(f"Failed to create batch job: {e}")
            raise
    
    async def start_batch_job(self, job_id: str) -> bool:
        """Start processing a batch job"""
        job = self.active_jobs.get(job_id)
        if not job:
            return False
        
        if job.status != BatchStatus.PENDING:
            return False
        
        # Start processing in background
        asyncio.create_task(self._process_batch_job(job))
        return True
    
    async def _process_batch_job(self, job: BatchJob):
        """Process a batch job"""
        async with self.job_semaphore:
            try:
                job.status = BatchStatus.PROCESSING
                job.started_at = datetime.utcnow()
                
                logger.info(f"Starting batch job {job.id}")
                
                with PerformanceLogger(f"batch_job_{job.id}") as perf:
                    perf.add_metadata(
                        job_id=job.id,
                        total_claims=job.total_claims
                    )
                    
                    # Process claims with concurrency control
                    semaphore = asyncio.Semaphore(self.max_concurrent_claims)
                    
                    async def process_claim_with_semaphore(claim: Claim) -> Optional[PipelineState]:
                        async with semaphore:
                            return await self._process_single_claim(job, claim)
                    
                    # Create tasks for all claims
                    tasks = [
                        process_claim_with_semaphore(claim)
                        for claim in job.claims
                    ]
                    
                    # Process with progress tracking
                    async for result in self._process_with_progress(job, tasks):
                        if result:
                            job.results.append(result)
                            job.successful_claims += 1
                        else:
                            job.failed_claims += 1
                        
                        job.processed_claims += 1
                        job.progress = job.processed_claims / job.total_claims
                
                # Mark as completed
                job.status = BatchStatus.COMPLETED
                job.completed_at = datetime.utcnow()
                
                logger.info(
                    f"Batch job {job.id} completed",
                    processing_time=job.processing_time,
                    successful_claims=job.successful_claims,
                    failed_claims=job.failed_claims
                )
                
            except Exception as e:
                job.status = BatchStatus.FAILED
                job.completed_at = datetime.utcnow()
                job.errors.append(f"Job processing failed: {str(e)}")
                
                logger.error(f"Batch job {job.id} failed: {e}")
            
            finally:
                # Move to history
                self.job_history[job.id] = job
                if job.id in self.active_jobs:
                    del self.active_jobs[job.id]
    
    async def _process_single_claim(self, job: BatchJob, claim: Claim) -> Optional[PipelineState]:
        """Process a single claim within a batch job"""
        try:
            pipeline = get_pipeline_controller()
            if not pipeline:
                raise Exception("Pipeline controller not available")
            
            result = await pipeline.process_claim(claim)
            return result
            
        except Exception as e:
            error_msg = f"Claim {claim.id} failed: {str(e)}"
            job.errors.append(error_msg)
            logger.warning(error_msg)
            return None
    
    async def _process_with_progress(
        self,
        job: BatchJob,
        tasks: List[asyncio.Task]
    ) -> AsyncGenerator[Optional[PipelineState], None]:
        """Process tasks with progress tracking"""
        # Use asyncio.as_completed for progress tracking
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                yield result
            except Exception as e:
                logger.warning(f"Task failed in batch job {job.id}: {e}")
                yield None
    
    async def cancel_batch_job(self, job_id: str) -> bool:
        """Cancel a batch job"""
        job = self.active_jobs.get(job_id) or self.job_history.get(job_id)
        if not job:
            return False
        
        if job.status in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED]:
            return False
        
        job.status = BatchStatus.CANCELLED
        job.completed_at = datetime.utcnow()
        job.errors.append("Job cancelled by user")
        
        # Move to history if active
        if job_id in self.active_jobs:
            self.job_history[job_id] = job
            del self.active_jobs[job_id]
        
        logger.info(f"Batch job {job_id} cancelled")
        return True
    
    def get_batch_job(self, job_id: str) -> Optional[BatchJob]:
        """Get batch job by ID"""
        return self.active_jobs.get(job_id) or self.job_history.get(job_id)
    
    def list_batch_jobs(
        self,
        status: Optional[BatchStatus] = None,
        limit: int = 50
    ) -> List[BatchJob]:
        """List batch jobs with optional filtering"""
        all_jobs = list(self.active_jobs.values()) + list(self.job_history.values())
        
        if status:
            all_jobs = [job for job in all_jobs if job.status == status]
        
        # Sort by creation time (newest first)
        all_jobs.sort(key=lambda j: j.created_at, reverse=True)
        
        return all_jobs[:limit]
    
    async def cleanup_old_jobs(self, days: int = 7):
        """Clean up old completed jobs"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        jobs_to_remove = []
        for job_id, job in self.job_history.items():
            if (job.completed_at and job.completed_at < cutoff_date and
                job.status in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED]):
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.job_history[job_id]
        
        logger.info(f"Cleaned up {len(jobs_to_remove)} old batch jobs")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get batch processing statistics"""
        all_jobs = list(self.active_jobs.values()) + list(self.job_history.values())
        
        stats = {
            "total_jobs": len(all_jobs),
            "active_jobs": len(self.active_jobs),
            "completed_jobs": len([j for j in all_jobs if j.status == BatchStatus.COMPLETED]),
            "failed_jobs": len([j for j in all_jobs if j.status == BatchStatus.FAILED]),
            "cancelled_jobs": len([j for j in all_jobs if j.status == BatchStatus.CANCELLED]),
            "total_claims_processed": sum(j.processed_claims for j in all_jobs),
            "total_successful_claims": sum(j.successful_claims for j in all_jobs),
            "total_failed_claims": sum(j.failed_claims for j in all_jobs),
        }
        
        # Calculate average processing time for completed jobs
        completed_jobs = [j for j in all_jobs if j.status == BatchStatus.COMPLETED and j.processing_time > 0]
        if completed_jobs:
            stats["average_job_processing_time"] = sum(j.processing_time for j in completed_jobs) / len(completed_jobs)
            stats["average_claims_per_job"] = sum(j.total_claims for j in completed_jobs) / len(completed_jobs)
        else:
            stats["average_job_processing_time"] = 0.0
            stats["average_claims_per_job"] = 0.0
        
        return stats


# Global batch processor instance
batch_processor: Optional[BatchProcessor] = None


def get_batch_processor() -> Optional[BatchProcessor]:
    """Get the global batch processor instance"""
    return batch_processor


def initialize_batch_processor(config: Dict[str, Any]) -> BatchProcessor:
    """Initialize the global batch processor"""
    global batch_processor
    batch_processor = BatchProcessor(config)
    return batch_processor