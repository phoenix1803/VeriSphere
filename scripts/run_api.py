#!/usr/bin/env python3
"""
REST API server for VeriSphere
"""
import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from defame.core.models import Claim
from defame.core.pipeline import initialize_pipeline_controller, get_pipeline_controller
from defame.core.agent_manager import initialize_agent_manager, get_agent_manager
from defame.core.mcp_orchestration import initialize_orchestrator
from defame.core.database import init_database, db_manager
from defame.batch.processor import initialize_batch_processor, get_batch_processor
from defame.auth.dependencies import get_current_user, require_authentication, require_permission
from defame.auth.models import User
from defame.auth.service import AuthService
from defame.security.rate_limiter import get_rate_limiter, RateLimitMiddleware
from defame.security.pii_detector import get_pii_detector
from defame.web.routes import web_router, setup_static_files
from defame.utils.logger import get_logger, setup_logging, set_correlation_id, clear_correlation_id
from config.globals import ClaimType, Priority, Verdict, get_config
import uuid

# Setup logging
setup_logging()
logger = get_logger(__name__)

# FastAPI app
app = FastAPI(
    title="VeriSphere API",
    description="Multi-Agent Misinformation Detection System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
system_initialized = False


# Pydantic models
class ClaimRequest(BaseModel):
    content: str = Field(..., description="Claim content to verify", min_length=1, max_length=10000)
    claim_type: str = Field(default="text", description="Type of claim", regex="^(text|image|multimodal)$")
    priority: str = Field(default="normal", description="Claim priority", regex="^(low|normal|high|critical)$")
    source: Optional[str] = Field(None, description="Source of the claim", max_length=500)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class ClaimResponse(BaseModel):
    claim_id: str
    status: str
    message: str
    estimated_completion: Optional[str] = None


class VerificationResult(BaseModel):
    claim_id: str
    claim_content: str
    verdict: str
    confidence: float
    processing_time: float
    agents_used: int
    evidence_count: int
    pipeline_complete: bool
    agent_results: List[Dict[str, Any]]
    errors: List[str]
    timestamp: str


class SystemStatus(BaseModel):
    status: str
    uptime: float
    agents: Dict[str, Any]
    database: Dict[str, Any]
    active_pipelines: int
    total_claims_processed: int


# Dependency for correlation ID
async def get_correlation_id():
    correlation_id = str(uuid.uuid4())
    set_correlation_id(correlation_id)
    try:
        yield correlation_id
    finally:
        clear_correlation_id()


# System initialization
async def initialize_system():
    """Initialize all system components"""
    global system_initialized
    
    if system_initialized:
        return True
    
    try:
        logger.info("Initializing VeriSphere API system")
        config = get_config()
        
        # Initialize database
        init_database()
        
        # Initialize authentication system
        AuthService.initialize_default_data()
        
        # Create default admin user if not exists
        try:
            AuthService.create_admin_user()
        except Exception as e:
            logger.info(f"Admin user already exists or creation failed: {e}")
        
        # Initialize security components
        rate_limiter = await get_rate_limiter()
        pii_detector = get_pii_detector({
            'enabled_types': ['email', 'phone', 'ssn', 'credit_card', 'name'],
            'min_confidence': 0.7
        })
        
        # Add rate limiting middleware
        app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)
        
        # Initialize components
        agent_manager = initialize_agent_manager({
            'ml_agent': {
                'enabled': True,
                'huggingface_api_key': config.secret_key,
                'confidence_threshold': 0.7
            },
            'wikipedia_agent': {
                'enabled': True,
                'max_search_results': 10
            },
            'coherence_agent': {
                'enabled': True,
                'consistency_threshold': 0.6
            },
            'webscrape_agent': {
                'enabled': True,
                'max_pages_per_claim': 5
            }
        })
        
        orchestrator = initialize_orchestrator({
            'consensus_threshold': 0.6,
            'min_agents_required': 2,
            'max_agents_per_claim': 4
        })
        
        pipeline_controller = initialize_pipeline_controller({
            'max_processing_time': 300,
            'enable_checkpoints': True,
            'checkpoint_interval': 30
        })
        
        batch_processor = initialize_batch_processor({
            'max_concurrent_jobs': 3,
            'max_concurrent_claims': 10,
            'job_timeout_hours': 24
        })
        
        system_initialized = True
        logger.info("VeriSphere API system initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        return False


# Include web routes
app.include_router(web_router)

# Setup static files
setup_static_files(app)

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    success = await initialize_system()
    if not success:
        logger.error("Failed to initialize system on startup")
        # Could exit here in production
    else:
        logger.info("VeriSphere API server started successfully")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        logger.info("Shutting down VeriSphere API server")
        
        # Shutdown components
        pipeline = get_pipeline_controller()
        if pipeline:
            await pipeline.shutdown()
        
        agent_manager = get_agent_manager()
        if agent_manager:
            await agent_manager.shutdown()
        
        logger.info("VeriSphere API server shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Health check endpoint
@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint"""
    try:
        # Check database
        db_healthy = db_manager.health_check()
        
        # Check agents
        agent_manager = get_agent_manager()
        agents_healthy = True
        if agent_manager:
            agent_status = agent_manager.get_agent_status()
            agents_healthy = agent_status['healthy_agents'] > 0
        
        if db_healthy and agents_healthy and system_initialized:
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="System not healthy"
            )
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )


# System status endpoint
@app.get("/api/v1/status", response_model=SystemStatus)
async def get_system_status(correlation_id: str = Depends(get_correlation_id)):
    """Get detailed system status"""
    try:
        if not system_initialized:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="System not initialized"
            )
        
        # Get agent status
        agent_manager = get_agent_manager()
        agent_status = agent_manager.get_agent_status() if agent_manager else {}
        
        # Get database stats
        db_stats = db_manager.get_stats()
        
        # Get pipeline status
        pipeline = get_pipeline_controller()
        active_pipelines = len(pipeline.get_active_pipelines()) if pipeline else 0
        
        return SystemStatus(
            status="operational" if system_initialized else "initializing",
            uptime=0.0,  # Could track actual uptime
            agents=agent_status,
            database=db_stats,
            active_pipelines=active_pipelines,
            total_claims_processed=db_stats.get('total_claims', 0)
        )
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system status: {str(e)}"
        )


# Submit claim for verification
@app.post("/api/v1/claims", response_model=ClaimResponse)
async def submit_claim(
    request: ClaimRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[User] = Depends(get_current_user),
    correlation_id: str = Depends(get_correlation_id)
):
    """Submit a claim for verification"""
    try:
        if not system_initialized:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="System not initialized"
            )
        
        # Check PII in content
        pii_detector = get_pii_detector()
        pii_analysis = pii_detector.analyze_pii_risk(request.content)
        
        if pii_analysis['risk_level'] in ['HIGH', 'MEDIUM']:
            # Redact PII
            redacted_content, pii_matches = pii_detector.redact_pii(request.content)
            logger.warning(f"PII detected and redacted in claim submission", 
                         risk_level=pii_analysis['risk_level'],
                         pii_types=pii_analysis['pii_types_found'])
            content = redacted_content
        else:
            content = request.content
        
        # Create claim object
        claim = Claim(
            content=content,
            claim_type=ClaimType(request.claim_type),
            priority=Priority(request.priority),
            source=request.source,
            metadata={
                **(request.metadata or {}),
                'submitted_by': str(current_user.id) if current_user else 'anonymous',
                'pii_analysis': pii_analysis
            }
        )
        
        logger.info(f"Claim submitted for verification: {claim.id}")
        
        # Start background processing
        pipeline = get_pipeline_controller()
        if not pipeline:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Pipeline controller not available"
            )
        
        # Add to background tasks
        background_tasks.add_task(process_claim_background, claim)
        
        return ClaimResponse(
            claim_id=claim.id,
            status="accepted",
            message="Claim accepted for verification",
            estimated_completion=(datetime.utcnow().isoformat() if claim.priority == Priority.CRITICAL 
                                else None)
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Failed to submit claim: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit claim: {str(e)}"
        )


async def process_claim_background(claim: Claim):
    """Background task to process claim"""
    try:
        pipeline = get_pipeline_controller()
        if pipeline:
            result = await pipeline.process_claim(claim)
            logger.info(f"Background processing completed for claim {claim.id}")
    except Exception as e:
        logger.error(f"Background processing failed for claim {claim.id}: {e}")


# Get claim status
@app.get("/api/v1/claims/{claim_id}/status")
async def get_claim_status(
    claim_id: str,
    correlation_id: str = Depends(get_correlation_id)
):
    """Get claim verification status"""
    try:
        pipeline = get_pipeline_controller()
        if not pipeline:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Pipeline controller not available"
            )
        
        status_info = await pipeline.get_pipeline_status(claim_id)
        if not status_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Claim {claim_id} not found"
            )
        
        return status_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get claim status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get claim status: {str(e)}"
        )


# Get claim results
@app.get("/api/v1/claims/{claim_id}/results", response_model=VerificationResult)
async def get_claim_results(
    claim_id: str,
    correlation_id: str = Depends(get_correlation_id)
):
    """Get claim verification results"""
    try:
        # This would typically query the database for completed results
        # For now, return a placeholder response
        
        # Check if claim exists and is completed
        pipeline = get_pipeline_controller()
        if pipeline:
            status_info = await pipeline.get_pipeline_status(claim_id)
            if not status_info:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Claim {claim_id} not found"
                )
            
            if status_info.get('status') != 'completed':
                raise HTTPException(
                    status_code=status.HTTP_202_ACCEPTED,
                    detail=f"Claim {claim_id} is still processing"
                )
        
        # Return placeholder result (in production, query database)
        return VerificationResult(
            claim_id=claim_id,
            claim_content="Sample claim content",
            verdict="inconclusive",
            confidence=0.5,
            processing_time=30.0,
            agents_used=3,
            evidence_count=5,
            pipeline_complete=True,
            agent_results=[],
            errors=[],
            timestamp=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get claim results: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get claim results: {str(e)}"
        )


# Cancel claim processing
@app.delete("/api/v1/claims/{claim_id}")
async def cancel_claim(
    claim_id: str,
    correlation_id: str = Depends(get_correlation_id)
):
    """Cancel claim processing"""
    try:
        pipeline = get_pipeline_controller()
        if not pipeline:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Pipeline controller not available"
            )
        
        success = await pipeline.cancel_pipeline(claim_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Claim {claim_id} not found or already completed"
            )
        
        return {"message": f"Claim {claim_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel claim: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel claim: {str(e)}"
        )


# List active claims
@app.get("/api/v1/claims/active")
async def get_active_claims(correlation_id: str = Depends(get_correlation_id)):
    """Get list of active claims"""
    try:
        pipeline = get_pipeline_controller()
        if not pipeline:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Pipeline controller not available"
            )
        
        active_pipelines = pipeline.get_active_pipelines()
        return {
            "active_claims": len(active_pipelines),
            "claims": active_pipelines
        }
        
    except Exception as e:
        logger.error(f"Failed to get active claims: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get active claims: {str(e)}"
        )


# Batch processing endpoints
class BatchJobRequest(BaseModel):
    name: str = Field(..., description="Job name")
    description: str = Field(default="", description="Job description")
    claims: List[Dict[str, Any]] = Field(..., description="List of claims to process")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

@app.post("/api/v1/batch/jobs")
async def create_batch_job(
    request: BatchJobRequest,
    current_user: User = Depends(require_permission("api.batch")),
    correlation_id: str = Depends(get_correlation_id)
):
    """Create a new batch processing job"""
    try:
        batch_processor = get_batch_processor()
        if not batch_processor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Batch processor not available"
            )
        
        job = await batch_processor.create_batch_job(
            name=request.name,
            claims_data=request.claims,
            description=request.description,
            metadata={
                **request.metadata,
                'created_by': str(current_user.id)
            }
        )
        
        # Start processing
        await batch_processor.start_batch_job(job.id)
        
        return {"job_id": job.id, "status": "started", "total_claims": job.total_claims}
        
    except Exception as e:
        logger.error(f"Failed to create batch job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/v1/batch/jobs/{job_id}")
async def get_batch_job(
    job_id: str,
    current_user: User = Depends(require_permission("api.batch")),
    correlation_id: str = Depends(get_correlation_id)
):
    """Get batch job status"""
    try:
        batch_processor = get_batch_processor()
        if not batch_processor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Batch processor not available"
            )
        
        job = batch_processor.get_batch_job(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Batch job not found"
            )
        
        return job.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get batch job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/v1/batch/jobs")
async def list_batch_jobs(
    status_filter: Optional[str] = None,
    limit: int = 50,
    current_user: User = Depends(require_permission("api.batch")),
    correlation_id: str = Depends(get_correlation_id)
):
    """List batch jobs"""
    try:
        batch_processor = get_batch_processor()
        if not batch_processor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Batch processor not available"
            )
        
        from defame.batch.processor import BatchStatus
        status_enum = BatchStatus(status_filter) if status_filter else None
        
        jobs = batch_processor.list_batch_jobs(status=status_enum, limit=limit)
        
        return {
            "jobs": [job.to_dict() for job in jobs],
            "total": len(jobs)
        }
        
    except Exception as e:
        logger.error(f"Failed to list batch jobs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.delete("/api/v1/batch/jobs/{job_id}")
async def cancel_batch_job(
    job_id: str,
    current_user: User = Depends(require_permission("api.batch")),
    correlation_id: str = Depends(get_correlation_id)
):
    """Cancel a batch job"""
    try:
        batch_processor = get_batch_processor()
        if not batch_processor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Batch processor not available"
            )
        
        success = await batch_processor.cancel_batch_job(job_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Batch job not found or cannot be cancelled"
            )
        
        return {"message": "Batch job cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel batch job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# File upload endpoint
@app.post("/api/v1/claims/upload")
async def upload_claim_file(
    file: UploadFile = File(...),
    claim_type: str = "image",
    priority: str = "normal",
    source: Optional[str] = None,
    current_user: Optional[User] = Depends(get_current_user),
    correlation_id: str = Depends(get_correlation_id)
):
    """Upload file for claim verification"""
    try:
        if not system_initialized:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="System not initialized"
            )
        
        # Validate file
        if file.size > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File too large"
            )
        
        # Read file content
        content = await file.read()
        
        # Create claim
        claim = Claim(
            content=content,
            claim_type=ClaimType(claim_type),
            priority=Priority(priority),
            source=source,
            metadata={
                'filename': file.filename,
                'content_type': file.content_type,
                'file_size': len(content),
                'submitted_by': str(current_user.id) if current_user else 'anonymous'
            }
        )
        
        # Process claim
        pipeline = get_pipeline_controller()
        if not pipeline:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Pipeline controller not available"
            )
        
        # Start background processing
        asyncio.create_task(process_claim_background(claim))
        
        return ClaimResponse(
            claim_id=claim.id,
            status="accepted",
            message="File uploaded and claim accepted for verification"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# PII analysis endpoint
@app.post("/api/v1/pii/analyze")
async def analyze_pii(
    request: Dict[str, str],
    current_user: User = Depends(require_permission("system.admin")),
    correlation_id: str = Depends(get_correlation_id)
):
    """Analyze text for PII"""
    try:
        text = request.get("text", "")
        if not text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text is required"
            )
        
        pii_detector = get_pii_detector()
        analysis = pii_detector.analyze_pii_risk(text)
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PII analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Rate limit info endpoint
@app.get("/api/v1/rate-limit/info")
async def get_rate_limit_info(
    request: Request,
    current_user: Optional[User] = Depends(get_current_user),
    correlation_id: str = Depends(get_correlation_id)
):
    """Get rate limit information for current user/IP"""
    try:
        rate_limiter = await get_rate_limiter()
        
        # Get identifier
        if current_user:
            identifier = f"user:{current_user.id}"
            rate_limit_type = "per_user"
        else:
            identifier = f"ip:{request.client.host}"
            rate_limit_type = "per_ip"
        
        from defame.security.rate_limiter import RateLimitType
        info = await rate_limiter.get_rate_limit_info(
            identifier, "api_general", RateLimitType(rate_limit_type)
        )
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to get rate limit info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics endpoint"""
    try:
        agent_manager = get_agent_manager()
        batch_processor = get_batch_processor()
        rate_limiter = await get_rate_limiter()
        
        prometheus_metrics = []
        
        if agent_manager:
            metrics = agent_manager.get_system_metrics()
            prometheus_metrics.extend([
                f"verisphere_claims_processed_total {metrics.claims_processed}",
                f"verisphere_average_processing_time_seconds {metrics.average_processing_time}",
                f"verisphere_accuracy_score {metrics.accuracy_score}",
                f"verisphere_error_rate {metrics.error_rate}",
                f"verisphere_throughput_per_minute {metrics.throughput_per_minute}"
            ])
            
            # Agent status
            agent_status = agent_manager.get_agent_status()
            prometheus_metrics.extend([
                f"verisphere_total_agents {agent_status['total_agents']}",
                f"verisphere_healthy_agents {agent_status['healthy_agents']}"
            ])
        
        if batch_processor:
            batch_stats = batch_processor.get_statistics()
            prometheus_metrics.extend([
                f"verisphere_batch_jobs_total {batch_stats['total_jobs']}",
                f"verisphere_batch_jobs_active {batch_stats['active_jobs']}",
                f"verisphere_batch_jobs_completed {batch_stats['completed_jobs']}",
                f"verisphere_batch_jobs_failed {batch_stats['failed_jobs']}"
            ])
        
        if rate_limiter:
            rate_stats = await rate_limiter.get_statistics()
            prometheus_metrics.extend([
                f"verisphere_rate_limiter_cache_entries {rate_stats['local_cache_entries']}",
                f"verisphere_rate_limiter_configured_limits {rate_stats['configured_limits']}"
            ])
        
        return "\n".join(prometheus_metrics) if prometheus_metrics else "# No metrics available"
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return f"# Error getting metrics: {str(e)}"


def main():
    """Main function to run the API server"""
    config = get_config()
    
    uvicorn.run(
        "scripts.run_api:app",
        host=config.api_host,
        port=config.api_port,
        workers=1,  # Single worker for now due to global state
        reload=config.debug,
        access_log=True,
        log_level="info" if not config.debug else "debug"
    )


if __name__ == "__main__":
    main()