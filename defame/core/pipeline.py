"""
DEFAME Pipeline Controller for six-stage verification process
"""
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from defame.core.models import Claim, PipelineState, VerificationResult
from defame.core.mcp_orchestration import get_orchestrator
from defame.core.database import DatabaseSession, ClaimModel, VerificationResultModel
from defame.utils.logger import get_logger, audit_logger
from config.globals import PipelineStage, Verdict

logger = get_logger(__name__)


class PipelineController:
    """Controller for the six-stage DEFAME verification pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_processing_time = config.get('max_processing_time', 300)  # 5 minutes
        self.checkpoint_interval = config.get('checkpoint_interval', 30)  # 30 seconds
        self.enable_checkpoints = config.get('enable_checkpoints', True)
        
        # Pipeline stage timeouts
        self.stage_timeouts = {
            PipelineStage.DETECTION: 10,
            PipelineStage.EVIDENCE: 120,
            PipelineStage.FACT_CHECKING: 60,
            PipelineStage.ANALYSIS: 30,
            PipelineStage.MITIGATION: 20,
            PipelineStage.EVALUATION: 10
        }
        
        # Active pipeline states
        self.active_pipelines: Dict[str, PipelineState] = {}
    
    async def process_claim(self, claim: Claim, metadata: Optional[Dict] = None) -> PipelineState:
        """Process a claim through the complete DEFAME pipeline"""
        try:
            logger.info(f"Starting pipeline processing for claim {claim.id}")
            
            # Initialize pipeline state
            state = PipelineState(
                claim_id=claim.id,
                current_stage=PipelineStage.DETECTION,
                start_time=datetime.utcnow(),
                metadata=metadata or {}
            )
            
            # Store in active pipelines
            self.active_pipelines[claim.id] = state
            
            # Save initial state to database
            await self._save_claim_to_database(claim)
            
            # Start checkpoint monitoring if enabled
            checkpoint_task = None
            if self.enable_checkpoints:
                checkpoint_task = asyncio.create_task(self._checkpoint_monitor(claim.id))
            
            try:
                # Execute pipeline with timeout
                state = await asyncio.wait_for(
                    self._execute_pipeline(claim, state),
                    timeout=self.max_processing_time
                )
                
            except asyncio.TimeoutError:
                logger.error(f"Pipeline timeout for claim {claim.id}")
                state.errors.append(f"Pipeline processing timed out after {self.max_processing_time} seconds")
                state.overall_verdict = Verdict.INCONCLUSIVE
                state.overall_confidence = 0.2
                state.end_time = datetime.utcnow()
            
            finally:
                # Cancel checkpoint monitoring
                if checkpoint_task:
                    checkpoint_task.cancel()
                    try:
                        await checkpoint_task
                    except asyncio.CancelledError:
                        pass
            
            # Save final results to database
            await self._save_results_to_database(state)
            
            # Remove from active pipelines
            self.active_pipelines.pop(claim.id, None)
            
            logger.info(
                f"Pipeline processing completed for claim {claim.id}",
                verdict=state.overall_verdict.value,
                confidence=state.overall_confidence,
                processing_time=state.processing_time
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Pipeline processing failed for claim {claim.id}: {e}")
            
            # Create error state
            error_state = PipelineState(
                claim_id=claim.id,
                current_stage=PipelineStage.EVALUATION,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                overall_verdict=Verdict.INCONCLUSIVE,
                overall_confidence=0.1,
                errors=[f"Pipeline error: {str(e)}"]
            )
            
            # Clean up
            self.active_pipelines.pop(claim.id, None)
            
            return error_state
    
    async def _execute_pipeline(self, claim: Claim, state: PipelineState) -> PipelineState:
        """Execute the complete pipeline stages"""
        try:
            orchestrator = get_orchestrator()
            if not orchestrator:
                raise Exception("MCP Orchestrator not available")
            
            # Execute orchestrated verification
            state = await orchestrator.orchestrate_verification(claim)
            
            # Ensure pipeline is marked as complete
            if state.current_stage != PipelineStage.EVALUATION:
                state.current_stage = PipelineStage.EVALUATION
                state.end_time = datetime.utcnow()
            
            return state
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            state.errors.append(f"Pipeline execution error: {str(e)}")
            state.current_stage = PipelineStage.EVALUATION
            state.end_time = datetime.utcnow()
            return state
    
    async def _checkpoint_monitor(self, claim_id: str):
        """Monitor pipeline progress and create checkpoints"""
        try:
            while claim_id in self.active_pipelines:
                await asyncio.sleep(self.checkpoint_interval)
                
                state = self.active_pipelines.get(claim_id)
                if state:
                    await self._create_checkpoint(state)
                    
                    # Check for stage timeouts
                    stage_duration = (datetime.utcnow() - state.start_time).total_seconds()
                    stage_timeout = self.stage_timeouts.get(state.current_stage, 60)
                    
                    if stage_duration > stage_timeout:
                        logger.warning(
                            f"Stage timeout warning for claim {claim_id}",
                            stage=state.current_stage.value,
                            duration=stage_duration,
                            timeout=stage_timeout
                        )
                        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Checkpoint monitoring failed for claim {claim_id}: {e}")
    
    async def _create_checkpoint(self, state: PipelineState):
        """Create a checkpoint of the current pipeline state"""
        try:
            checkpoint_data = {
                'claim_id': state.claim_id,
                'current_stage': state.current_stage.value,
                'progress': self._calculate_progress(state),
                'timestamp': datetime.utcnow().isoformat(),
                'agent_results_count': len(state.agent_results),
                'errors_count': len(state.errors)
            }
            
            # Log checkpoint
            logger.debug(f"Pipeline checkpoint for claim {state.claim_id}", **checkpoint_data)
            
            # Could save to database or cache for recovery
            # For now, just log the checkpoint
            
        except Exception as e:
            logger.warning(f"Failed to create checkpoint: {e}")
    
    def _calculate_progress(self, state: PipelineState) -> float:
        """Calculate pipeline progress percentage"""
        stage_weights = {
            PipelineStage.DETECTION: 0.1,
            PipelineStage.EVIDENCE: 0.3,
            PipelineStage.FACT_CHECKING: 0.2,
            PipelineStage.ANALYSIS: 0.2,
            PipelineStage.MITIGATION: 0.1,
            PipelineStage.EVALUATION: 0.1
        }
        
        completed_weight = 0.0
        for stage, weight in stage_weights.items():
            if stage in state.stage_results or state.current_stage.value > stage.value:
                completed_weight += weight
            elif state.current_stage == stage:
                completed_weight += weight * 0.5  # Partial completion
        
        return min(1.0, completed_weight)
    
    async def _save_claim_to_database(self, claim: Claim):
        """Save claim to database"""
        try:
            with DatabaseSession() as db:
                db_claim = ClaimModel(
                    id=claim.id,
                    content=str(claim.content),
                    claim_type=claim.claim_type.value,
                    source=claim.source,
                    priority=claim.priority.value,
                    status='processing',
                    metadata=claim.metadata
                )
                db.add(db_claim)
                
        except Exception as e:
            logger.error(f"Failed to save claim to database: {e}")
    
    async def _save_results_to_database(self, state: PipelineState):
        """Save pipeline results to database"""
        try:
            with DatabaseSession() as db:
                # Update claim status
                claim = db.query(ClaimModel).filter(ClaimModel.id == state.claim_id).first()
                if claim:
                    claim.status = 'completed' if state.is_complete else 'failed'
                    claim.updated_at = datetime.utcnow()
                
                # Save agent results
                for result in state.agent_results:
                    db_result = VerificationResultModel(
                        id=result.id,
                        claim_id=result.claim_id,
                        agent_id=result.agent_id,
                        agent_type=result.agent_type.value,
                        confidence=result.confidence,
                        verdict=result.verdict.value,
                        reasoning=result.reasoning,
                        processing_time=result.processing_time,
                        evidence=[e.to_dict() for e in result.evidence],
                        metadata=result.metadata
                    )
                    db.add(db_result)
                
        except Exception as e:
            logger.error(f"Failed to save results to database: {e}")
    
    async def get_pipeline_status(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a pipeline"""
        try:
            state = self.active_pipelines.get(claim_id)
            if not state:
                # Check database for completed pipelines
                with DatabaseSession() as db:
                    claim = db.query(ClaimModel).filter(ClaimModel.id == claim_id).first()
                    if claim:
                        return {
                            'claim_id': claim_id,
                            'status': claim.status,
                            'completed': claim.status in ['completed', 'failed'],
                            'updated_at': claim.updated_at.isoformat()
                        }
                return None
            
            return {
                'claim_id': claim_id,
                'current_stage': state.current_stage.value,
                'progress': self._calculate_progress(state),
                'processing_time': state.processing_time,
                'estimated_completion': state.estimated_completion.isoformat() if state.estimated_completion else None,
                'agent_results_count': len(state.agent_results),
                'errors_count': len(state.errors),
                'status': 'processing'
            }
            
        except Exception as e:
            logger.error(f"Failed to get pipeline status: {e}")
            return None
    
    async def cancel_pipeline(self, claim_id: str) -> bool:
        """Cancel an active pipeline"""
        try:
            if claim_id in self.active_pipelines:
                state = self.active_pipelines[claim_id]
                state.errors.append("Pipeline cancelled by user")
                state.current_stage = PipelineStage.EVALUATION
                state.end_time = datetime.utcnow()
                state.overall_verdict = Verdict.INCONCLUSIVE
                state.overall_confidence = 0.1
                
                # Save cancelled state
                await self._save_results_to_database(state)
                
                # Remove from active pipelines
                self.active_pipelines.pop(claim_id, None)
                
                logger.info(f"Pipeline cancelled for claim {claim_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel pipeline: {e}")
            return False
    
    async def resume_pipeline(self, claim_id: str) -> Optional[PipelineState]:
        """Resume a pipeline from checkpoint (if supported)"""
        try:
            # This would load checkpoint data and resume processing
            # For now, return None (not implemented)
            logger.info(f"Pipeline resume requested for claim {claim_id} (not implemented)")
            return None
            
        except Exception as e:
            logger.error(f"Failed to resume pipeline: {e}")
            return None
    
    def get_active_pipelines(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all active pipelines"""
        active_status = {}
        
        for claim_id, state in self.active_pipelines.items():
            active_status[claim_id] = {
                'current_stage': state.current_stage.value,
                'progress': self._calculate_progress(state),
                'processing_time': state.processing_time,
                'agent_results_count': len(state.agent_results),
                'errors_count': len(state.errors)
            }
        
        return active_status
    
    async def shutdown(self):
        """Shutdown pipeline controller and cleanup"""
        try:
            logger.info("Shutting down pipeline controller")
            
            # Cancel all active pipelines
            for claim_id in list(self.active_pipelines.keys()):
                await self.cancel_pipeline(claim_id)
            
            logger.info("Pipeline controller shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during pipeline shutdown: {e}")


# Global pipeline controller instance
pipeline_controller: Optional[PipelineController] = None


def get_pipeline_controller() -> Optional[PipelineController]:
    """Get the global pipeline controller instance"""
    return pipeline_controller


def initialize_pipeline_controller(config: Dict[str, Any]) -> PipelineController:
    """Initialize the global pipeline controller"""
    global pipeline_controller
    pipeline_controller = PipelineController(config)
    return pipeline_controller