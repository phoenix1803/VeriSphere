"""
MCP (Model Context Protocol) Orchestration Engine for intelligent agent coordination
"""
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import statistics

from defame.core.interfaces import BaseOrchestrator
from defame.core.models import Claim, VerificationResult, PipelineState, Evidence
from defame.core.agent_manager import get_agent_manager
from defame.utils.logger import get_logger, audit_logger
from defame.utils.helpers import calculate_weighted_average
from config.globals import Verdict, PipelineStage, Priority

logger = get_logger(__name__)


class MCPOrchestrator(BaseOrchestrator):
    """MCP-based orchestration engine for intelligent agent coordination"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.consensus_threshold = config.get('consensus_threshold', 0.6)
        self.min_agents_required = config.get('min_agents_required', 2)
        self.max_agents_per_claim = config.get('max_agents_per_claim', 4)
        self.confidence_weights = config.get('confidence_weights', {
            'ml_agent': 0.3,
            'wikipedia_agent': 0.25,
            'coherence_agent': 0.2,
            'webscrape_agent': 0.25
        })
        
        # Verdict priority for conflict resolution
        self.verdict_priority = {
            Verdict.FALSE: 4,
            Verdict.MISLEADING: 3,
            Verdict.INCONCLUSIVE: 2,
            Verdict.TRUE: 1
        }
    
    async def orchestrate_verification(self, claim: Claim) -> PipelineState:
        """Orchestrate the complete verification process"""
        try:
            logger.info(f"Starting MCP orchestration for claim {claim.id}")
            
            # Initialize pipeline state
            state = PipelineState(
                claim_id=claim.id,
                current_stage=PipelineStage.DETECTION,
                start_time=datetime.utcnow()
            )
            
            # Stage 1: Detection and Analysis
            state = await self._detection_stage(claim, state)
            
            # Stage 2: Agent Selection and Coordination
            state = await self._agent_coordination_stage(claim, state)
            
            # Stage 3: Evidence Aggregation
            state = await self._evidence_aggregation_stage(claim, state)
            
            # Stage 4: Consensus Building
            state = await self._consensus_building_stage(claim, state)
            
            # Stage 5: Final Assessment
            state = await self._final_assessment_stage(claim, state)
            
            # Complete the pipeline
            state.current_stage = PipelineStage.EVALUATION
            state.end_time = datetime.utcnow()
            
            logger.info(
                f"MCP orchestration completed",
                claim_id=claim.id,
                final_verdict=state.overall_verdict.value,
                confidence=state.overall_confidence,
                processing_time=state.processing_time
            )
            
            return state
            
        except Exception as e:
            logger.error(f"MCP orchestration failed for claim {claim.id}: {e}")
            state.errors.append(f"Orchestration error: {str(e)}")
            state.current_stage = PipelineStage.EVALUATION
            state.end_time = datetime.utcnow()
            return state
    
    async def _detection_stage(self, claim: Claim, state: PipelineState) -> PipelineState:
        """Stage 1: Claim detection and initial analysis"""
        try:
            state.current_stage = PipelineStage.DETECTION
            
            # Analyze claim characteristics
            claim_analysis = await self._analyze_claim_characteristics(claim)
            state.stage_results[PipelineStage.DETECTION] = claim_analysis
            
            # Estimate processing complexity
            complexity = self._estimate_processing_complexity(claim, claim_analysis)
            state.metadata['processing_complexity'] = complexity
            
            # Set estimated completion time
            base_time = 30  # Base processing time in seconds
            complexity_multiplier = {'low': 1.0, 'medium': 1.5, 'high': 2.0}.get(complexity, 1.5)
            estimated_seconds = base_time * complexity_multiplier
            state.estimated_completion = datetime.utcnow().replace(
                second=int(datetime.utcnow().second + estimated_seconds)
            )
            
            audit_logger.log_pipeline_stage(
                claim.id, 
                PipelineStage.DETECTION.value, 
                "completed", 
                0.0
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Detection stage failed: {e}")
            state.errors.append(f"Detection stage error: {str(e)}")
            return state
    
    async def _analyze_claim_characteristics(self, claim: Claim) -> Dict[str, Any]:
        """Analyze claim characteristics for agent selection"""
        try:
            characteristics = {
                'claim_type': claim.claim_type.value,
                'content_length': len(str(claim.content)),
                'priority': claim.priority.value,
                'has_urls': False,
                'has_numbers': False,
                'has_dates': False,
                'language': 'en',  # Default, could be detected
                'complexity_indicators': []
            }
            
            if isinstance(claim.content, str):
                content = claim.content.lower()
                
                # Check for URLs
                if 'http' in content or 'www.' in content:
                    characteristics['has_urls'] = True
                    characteristics['complexity_indicators'].append('contains_urls')
                
                # Check for numbers/statistics
                import re
                if re.search(r'\d+', content):
                    characteristics['has_numbers'] = True
                    characteristics['complexity_indicators'].append('contains_numbers')
                
                # Check for dates
                date_patterns = [r'\d{4}', r'\d{1,2}/\d{1,2}', r'january|february|march|april|may|june|july|august|september|october|november|december']
                if any(re.search(pattern, content) for pattern in date_patterns):
                    characteristics['has_dates'] = True
                    characteristics['complexity_indicators'].append('contains_dates')
                
                # Check for complex claims
                complex_indicators = ['because', 'therefore', 'however', 'although', 'despite', 'according to']
                if any(indicator in content for indicator in complex_indicators):
                    characteristics['complexity_indicators'].append('complex_reasoning')
                
                # Check for emotional language
                emotional_words = ['amazing', 'terrible', 'shocking', 'incredible', 'unbelievable', 'outrageous']
                if any(word in content for word in emotional_words):
                    characteristics['complexity_indicators'].append('emotional_language')
            
            return characteristics
            
        except Exception as e:
            logger.warning(f"Claim analysis failed: {e}")
            return {'claim_type': claim.claim_type.value, 'complexity_indicators': []}
    
    def _estimate_processing_complexity(self, claim: Claim, analysis: Dict[str, Any]) -> str:
        """Estimate processing complexity based on claim characteristics"""
        complexity_score = 0
        
        # Base complexity by claim type
        if claim.claim_type.value == 'image':
            complexity_score += 2
        elif claim.claim_type.value == 'multimodal':
            complexity_score += 3
        
        # Content length
        content_length = analysis.get('content_length', 0)
        if content_length > 500:
            complexity_score += 2
        elif content_length > 200:
            complexity_score += 1
        
        # Complexity indicators
        indicators = analysis.get('complexity_indicators', [])
        complexity_score += len(indicators)
        
        # Priority
        if claim.priority.value == 'high':
            complexity_score += 1
        elif claim.priority.value == 'critical':
            complexity_score += 2
        
        # Determine complexity level
        if complexity_score <= 2:
            return 'low'
        elif complexity_score <= 5:
            return 'medium'
        else:
            return 'high'
    
    async def _agent_coordination_stage(self, claim: Claim, state: PipelineState) -> PipelineState:
        """Stage 2: Select and coordinate agents"""
        try:
            state.current_stage = PipelineStage.EVIDENCE
            
            agent_manager = get_agent_manager()
            if not agent_manager:
                raise Exception("Agent manager not available")
            
            # Select agents based on claim characteristics
            selected_agents = await self._intelligent_agent_selection(claim, state)
            
            if not selected_agents:
                raise Exception("No suitable agents available")
            
            # Execute agents with coordination
            agent_results = await agent_manager.execute_agents(claim, selected_agents)
            state.agent_results = agent_results
            
            # Store stage results
            state.stage_results[PipelineStage.EVIDENCE] = {
                'agents_selected': len(selected_agents),
                'agents_executed': len(agent_results),
                'agent_types': [r.agent_type.value for r in agent_results]
            }
            
            audit_logger.log_pipeline_stage(
                claim.id,
                PipelineStage.EVIDENCE.value,
                "completed",
                sum(r.processing_time for r in agent_results)
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Agent coordination stage failed: {e}")
            state.errors.append(f"Agent coordination error: {str(e)}")
            return state
    
    async def _intelligent_agent_selection(self, claim: Claim, state: PipelineState) -> List:
        """Intelligently select agents based on claim characteristics and context"""
        try:
            agent_manager = get_agent_manager()
            claim_analysis = state.stage_results.get(PipelineStage.DETECTION, {})
            
            # Get all suitable agents
            all_suitable = await agent_manager.select_agents_for_claim(claim)
            
            if not all_suitable:
                return []
            
            # Apply intelligent selection based on claim characteristics
            selected_agents = []
            
            # Always include ML agent if available (primary fact-checker)
            ml_agents = [a for a in all_suitable if a.agent_type.value == 'ml_agent']
            if ml_agents:
                selected_agents.extend(ml_agents[:1])  # Take first ML agent
            
            # Select based on claim characteristics
            if claim_analysis.get('has_urls') or claim_analysis.get('complexity_indicators'):
                # Complex claims need web scraping
                webscrape_agents = [a for a in all_suitable if a.agent_type.value == 'webscrape_agent']
                if webscrape_agents:
                    selected_agents.extend(webscrape_agents[:1])
            
            # For factual claims, include Wikipedia
            if not any('emotional_language' in claim_analysis.get('complexity_indicators', [])):
                wiki_agents = [a for a in all_suitable if a.agent_type.value == 'wikipedia_agent']
                if wiki_agents:
                    selected_agents.extend(wiki_agents[:1])
            
            # For complex reasoning, include coherence agent
            if 'complex_reasoning' in claim_analysis.get('complexity_indicators', []):
                coherence_agents = [a for a in all_suitable if a.agent_type.value == 'coherence_agent']
                if coherence_agents:
                    selected_agents.extend(coherence_agents[:1])
            
            # Ensure minimum agents
            if len(selected_agents) < self.min_agents_required:
                remaining_agents = [a for a in all_suitable if a not in selected_agents]
                needed = self.min_agents_required - len(selected_agents)
                selected_agents.extend(remaining_agents[:needed])
            
            # Limit maximum agents
            selected_agents = selected_agents[:self.max_agents_per_claim]
            
            logger.info(
                f"Selected {len(selected_agents)} agents for claim {claim.id}",
                agent_types=[a.agent_type.value for a in selected_agents]
            )
            
            return selected_agents
            
        except Exception as e:
            logger.error(f"Agent selection failed: {e}")
            return []
    
    async def _evidence_aggregation_stage(self, claim: Claim, state: PipelineState) -> PipelineState:
        """Stage 3: Aggregate evidence from all agents"""
        try:
            state.current_stage = PipelineStage.FACT_CHECKING
            
            if not state.agent_results:
                state.errors.append("No agent results to aggregate")
                return state
            
            # Collect all evidence
            all_evidence = []
            for result in state.agent_results:
                all_evidence.extend(result.evidence)
            
            # Deduplicate and rank evidence
            unique_evidence = self._deduplicate_evidence(all_evidence)
            ranked_evidence = self._rank_evidence(unique_evidence)
            
            # Store aggregated evidence
            state.stage_results[PipelineStage.FACT_CHECKING] = {
                'total_evidence': len(all_evidence),
                'unique_evidence': len(unique_evidence),
                'top_evidence': ranked_evidence[:10]  # Top 10 pieces
            }
            
            audit_logger.log_pipeline_stage(
                claim.id,
                PipelineStage.FACT_CHECKING.value,
                "completed",
                0.0
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Evidence aggregation failed: {e}")
            state.errors.append(f"Evidence aggregation error: {str(e)}")
            return state
    
    def _deduplicate_evidence(self, evidence_list: List[Evidence]) -> List[Evidence]:
        """Remove duplicate evidence based on content similarity"""
        if not evidence_list:
            return []
        
        unique_evidence = []
        
        for evidence in evidence_list:
            is_duplicate = False
            
            for existing in unique_evidence:
                # Simple deduplication based on source and content similarity
                if (evidence.source == existing.source or 
                    self._calculate_content_similarity(evidence.content, existing.content) > 0.8):
                    is_duplicate = True
                    # Keep the one with higher credibility
                    if evidence.credibility_score > existing.credibility_score:
                        unique_evidence.remove(existing)
                        unique_evidence.append(evidence)
                    break
            
            if not is_duplicate:
                unique_evidence.append(evidence)
        
        return unique_evidence
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two pieces of content"""
        if not content1 or not content2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _rank_evidence(self, evidence_list: List[Evidence]) -> List[Evidence]:
        """Rank evidence by quality score"""
        def evidence_score(evidence: Evidence) -> float:
            # Combined score of credibility and relevance
            return (evidence.credibility_score * 0.6) + (evidence.relevance_score * 0.4)
        
        return sorted(evidence_list, key=evidence_score, reverse=True)
    
    async def _consensus_building_stage(self, claim: Claim, state: PipelineState) -> PipelineState:
        """Stage 4: Build consensus from agent results"""
        try:
            state.current_stage = PipelineStage.ANALYSIS
            
            if not state.agent_results:
                state.errors.append("No agent results for consensus building")
                return state
            
            # Analyze agent verdicts
            verdict_analysis = self._analyze_agent_verdicts(state.agent_results)
            
            # Calculate weighted consensus
            consensus_verdict, consensus_confidence = self._calculate_weighted_consensus(state.agent_results)
            
            # Detect and resolve conflicts
            conflict_resolution = self._resolve_verdict_conflicts(state.agent_results)
            
            # Store consensus results
            state.stage_results[PipelineStage.ANALYSIS] = {
                'verdict_analysis': verdict_analysis,
                'consensus_verdict': consensus_verdict.value,
                'consensus_confidence': consensus_confidence,
                'conflict_resolution': conflict_resolution
            }
            
            audit_logger.log_pipeline_stage(
                claim.id,
                PipelineStage.ANALYSIS.value,
                "completed",
                0.0
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Consensus building failed: {e}")
            state.errors.append(f"Consensus building error: {str(e)}")
            return state
    
    def _analyze_agent_verdicts(self, results: List[VerificationResult]) -> Dict[str, Any]:
        """Analyze distribution of agent verdicts"""
        verdict_counts = {}
        confidence_by_verdict = {}
        
        for result in results:
            verdict = result.verdict.value
            
            if verdict not in verdict_counts:
                verdict_counts[verdict] = 0
                confidence_by_verdict[verdict] = []
            
            verdict_counts[verdict] += 1
            confidence_by_verdict[verdict].append(result.confidence)
        
        # Calculate average confidence per verdict
        avg_confidence_by_verdict = {}
        for verdict, confidences in confidence_by_verdict.items():
            avg_confidence_by_verdict[verdict] = sum(confidences) / len(confidences)
        
        return {
            'verdict_counts': verdict_counts,
            'avg_confidence_by_verdict': avg_confidence_by_verdict,
            'total_agents': len(results),
            'agreement_level': max(verdict_counts.values()) / len(results) if results else 0
        }
    
    def _calculate_weighted_consensus(self, results: List[VerificationResult]) -> Tuple[Verdict, float]:
        """Calculate weighted consensus based on agent types and confidence"""
        if not results:
            return Verdict.INCONCLUSIVE, 0.0
        
        # Calculate weighted scores for each verdict
        verdict_scores = {}
        total_weight = 0
        
        for result in results:
            agent_type = result.agent_type.value
            weight = self.confidence_weights.get(agent_type, 0.25)
            verdict = result.verdict
            
            # Adjust weight by confidence
            adjusted_weight = weight * result.confidence
            
            if verdict not in verdict_scores:
                verdict_scores[verdict] = 0
            
            verdict_scores[verdict] += adjusted_weight
            total_weight += adjusted_weight
        
        if total_weight == 0:
            return Verdict.INCONCLUSIVE, 0.0
        
        # Normalize scores
        for verdict in verdict_scores:
            verdict_scores[verdict] /= total_weight
        
        # Find consensus verdict
        consensus_verdict = max(verdict_scores.keys(), key=lambda v: verdict_scores[v])
        consensus_confidence = verdict_scores[consensus_verdict]
        
        return consensus_verdict, consensus_confidence
    
    def _resolve_verdict_conflicts(self, results: List[VerificationResult]) -> Dict[str, Any]:
        """Resolve conflicts between agent verdicts"""
        if len(results) < 2:
            return {'conflicts': False, 'resolution': 'single_agent'}
        
        verdicts = [r.verdict for r in results]
        unique_verdicts = set(verdicts)
        
        if len(unique_verdicts) == 1:
            return {'conflicts': False, 'resolution': 'unanimous'}
        
        # There are conflicts - apply resolution strategy
        conflict_resolution = {
            'conflicts': True,
            'conflicting_verdicts': list(unique_verdicts),
            'resolution_strategy': 'weighted_priority'
        }
        
        # Priority-based resolution (FALSE > MISLEADING > INCONCLUSIVE > TRUE)
        # This is conservative - prefer to flag potential misinformation
        highest_priority_verdict = min(unique_verdicts, key=lambda v: self.verdict_priority[v])
        
        conflict_resolution['resolved_verdict'] = highest_priority_verdict.value
        conflict_resolution['reasoning'] = f"Resolved conflict using priority system: {highest_priority_verdict.value} has highest priority"
        
        return conflict_resolution
    
    async def _final_assessment_stage(self, claim: Claim, state: PipelineState) -> PipelineState:
        """Stage 5: Final assessment and verdict"""
        try:
            state.current_stage = PipelineStage.MITIGATION
            
            # Get consensus results
            analysis_results = state.stage_results.get(PipelineStage.ANALYSIS, {})
            
            if not analysis_results:
                state.overall_verdict = Verdict.INCONCLUSIVE
                state.overall_confidence = 0.3
                state.errors.append("No analysis results for final assessment")
                return state
            
            # Set final verdict and confidence
            consensus_verdict_str = analysis_results.get('consensus_verdict', 'inconclusive')
            state.overall_verdict = Verdict(consensus_verdict_str)
            state.overall_confidence = analysis_results.get('consensus_confidence', 0.3)
            
            # Apply final adjustments based on evidence quality
            evidence_results = state.stage_results.get(PipelineStage.FACT_CHECKING, {})
            if evidence_results:
                evidence_count = evidence_results.get('unique_evidence', 0)
                if evidence_count < 3:  # Low evidence count
                    state.overall_confidence *= 0.8  # Reduce confidence
                elif evidence_count > 10:  # High evidence count
                    state.overall_confidence = min(1.0, state.overall_confidence * 1.1)  # Boost confidence
            
            # Final confidence bounds
            state.overall_confidence = max(0.1, min(1.0, state.overall_confidence))
            
            # Store final assessment
            state.stage_results[PipelineStage.MITIGATION] = {
                'final_verdict': state.overall_verdict.value,
                'final_confidence': state.overall_confidence,
                'evidence_adjustment': evidence_results.get('unique_evidence', 0),
                'processing_complete': True
            }
            
            audit_logger.log_verification_completed(
                claim.id,
                state.overall_verdict.value,
                state.overall_confidence,
                state.processing_time
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Final assessment failed: {e}")
            state.errors.append(f"Final assessment error: {str(e)}")
            state.overall_verdict = Verdict.INCONCLUSIVE
            state.overall_confidence = 0.2
            return state
    
    def select_agents(self, claim: Claim) -> List:
        """Select appropriate agents for a claim (interface requirement)"""
        # This is handled by _intelligent_agent_selection in the orchestration flow
        # This method is kept for interface compatibility
        return []


# Global orchestrator instance
orchestrator: Optional[MCPOrchestrator] = None


def get_orchestrator() -> Optional[MCPOrchestrator]:
    """Get the global orchestrator instance"""
    return orchestrator


def initialize_orchestrator(config: Dict[str, Any]) -> MCPOrchestrator:
    """Initialize the global orchestrator"""
    global orchestrator
    orchestrator = MCPOrchestrator(config)
    return orchestrator