"""
Coherence Agent for logical consistency and contradiction detection
"""
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import re

from defame.core.interfaces import BaseAgent
from defame.core.models import Claim, VerificationResult, Evidence, AgentCapability
from defame.utils.logger import get_logger
from defame.utils.helpers import sanitize_text, calculate_text_similarity
from config.globals import AgentType, ClaimType, Verdict

logger = get_logger(__name__)


class CoherenceAgent(BaseAgent):
    """Agent for logical consistency and coherence checking"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.consistency_threshold = config.get('consistency_threshold', 0.6)
        self.temporal_window_hours = config.get('temporal_window_hours', 24)
        self.contradiction_threshold = config.get('contradiction_threshold', 0.8)
        
        # Logical inconsistency patterns
        self.contradiction_patterns = [
            (r'\b(always|never|all|none|every|no)\b', r'\b(sometimes|some|few|many|most)\b'),
            (r'\b(impossible|cannot|never)\b', r'\b(possible|can|might|could)\b'),
            (r'\b(true|fact|proven|confirmed)\b', r'\b(false|lie|disproven|debunked)\b'),
            (r'\b(increase|rise|grow|more)\b', r'\b(decrease|fall|shrink|less)\b'),
        ]
    
    def _get_agent_type(self) -> AgentType:
        return AgentType.COHERENCE_AGENT
    
    async def verify_claim(self, claim: Claim, metadata: Optional[Dict] = None) -> VerificationResult:
        """Verify claim for logical consistency and coherence"""
        start_time = datetime.utcnow()
        
        try:
            claim_text = self._prepare_claim_text(claim)
            if not claim_text:
                return self._create_error_result(claim, "No text content to analyze")
            
            evidence_list = []
            
            # Perform different coherence checks
            internal_consistency = await self._check_internal_consistency(claim_text)
            if internal_consistency:
                evidence_list.append(internal_consistency)
            
            temporal_consistency = await self._check_temporal_consistency(claim_text)
            if temporal_consistency:
                evidence_list.append(temporal_consistency)
            
            logical_structure = await self._analyze_logical_structure(claim_text)
            if logical_structure:
                evidence_list.append(logical_structure)
            
            # Calculate overall assessment
            verdict, confidence, reasoning = self._calculate_assessment(claim_text, evidence_list)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return VerificationResult(
                claim_id=claim.id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                confidence=confidence,
                verdict=verdict,
                evidence=evidence_list,
                reasoning=reasoning,
                processing_time=processing_time,
                metadata={'coherence_checks': len(evidence_list)}
            )
            
        except Exception as e:
            logger.error(f"Coherence verification failed: {e}")
            return self._create_error_result(claim, str(e))
    
    def _prepare_claim_text(self, claim: Claim) -> str:
        """Prepare claim text for analysis"""
        if isinstance(claim.content, str):
            return sanitize_text(claim.content, 2000)
        return claim.metadata.get('extracted_text', '')
    
    async def _check_internal_consistency(self, text: str) -> Optional[Evidence]:
        """Check for internal contradictions within the text"""
        try:
            contradictions = []
            sentences = re.split(r'[.!?]+', text)
            
            # Check for direct contradictions using patterns
            for pattern1, pattern2 in self.contradiction_patterns:
                matches1 = re.findall(pattern1, text.lower())
                matches2 = re.findall(pattern2, text.lower())
                
                if matches1 and matches2:
                    contradictions.append({
                        'type': 'pattern_contradiction',
                        'pattern1': pattern1,
                        'pattern2': pattern2,
                        'matches1': matches1,
                        'matches2': matches2
                    })
            
            # Check for contradictory statements between sentences
            for i, sent1 in enumerate(sentences):
                for j, sent2 in enumerate(sentences[i+1:], i+1):
                    if len(sent1.strip()) > 10 and len(sent2.strip()) > 10:
                        # Simple contradiction detection based on negation
                        if self._are_contradictory(sent1, sent2):
                            contradictions.append({
                                'type': 'sentence_contradiction',
                                'sentence1': sent1.strip(),
                                'sentence2': sent2.strip()
                            })
            
            if contradictions:
                credibility_score = max(0.1, 1.0 - (len(contradictions) * 0.2))
                content = f"Internal contradictions detected: {len(contradictions)} inconsistencies found"
            else:
                credibility_score = 0.8
                content = "No internal contradictions detected"
            
            return Evidence(
                source='Coherence Analysis - Internal Consistency',
                content=content,
                credibility_score=credibility_score,
                relevance_score=0.9,
                evidence_type='internal_consistency',
                metadata={
                    'contradictions': contradictions,
                    'contradiction_count': len(contradictions)
                }
            )
            
        except Exception as e:
            logger.warning(f"Internal consistency check failed: {e}")
            return None
    
    def _are_contradictory(self, sent1: str, sent2: str) -> bool:
        """Simple contradiction detection between two sentences"""
        # Look for negation patterns
        negation_words = ['not', 'no', 'never', 'none', 'nothing', 'nobody', 'nowhere']
        
        sent1_lower = sent1.lower()
        sent2_lower = sent2.lower()
        
        # Check if one sentence has negation and they share similar content
        sent1_has_negation = any(neg in sent1_lower for neg in negation_words)
        sent2_has_negation = any(neg in sent2_lower for neg in negation_words)
        
        if sent1_has_negation != sent2_has_negation:  # One has negation, other doesn't
            # Check for content similarity
            similarity = calculate_text_similarity(sent1, sent2)
            if similarity > 0.3:  # Similar content but opposite polarity
                return True
        
        return False
    
    async def _check_temporal_consistency(self, text: str) -> Optional[Evidence]:
        """Check for temporal consistency in claims"""
        try:
            # Extract temporal references
            temporal_patterns = [
                r'\b(\d{4})\b',  # Years
                r'\b(yesterday|today|tomorrow)\b',
                r'\b(last|next|this)\s+(year|month|week|day)\b',
                r'\b(before|after|during|since|until)\b',
                r'\b(always|never|sometimes|often|rarely)\b'
            ]
            
            temporal_references = []
            for pattern in temporal_patterns:
                matches = re.findall(pattern, text.lower())
                temporal_references.extend(matches)
            
            # Simple temporal consistency check
            inconsistencies = []
            
            # Check for impossible temporal claims
            current_year = datetime.now().year
            years = [int(match) for match in re.findall(r'\b(\d{4})\b', text) 
                    if match.isdigit() and 1900 <= int(match) <= current_year + 10]
            
            if years:
                # Check for future events stated as past
                future_years = [y for y in years if y > current_year]
                if future_years and any(word in text.lower() for word in ['happened', 'occurred', 'was', 'did']):
                    inconsistencies.append("Future events described in past tense")
            
            if temporal_references:
                credibility_score = 0.7 if not inconsistencies else max(0.3, 0.7 - len(inconsistencies) * 0.2)
                content = f"Temporal analysis: {len(temporal_references)} time references found"
                if inconsistencies:
                    content += f", {len(inconsistencies)} inconsistencies detected"
            else:
                credibility_score = 0.6
                content = "No temporal references found for consistency analysis"
            
            return Evidence(
                source='Coherence Analysis - Temporal Consistency',
                content=content,
                credibility_score=credibility_score,
                relevance_score=0.7,
                evidence_type='temporal_consistency',
                metadata={
                    'temporal_references': temporal_references,
                    'inconsistencies': inconsistencies,
                    'years_mentioned': years
                }
            )
            
        except Exception as e:
            logger.warning(f"Temporal consistency check failed: {e}")
            return None
    
    async def _analyze_logical_structure(self, text: str) -> Optional[Evidence]:
        """Analyze the logical structure of the claim"""
        try:
            # Look for logical connectors and structure
            logical_connectors = {
                'causal': ['because', 'since', 'therefore', 'thus', 'hence', 'so', 'as a result'],
                'conditional': ['if', 'unless', 'provided that', 'assuming', 'suppose'],
                'contrast': ['but', 'however', 'although', 'despite', 'nevertheless', 'yet'],
                'addition': ['and', 'also', 'furthermore', 'moreover', 'additionally'],
                'temporal': ['when', 'while', 'before', 'after', 'during', 'until']
            }
            
            structure_analysis = {}
            text_lower = text.lower()
            
            for category, connectors in logical_connectors.items():
                count = sum(1 for connector in connectors if connector in text_lower)
                structure_analysis[category] = count
            
            # Analyze argument structure
            total_connectors = sum(structure_analysis.values())
            
            # Check for logical fallacies (simple patterns)
            fallacy_patterns = [
                (r'\b(all|every|always)\b.*\b(are|is)\b', 'overgeneralization'),
                (r'\b(never|no|none)\b.*\b(can|will|are|is)\b', 'absolute_statement'),
                (r'\b(everyone|everybody|nobody)\b', 'hasty_generalization'),
                (r'\b(obviously|clearly|certainly)\b', 'appeal_to_obviousness')
            ]
            
            potential_fallacies = []
            for pattern, fallacy_type in fallacy_patterns:
                if re.search(pattern, text_lower):
                    potential_fallacies.append(fallacy_type)
            
            # Calculate credibility based on logical structure
            base_credibility = 0.6
            
            if total_connectors > 0:
                base_credibility += 0.1  # Has logical structure
            
            if structure_analysis['causal'] > 0:
                base_credibility += 0.05  # Has causal reasoning
            
            if len(potential_fallacies) > 0:
                base_credibility -= len(potential_fallacies) * 0.1  # Penalize fallacies
            
            credibility_score = max(0.1, min(1.0, base_credibility))
            
            content = f"Logical structure analysis: {total_connectors} logical connectors found"
            if potential_fallacies:
                content += f", {len(potential_fallacies)} potential logical fallacies detected"
            
            return Evidence(
                source='Coherence Analysis - Logical Structure',
                content=content,
                credibility_score=credibility_score,
                relevance_score=0.8,
                evidence_type='logical_structure',
                metadata={
                    'structure_analysis': structure_analysis,
                    'total_connectors': total_connectors,
                    'potential_fallacies': potential_fallacies
                }
            )
            
        except Exception as e:
            logger.warning(f"Logical structure analysis failed: {e}")
            return None
    
    def _calculate_assessment(self, claim_text: str, evidence_list: List[Evidence]) -> tuple[Verdict, float, str]:
        """Calculate overall coherence assessment"""
        if not evidence_list:
            return Verdict.INCONCLUSIVE, 0.3, "No coherence analysis results available"
        
        # Calculate average credibility
        avg_credibility = sum(e.credibility_score for e in evidence_list) / len(evidence_list)
        
        # Check for major inconsistencies
        internal_evidence = [e for e in evidence_list if e.evidence_type == 'internal_consistency']
        has_contradictions = False
        
        if internal_evidence:
            contradiction_count = internal_evidence[0].metadata.get('contradiction_count', 0)
            has_contradictions = contradiction_count > 0
        
        # Determine verdict
        if has_contradictions:
            verdict = Verdict.FALSE if avg_credibility < 0.4 else Verdict.MISLEADING
            reasoning = f"Logical inconsistencies detected in claim structure"
        elif avg_credibility > 0.7:
            verdict = Verdict.INCONCLUSIVE  # Coherent but doesn't prove truth
            reasoning = f"Claim shows logical coherence and consistency"
        else:
            verdict = Verdict.INCONCLUSIVE
            reasoning = f"Mixed coherence indicators found"
        
        confidence = avg_credibility
        
        return verdict, confidence, reasoning
    
    def _create_error_result(self, claim: Claim, error_message: str) -> VerificationResult:
        """Create error result"""
        return VerificationResult(
            claim_id=claim.id,
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            confidence=0.0,
            verdict=Verdict.INCONCLUSIVE,
            reasoning=f"Coherence Agent error: {error_message}",
            processing_time=0.0
        )
    
    def get_capabilities(self) -> AgentCapability:
        """Return agent capabilities"""
        return AgentCapability(
            agent_type=self.agent_type,
            supported_claim_types=[ClaimType.TEXT, ClaimType.MULTIMODAL],
            confidence_range=(0.1, 1.0),
            processing_time_estimate=10.0,
            description="Logical consistency checking, contradiction detection, and coherence analysis"
        )