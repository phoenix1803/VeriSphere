"""
ML Agent for deep learning-based claim verification
"""
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import re

from defame.core.interfaces import BaseAgent
from defame.core.models import Claim, VerificationResult, Evidence, AgentCapability
from defame.utils.logger import get_logger
from defame.utils.helpers import retry_async, sanitize_text, calculate_text_similarity
from config.globals import AgentType, ClaimType, Verdict

logger = get_logger(__name__)


class MLAgent(BaseAgent):
    """Machine Learning agent for claim verification using transformer models"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.huggingface_api_key = config.get('huggingface_api_key')
        self.base_url = config.get('base_url', 'https://api-inference.huggingface.co')
        self.timeout = config.get('timeout', 60)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.max_sequence_length = config.get('max_sequence_length', 512)
        
        # Model configurations
        self.models = {
            'fact_check': config.get('fact_check_model', 'facebook/bart-large-mnli'),
            'sentiment': config.get('sentiment_model', 'cardiffnlp/twitter-roberta-base-sentiment-latest'),
            'ner': config.get('ner_model', 'dbmdz/bert-large-cased-finetuned-conll03-english'),
            'classification': config.get('classification_model', 'microsoft/DialoGPT-medium')
        }
        
        # Fact-checking templates for MNLI
        self.fact_check_templates = [
            "This claim is factually correct.",
            "This claim is factually incorrect.",
            "This claim is misleading or partially true.",
            "This claim cannot be verified with available information."
        ]
    
    def _get_agent_type(self) -> AgentType:
        """Return agent type"""
        return AgentType.ML_AGENT
    
    async def verify_claim(self, claim: Claim, metadata: Optional[Dict] = None) -> VerificationResult:
        """Verify claim using ML models"""
        start_time = datetime.utcnow()
        
        try:
            logger.info(
                "Starting ML verification",
                claim_id=claim.id,
                claim_type=claim.claim_type.value
            )
            
            # Prepare claim text
            claim_text = self._prepare_claim_text(claim)
            if not claim_text:
                return self._create_error_result(claim, "No text content to analyze")
            
            evidence_list = []
            
            # Perform different types of ML analysis
            analyses = await asyncio.gather(
                self._fact_check_analysis(claim_text),
                self._sentiment_analysis(claim_text),
                self._entity_analysis(claim_text),
                self._linguistic_analysis(claim_text),
                return_exceptions=True
            )
            
            # Process analysis results
            fact_check_result, sentiment_result, entity_result, linguistic_result = analyses
            
            # Create evidence from each analysis
            if not isinstance(fact_check_result, Exception) and fact_check_result:
                evidence_list.append(self._create_fact_check_evidence(fact_check_result))
            
            if not isinstance(sentiment_result, Exception) and sentiment_result:
                evidence_list.append(self._create_sentiment_evidence(sentiment_result))
            
            if not isinstance(entity_result, Exception) and entity_result:
                evidence_list.append(self._create_entity_evidence(entity_result))
            
            if not isinstance(linguistic_result, Exception) and linguistic_result:
                evidence_list.append(self._create_linguistic_evidence(linguistic_result))
            
            # Calculate overall verdict and confidence
            verdict, confidence, reasoning = self._calculate_overall_assessment(
                claim_text, evidence_list, analyses
            )
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            result = VerificationResult(
                claim_id=claim.id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                confidence=confidence,
                verdict=verdict,
                evidence=[e for e in evidence_list if e],
                reasoning=reasoning,
                processing_time=processing_time,
                metadata={
                    'models_used': list(self.models.keys()),
                    'claim_length': len(claim_text),
                    'analysis_count': len([a for a in analyses if not isinstance(a, Exception)])
                }
            )
            
            logger.info(
                "ML verification completed",
                claim_id=claim.id,
                verdict=verdict.value,
                confidence=confidence,
                processing_time=processing_time
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "ML verification failed",
                claim_id=claim.id,
                error=str(e)
            )
            return self._create_error_result(claim, str(e))
    
    def _prepare_claim_text(self, claim: Claim) -> str:
        """Prepare claim text for analysis"""
        if isinstance(claim.content, str):
            text = sanitize_text(claim.content, self.max_sequence_length)
        else:
            # For non-text claims, try to extract text from metadata
            text = claim.metadata.get('extracted_text', '')
        
        return text.strip()
    
    @retry_async(max_attempts=3, delay=2.0)
    async def _fact_check_analysis(self, claim_text: str) -> Optional[Dict[str, Any]]:
        """Perform fact-checking using MNLI model"""
        try:
            model = self.models['fact_check']
            
            # Create premise-hypothesis pairs for fact-checking
            results = []
            for template in self.fact_check_templates:
                payload = {
                    "inputs": {
                        "premise": claim_text,
                        "hypothesis": template
                    }
                }
                
                result = await self._query_huggingface_model(model, payload)
                if result:
                    results.append({
                        'template': template,
                        'scores': result
                    })
            
            return {
                'model': model,
                'results': results,
                'claim_text': claim_text
            }
            
        except Exception as e:
            logger.warning(f"Fact-check analysis failed: {e}")
            return None
    
    @retry_async(max_attempts=3, delay=2.0)
    async def _sentiment_analysis(self, claim_text: str) -> Optional[Dict[str, Any]]:
        """Perform sentiment analysis"""
        try:
            model = self.models['sentiment']
            payload = {"inputs": claim_text}
            
            result = await self._query_huggingface_model(model, payload)
            if result:
                return {
                    'model': model,
                    'results': result,
                    'claim_text': claim_text
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return None
    
    @retry_async(max_attempts=3, delay=2.0)
    async def _entity_analysis(self, claim_text: str) -> Optional[Dict[str, Any]]:
        """Perform named entity recognition"""
        try:
            model = self.models['ner']
            payload = {"inputs": claim_text}
            
            result = await self._query_huggingface_model(model, payload)
            if result:
                return {
                    'model': model,
                    'results': result,
                    'claim_text': claim_text
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Entity analysis failed: {e}")
            return None
    
    async def _linguistic_analysis(self, claim_text: str) -> Optional[Dict[str, Any]]:
        """Perform linguistic pattern analysis"""
        try:
            # Simple linguistic analysis without external API
            patterns = {
                'certainty_indicators': len(re.findall(r'\b(definitely|certainly|absolutely|clearly|obviously)\b', claim_text.lower())),
                'uncertainty_indicators': len(re.findall(r'\b(maybe|perhaps|possibly|allegedly|reportedly)\b', claim_text.lower())),
                'superlatives': len(re.findall(r'\b(best|worst|most|least|always|never)\b', claim_text.lower())),
                'emotional_language': len(re.findall(r'\b(amazing|terrible|shocking|incredible|unbelievable)\b', claim_text.lower())),
                'question_marks': claim_text.count('?'),
                'exclamation_marks': claim_text.count('!'),
                'word_count': len(claim_text.split()),
                'sentence_count': len(re.split(r'[.!?]+', claim_text))
            }
            
            return {
                'model': 'linguistic_patterns',
                'results': patterns,
                'claim_text': claim_text
            }
            
        except Exception as e:
            logger.warning(f"Linguistic analysis failed: {e}")
            return None
    
    async def _query_huggingface_model(self, model: str, payload: Dict[str, Any]) -> Optional[Any]:
        """Query HuggingFace model API"""
        headers = {
            'Authorization': f'Bearer {self.huggingface_api_key}',
            'Content-Type': 'application/json'
        }
        
        url = f"{self.base_url}/models/{model}"
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.warning(f"HuggingFace API error {response.status}: {error_text}")
                    return None
    
    def _create_fact_check_evidence(self, result: Dict[str, Any]) -> Evidence:
        """Create evidence from fact-checking analysis"""
        try:
            # Find the most confident prediction
            best_result = None
            best_score = 0
            
            for item in result['results']:
                if item['scores']:
                    # Assuming MNLI returns entailment/contradiction/neutral scores
                    max_score = max(item['scores'], key=lambda x: x.get('score', 0))
                    if max_score.get('score', 0) > best_score:
                        best_score = max_score.get('score', 0)
                        best_result = {
                            'template': item['template'],
                            'prediction': max_score
                        }
            
            if best_result:
                content = f"Fact-check analysis: {best_result['template']} (confidence: {best_score:.2f})"
                credibility_score = min(best_score + 0.1, 1.0)  # Boost ML credibility slightly
            else:
                content = "Fact-check analysis completed but inconclusive"
                credibility_score = 0.5
            
            return Evidence(
                source='ML Fact-Check Model',
                content=content,
                credibility_score=credibility_score,
                relevance_score=0.9,
                evidence_type='ml_fact_check',
                metadata={
                    'model': result['model'],
                    'best_result': best_result,
                    'all_results': result['results']
                }
            )
            
        except Exception as e:
            logger.warning(f"Failed to create fact-check evidence: {e}")
            return None
    
    def _create_sentiment_evidence(self, result: Dict[str, Any]) -> Evidence:
        """Create evidence from sentiment analysis"""
        try:
            results = result['results']
            if results and isinstance(results, list) and results[0]:
                top_sentiment = results[0][0] if isinstance(results[0], list) else results[0]
                
                label = top_sentiment.get('label', 'UNKNOWN')
                score = top_sentiment.get('score', 0.0)
                
                content = f"Sentiment analysis: {label} (confidence: {score:.2f})"
                
                # Negative sentiment might indicate bias or emotional manipulation
                credibility_adjustment = 0.0
                if label.upper() in ['NEGATIVE', 'ANGER', 'FEAR']:
                    credibility_adjustment = -0.1
                elif label.upper() in ['POSITIVE', 'JOY']:
                    credibility_adjustment = 0.05
                
                return Evidence(
                    source='ML Sentiment Model',
                    content=content,
                    credibility_score=max(0.6 + credibility_adjustment, 0.1),
                    relevance_score=0.6,
                    evidence_type='sentiment_analysis',
                    metadata={
                        'model': result['model'],
                        'sentiment': label,
                        'confidence': score,
                        'all_results': results
                    }
                )
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to create sentiment evidence: {e}")
            return None
    
    def _create_entity_evidence(self, result: Dict[str, Any]) -> Evidence:
        """Create evidence from entity analysis"""
        try:
            results = result['results']
            if results and isinstance(results, list):
                # Extract unique entities
                entities = {}
                for entity in results:
                    entity_text = entity.get('word', '').strip()
                    entity_type = entity.get('entity_group', entity.get('entity', 'UNKNOWN'))
                    confidence = entity.get('score', 0.0)
                    
                    if entity_text and confidence > 0.5:
                        if entity_text not in entities or entities[entity_text]['confidence'] < confidence:
                            entities[entity_text] = {
                                'type': entity_type,
                                'confidence': confidence
                            }
                
                if entities:
                    entity_list = [f"{text} ({info['type']})" for text, info in entities.items()]
                    content = f"Named entities found: {', '.join(entity_list[:5])}"  # Top 5
                    
                    return Evidence(
                        source='ML Entity Recognition',
                        content=content,
                        credibility_score=0.7,
                        relevance_score=0.8,
                        evidence_type='entity_recognition',
                        metadata={
                            'model': result['model'],
                            'entities': entities,
                            'entity_count': len(entities)
                        }
                    )
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to create entity evidence: {e}")
            return None
    
    def _create_linguistic_evidence(self, result: Dict[str, Any]) -> Evidence:
        """Create evidence from linguistic analysis"""
        try:
            patterns = result['results']
            
            # Calculate linguistic credibility score
            credibility_indicators = []
            
            # High certainty without evidence might indicate bias
            if patterns['certainty_indicators'] > 2:
                credibility_indicators.append("High certainty language detected")
            
            # Uncertainty indicators might suggest more balanced reporting
            if patterns['uncertainty_indicators'] > 0:
                credibility_indicators.append("Uncertainty indicators present")
            
            # Excessive superlatives might indicate bias
            if patterns['superlatives'] > 3:
                credibility_indicators.append("Excessive superlative language")
            
            # Emotional language might indicate bias
            if patterns['emotional_language'] > 2:
                credibility_indicators.append("Emotional language detected")
            
            # Calculate base credibility
            base_credibility = 0.6
            if patterns['uncertainty_indicators'] > patterns['certainty_indicators']:
                base_credibility += 0.1  # More balanced
            if patterns['emotional_language'] > 3:
                base_credibility -= 0.15  # Too emotional
            if patterns['superlatives'] > 3:
                base_credibility -= 0.1  # Too many superlatives
            
            content = f"Linguistic analysis: {len(credibility_indicators)} patterns detected"
            if credibility_indicators:
                content += f" - {', '.join(credibility_indicators[:3])}"
            
            return Evidence(
                source='ML Linguistic Analysis',
                content=content,
                credibility_score=max(0.1, min(1.0, base_credibility)),
                relevance_score=0.5,
                evidence_type='linguistic_analysis',
                metadata={
                    'patterns': patterns,
                    'indicators': credibility_indicators
                }
            )
            
        except Exception as e:
            logger.warning(f"Failed to create linguistic evidence: {e}")
            return None
    
    def _calculate_overall_assessment(self, claim_text: str, evidence_list: List[Evidence], analyses: List[Any]) -> tuple[Verdict, float, str]:
        """Calculate overall verdict and confidence"""
        try:
            # Collect confidence scores from evidence
            confidences = [e.credibility_score for e in evidence_list if e]
            
            if not confidences:
                return Verdict.INCONCLUSIVE, 0.3, "No ML analysis results available"
            
            # Calculate weighted average confidence
            overall_confidence = sum(confidences) / len(confidences)
            
            # Determine verdict based on fact-checking results
            verdict = Verdict.INCONCLUSIVE
            reasoning_parts = []
            
            # Check fact-checking evidence
            fact_check_evidence = [e for e in evidence_list if e and e.evidence_type == 'ml_fact_check']
            if fact_check_evidence:
                fc_evidence = fact_check_evidence[0]
                best_result = fc_evidence.metadata.get('best_result', {})
                
                if best_result:
                    template = best_result.get('template', '')
                    if 'factually correct' in template.lower():
                        verdict = Verdict.TRUE
                        reasoning_parts.append("ML fact-checking indicates claim is likely true")
                    elif 'factually incorrect' in template.lower():
                        verdict = Verdict.FALSE
                        reasoning_parts.append("ML fact-checking indicates claim is likely false")
                    elif 'misleading' in template.lower():
                        verdict = Verdict.MISLEADING
                        reasoning_parts.append("ML fact-checking indicates claim is misleading")
            
            # Adjust confidence based on linguistic patterns
            linguistic_evidence = [e for e in evidence_list if e and e.evidence_type == 'linguistic_analysis']
            if linguistic_evidence:
                patterns = linguistic_evidence[0].metadata.get('patterns', {})
                if patterns.get('emotional_language', 0) > 3 or patterns.get('superlatives', 0) > 3:
                    overall_confidence *= 0.9  # Reduce confidence for emotional/superlative language
                    reasoning_parts.append("Linguistic patterns suggest potential bias")
            
            # Adjust confidence based on sentiment
            sentiment_evidence = [e for e in evidence_list if e and e.evidence_type == 'sentiment_analysis']
            if sentiment_evidence:
                sentiment = sentiment_evidence[0].metadata.get('sentiment', '')
                if sentiment.upper() in ['NEGATIVE', 'ANGER']:
                    overall_confidence *= 0.95
                    reasoning_parts.append("Negative sentiment detected")
            
            # Ensure confidence is within bounds
            overall_confidence = max(0.1, min(1.0, overall_confidence))
            
            # Create reasoning
            if not reasoning_parts:
                reasoning_parts.append("ML analysis completed with standard confidence")
            
            reasoning = f"ML Agent analysis: {'; '.join(reasoning_parts)}. Overall confidence: {overall_confidence:.2f}"
            
            return verdict, overall_confidence, reasoning
            
        except Exception as e:
            logger.warning(f"Failed to calculate overall assessment: {e}")
            return Verdict.INCONCLUSIVE, 0.3, f"Assessment calculation failed: {str(e)}"
    
    def _create_error_result(self, claim: Claim, error_message: str) -> VerificationResult:
        """Create error result"""
        return VerificationResult(
            claim_id=claim.id,
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            confidence=0.0,
            verdict=Verdict.INCONCLUSIVE,
            reasoning=f"ML Agent error: {error_message}",
            processing_time=0.0
        )
    
    def get_capabilities(self) -> AgentCapability:
        """Return agent capabilities"""
        return AgentCapability(
            agent_type=self.agent_type,
            supported_claim_types=[ClaimType.TEXT, ClaimType.MULTIMODAL],
            confidence_range=(0.1, 1.0),
            processing_time_estimate=30.0,
            description="Deep learning-based claim verification using transformer models for fact-checking, sentiment analysis, and linguistic pattern detection"
        )


# Example usage and testing
if __name__ == "__main__":
    import os
    from defame.core.models import Claim
    from config.globals import ClaimType
    
    async def test_ml_agent():
        # Test configuration
        config = {
            'huggingface_api_key': os.getenv('HUGGINGFACE_API_KEY', 'test-key'),
            'confidence_threshold': 0.7
        }
        
        ml_agent = MLAgent(config)
        
        # Test claim
        test_claim = Claim(
            content="The Earth is flat and NASA is hiding the truth from everyone.",
            claim_type=ClaimType.TEXT
        )
        
        # Test verification
        result = await ml_agent.verify_claim(test_claim)
        
        print(f"ML Agent Result:")
        print(f"Verdict: {result.verdict.value}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Evidence count: {len(result.evidence)}")
        print(f"Processing time: {result.processing_time:.2f}s")
        
        for i, evidence in enumerate(result.evidence):
            print(f"\nEvidence {i+1}: {evidence.source}")
            print(f"  Type: {evidence.evidence_type}")
            print(f"  Content: {evidence.content}")
            print(f"  Credibility: {evidence.credibility_score:.2f}")
    
    # Run test if API key is available
    if os.getenv('HUGGINGFACE_API_KEY'):
        asyncio.run(test_ml_agent())
    else:
        print("Set HUGGINGFACE_API_KEY environment variable to test ML agent")