"""
Wikipedia Agent for knowledge-based claim verification
"""
import asyncio
import aiohttp
import wikipedia
from typing import Dict, List, Any, Optional
from datetime import datetime
import re

from defame.core.interfaces import BaseAgent
from defame.core.models import Claim, VerificationResult, Evidence, AgentCapability
from defame.utils.logger import get_logger
from defame.utils.helpers import retry_async, sanitize_text, calculate_text_similarity
from config.globals import AgentType, ClaimType, Verdict

logger = get_logger(__name__)


class WikipediaAgent(BaseAgent):
    """Wikipedia-based knowledge verification agent"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_search_results = config.get('max_search_results', 10)
        self.similarity_threshold = config.get('similarity_threshold', 0.8)
        self.language = config.get('language', 'en')
        wikipedia.set_lang(self.language)
    
    def _get_agent_type(self) -> AgentType:
        return AgentType.WIKIPEDIA_AGENT
    
    async def verify_claim(self, claim: Claim, metadata: Optional[Dict] = None) -> VerificationResult:
        """Verify claim against Wikipedia knowledge"""
        start_time = datetime.utcnow()
        
        try:
            claim_text = self._prepare_claim_text(claim)
            if not claim_text:
                return self._create_error_result(claim, "No text content to analyze")
            
            # Extract key entities and topics
            entities = self._extract_entities(claim_text)
            evidence_list = []
            
            # Search Wikipedia for each entity
            for entity in entities[:5]:  # Limit to top 5 entities
                try:
                    wiki_evidence = await self._search_wikipedia(claim_text, entity)
                    evidence_list.extend(wiki_evidence)
                except Exception as e:
                    logger.warning(f"Wikipedia search failed for {entity}: {e}")
            
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
                metadata={'entities_searched': entities[:5]}
            )
            
        except Exception as e:
            logger.error(f"Wikipedia verification failed: {e}")
            return self._create_error_result(claim, str(e))
    
    def _prepare_claim_text(self, claim: Claim) -> str:
        """Prepare claim text for analysis"""
        if isinstance(claim.content, str):
            return sanitize_text(claim.content, 1000)
        return claim.metadata.get('extracted_text', '')
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract potential entities from text"""
        # Simple entity extraction - in production, use NER
        entities = []
        
        # Extract capitalized words/phrases
        capitalized_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        potential_entities = re.findall(capitalized_pattern, text)
        
        # Filter and clean entities
        for entity in potential_entities:
            if len(entity) > 2 and entity not in ['The', 'This', 'That', 'There']:
                entities.append(entity.strip())
        
        # Add the full claim as a search term
        entities.append(text[:100])  # First 100 chars
        
        return list(set(entities))  # Remove duplicates
    
    async def _search_wikipedia(self, claim_text: str, query: str) -> List[Evidence]:
        """Search Wikipedia for information about the query"""
        evidence_list = []
        
        try:
            # Search for pages
            search_results = wikipedia.search(query, results=self.max_search_results)
            
            for page_title in search_results[:3]:  # Top 3 results
                try:
                    # Get page content
                    page = wikipedia.page(page_title)
                    
                    # Calculate relevance
                    relevance = calculate_text_similarity(claim_text, page.content[:1000])
                    
                    if relevance > 0.1:  # Minimum relevance threshold
                        evidence = Evidence(
                            source=f'Wikipedia: {page.title}',
                            content=f"{page.summary[:500]}...",
                            url=page.url,
                            credibility_score=0.75,  # Wikipedia is generally credible
                            relevance_score=relevance,
                            evidence_type='wikipedia_article',
                            metadata={
                                'page_title': page.title,
                                'page_id': getattr(page, 'pageid', None),
                                'categories': getattr(page, 'categories', [])[:5],
                                'references': len(getattr(page, 'references', [])),
                                'query': query
                            }
                        )
                        evidence_list.append(evidence)
                
                except wikipedia.exceptions.DisambiguationError as e:
                    # Handle disambiguation by taking the first option
                    try:
                        page = wikipedia.page(e.options[0])
                        relevance = calculate_text_similarity(claim_text, page.content[:1000])
                        
                        if relevance > 0.1:
                            evidence = Evidence(
                                source=f'Wikipedia: {page.title}',
                                content=f"{page.summary[:500]}...",
                                url=page.url,
                                credibility_score=0.7,  # Slightly lower for disambiguation
                                relevance_score=relevance,
                                evidence_type='wikipedia_article',
                                metadata={
                                    'page_title': page.title,
                                    'disambiguation': True,
                                    'query': query
                                }
                            )
                            evidence_list.append(evidence)
                    except Exception:
                        pass
                
                except Exception as e:
                    logger.warning(f"Failed to process Wikipedia page {page_title}: {e}")
        
        except Exception as e:
            logger.warning(f"Wikipedia search failed for query '{query}': {e}")
        
        return evidence_list
    
    def _calculate_assessment(self, claim_text: str, evidence_list: List[Evidence]) -> tuple[Verdict, float, str]:
        """Calculate overall assessment based on Wikipedia evidence"""
        if not evidence_list:
            return Verdict.INCONCLUSIVE, 0.3, "No relevant Wikipedia articles found"
        
        # Calculate average relevance and credibility
        avg_relevance = sum(e.relevance_score for e in evidence_list) / len(evidence_list)
        avg_credibility = sum(e.credibility_score for e in evidence_list) / len(evidence_list)
        
        # Overall confidence is combination of relevance and credibility
        confidence = (avg_relevance + avg_credibility) / 2
        
        # Determine verdict based on evidence quality
        if confidence > 0.7 and avg_relevance > 0.5:
            verdict = Verdict.TRUE  # Strong Wikipedia support
            reasoning = f"Strong Wikipedia evidence found with {len(evidence_list)} relevant articles"
        elif confidence > 0.5:
            verdict = Verdict.INCONCLUSIVE  # Some support but not conclusive
            reasoning = f"Partial Wikipedia evidence found with {len(evidence_list)} articles"
        else:
            verdict = Verdict.INCONCLUSIVE  # Weak evidence
            reasoning = f"Limited Wikipedia evidence found with {len(evidence_list)} articles"
        
        return verdict, confidence, reasoning
    
    def _create_error_result(self, claim: Claim, error_message: str) -> VerificationResult:
        """Create error result"""
        return VerificationResult(
            claim_id=claim.id,
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            confidence=0.0,
            verdict=Verdict.INCONCLUSIVE,
            reasoning=f"Wikipedia Agent error: {error_message}",
            processing_time=0.0
        )
    
    def get_capabilities(self) -> AgentCapability:
        """Return agent capabilities"""
        return AgentCapability(
            agent_type=self.agent_type,
            supported_claim_types=[ClaimType.TEXT, ClaimType.MULTIMODAL],
            confidence_range=(0.1, 1.0),
            processing_time_estimate=15.0,
            description="Knowledge verification using Wikipedia articles and cross-referencing"
        )