"""
WebScrape Agent for real-time information gathering and source credibility assessment
"""
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from defame.core.interfaces import BaseAgent
from defame.core.models import Claim, VerificationResult, Evidence, AgentCapability
from defame.evidence_retrieval.tools.search_tool import SearchTool
from defame.evidence_retrieval.tools.firecrawl_scraper import FirecrawlScraper
from defame.utils.logger import get_logger
from defame.utils.helpers import sanitize_text, extract_urls, get_domain
from config.globals import AgentType, ClaimType, Verdict

logger = get_logger(__name__)


class WebScrapeAgent(BaseAgent):
    """Agent for web scraping and real-time information gathering"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_pages_per_claim = config.get('max_pages_per_claim', 5)
        self.timeout_seconds = config.get('timeout_seconds', 30)
        self.respect_robots_txt = config.get('respect_robots_txt', True)
        
        # Initialize tools
        self.search_tool = SearchTool(config.get('search_tool', {}))
        self.scraper_tool = FirecrawlScraper(config.get('scraper_tool', {}))
        
        # Source credibility mapping
        self.source_credibility = {
            'news_sites': 0.8,
            'academic_sites': 0.9,
            'government_sites': 0.85,
            'fact_check_sites': 0.9,
            'social_media': 0.3,
            'blogs': 0.4,
            'unknown': 0.5
        }
    
    def _get_agent_type(self) -> AgentType:
        return AgentType.WEBSCRAPE_AGENT
    
    async def verify_claim(self, claim: Claim, metadata: Optional[Dict] = None) -> VerificationResult:
        """Verify claim through web scraping and information gathering"""
        start_time = datetime.utcnow()
        
        try:
            claim_text = self._prepare_claim_text(claim)
            if not claim_text:
                return self._create_error_result(claim, "No text content to analyze")
            
            evidence_list = []
            
            # Step 1: Search for relevant web sources
            search_evidence = await self._search_for_sources(claim, claim_text)
            evidence_list.extend(search_evidence)
            
            # Step 2: Scrape specific URLs if provided
            if metadata and metadata.get('urls'):
                scrape_evidence = await self._scrape_provided_urls(claim, metadata['urls'], claim_text)
                evidence_list.extend(scrape_evidence)
            
            # Step 3: Analyze source credibility
            credibility_evidence = await self._analyze_source_credibility(claim, evidence_list)
            if credibility_evidence:
                evidence_list.append(credibility_evidence)
            
            # Step 4: Check for recent information
            recency_evidence = await self._check_information_recency(claim, claim_text)
            if recency_evidence:
                evidence_list.append(recency_evidence)
            
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
                metadata={
                    'sources_checked': len([e for e in evidence_list if e.evidence_type in ['web_search', 'web_content']]),
                    'scraping_successful': len([e for e in evidence_list if e.evidence_type == 'web_content'])
                }
            )
            
        except Exception as e:
            logger.error(f"WebScrape verification failed: {e}")
            return self._create_error_result(claim, str(e))
    
    def _prepare_claim_text(self, claim: Claim) -> str:
        """Prepare claim text for analysis"""
        if isinstance(claim.content, str):
            return sanitize_text(claim.content, 1000)
        return claim.metadata.get('extracted_text', '')
    
    async def _search_for_sources(self, claim: Claim, claim_text: str) -> List[Evidence]:
        """Search for relevant web sources"""
        try:
            # Generate search queries
            search_queries = self._generate_search_queries(claim_text)
            evidence_list = []
            
            for query in search_queries[:3]:  # Limit to 3 queries
                try:
                    search_results = await self.search_tool.gather_evidence(claim, query)
                    evidence_list.extend(search_results[:2])  # Top 2 results per query
                except Exception as e:
                    logger.warning(f"Search failed for query '{query}': {e}")
            
            return evidence_list
            
        except Exception as e:
            logger.warning(f"Source search failed: {e}")
            return []
    
    def _generate_search_queries(self, claim_text: str) -> List[str]:
        """Generate search queries from claim text"""
        queries = []
        
        # Original claim
        queries.append(claim_text[:100])
        
        # Fact-check specific query
        queries.append(f'"{claim_text[:50]}" fact check')
        
        # News search query
        queries.append(f'"{claim_text[:50]}" news')
        
        # Debunk search query
        queries.append(f'"{claim_text[:50]}" debunk false true')
        
        return queries
    
    async def _scrape_provided_urls(self, claim: Claim, urls: List[str], claim_text: str) -> List[Evidence]:
        """Scrape content from provided URLs"""
        try:
            evidence_list = await self.scraper_tool.scrape_multiple_urls(claim, urls, claim_text)
            return evidence_list
        except Exception as e:
            logger.warning(f"URL scraping failed: {e}")
            return []
    
    async def _analyze_source_credibility(self, claim: Claim, evidence_list: List[Evidence]) -> Optional[Evidence]:
        """Analyze the credibility of sources found"""
        try:
            if not evidence_list:
                return None
            
            source_analysis = {}
            total_sources = 0
            
            for evidence in evidence_list:
                if evidence.url:
                    domain = get_domain(evidence.url)
                    if domain:
                        source_type = self._classify_source_type(domain)
                        if source_type not in source_analysis:
                            source_analysis[source_type] = {
                                'count': 0,
                                'domains': [],
                                'avg_credibility': 0.0
                            }
                        
                        source_analysis[source_type]['count'] += 1
                        source_analysis[source_type]['domains'].append(domain)
                        source_analysis[source_type]['avg_credibility'] += evidence.credibility_score
                        total_sources += 1
            
            # Calculate average credibility for each source type
            for source_type in source_analysis:
                if source_analysis[source_type]['count'] > 0:
                    source_analysis[source_type]['avg_credibility'] /= source_analysis[source_type]['count']
            
            # Overall credibility assessment
            if total_sources > 0:
                overall_credibility = sum(
                    info['avg_credibility'] * info['count'] 
                    for info in source_analysis.values()
                ) / total_sources
                
                high_credibility_sources = sum(
                    info['count'] for source_type, info in source_analysis.items()
                    if source_type in ['news_sites', 'academic_sites', 'fact_check_sites']
                )
                
                content = f"Source credibility analysis: {total_sources} sources analyzed, "
                content += f"{high_credibility_sources} high-credibility sources found"
                
                return Evidence(
                    source='WebScrape Agent - Source Analysis',
                    content=content,
                    credibility_score=overall_credibility,
                    relevance_score=0.8,
                    evidence_type='source_credibility',
                    metadata={
                        'source_analysis': source_analysis,
                        'total_sources': total_sources,
                        'high_credibility_count': high_credibility_sources,
                        'overall_credibility': overall_credibility
                    }
                )
            
            return None
            
        except Exception as e:
            logger.warning(f"Source credibility analysis failed: {e}")
            return None
    
    def _classify_source_type(self, domain: str) -> str:
        """Classify source type based on domain"""
        domain_lower = domain.lower()
        
        # Academic sites
        if any(tld in domain_lower for tld in ['.edu', '.ac.', 'university', 'college']):
            return 'academic_sites'
        
        # Government sites
        if any(tld in domain_lower for tld in ['.gov', '.mil']):
            return 'government_sites'
        
        # Fact-check sites
        fact_check_domains = ['factcheck.org', 'snopes.com', 'politifact.com', 'fullfact.org']
        if any(fc_domain in domain_lower for fc_domain in fact_check_domains):
            return 'fact_check_sites'
        
        # News sites (simplified)
        news_indicators = ['news', 'times', 'post', 'herald', 'tribune', 'guardian', 'reuters', 'ap.org', 'bbc']
        if any(indicator in domain_lower for indicator in news_indicators):
            return 'news_sites'
        
        # Social media
        social_domains = ['twitter.com', 'facebook.com', 'instagram.com', 'tiktok.com', 'youtube.com']
        if any(social in domain_lower for social in social_domains):
            return 'social_media'
        
        # Blogs
        blog_indicators = ['blog', 'wordpress', 'blogspot', 'medium.com']
        if any(blog in domain_lower for blog in blog_indicators):
            return 'blogs'
        
        return 'unknown'
    
    async def _check_information_recency(self, claim: Claim, claim_text: str) -> Optional[Evidence]:
        """Check for recent information about the claim"""
        try:
            # Search for recent news
            recent_query = f'"{claim_text[:50]}" after:2023-01-01'  # Last year
            recent_evidence = await self.search_tool.gather_evidence(claim, recent_query, search_type='news')
            
            if recent_evidence:
                recent_count = len(recent_evidence)
                avg_relevance = sum(e.relevance_score for e in recent_evidence) / recent_count
                
                content = f"Recent information analysis: {recent_count} recent sources found"
                credibility_score = min(0.8, 0.5 + (avg_relevance * 0.3))
                
                return Evidence(
                    source='WebScrape Agent - Recency Check',
                    content=content,
                    credibility_score=credibility_score,
                    relevance_score=avg_relevance,
                    evidence_type='information_recency',
                    metadata={
                        'recent_sources_count': recent_count,
                        'avg_relevance': avg_relevance,
                        'search_query': recent_query
                    }
                )
            
            return None
            
        except Exception as e:
            logger.warning(f"Information recency check failed: {e}")
            return None
    
    def _calculate_assessment(self, claim_text: str, evidence_list: List[Evidence]) -> tuple[Verdict, float, str]:
        """Calculate overall assessment based on web evidence"""
        if not evidence_list:
            return Verdict.INCONCLUSIVE, 0.3, "No web sources found for verification"
        
        # Separate different types of evidence
        web_evidence = [e for e in evidence_list if e.evidence_type in ['web_search', 'web_content']]
        credibility_evidence = [e for e in evidence_list if e.evidence_type == 'source_credibility']
        
        if not web_evidence:
            return Verdict.INCONCLUSIVE, 0.3, "No relevant web content found"
        
        # Calculate average relevance and credibility
        avg_relevance = sum(e.relevance_score for e in web_evidence) / len(web_evidence)
        avg_credibility = sum(e.credibility_score for e in web_evidence) / len(web_evidence)
        
        # Adjust based on source credibility analysis
        if credibility_evidence:
            source_credibility = credibility_evidence[0].credibility_score
            avg_credibility = (avg_credibility + source_credibility) / 2
        
        # Overall confidence
        confidence = (avg_relevance + avg_credibility) / 2
        
        # Determine verdict based on evidence quality and quantity
        high_quality_sources = len([e for e in web_evidence if e.credibility_score > 0.7])
        
        if confidence > 0.7 and high_quality_sources >= 2:
            verdict = Verdict.TRUE if avg_relevance > 0.6 else Verdict.INCONCLUSIVE
            reasoning = f"Strong web evidence from {len(web_evidence)} sources, {high_quality_sources} high-quality"
        elif confidence > 0.5:
            verdict = Verdict.INCONCLUSIVE
            reasoning = f"Moderate web evidence from {len(web_evidence)} sources"
        else:
            verdict = Verdict.INCONCLUSIVE
            reasoning = f"Limited web evidence from {len(web_evidence)} sources"
        
        return verdict, confidence, reasoning
    
    def _create_error_result(self, claim: Claim, error_message: str) -> VerificationResult:
        """Create error result"""
        return VerificationResult(
            claim_id=claim.id,
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            confidence=0.0,
            verdict=Verdict.INCONCLUSIVE,
            reasoning=f"WebScrape Agent error: {error_message}",
            processing_time=0.0
        )
    
    def get_capabilities(self) -> AgentCapability:
        """Return agent capabilities"""
        return AgentCapability(
            agent_type=self.agent_type,
            supported_claim_types=[ClaimType.TEXT, ClaimType.MULTIMODAL],
            confidence_range=(0.1, 1.0),
            processing_time_estimate=45.0,
            description="Real-time web information gathering, source credibility assessment, and content analysis"
        )