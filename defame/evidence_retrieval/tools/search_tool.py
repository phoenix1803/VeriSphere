"""
Web search tool using Serper API for evidence gathering
"""
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
from datetime import datetime
import re

from defame.core.interfaces import BaseEvidenceTool
from defame.core.models import Claim, Evidence
from defame.utils.logger import get_logger
from defame.utils.helpers import retry_async, RateLimiter, sanitize_text, extract_urls, get_domain

logger = get_logger(__name__)


class SearchTool(BaseEvidenceTool):
    """Web search tool using Serper API"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('serper_api_key')
        self.base_url = config.get('base_url', 'https://google.serper.dev')
        self.timeout = config.get('timeout', 30)
        self.max_results = config.get('max_results', 10)
        self.rate_limiter = RateLimiter(
            max_calls=config.get('rate_limit_per_minute', 60),
            time_window=60.0
        )
        
        # Domain credibility scores (can be loaded from config)
        self.domain_credibility = {
            'reuters.com': 0.95,
            'ap.org': 0.95,
            'bbc.com': 0.90,
            'cnn.com': 0.85,
            'nytimes.com': 0.85,
            'washingtonpost.com': 0.85,
            'theguardian.com': 0.85,
            'npr.org': 0.85,
            'factcheck.org': 0.90,
            'snopes.com': 0.85,
            'politifact.com': 0.85,
            'wikipedia.org': 0.75,
            'nature.com': 0.95,
            'science.org': 0.95,
            'pubmed.ncbi.nlm.nih.gov': 0.90,
        }
    
    async def gather_evidence(self, claim: Claim, query: str, **kwargs) -> List[Evidence]:
        """
        Gather evidence using web search
        
        Args:
            claim: The claim being verified
            query: Search query
            **kwargs: Additional parameters (search_type, country, etc.)
            
        Returns:
            List of Evidence objects from search results
        """
        try:
            await self.rate_limiter.acquire()
            
            # Prepare search parameters
            search_params = {
                'q': sanitize_text(query),
                'num': min(self.max_results, kwargs.get('max_results', self.max_results)),
                'gl': kwargs.get('country', 'us'),
                'hl': kwargs.get('language', 'en'),
            }
            
            # Add search type if specified
            search_type = kwargs.get('search_type', 'search')
            if search_type in ['news', 'images', 'videos']:
                search_params['type'] = search_type
            
            logger.info(
                "Performing web search",
                claim_id=claim.id,
                query=query,
                search_type=search_type
            )
            
            # Perform search
            results = await self._perform_search(search_params)
            
            # Convert results to evidence
            evidence_list = await self._process_search_results(claim, results, query)
            
            logger.info(
                "Web search completed",
                claim_id=claim.id,
                evidence_count=len(evidence_list),
                query=query
            )
            
            return evidence_list
            
        except Exception as e:
            logger.error(
                "Web search failed",
                claim_id=claim.id,
                query=query,
                error=str(e)
            )
            return []
    
    @retry_async(max_attempts=3, delay=1.0, backoff=2.0)
    async def _perform_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform the actual search API call"""
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.post(
                f"{self.base_url}/search",
                json=params,
                headers=headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Search API error {response.status}: {error_text}")
    
    async def _process_search_results(self, claim: Claim, results: Dict[str, Any], query: str) -> List[Evidence]:
        """Process search results into Evidence objects"""
        evidence_list = []
        
        # Process organic results
        organic_results = results.get('organic', [])
        for result in organic_results:
            evidence = await self._create_evidence_from_result(claim, result, query, 'organic')
            if evidence:
                evidence_list.append(evidence)
        
        # Process news results if available
        news_results = results.get('news', [])
        for result in news_results:
            evidence = await self._create_evidence_from_result(claim, result, query, 'news')
            if evidence:
                evidence_list.append(evidence)
        
        # Process knowledge graph if available
        knowledge_graph = results.get('knowledgeGraph', {})
        if knowledge_graph:
            evidence = await self._create_evidence_from_knowledge_graph(claim, knowledge_graph, query)
            if evidence:
                evidence_list.append(evidence)
        
        # Sort by relevance and credibility
        evidence_list.sort(key=lambda e: (e.credibility_score + e.relevance_score) / 2, reverse=True)
        
        return evidence_list
    
    async def _create_evidence_from_result(self, claim: Claim, result: Dict[str, Any], query: str, result_type: str) -> Optional[Evidence]:
        """Create Evidence object from search result"""
        try:
            url = result.get('link', '')
            title = result.get('title', '')
            snippet = result.get('snippet', '')
            
            if not url or not title:
                return None
            
            # Calculate credibility score based on domain
            domain = get_domain(url)
            credibility_score = self.domain_credibility.get(domain, 0.5)
            
            # Adjust credibility for news results
            if result_type == 'news':
                credibility_score = min(credibility_score + 0.1, 1.0)
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance(query, title, snippet)
            
            # Create evidence
            evidence = Evidence(
                source=domain or url,
                content=f"{title}\n\n{snippet}",
                url=url,
                credibility_score=credibility_score,
                relevance_score=relevance_score,
                evidence_type='web_search',
                metadata={
                    'search_query': query,
                    'result_type': result_type,
                    'title': title,
                    'snippet': snippet,
                    'domain': domain,
                    'position': result.get('position', 0),
                    'date': result.get('date'),
                }
            )
            
            return evidence
            
        except Exception as e:
            logger.warning(f"Failed to create evidence from result: {e}")
            return None
    
    async def _create_evidence_from_knowledge_graph(self, claim: Claim, kg: Dict[str, Any], query: str) -> Optional[Evidence]:
        """Create Evidence from knowledge graph result"""
        try:
            title = kg.get('title', '')
            description = kg.get('description', '')
            
            if not title and not description:
                return None
            
            content = f"{title}\n\n{description}"
            
            # Knowledge graphs are generally reliable
            credibility_score = 0.85
            relevance_score = self._calculate_relevance(query, title, description)
            
            evidence = Evidence(
                source='Google Knowledge Graph',
                content=content,
                credibility_score=credibility_score,
                relevance_score=relevance_score,
                evidence_type='knowledge_graph',
                metadata={
                    'search_query': query,
                    'title': title,
                    'description': description,
                    'type': kg.get('type'),
                    'website': kg.get('website'),
                }
            )
            
            return evidence
            
        except Exception as e:
            logger.warning(f"Failed to create evidence from knowledge graph: {e}")
            return None
    
    def _calculate_relevance(self, query: str, title: str, snippet: str) -> float:
        """Calculate relevance score between query and result"""
        query_words = set(query.lower().split())
        title_words = set(title.lower().split())
        snippet_words = set(snippet.lower().split())
        
        # Calculate word overlap
        title_overlap = len(query_words.intersection(title_words)) / len(query_words) if query_words else 0
        snippet_overlap = len(query_words.intersection(snippet_words)) / len(query_words) if query_words else 0
        
        # Weight title more heavily than snippet
        relevance = (title_overlap * 0.7) + (snippet_overlap * 0.3)
        
        return min(relevance, 1.0)
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Return tool information"""
        return {
            'tool_id': self.tool_id,
            'name': 'SearchTool',
            'description': 'Web search using Serper API',
            'capabilities': [
                'web_search',
                'news_search',
                'knowledge_graph',
                'domain_credibility_scoring'
            ],
            'supported_languages': ['en', 'es', 'fr', 'de', 'zh', 'ja'],
            'rate_limit': '60 requests per minute',
            'max_results': self.max_results
        }
    
    async def health_check(self) -> bool:
        """Check if search API is available"""
        try:
            # Perform a simple test search
            test_params = {
                'q': 'test',
                'num': 1
            }
            
            result = await self._perform_search(test_params)
            return 'organic' in result or 'searchParameters' in result
            
        except Exception as e:
            logger.error(f"Search tool health check failed: {e}")
            self.is_available = False
            return False
    
    async def search_news(self, claim: Claim, query: str, **kwargs) -> List[Evidence]:
        """Specialized news search"""
        kwargs['search_type'] = 'news'
        return await self.gather_evidence(claim, query, **kwargs)
    
    async def search_recent(self, claim: Claim, query: str, days: int = 7, **kwargs) -> List[Evidence]:
        """Search for recent results within specified days"""
        # Add time filter to query
        time_query = f"{query} after:{(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')}"
        return await self.gather_evidence(claim, time_query, **kwargs)
    
    async def fact_check_search(self, claim: Claim, query: str, **kwargs) -> List[Evidence]:
        """Search specifically for fact-checking sources"""
        fact_check_query = f"{query} site:factcheck.org OR site:snopes.com OR site:politifact.com"
        return await self.gather_evidence(claim, fact_check_query, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    import os
    from defame.core.models import Claim
    from config.globals import ClaimType
    
    async def test_search_tool():
        # Test configuration
        config = {
            'serper_api_key': os.getenv('SERPER_API_KEY', 'test-key'),
            'max_results': 5
        }
        
        search_tool = SearchTool(config)
        
        # Test claim
        test_claim = Claim(
            content="The Earth is round",
            claim_type=ClaimType.TEXT
        )
        
        # Test search
        evidence = await search_tool.gather_evidence(
            test_claim,
            "Earth round sphere scientific evidence"
        )
        
        print(f"Found {len(evidence)} pieces of evidence:")
        for i, e in enumerate(evidence[:3]):
            print(f"{i+1}. {e.source} (credibility: {e.credibility_score:.2f}, relevance: {e.relevance_score:.2f})")
            print(f"   {e.content[:100]}...")
            print()
    
    # Run test if API key is available
    if os.getenv('SERPER_API_KEY'):
        asyncio.run(test_search_tool())
    else:
        print("Set SERPER_API_KEY environment variable to test search tool")