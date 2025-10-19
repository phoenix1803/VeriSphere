"""
Web scraping tool using Firecrawl for content extraction
"""
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
from datetime import datetime
from urllib.parse import urljoin, urlparse
import re

from defame.core.interfaces import BaseEvidenceTool
from defame.core.models import Claim, Evidence
from defame.utils.logger import get_logger
from defame.utils.helpers import retry_async, RateLimiter, sanitize_text, is_valid_url, get_domain

logger = get_logger(__name__)


class FirecrawlScraper(BaseEvidenceTool):
    """Web scraping tool using Firecrawl"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.firecrawl_url = config.get('firecrawl_url', 'http://localhost:3002')
        self.timeout = config.get('timeout', 60)
        self.max_pages = config.get('max_pages_per_claim', 5)
        self.respect_robots = config.get('respect_robots_txt', True)
        self.rate_limiter = RateLimiter(
            max_calls=config.get('rate_limit_per_minute', 30),
            time_window=60.0
        )
        
        # Domain credibility mapping
        self.domain_credibility = {
            'reuters.com': 0.95, 'ap.org': 0.95, 'bbc.com': 0.90,
            'cnn.com': 0.85, 'nytimes.com': 0.85, 'washingtonpost.com': 0.85,
            'theguardian.com': 0.85, 'npr.org': 0.85, 'wikipedia.org': 0.75,
            'nature.com': 0.95, 'science.org': 0.95
        }
    
    async def gather_evidence(self, claim: Claim, query: str, **kwargs) -> List[Evidence]:
        """Gather evidence by scraping web pages"""
        try:
            evidence_list = []
            
            # Get URLs to scrape
            urls = kwargs.get('urls', [])
            if not urls:
                # Extract URLs from claim content if available
                urls = self._extract_urls_from_text(f"{claim.content} {query}")
            
            if not urls:
                return []
            
            # Limit number of pages
            urls = urls[:self.max_pages]
            
            logger.info(
                "Starting web scraping",
                claim_id=claim.id,
                url_count=len(urls)
            )
            
            # Scrape each URL
            for url in urls:
                if not is_valid_url(url):
                    continue
                
                await self.rate_limiter.acquire()
                
                try:
                    evidence = await self._scrape_url(claim, url, query)
                    if evidence:
                        evidence_list.append(evidence)
                except Exception as e:
                    logger.warning(f"Failed to scrape {url}: {e}")
            
            logger.info(
                "Web scraping completed",
                claim_id=claim.id,
                evidence_count=len(evidence_list)
            )
            
            return evidence_list
            
        except Exception as e:
            logger.error(f"Web scraping failed: {e}")
            return []
    
    def _extract_urls_from_text(self, text: str) -> List[str]:
        """Extract URLs from text content"""
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        return url_pattern.findall(text)
    
    @retry_async(max_attempts=3, delay=2.0)
    async def _scrape_url(self, claim: Claim, url: str, query: str) -> Optional[Evidence]:
        """Scrape a single URL using Firecrawl"""
        try:
            # Prepare Firecrawl request
            scrape_data = {
                'url': url,
                'formats': ['markdown', 'html'],
                'onlyMainContent': True,
                'includeTags': ['title', 'meta', 'article', 'main'],
                'excludeTags': ['nav', 'footer', 'aside', 'script', 'style'],
                'waitFor': 2000,  # Wait 2 seconds for dynamic content
                'timeout': self.timeout * 1000
            }
            
            # Make request to Firecrawl
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(
                    f"{self.firecrawl_url}/v0/scrape",
                    json=scrape_data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return await self._process_scrape_result(claim, result, url, query)
                    else:
                        error_text = await response.text()
                        logger.warning(f"Firecrawl error {response.status} for {url}: {error_text}")
                        return None
            
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")
            return None
    
    async def _process_scrape_result(self, claim: Claim, result: Dict[str, Any], url: str, query: str) -> Optional[Evidence]:
        """Process Firecrawl scraping result"""
        try:
            data = result.get('data', {})
            
            # Extract content
            title = data.get('title', '')
            content = data.get('markdown', '') or data.get('content', '')
            metadata = data.get('metadata', {})
            
            if not content:
                return None
            
            # Clean and truncate content
            content = sanitize_text(content, max_length=5000)
            
            # Calculate relevance score
            relevance_score = self._calculate_content_relevance(query, title, content)
            
            # Get domain credibility
            domain = get_domain(url)
            credibility_score = self.domain_credibility.get(domain, 0.6)
            
            # Adjust credibility based on content quality indicators
            credibility_score = self._adjust_credibility_score(credibility_score, content, metadata)
            
            # Create evidence
            evidence = Evidence(
                source=domain or url,
                content=f"{title}\n\n{content[:1000]}{'...' if len(content) > 1000 else ''}",
                url=url,
                credibility_score=credibility_score,
                relevance_score=relevance_score,
                evidence_type='web_content',
                metadata={
                    'title': title,
                    'domain': domain,
                    'word_count': len(content.split()),
                    'scrape_timestamp': datetime.utcnow().isoformat(),
                    'page_metadata': metadata,
                    'content_length': len(content)
                }
            )
            
            return evidence
            
        except Exception as e:
            logger.warning(f"Failed to process scrape result for {url}: {e}")
            return None
    
    def _calculate_content_relevance(self, query: str, title: str, content: str) -> float:
        """Calculate relevance score between query and scraped content"""
        if not query:
            return 0.5
        
        query_words = set(query.lower().split())
        title_words = set(title.lower().split())
        content_words = set(content.lower().split())
        
        # Calculate word overlap
        title_overlap = len(query_words.intersection(title_words)) / len(query_words) if query_words else 0
        content_overlap = len(query_words.intersection(content_words)) / len(query_words) if query_words else 0
        
        # Weight title more heavily
        relevance = (title_overlap * 0.6) + (content_overlap * 0.4)
        
        return min(relevance, 1.0)
    
    def _adjust_credibility_score(self, base_score: float, content: str, metadata: Dict[str, Any]) -> float:
        """Adjust credibility score based on content quality indicators"""
        score = base_score
        
        # Check for quality indicators
        quality_indicators = [
            'published', 'author', 'date', 'source', 'references',
            'citation', 'study', 'research', 'data', 'evidence'
        ]
        
        content_lower = content.lower()
        indicator_count = sum(1 for indicator in quality_indicators if indicator in content_lower)
        
        # Boost score for quality indicators
        score += (indicator_count * 0.02)
        
        # Check for problematic indicators
        problematic_indicators = [
            'opinion', 'blog', 'rumor', 'unconfirmed', 'allegedly',
            'claims', 'reportedly', 'sources say'
        ]
        
        problematic_count = sum(1 for indicator in problematic_indicators if indicator in content_lower)
        
        # Reduce score for problematic indicators
        score -= (problematic_count * 0.05)
        
        # Check content length (very short content is less reliable)
        if len(content) < 200:
            score -= 0.1
        
        # Check for author information
        if metadata.get('author') or 'author:' in content_lower or 'by ' in content_lower:
            score += 0.05
        
        # Check for publication date
        if metadata.get('publishedTime') or any(word in content_lower for word in ['published', 'updated', 'posted']):
            score += 0.05
        
        return max(0.0, min(1.0, score))
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Return tool information"""
        return {
            'tool_id': self.tool_id,
            'name': 'FirecrawlScraper',
            'description': 'Web content scraping using Firecrawl',
            'capabilities': [
                'web_content_extraction',
                'metadata_extraction',
                'content_quality_assessment',
                'domain_credibility_scoring'
            ],
            'max_pages_per_claim': self.max_pages,
            'respects_robots_txt': self.respect_robots,
            'rate_limit': '30 requests per minute'
        }
    
    async def health_check(self) -> bool:
        """Check if Firecrawl service is available"""
        try:
            # Test with a simple scrape request
            test_data = {
                'url': 'https://example.com',
                'formats': ['markdown'],
                'timeout': 10000
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.post(
                    f"{self.firecrawl_url}/v0/scrape",
                    json=test_data
                ) as response:
                    # Accept both success and reasonable error responses
                    return response.status in [200, 400, 429]  # 400 might be invalid URL, 429 rate limit
            
        except Exception as e:
            logger.error(f"Firecrawl health check failed: {e}")
            self.is_available = False
            return False
    
    async def scrape_multiple_urls(self, claim: Claim, urls: List[str], query: str = "") -> List[Evidence]:
        """Scrape multiple URLs concurrently"""
        evidence_list = []
        
        # Limit concurrent requests
        semaphore = asyncio.Semaphore(3)
        
        async def scrape_with_semaphore(url: str) -> Optional[Evidence]:
            async with semaphore:
                await self.rate_limiter.acquire()
                return await self._scrape_url(claim, url, query)
        
        # Execute scraping tasks
        tasks = [scrape_with_semaphore(url) for url in urls[:self.max_pages]]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect successful results
        for result in results:
            if isinstance(result, Evidence):
                evidence_list.append(result)
            elif isinstance(result, Exception):
                logger.warning(f"Scraping task failed: {result}")
        
        return evidence_list


# Example usage and testing
if __name__ == "__main__":
    import os
    from defame.core.models import Claim
    from config.globals import ClaimType
    
    async def test_firecrawl_scraper():
        # Test configuration
        config = {
            'firecrawl_url': 'http://localhost:3002',
            'max_pages_per_claim': 2
        }
        
        scraper = FirecrawlScraper(config)
        
        # Test claim
        test_claim = Claim(
            content="Climate change is caused by human activities",
            claim_type=ClaimType.TEXT
        )
        
        # Test URLs
        test_urls = [
            'https://climate.nasa.gov/causes/',
            'https://www.ipcc.ch/report/ar6/wg1/'
        ]
        
        # Test scraping
        evidence = await scraper.scrape_multiple_urls(
            test_claim,
            test_urls,
            "climate change human causes"
        )
        
        print(f"Scraped {len(evidence)} pieces of evidence:")
        for i, e in enumerate(evidence):
            print(f"{i+1}. {e.source}")
            print(f"   Credibility: {e.credibility_score:.2f}, Relevance: {e.relevance_score:.2f}")
            print(f"   {e.content[:150]}...")
            print()
    
    # Run test
    asyncio.run(test_firecrawl_scraper())