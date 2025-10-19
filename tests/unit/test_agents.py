"""
Unit tests for VeriSphere agents
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
import aiohttp

from defame.agents.ml_agent import MLAgent
from defame.agents.wikipedia_agent import WikipediaAgent
from defame.agents.coherence_agent import CoherenceAgent
from defame.agents.webscrape_agent import WebScrapeAgent
from defame.core.models import Claim, Evidence, VerificationResult
from config.globals import ClaimType, AgentType, Verdict


class TestMLAgent:
    """Test ML Agent"""
    
    @pytest.fixture
    def ml_agent_config(self):
        return {
            'huggingface_api_key': 'test-key',
            'confidence_threshold': 0.7,
            'timeout': 30
        }
    
    @pytest.fixture
    def ml_agent(self, ml_agent_config):
        return MLAgent(ml_agent_config)
    
    def test_ml_agent_initialization(self, ml_agent):
        """Test ML agent initialization"""
        assert ml_agent.agent_type == AgentType.ML_AGENT
        assert ml_agent.huggingface_api_key == 'test-key'
        assert ml_agent.confidence_threshold == 0.7
        assert ml_agent.timeout == 30
    
    def test_ml_agent_capabilities(self, ml_agent):
        """Test ML agent capabilities"""
        capabilities = ml_agent.get_capabilities()
        
        assert capabilities.agent_type == AgentType.ML_AGENT
        assert ClaimType.TEXT in capabilities.supported_claim_types
        assert ClaimType.MULTIMODAL in capabilities.supported_claim_types
        assert capabilities.confidence_range == (0.1, 1.0)
        assert capabilities.processing_time_estimate == 30.0
    
    @pytest.mark.asyncio
    async def test_ml_agent_verify_claim_success(self, ml_agent, sample_claim):
        """Test successful ML agent verification"""
        # Mock HuggingFace API responses
        mock_responses = {
            'fact_check': [{'scores': [{'label': 'ENTAILMENT', 'score': 0.8}]}],
            'sentiment': [[{'label': 'NEUTRAL', 'score': 0.7}]],
            'ner': [{'word': 'Earth', 'entity': 'LOCATION', 'score': 0.9}]
        }
        
        with patch.object(ml_agent, '_query_huggingface_model') as mock_query:
            mock_query.side_effect = lambda model, payload: mock_responses.get(
                'fact_check' if 'mnli' in model else 
                'sentiment' if 'sentiment' in model else 'ner',
                {}
            )
            
            result = await ml_agent.verify_claim(sample_claim)
            
            assert isinstance(result, VerificationResult)
            assert result.claim_id == sample_claim.id
            assert result.agent_type == AgentType.ML_AGENT
            assert 0.0 <= result.confidence <= 1.0
            assert result.verdict in [Verdict.TRUE, Verdict.FALSE, Verdict.INCONCLUSIVE, Verdict.MISLEADING]
            assert len(result.evidence) > 0
    
    @pytest.mark.asyncio
    async def test_ml_agent_verify_claim_api_failure(self, ml_agent, sample_claim):
        """Test ML agent with API failure"""
        with patch.object(ml_agent, '_query_huggingface_model') as mock_query:
            mock_query.side_effect = Exception("API Error")
            
            result = await ml_agent.verify_claim(sample_claim)
            
            assert result.verdict == Verdict.INCONCLUSIVE
            assert result.confidence == 0.0
            assert "ML Agent error" in result.reasoning
    
    def test_ml_agent_prepare_claim_text(self, ml_agent):
        """Test claim text preparation"""
        # Test with string content
        claim = Claim(content="Test claim content", claim_type=ClaimType.TEXT)
        text = ml_agent._prepare_claim_text(claim)
        assert text == "Test claim content"
        
        # Test with binary content
        claim_binary = Claim(content=b"binary data", claim_type=ClaimType.IMAGE)
        text_binary = ml_agent._prepare_claim_text(claim_binary)
        assert text_binary == ""  # Should return empty for binary without extracted text
    
    @pytest.mark.asyncio
    async def test_ml_agent_linguistic_analysis(self, ml_agent):
        """Test linguistic analysis"""
        text = "This is definitely the best solution ever! It's absolutely amazing and incredible."
        
        result = await ml_agent._linguistic_analysis(text)
        
        assert result is not None
        assert result['model'] == 'linguistic_patterns'
        patterns = result['results']
        assert patterns['certainty_indicators'] > 0  # "definitely", "absolutely"
        assert patterns['superlatives'] > 0  # "best", "ever"
        assert patterns['emotional_language'] > 0  # "amazing", "incredible"


class TestWikipediaAgent:
    """Test Wikipedia Agent"""
    
    @pytest.fixture
    def wikipedia_agent_config(self):
        return {
            'max_search_results': 5,
            'similarity_threshold': 0.8,
            'language': 'en'
        }
    
    @pytest.fixture
    def wikipedia_agent(self, wikipedia_agent_config):
        return WikipediaAgent(wikipedia_agent_config)
    
    def test_wikipedia_agent_initialization(self, wikipedia_agent):
        """Test Wikipedia agent initialization"""
        assert wikipedia_agent.agent_type == AgentType.WIKIPEDIA_AGENT
        assert wikipedia_agent.max_search_results == 5
        assert wikipedia_agent.similarity_threshold == 0.8
        assert wikipedia_agent.language == 'en'
    
    def test_wikipedia_agent_capabilities(self, wikipedia_agent):
        """Test Wikipedia agent capabilities"""
        capabilities = wikipedia_agent.get_capabilities()
        
        assert capabilities.agent_type == AgentType.WIKIPEDIA_AGENT
        assert ClaimType.TEXT in capabilities.supported_claim_types
        assert capabilities.processing_time_estimate == 15.0
    
    def test_extract_entities(self, wikipedia_agent):
        """Test entity extraction"""
        text = "Barack Obama was the President of the United States"
        entities = wikipedia_agent._extract_entities(text)
        
        assert "Barack Obama" in entities
        assert "President" in entities
        assert "United States" in entities
    
    @pytest.mark.asyncio
    async def test_wikipedia_agent_verify_claim_success(self, wikipedia_agent, sample_claim):
        """Test successful Wikipedia verification"""
        # Mock Wikipedia search and page results
        mock_page = Mock()
        mock_page.title = "Earth"
        mock_page.summary = "Earth is the third planet from the Sun"
        mock_page.content = "Earth is a terrestrial planet with a solid surface"
        mock_page.url = "https://en.wikipedia.org/wiki/Earth"
        
        with patch('wikipedia.search') as mock_search, \
             patch('wikipedia.page') as mock_page_func:
            
            mock_search.return_value = ["Earth", "Solar System"]
            mock_page_func.return_value = mock_page
            
            result = await wikipedia_agent.verify_claim(sample_claim)
            
            assert isinstance(result, VerificationResult)
            assert result.agent_type == AgentType.WIKIPEDIA_AGENT
            assert len(result.evidence) > 0
            
            # Check evidence
            evidence = result.evidence[0]
            assert evidence.source == "en.wikipedia.org"
            assert evidence.evidence_type == "wikipedia_article"
            assert evidence.url == "https://en.wikipedia.org/wiki/Earth"
    
    @pytest.mark.asyncio
    async def test_wikipedia_agent_disambiguation_handling(self, wikipedia_agent, sample_claim):
        """Test handling of Wikipedia disambiguation pages"""
        import wikipedia
        
        with patch('wikipedia.search') as mock_search, \
             patch('wikipedia.page') as mock_page_func:
            
            mock_search.return_value = ["Earth (disambiguation)"]
            
            # Simulate disambiguation error
            disambiguation_error = wikipedia.exceptions.DisambiguationError(
                "Earth", ["Earth (planet)", "Earth (element)"]
            )
            mock_page_func.side_effect = disambiguation_error
            
            result = await wikipedia_agent.verify_claim(sample_claim)
            
            # Should handle gracefully
            assert isinstance(result, VerificationResult)


class TestCoherenceAgent:
    """Test Coherence Agent"""
    
    @pytest.fixture
    def coherence_agent_config(self):
        return {
            'consistency_threshold': 0.6,
            'temporal_window_hours': 24,
            'contradiction_threshold': 0.8
        }
    
    @pytest.fixture
    def coherence_agent(self, coherence_agent_config):
        return CoherenceAgent(coherence_agent_config)
    
    def test_coherence_agent_initialization(self, coherence_agent):
        """Test Coherence agent initialization"""
        assert coherence_agent.agent_type == AgentType.COHERENCE_AGENT
        assert coherence_agent.consistency_threshold == 0.6
        assert coherence_agent.temporal_window_hours == 24
    
    def test_coherence_agent_capabilities(self, coherence_agent):
        """Test Coherence agent capabilities"""
        capabilities = coherence_agent.get_capabilities()
        
        assert capabilities.agent_type == AgentType.COHERENCE_AGENT
        assert ClaimType.TEXT in capabilities.supported_claim_types
        assert capabilities.processing_time_estimate == 10.0
    
    @pytest.mark.asyncio
    async def test_coherence_agent_internal_consistency(self, coherence_agent):
        """Test internal consistency checking"""
        # Test contradictory text
        contradictory_text = "All cats are black. Some cats are white."
        evidence = await coherence_agent._check_internal_consistency(contradictory_text)
        
        assert evidence is not None
        assert evidence.evidence_type == "internal_consistency"
        assert evidence.credibility_score < 0.8  # Should detect contradiction
    
    @pytest.mark.asyncio
    async def test_coherence_agent_temporal_consistency(self, coherence_agent):
        """Test temporal consistency checking"""
        # Test with temporal references
        temporal_text = "In 2025, the event happened last year in 2020."
        evidence = await coherence_agent._check_temporal_consistency(temporal_text)
        
        assert evidence is not None
        assert evidence.evidence_type == "temporal_consistency"
        # Should detect temporal inconsistency
    
    @pytest.mark.asyncio
    async def test_coherence_agent_logical_structure(self, coherence_agent):
        """Test logical structure analysis"""
        # Test with logical connectors
        logical_text = "Because the sun is hot, therefore water evaporates. However, ice melts slowly."
        evidence = await coherence_agent._analyze_logical_structure(logical_text)
        
        assert evidence is not None
        assert evidence.evidence_type == "logical_structure"
        assert evidence.credibility_score > 0.5  # Should recognize logical structure
    
    def test_are_contradictory(self, coherence_agent):
        """Test contradiction detection between sentences"""
        sent1 = "The sky is blue"
        sent2 = "The sky is not blue"
        
        assert coherence_agent._are_contradictory(sent1, sent2)
        
        sent3 = "The sky is blue"
        sent4 = "The ocean is deep"
        
        assert not coherence_agent._are_contradictory(sent3, sent4)


class TestWebScrapeAgent:
    """Test WebScrape Agent"""
    
    @pytest.fixture
    def webscrape_agent_config(self):
        return {
            'max_pages_per_claim': 3,
            'timeout_seconds': 30,
            'respect_robots_txt': True,
            'search_tool': {'serper_api_key': 'test-key'},
            'scraper_tool': {'firecrawl_url': 'http://localhost:3002'}
        }
    
    @pytest.fixture
    def webscrape_agent(self, webscrape_agent_config):
        return WebScrapeAgent(webscrape_agent_config)
    
    def test_webscrape_agent_initialization(self, webscrape_agent):
        """Test WebScrape agent initialization"""
        assert webscrape_agent.agent_type == AgentType.WEBSCRAPE_AGENT
        assert webscrape_agent.max_pages_per_claim == 3
        assert webscrape_agent.timeout_seconds == 30
        assert webscrape_agent.respect_robots_txt == True
    
    def test_webscrape_agent_capabilities(self, webscrape_agent):
        """Test WebScrape agent capabilities"""
        capabilities = webscrape_agent.get_capabilities()
        
        assert capabilities.agent_type == AgentType.WEBSCRAPE_AGENT
        assert ClaimType.TEXT in capabilities.supported_claim_types
        assert capabilities.processing_time_estimate == 45.0
    
    def test_classify_source_type(self, webscrape_agent):
        """Test source type classification"""
        # Test academic domain
        assert webscrape_agent._classify_source_type("harvard.edu") == "academic_sites"
        
        # Test government domain
        assert webscrape_agent._classify_source_type("cdc.gov") == "government_sites"
        
        # Test fact-check domain
        assert webscrape_agent._classify_source_type("snopes.com") == "fact_check_sites"
        
        # Test news domain
        assert webscrape_agent._classify_source_type("bbc.com") == "news_sites"
        
        # Test social media
        assert webscrape_agent._classify_source_type("twitter.com") == "social_media"
        
        # Test unknown domain
        assert webscrape_agent._classify_source_type("random-site.com") == "unknown"
    
    def test_generate_search_queries(self, webscrape_agent):
        """Test search query generation"""
        claim_text = "Climate change is caused by human activities"
        queries = webscrape_agent._generate_search_queries(claim_text)
        
        assert len(queries) > 0
        assert any("fact check" in query.lower() for query in queries)
        assert any("debunk" in query.lower() for query in queries)
        assert claim_text[:50] in queries[0]  # Original claim should be first
    
    @pytest.mark.asyncio
    async def test_webscrape_agent_verify_claim_success(self, webscrape_agent, sample_claim):
        """Test successful web scraping verification"""
        # Mock search tool results
        mock_search_evidence = [
            Evidence(
                source="example.com",
                content="Test search result content",
                url="https://example.com/article",
                credibility_score=0.8,
                relevance_score=0.7,
                evidence_type="web_search"
            )
        ]
        
        with patch.object(webscrape_agent.search_tool, 'gather_evidence') as mock_search:
            mock_search.return_value = mock_search_evidence
            
            result = await webscrape_agent.verify_claim(sample_claim)
            
            assert isinstance(result, VerificationResult)
            assert result.agent_type == AgentType.WEBSCRAPE_AGENT
            assert len(result.evidence) > 0
    
    @pytest.mark.asyncio
    async def test_webscrape_agent_source_credibility_analysis(self, webscrape_agent, sample_claim):
        """Test source credibility analysis"""
        # Create evidence with different source types
        evidence_list = [
            Evidence(source="reuters.com", url="https://reuters.com/article", credibility_score=0.9),
            Evidence(source="twitter.com", url="https://twitter.com/post", credibility_score=0.3),
            Evidence(source="nature.com", url="https://nature.com/article", credibility_score=0.95)
        ]
        
        credibility_evidence = await webscrape_agent._analyze_source_credibility(sample_claim, evidence_list)
        
        assert credibility_evidence is not None
        assert credibility_evidence.evidence_type == "source_credibility"
        
        # Should have high credibility due to Reuters and Nature
        assert credibility_evidence.credibility_score > 0.7
        
        metadata = credibility_evidence.metadata
        assert "source_analysis" in metadata
        assert metadata["total_sources"] == 3
        assert metadata["high_credibility_count"] >= 2  # Reuters and Nature


# Integration tests for agent interactions
class TestAgentIntegration:
    """Test agent integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_multiple_agents_same_claim(self, sample_claim):
        """Test multiple agents processing the same claim"""
        # Initialize agents with minimal config
        ml_agent = MLAgent({'huggingface_api_key': 'test'})
        coherence_agent = CoherenceAgent({})
        
        # Mock external dependencies
        with patch.object(ml_agent, '_query_huggingface_model') as mock_ml, \
             patch('wikipedia.search') as mock_wiki_search:
            
            mock_ml.return_value = [{'label': 'ENTAILMENT', 'score': 0.8}]
            mock_wiki_search.return_value = []
            
            # Process claim with both agents
            ml_result = await ml_agent.verify_claim(sample_claim)
            coherence_result = await coherence_agent.verify_claim(sample_claim)
            
            # Both should return valid results
            assert isinstance(ml_result, VerificationResult)
            assert isinstance(coherence_result, VerificationResult)
            
            # Both should have the same claim ID
            assert ml_result.claim_id == coherence_result.claim_id == sample_claim.id
            
            # Should have different agent types
            assert ml_result.agent_type != coherence_result.agent_type
    
    @pytest.mark.asyncio
    async def test_agent_error_handling(self, sample_claim):
        """Test agent error handling"""
        agent = MLAgent({'huggingface_api_key': 'invalid-key'})
        
        # Should handle errors gracefully
        result = await agent.verify_claim(sample_claim)
        
        assert isinstance(result, VerificationResult)
        assert result.verdict == Verdict.INCONCLUSIVE
        assert result.confidence == 0.0
        assert "error" in result.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_agent_timeout_handling(self, sample_claim):
        """Test agent timeout handling"""
        agent = MLAgent({'huggingface_api_key': 'test', 'timeout': 0.001})  # Very short timeout
        
        with patch.object(agent, '_query_huggingface_model') as mock_query:
            # Simulate slow response
            async def slow_response(*args, **kwargs):
                await asyncio.sleep(1)  # Longer than timeout
                return {}
            
            mock_query.side_effect = slow_response
            
            result = await agent._process_with_timeout(sample_claim, timeout=0.001)
            
            assert isinstance(result, VerificationResult)
            assert "timed out" in result.reasoning.lower()