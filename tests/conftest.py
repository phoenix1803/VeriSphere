"""
Pytest configuration and fixtures for VeriSphere
"""
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import Mock, AsyncMock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from defame.core.database import Base, DatabaseSession
from defame.core.models import Claim
from defame.core.agent_manager import AgentManager
from defame.core.pipeline import PipelineController
from defame.core.mcp_orchestration import MCPOrchestrator
from defame.auth.service import AuthService
from config.globals import ClaimType, Priority


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def test_db():
    """Create test database"""
    # Use in-memory SQLite for tests
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    yield TestSessionLocal
    
    Base.metadata.drop_all(engine)


@pytest.fixture
def db_session(test_db):
    """Create database session for tests"""
    session = test_db()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def test_config():
    """Test configuration"""
    return {
        'database_url': 'sqlite:///:memory:',
        'redis_url': 'redis://localhost:6379/1',
        'debug': True,
        'log_level': 'DEBUG',
        'max_concurrent_claims': 5,
        'agent_timeout_seconds': 30,
    }


@pytest.fixture
def mock_agent():
    """Mock agent for testing"""
    agent = Mock()
    agent.agent_id = "test_agent_123"
    agent.agent_type = Mock()
    agent.agent_type.value = "test_agent"
    agent.is_healthy = True
    agent.processing_count = 0
    agent.error_count = 0
    
    # Mock async methods
    agent.verify_claim = AsyncMock()
    agent.health_check = AsyncMock(return_value=True)
    agent.get_capabilities = Mock()
    
    return agent


@pytest.fixture
def sample_claim():
    """Sample claim for testing"""
    return Claim(
        content="The Earth is round and orbits the Sun",
        claim_type=ClaimType.TEXT,
        priority=Priority.NORMAL,
        source="test_source"
    )


@pytest.fixture
def sample_claims():
    """Multiple sample claims for testing"""
    return [
        Claim(
            content="The Earth is round",
            claim_type=ClaimType.TEXT,
            priority=Priority.NORMAL
        ),
        Claim(
            content="Water boils at 100°C at sea level",
            claim_type=ClaimType.TEXT,
            priority=Priority.LOW
        ),
        Claim(
            content="The moon landing was faked",
            claim_type=ClaimType.TEXT,
            priority=Priority.HIGH
        )
    ]


@pytest.fixture
def agent_manager(test_config, mock_agent):
    """Agent manager for testing"""
    manager = AgentManager(test_config)
    # Replace real agents with mock
    manager.agents = {mock_agent.agent_id: mock_agent}
    return manager


@pytest.fixture
def pipeline_controller(test_config):
    """Pipeline controller for testing"""
    return PipelineController(test_config)


@pytest.fixture
def mcp_orchestrator(test_config):
    """MCP orchestrator for testing"""
    return MCPOrchestrator(test_config)


@pytest.fixture
def test_user(db_session):
    """Create test user"""
    # Initialize auth data first
    AuthService.initialize_default_data()
    
    user = AuthService.create_user(
        username="testuser",
        email="test@example.com",
        password="testpass123",
        full_name="Test User"
    )
    return user


@pytest.fixture
def api_client(test_config):
    """FastAPI test client"""
    from scripts.run_api import app
    return TestClient(app)


@pytest.fixture
def mock_external_apis():
    """Mock external API responses"""
    mocks = {
        'serper_api': Mock(),
        'huggingface_api': Mock(),
        'google_vision_api': Mock(),
        'firecrawl_api': Mock(),
        'wikipedia_api': Mock()
    }
    
    # Configure mock responses
    mocks['serper_api'].post.return_value.status_code = 200
    mocks['serper_api'].post.return_value.json.return_value = {
        'organic': [
            {
                'title': 'Test Result',
                'link': 'https://example.com',
                'snippet': 'Test snippet'
            }
        ]
    }
    
    mocks['huggingface_api'].post.return_value.status_code = 200
    mocks['huggingface_api'].post.return_value.json.return_value = [
        {'label': 'ENTAILMENT', 'score': 0.8}
    ]
    
    return mocks


@pytest.fixture
def performance_test_config():
    """Configuration for performance tests"""
    return {
        'max_claims': 100,
        'concurrent_claims': 10,
        'timeout_seconds': 300,
        'target_throughput': 10,  # claims per second
        'max_response_time': 30,  # seconds
    }


# Async test helpers
@pytest.fixture
def async_test():
    """Decorator for async tests"""
    def decorator(func):
        return pytest.mark.asyncio(func)
    return decorator


# Test data generators
@pytest.fixture
def claim_generator():
    """Generate test claims"""
    def generate(count: int = 10, claim_type: ClaimType = ClaimType.TEXT):
        claims = []
        for i in range(count):
            claim = Claim(
                content=f"Test claim number {i+1}",
                claim_type=claim_type,
                priority=Priority.NORMAL,
                source=f"test_source_{i}"
            )
            claims.append(claim)
        return claims
    return generate


# Mock services
@pytest.fixture
def mock_search_service():
    """Mock search service"""
    service = Mock()
    service.search = AsyncMock(return_value=[
        {
            'title': 'Test Search Result',
            'url': 'https://example.com',
            'snippet': 'Test snippet content'
        }
    ])
    return service


@pytest.fixture
def mock_ml_service():
    """Mock ML service"""
    service = Mock()
    service.classify = AsyncMock(return_value={
        'verdict': 'true',
        'confidence': 0.85,
        'reasoning': 'Test ML reasoning'
    })
    return service


# Integration test fixtures
@pytest.fixture
def integration_test_env():
    """Setup for integration tests"""
    return {
        'database_ready': True,
        'redis_ready': True,
        'external_apis_ready': False,  # Set to True when testing with real APIs
    }


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test"""
    yield
    # Cleanup code here if needed
    pass