"""
Global configuration and constants for VeriSphere
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field
from enum import Enum


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class ClaimType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"


class Verdict(str, Enum):
    TRUE = "true"
    FALSE = "false"
    INCONCLUSIVE = "inconclusive"
    MISLEADING = "misleading"


class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class PipelineStage(str, Enum):
    DETECTION = "detection"
    EVIDENCE = "evidence"
    FACT_CHECKING = "fact_checking"
    ANALYSIS = "analysis"
    MITIGATION = "mitigation"
    EVALUATION = "evaluation"


class AgentType(str, Enum):
    ML_AGENT = "ml_agent"
    WIKIPEDIA_AGENT = "wikipedia_agent"
    COHERENCE_AGENT = "coherence_agent"
    WEBSCRAPE_AGENT = "webscrape_agent"


class GlobalConfig(BaseSettings):
    """Global configuration settings"""
    
    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=True)
    log_level: str = Field(default="INFO")
    
    # Database
    database_url: str = Field(default="postgresql://verisphere:verisphere123@localhost:5432/verisphere")
    redis_url: str = Field(default="redis://localhost:6379/0")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=4)
    
    # Security
    secret_key: str = Field(default="dev-secret-key")
    jwt_secret_key: str = Field(default="dev-jwt-secret")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expire_minutes: int = Field(default=1440)
    
    # Processing
    max_concurrent_claims: int = Field(default=50)
    claim_timeout_seconds: int = Field(default=300)
    agent_timeout_seconds: int = Field(default=60)
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=100)
    rate_limit_burst: int = Field(default=20)
    
    # Cache
    cache_ttl_seconds: int = Field(default=3600)
    search_cache_ttl_seconds: int = Field(default=1800)
    
    # File Upload
    upload_max_size: int = Field(default=10485760)  # 10MB
    allowed_image_types: str = Field(default="jpg,jpeg,png,webp,gif")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global constants
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "out"
LOGS_DIR = OUTPUT_DIR / "logs"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Ensure directories exist
for directory in [OUTPUT_DIR, LOGS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Agent configuration
AGENT_CONFIGS = {
    AgentType.ML_AGENT: {
        "model_name": "bert-base-uncased",
        "confidence_threshold": 0.7,
        "max_sequence_length": 512,
    },
    AgentType.WIKIPEDIA_AGENT: {
        "max_search_results": 10,
        "similarity_threshold": 0.8,
        "language": "en",
    },
    AgentType.COHERENCE_AGENT: {
        "consistency_threshold": 0.6,
        "temporal_window_hours": 24,
        "contradiction_threshold": 0.8,
    },
    AgentType.WEBSCRAPE_AGENT: {
        "max_pages_per_claim": 5,
        "timeout_seconds": 30,
        "respect_robots_txt": True,
    }
}

# Pipeline configuration
PIPELINE_CONFIG = {
    "max_evidence_sources": 20,
    "min_confidence_threshold": 0.5,
    "consensus_weight_threshold": 0.6,
    "stage_timeout_seconds": 120,
}

# Crisis communication thresholds
CRISIS_THRESHOLDS = {
    "viral_share_count": 1000,
    "high_impact_keywords": [
        "emergency", "disaster", "outbreak", "attack", "crisis",
        "breaking", "urgent", "alert", "warning", "evacuation"
    ],
    "sentiment_volatility_threshold": 0.8,
    "misinformation_confidence_threshold": 0.9,
}

# Blockchain configuration
BLOCKCHAIN_CONFIG = {
    "network": "testnet",
    "gas_limit": 100000,
    "confirmation_blocks": 3,
    "retry_attempts": 3,
}

# Load global configuration instance
config = GlobalConfig()


def get_config() -> GlobalConfig:
    """Get global configuration instance"""
    return config


def get_agent_config(agent_type: AgentType) -> Dict[str, Any]:
    """Get configuration for specific agent type"""
    return AGENT_CONFIGS.get(agent_type, {})


def get_pipeline_config() -> Dict[str, Any]:
    """Get pipeline configuration"""
    return PIPELINE_CONFIG


def get_crisis_config() -> Dict[str, Any]:
    """Get crisis communication configuration"""
    return CRISIS_THRESHOLDS


def get_blockchain_config() -> Dict[str, Any]:
    """Get blockchain configuration"""
    return BLOCKCHAIN_CONFIG