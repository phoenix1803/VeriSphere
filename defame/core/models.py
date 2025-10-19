"""
Core data models for VeriSphere misinformation detection system
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import uuid
import json
from pathlib import Path

from config.globals import ClaimType, Verdict, Priority, PipelineStage, AgentType


@dataclass
class Claim:
    """Represents a claim to be fact-checked"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Union[str, bytes] = ""
    claim_type: ClaimType = ClaimType.TEXT
    source: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: Priority = Priority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert claim to dictionary for serialization"""
        return {
            "id": self.id,
            "content": self.content if isinstance(self.content, str) else "<binary_data>",
            "claim_type": self.claim_type.value,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "metadata": self.metadata,
            "status": self.status
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Claim':
        """Create claim from dictionary"""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            content=data.get("content", ""),
            claim_type=ClaimType(data.get("claim_type", "text")),
            source=data.get("source"),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat())),
            priority=Priority(data.get("priority", "normal")),
            metadata=data.get("metadata", {}),
            status=data.get("status", "pending")
        )


@dataclass
class Evidence:
    """Represents a piece of evidence for or against a claim"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = ""
    content: str = ""
    url: Optional[str] = None
    credibility_score: float = 0.0
    relevance_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    evidence_type: str = "text"  # text, image, video, document
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert evidence to dictionary"""
        return {
            "id": self.id,
            "source": self.source,
            "content": self.content,
            "url": self.url,
            "credibility_score": self.credibility_score,
            "relevance_score": self.relevance_score,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "evidence_type": self.evidence_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Evidence':
        """Create evidence from dictionary"""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            source=data.get("source", ""),
            content=data.get("content", ""),
            url=data.get("url"),
            credibility_score=data.get("credibility_score", 0.0),
            relevance_score=data.get("relevance_score", 0.0),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat())),
            metadata=data.get("metadata", {}),
            evidence_type=data.get("evidence_type", "text")
        )


@dataclass
class VerificationResult:
    """Result from an agent's verification process"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    claim_id: str = ""
    agent_id: str = ""
    agent_type: AgentType = AgentType.ML_AGENT
    confidence: float = 0.0  # 0.0 to 1.0
    verdict: Verdict = Verdict.INCONCLUSIVE
    evidence: List[Evidence] = field(default_factory=list)
    reasoning: str = ""
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "id": self.id,
            "claim_id": self.claim_id,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "confidence": self.confidence,
            "verdict": self.verdict.value,
            "evidence": [e.to_dict() for e in self.evidence],
            "reasoning": self.reasoning,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VerificationResult':
        """Create result from dictionary"""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            claim_id=data.get("claim_id", ""),
            agent_id=data.get("agent_id", ""),
            agent_type=AgentType(data.get("agent_type", "ml_agent")),
            confidence=data.get("confidence", 0.0),
            verdict=Verdict(data.get("verdict", "inconclusive")),
            evidence=[Evidence.from_dict(e) for e in data.get("evidence", [])],
            reasoning=data.get("reasoning", ""),
            processing_time=data.get("processing_time", 0.0),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat())),
            metadata=data.get("metadata", {})
        )


@dataclass
class PipelineState:
    """State of the verification pipeline for a claim"""
    claim_id: str = ""
    current_stage: PipelineStage = PipelineStage.DETECTION
    stage_results: Dict[PipelineStage, Any] = field(default_factory=dict)
    overall_confidence: float = 0.0
    overall_verdict: Verdict = Verdict.INCONCLUSIVE
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    agent_results: List[VerificationResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_complete(self) -> bool:
        """Check if pipeline is complete"""
        return self.current_stage == PipelineStage.EVALUATION and self.end_time is not None
    
    @property
    def processing_time(self) -> float:
        """Get total processing time in seconds"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.utcnow() - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary"""
        return {
            "claim_id": self.claim_id,
            "current_stage": self.current_stage.value,
            "stage_results": {k.value: v for k, v in self.stage_results.items()},
            "overall_confidence": self.overall_confidence,
            "overall_verdict": self.overall_verdict.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "estimated_completion": self.estimated_completion.isoformat() if self.estimated_completion else None,
            "agent_results": [r.to_dict() for r in self.agent_results],
            "errors": self.errors,
            "metadata": self.metadata,
            "is_complete": self.is_complete,
            "processing_time": self.processing_time
        }


@dataclass
class AgentCapability:
    """Describes what an agent can do"""
    agent_type: AgentType
    supported_claim_types: List[ClaimType]
    confidence_range: tuple[float, float]  # min, max confidence this agent can provide
    processing_time_estimate: float  # seconds
    description: str = ""
    
    def can_handle(self, claim_type: ClaimType) -> bool:
        """Check if agent can handle this claim type"""
        return claim_type in self.supported_claim_types


@dataclass
class ProcessingMetrics:
    """Metrics for monitoring system performance"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    claims_processed: int = 0
    average_processing_time: float = 0.0
    accuracy_score: float = 0.0
    agent_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    error_rate: float = 0.0
    throughput_per_minute: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "claims_processed": self.claims_processed,
            "average_processing_time": self.average_processing_time,
            "accuracy_score": self.accuracy_score,
            "agent_performance": self.agent_performance,
            "error_rate": self.error_rate,
            "throughput_per_minute": self.throughput_per_minute
        }


@dataclass
class AuditEntry:
    """Blockchain audit trail entry"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    claim_id: str = ""
    action: str = ""
    actor: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    blockchain_hash: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit entry to dictionary"""
        return {
            "id": self.id,
            "claim_id": self.claim_id,
            "action": self.action,
            "actor": self.actor,
            "timestamp": self.timestamp.isoformat(),
            "blockchain_hash": self.blockchain_hash,
            "details": self.details
        }


class ModelEncoder(json.JSONEncoder):
    """Custom JSON encoder for VeriSphere models"""
    
    def default(self, obj):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, bytes):
            return "<binary_data>"
        return super().default(obj)


def save_model_to_file(model: Any, filepath: Union[str, Path]) -> None:
    """Save a model to JSON file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(model, f, cls=ModelEncoder, indent=2, ensure_ascii=False)


def load_model_from_file(filepath: Union[str, Path], model_class: type) -> Any:
    """Load a model from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if hasattr(model_class, 'from_dict'):
        return model_class.from_dict(data)
    return data