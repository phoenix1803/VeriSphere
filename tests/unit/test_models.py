"""
Unit tests for VeriSphere core models
"""
import pytest
from datetime import datetime
import json

from defame.core.models import (
    Claim, Evidence, VerificationResult, PipelineState, 
    AgentCapability, ProcessingMetrics, AuditEntry, ModelEncoder
)
from config.globals import ClaimType, Verdict, Priority, PipelineStage, AgentType


class TestClaim:
    """Test Claim model"""
    
    def test_claim_creation(self):
        """Test basic claim creation"""
        claim = Claim(
            content="Test claim content",
            claim_type=ClaimType.TEXT,
            priority=Priority.HIGH,
            source="test_source"
        )
        
        assert claim.content == "Test claim content"
        assert claim.claim_type == ClaimType.TEXT
        assert claim.priority == Priority.HIGH
        assert claim.source == "test_source"
        assert claim.status == "pending"
        assert isinstance(claim.timestamp, datetime)
        assert len(claim.id) > 0
    
    def test_claim_to_dict(self):
        """Test claim serialization to dictionary"""
        claim = Claim(
            content="Test content",
            claim_type=ClaimType.IMAGE,
            priority=Priority.CRITICAL
        )
        
        claim_dict = claim.to_dict()
        
        assert claim_dict["content"] == "Test content"
        assert claim_dict["claim_type"] == "image"
        assert claim_dict["priority"] == "critical"
        assert claim_dict["status"] == "pending"
        assert "timestamp" in claim_dict
        assert "id" in claim_dict
    
    def test_claim_from_dict(self):
        """Test claim deserialization from dictionary"""
        data = {
            "id": "test-id-123",
            "content": "Test content",
            "claim_type": "text",
            "priority": "normal",
            "source": "test_source",
            "status": "processing"
        }
        
        claim = Claim.from_dict(data)
        
        assert claim.id == "test-id-123"
        assert claim.content == "Test content"
        assert claim.claim_type == ClaimType.TEXT
        assert claim.priority == Priority.NORMAL
        assert claim.source == "test_source"
        assert claim.status == "processing"
    
    def test_claim_binary_content(self):
        """Test claim with binary content"""
        binary_data = b"binary image data"
        claim = Claim(
            content=binary_data,
            claim_type=ClaimType.IMAGE
        )
        
        assert claim.content == binary_data
        
        # Test serialization handles binary data
        claim_dict = claim.to_dict()
        assert claim_dict["content"] == "<binary_data>"


class TestEvidence:
    """Test Evidence model"""
    
    def test_evidence_creation(self):
        """Test evidence creation"""
        evidence = Evidence(
            source="Test Source",
            content="Test evidence content",
            url="https://example.com",
            credibility_score=0.8,
            relevance_score=0.9,
            evidence_type="web_search"
        )
        
        assert evidence.source == "Test Source"
        assert evidence.content == "Test evidence content"
        assert evidence.url == "https://example.com"
        assert evidence.credibility_score == 0.8
        assert evidence.relevance_score == 0.9
        assert evidence.evidence_type == "web_search"
        assert isinstance(evidence.timestamp, datetime)
    
    def test_evidence_serialization(self):
        """Test evidence serialization"""
        evidence = Evidence(
            source="Test Source",
            content="Test content",
            credibility_score=0.75
        )
        
        evidence_dict = evidence.to_dict()
        
        assert evidence_dict["source"] == "Test Source"
        assert evidence_dict["content"] == "Test content"
        assert evidence_dict["credibility_score"] == 0.75
        assert "timestamp" in evidence_dict
        assert "id" in evidence_dict
    
    def test_evidence_from_dict(self):
        """Test evidence deserialization"""
        data = {
            "id": "evidence-123",
            "source": "Wikipedia",
            "content": "Wikipedia content",
            "credibility_score": 0.85,
            "relevance_score": 0.7,
            "evidence_type": "wikipedia_article"
        }
        
        evidence = Evidence.from_dict(data)
        
        assert evidence.id == "evidence-123"
        assert evidence.source == "Wikipedia"
        assert evidence.content == "Wikipedia content"
        assert evidence.credibility_score == 0.85
        assert evidence.relevance_score == 0.7
        assert evidence.evidence_type == "wikipedia_article"


class TestVerificationResult:
    """Test VerificationResult model"""
    
    def test_verification_result_creation(self):
        """Test verification result creation"""
        evidence = Evidence(source="Test", content="Test evidence")
        
        result = VerificationResult(
            claim_id="claim-123",
            agent_id="agent-456",
            agent_type=AgentType.ML_AGENT,
            confidence=0.85,
            verdict=Verdict.TRUE,
            evidence=[evidence],
            reasoning="Test reasoning",
            processing_time=2.5
        )
        
        assert result.claim_id == "claim-123"
        assert result.agent_id == "agent-456"
        assert result.agent_type == AgentType.ML_AGENT
        assert result.confidence == 0.85
        assert result.verdict == Verdict.TRUE
        assert len(result.evidence) == 1
        assert result.reasoning == "Test reasoning"
        assert result.processing_time == 2.5
    
    def test_verification_result_serialization(self):
        """Test verification result serialization"""
        evidence = Evidence(source="Test", content="Test evidence")
        
        result = VerificationResult(
            claim_id="claim-123",
            agent_type=AgentType.WIKIPEDIA_AGENT,
            verdict=Verdict.FALSE,
            evidence=[evidence]
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["claim_id"] == "claim-123"
        assert result_dict["agent_type"] == "wikipedia_agent"
        assert result_dict["verdict"] == "false"
        assert len(result_dict["evidence"]) == 1
        assert isinstance(result_dict["evidence"][0], dict)


class TestPipelineState:
    """Test PipelineState model"""
    
    def test_pipeline_state_creation(self):
        """Test pipeline state creation"""
        state = PipelineState(
            claim_id="claim-123",
            current_stage=PipelineStage.EVIDENCE,
            overall_confidence=0.7,
            overall_verdict=Verdict.INCONCLUSIVE
        )
        
        assert state.claim_id == "claim-123"
        assert state.current_stage == PipelineStage.EVIDENCE
        assert state.overall_confidence == 0.7
        assert state.overall_verdict == Verdict.INCONCLUSIVE
        assert not state.is_complete
        assert isinstance(state.start_time, datetime)
    
    def test_pipeline_state_completion(self):
        """Test pipeline state completion detection"""
        state = PipelineState(
            claim_id="claim-123",
            current_stage=PipelineStage.EVALUATION,
            end_time=datetime.utcnow()
        )
        
        assert state.is_complete
        assert state.processing_time > 0
    
    def test_pipeline_state_serialization(self):
        """Test pipeline state serialization"""
        result = VerificationResult(
            claim_id="claim-123",
            agent_type=AgentType.ML_AGENT,
            verdict=Verdict.TRUE
        )
        
        state = PipelineState(
            claim_id="claim-123",
            current_stage=PipelineStage.ANALYSIS,
            agent_results=[result]
        )
        
        state_dict = state.to_dict()
        
        assert state_dict["claim_id"] == "claim-123"
        assert state_dict["current_stage"] == "analysis"
        assert len(state_dict["agent_results"]) == 1
        assert state_dict["is_complete"] == False
        assert "processing_time" in state_dict


class TestAgentCapability:
    """Test AgentCapability model"""
    
    def test_agent_capability_creation(self):
        """Test agent capability creation"""
        capability = AgentCapability(
            agent_type=AgentType.ML_AGENT,
            supported_claim_types=[ClaimType.TEXT, ClaimType.MULTIMODAL],
            confidence_range=(0.1, 1.0),
            processing_time_estimate=30.0,
            description="ML-based verification"
        )
        
        assert capability.agent_type == AgentType.ML_AGENT
        assert len(capability.supported_claim_types) == 2
        assert capability.confidence_range == (0.1, 1.0)
        assert capability.processing_time_estimate == 30.0
        assert capability.description == "ML-based verification"
    
    def test_can_handle_claim_type(self):
        """Test claim type handling check"""
        capability = AgentCapability(
            agent_type=AgentType.COHERENCE_AGENT,
            supported_claim_types=[ClaimType.TEXT],
            confidence_range=(0.0, 1.0),
            processing_time_estimate=10.0
        )
        
        assert capability.can_handle(ClaimType.TEXT)
        assert not capability.can_handle(ClaimType.IMAGE)


class TestProcessingMetrics:
    """Test ProcessingMetrics model"""
    
    def test_processing_metrics_creation(self):
        """Test processing metrics creation"""
        metrics = ProcessingMetrics(
            claims_processed=100,
            average_processing_time=25.5,
            accuracy_score=0.92,
            error_rate=0.05,
            throughput_per_minute=2.4
        )
        
        assert metrics.claims_processed == 100
        assert metrics.average_processing_time == 25.5
        assert metrics.accuracy_score == 0.92
        assert metrics.error_rate == 0.05
        assert metrics.throughput_per_minute == 2.4
        assert isinstance(metrics.timestamp, datetime)
    
    def test_processing_metrics_serialization(self):
        """Test processing metrics serialization"""
        metrics = ProcessingMetrics(
            claims_processed=50,
            accuracy_score=0.88,
            agent_performance={"agent1": {"requests": 25, "success": 23}}
        )
        
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict["claims_processed"] == 50
        assert metrics_dict["accuracy_score"] == 0.88
        assert "agent_performance" in metrics_dict
        assert "timestamp" in metrics_dict


class TestAuditEntry:
    """Test AuditEntry model"""
    
    def test_audit_entry_creation(self):
        """Test audit entry creation"""
        entry = AuditEntry(
            claim_id="claim-123",
            action="claim_submitted",
            actor="user-456",
            blockchain_hash="0x123abc",
            details={"source": "api", "priority": "high"}
        )
        
        assert entry.claim_id == "claim-123"
        assert entry.action == "claim_submitted"
        assert entry.actor == "user-456"
        assert entry.blockchain_hash == "0x123abc"
        assert entry.details["source"] == "api"
        assert isinstance(entry.timestamp, datetime)
    
    def test_audit_entry_serialization(self):
        """Test audit entry serialization"""
        entry = AuditEntry(
            claim_id="claim-123",
            action="verification_completed",
            actor="system"
        )
        
        entry_dict = entry.to_dict()
        
        assert entry_dict["claim_id"] == "claim-123"
        assert entry_dict["action"] == "verification_completed"
        assert entry_dict["actor"] == "system"
        assert "timestamp" in entry_dict
        assert "id" in entry_dict


class TestModelEncoder:
    """Test ModelEncoder for JSON serialization"""
    
    def test_model_encoder_with_models(self):
        """Test encoder with model objects"""
        claim = Claim(content="Test", claim_type=ClaimType.TEXT)
        evidence = Evidence(source="Test", content="Evidence")
        
        data = {
            "claim": claim,
            "evidence": evidence,
            "timestamp": datetime.utcnow()
        }
        
        # Should not raise exception
        json_str = json.dumps(data, cls=ModelEncoder)
        assert isinstance(json_str, str)
        assert "Test" in json_str
    
    def test_model_encoder_with_enums(self):
        """Test encoder with enum values"""
        data = {
            "claim_type": ClaimType.IMAGE,
            "verdict": Verdict.TRUE,
            "priority": Priority.HIGH
        }
        
        json_str = json.dumps(data, cls=ModelEncoder)
        parsed = json.loads(json_str)
        
        assert parsed["claim_type"] == "image"
        assert parsed["verdict"] == "true"
        assert parsed["priority"] == "high"
    
    def test_model_encoder_with_binary_data(self):
        """Test encoder with binary data"""
        data = {
            "image_data": b"binary image data",
            "text": "normal text"
        }
        
        json_str = json.dumps(data, cls=ModelEncoder)
        parsed = json.loads(json_str)
        
        assert parsed["image_data"] == "<binary_data>"
        assert parsed["text"] == "normal text"


class TestModelUtilities:
    """Test model utility functions"""
    
    def test_save_and_load_model(self, temp_dir):
        """Test saving and loading models to/from files"""
        from defame.core.models import save_model_to_file, load_model_from_file
        
        claim = Claim(
            content="Test claim",
            claim_type=ClaimType.TEXT,
            priority=Priority.HIGH
        )
        
        file_path = temp_dir / "test_claim.json"
        
        # Save model
        save_model_to_file(claim, file_path)
        assert file_path.exists()
        
        # Load model
        loaded_data = load_model_from_file(file_path, Claim)
        loaded_claim = Claim.from_dict(loaded_data)
        
        assert loaded_claim.content == claim.content
        assert loaded_claim.claim_type == claim.claim_type
        assert loaded_claim.priority == claim.priority