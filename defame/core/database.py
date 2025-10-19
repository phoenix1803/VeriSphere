"""
Database models and schema for VeriSphere
"""
from sqlalchemy import (
    create_engine, Column, String, Text, DateTime, Float, Integer, 
    Boolean, JSON, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import uuid
import json

from config.globals import get_config

# Database configuration
config = get_config()
engine = create_engine(
    config.database_url,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ClaimModel(Base):
    """Database model for claims"""
    __tablename__ = "claims"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content = Column(Text, nullable=False)
    claim_type = Column(String(20), nullable=False, index=True)
    source = Column(String(500), nullable=True)
    priority = Column(String(10), nullable=False, default="normal", index=True)
    status = Column(String(20), nullable=False, default="pending", index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    verification_results = relationship("VerificationResultModel", back_populates="claim", cascade="all, delete-orphan")
    audit_entries = relationship("AuditTrailModel", back_populates="claim", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_claims_type_status', 'claim_type', 'status'),
        Index('idx_claims_created_priority', 'created_at', 'priority'),
    )


class VerificationResultModel(Base):
    """Database model for verification results"""
    __tablename__ = "verification_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id = Column(UUID(as_uuid=True), ForeignKey('claims.id'), nullable=False, index=True)
    agent_id = Column(String(100), nullable=False, index=True)
    agent_type = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False, index=True)
    verdict = Column(String(20), nullable=False, index=True)
    reasoning = Column(Text, nullable=True)
    processing_time = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    evidence = Column(JSON, nullable=True)
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    claim = relationship("ClaimModel", back_populates="verification_results")
    
    # Indexes
    __table_args__ = (
        Index('idx_results_claim_agent', 'claim_id', 'agent_type'),
        Index('idx_results_confidence_verdict', 'confidence', 'verdict'),
    )


class AuditTrailModel(Base):
    """Database model for audit trail"""
    __tablename__ = "audit_trail"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id = Column(UUID(as_uuid=True), ForeignKey('claims.id'), nullable=False, index=True)
    action = Column(String(50), nullable=False, index=True)
    actor = Column(String(100), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    blockchain_hash = Column(String(64), nullable=True, unique=True)
    details = Column(JSON, nullable=True)
    
    # Relationships
    claim = relationship("ClaimModel", back_populates="audit_entries")
    
    # Indexes
    __table_args__ = (
        Index('idx_audit_claim_action', 'claim_id', 'action'),
        Index('idx_audit_timestamp_actor', 'timestamp', 'actor'),
    )


class AgentMetricsModel(Base):
    """Database model for agent performance metrics"""
    __tablename__ = "agent_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(String(100), nullable=False, index=True)
    agent_type = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    claims_processed = Column(Integer, nullable=False, default=0)
    average_processing_time = Column(Float, nullable=False, default=0.0)
    accuracy_score = Column(Float, nullable=True)
    error_rate = Column(Float, nullable=False, default=0.0)
    throughput_per_minute = Column(Float, nullable=False, default=0.0)
    metadata = Column(JSON, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_metrics_agent_timestamp', 'agent_id', 'timestamp'),
        Index('idx_metrics_type_timestamp', 'agent_type', 'timestamp'),
    )


class SystemMetricsModel(Base):
    """Database model for system-wide metrics"""
    __tablename__ = "system_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    total_claims_processed = Column(Integer, nullable=False, default=0)
    average_processing_time = Column(Float, nullable=False, default=0.0)
    system_accuracy = Column(Float, nullable=True)
    active_agents = Column(Integer, nullable=False, default=0)
    queue_depth = Column(Integer, nullable=False, default=0)
    error_rate = Column(Float, nullable=False, default=0.0)
    throughput_per_minute = Column(Float, nullable=False, default=0.0)
    metadata = Column(JSON, nullable=True)


class EvidenceModel(Base):
    """Database model for evidence storage"""
    __tablename__ = "evidence"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id = Column(UUID(as_uuid=True), ForeignKey('claims.id'), nullable=False, index=True)
    source = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    url = Column(String(1000), nullable=True)
    evidence_type = Column(String(20), nullable=False, default="text", index=True)
    credibility_score = Column(Float, nullable=False, default=0.0, index=True)
    relevance_score = Column(Float, nullable=False, default=0.0, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    metadata = Column(JSON, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_evidence_claim_type', 'claim_id', 'evidence_type'),
        Index('idx_evidence_scores', 'credibility_score', 'relevance_score'),
    )


# Database utility functions
def get_db() -> Session:
    """Get database session"""
    db = SessionLocal()
    try:
        return db
    finally:
        pass  # Session will be closed by caller


def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)


def drop_tables():
    """Drop all database tables"""
    Base.metadata.drop_all(bind=engine)


def init_database():
    """Initialize database with tables and indexes"""
    create_tables()
    
    # Create additional indexes for performance
    with engine.connect() as conn:
        # Composite indexes for common queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_claims_status_priority_created 
            ON claims(status, priority, created_at DESC)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_results_claim_confidence_desc 
            ON verification_results(claim_id, confidence DESC)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_claim_timestamp_desc 
            ON audit_trail(claim_id, timestamp DESC)
        """)
        
        conn.commit()


class DatabaseManager:
    """Database operations manager"""
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
    
    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()
    
    def health_check(self) -> bool:
        """Check database connectivity"""
        try:
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
                return True
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with self.get_session() as db:
                stats = {
                    "total_claims": db.query(ClaimModel).count(),
                    "pending_claims": db.query(ClaimModel).filter(ClaimModel.status == "pending").count(),
                    "completed_claims": db.query(ClaimModel).filter(ClaimModel.status == "completed").count(),
                    "total_results": db.query(VerificationResultModel).count(),
                    "total_audit_entries": db.query(AuditTrailModel).count(),
                    "total_evidence": db.query(EvidenceModel).count(),
                }
                return stats
        except Exception as e:
            return {"error": str(e)}
    
    def cleanup_old_data(self, days: int = 30) -> int:
        """Clean up old data beyond retention period"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            with self.get_session() as db:
                # Delete old completed claims and related data
                old_claims = db.query(ClaimModel).filter(
                    ClaimModel.status == "completed",
                    ClaimModel.updated_at < cutoff_date
                ).all()
                
                count = len(old_claims)
                for claim in old_claims:
                    db.delete(claim)
                
                db.commit()
                return count
        except Exception:
            return 0


# Global database manager instance
db_manager = DatabaseManager()


# Context manager for database sessions
class DatabaseSession:
    """Context manager for database sessions"""
    
    def __init__(self):
        self.db = None
    
    def __enter__(self) -> Session:
        self.db = db_manager.get_session()
        return self.db
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.db:
            if exc_type is not None:
                self.db.rollback()
            else:
                self.db.commit()
            self.db.close()


# Migration utilities
def run_migrations():
    """Run database migrations"""
    # This would typically use Alembic for production
    # For now, just ensure tables exist
    init_database()


if __name__ == "__main__":
    # Initialize database when run directly
    init_database()
    print("Database initialized successfully")