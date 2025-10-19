"""
Structured logging system for VeriSphere
"""
import logging
import structlog
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from pythonjsonlogger import jsonlogger

from config.globals import get_config, LOGS_DIR

config = get_config()


class CorrelationIDProcessor:
    """Add correlation ID to log records"""
    
    def __init__(self):
        self.correlation_id = None
    
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for current context"""
        self.correlation_id = correlation_id
    
    def __call__(self, logger, method_name, event_dict):
        if self.correlation_id:
            event_dict['correlation_id'] = self.correlation_id
        return event_dict


class TimestampProcessor:
    """Add timestamp to log records"""
    
    def __call__(self, logger, method_name, event_dict):
        event_dict['timestamp'] = datetime.utcnow().isoformat()
        return event_dict


class ServiceInfoProcessor:
    """Add service information to log records"""
    
    def __call__(self, logger, method_name, event_dict):
        event_dict['service'] = 'verisphere'
        event_dict['environment'] = config.environment.value
        return event_dict


# Global correlation ID processor
correlation_processor = CorrelationIDProcessor()


def setup_logging():
    """Setup structured logging configuration"""
    
    # Ensure logs directory exists
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            TimestampProcessor(),
            ServiceInfoProcessor(),
            correlation_processor,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, config.log_level.upper())
    )
    
    # File handlers
    file_handler = logging.FileHandler(LOGS_DIR / "verisphere.log")
    file_handler.setFormatter(jsonlogger.JsonFormatter())
    
    error_handler = logging.FileHandler(LOGS_DIR / "errors.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(jsonlogger.JsonFormatter())
    
    # Add handlers to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)


def get_logger(name: str = None) -> structlog.BoundLogger:
    """Get a structured logger instance"""
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class"""
    
    @property
    def logger(self) -> structlog.BoundLogger:
        """Get logger for this class"""
        return get_logger(self.__class__.__name__)


class PerformanceLogger:
    """Logger for performance metrics and timing"""
    
    def __init__(self, operation: str, logger: Optional[structlog.BoundLogger] = None):
        self.operation = operation
        self.logger = logger or get_logger("performance")
        self.start_time = None
        self.metadata = {}
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        self.logger.info("operation_started", operation=self.operation)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.utcnow()
        duration = (end_time - self.start_time).total_seconds()
        
        log_data = {
            "operation": self.operation,
            "duration_seconds": duration,
            "success": exc_type is None,
            **self.metadata
        }
        
        if exc_type is not None:
            log_data["error"] = str(exc_val)
            self.logger.error("operation_failed", **log_data)
        else:
            self.logger.info("operation_completed", **log_data)
    
    def add_metadata(self, **kwargs):
        """Add metadata to the performance log"""
        self.metadata.update(kwargs)


class AuditLogger:
    """Logger for audit events"""
    
    def __init__(self):
        self.logger = get_logger("audit")
    
    def log_claim_submitted(self, claim_id: str, claim_type: str, source: Optional[str] = None):
        """Log claim submission"""
        self.logger.info(
            "claim_submitted",
            claim_id=claim_id,
            claim_type=claim_type,
            source=source,
            event_type="claim_lifecycle"
        )
    
    def log_agent_started(self, claim_id: str, agent_id: str, agent_type: str):
        """Log agent verification start"""
        self.logger.info(
            "agent_verification_started",
            claim_id=claim_id,
            agent_id=agent_id,
            agent_type=agent_type,
            event_type="agent_lifecycle"
        )
    
    def log_agent_completed(self, claim_id: str, agent_id: str, verdict: str, confidence: float):
        """Log agent verification completion"""
        self.logger.info(
            "agent_verification_completed",
            claim_id=claim_id,
            agent_id=agent_id,
            verdict=verdict,
            confidence=confidence,
            event_type="agent_lifecycle"
        )
    
    def log_pipeline_stage(self, claim_id: str, stage: str, status: str, duration: float):
        """Log pipeline stage completion"""
        self.logger.info(
            "pipeline_stage_completed",
            claim_id=claim_id,
            stage=stage,
            status=status,
            duration_seconds=duration,
            event_type="pipeline_lifecycle"
        )
    
    def log_verification_completed(self, claim_id: str, final_verdict: str, confidence: float, processing_time: float):
        """Log complete verification"""
        self.logger.info(
            "verification_completed",
            claim_id=claim_id,
            final_verdict=final_verdict,
            final_confidence=confidence,
            total_processing_time=processing_time,
            event_type="claim_lifecycle"
        )
    
    def log_error(self, claim_id: str, error_type: str, error_message: str, component: str):
        """Log error events"""
        self.logger.error(
            "system_error",
            claim_id=claim_id,
            error_type=error_type,
            error_message=error_message,
            component=component,
            event_type="error"
        )
    
    def log_security_event(self, event_type: str, user_id: Optional[str] = None, ip_address: Optional[str] = None, details: Optional[Dict] = None):
        """Log security-related events"""
        self.logger.warning(
            "security_event",
            security_event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            details=details or {},
            event_type="security"
        )


class MetricsLogger:
    """Logger for system metrics"""
    
    def __init__(self):
        self.logger = get_logger("metrics")
    
    def log_system_metrics(self, metrics: Dict[str, Any]):
        """Log system-wide metrics"""
        self.logger.info(
            "system_metrics",
            **metrics,
            metric_type="system"
        )
    
    def log_agent_metrics(self, agent_id: str, agent_type: str, metrics: Dict[str, Any]):
        """Log agent-specific metrics"""
        self.logger.info(
            "agent_metrics",
            agent_id=agent_id,
            agent_type=agent_type,
            **metrics,
            metric_type="agent"
        )
    
    def log_api_metrics(self, endpoint: str, method: str, status_code: int, response_time: float):
        """Log API request metrics"""
        self.logger.info(
            "api_request",
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time_seconds=response_time,
            metric_type="api"
        )


# Global logger instances
audit_logger = AuditLogger()
metrics_logger = MetricsLogger()


def set_correlation_id(correlation_id: str):
    """Set correlation ID for current request/operation"""
    correlation_processor.set_correlation_id(correlation_id)


def clear_correlation_id():
    """Clear correlation ID"""
    correlation_processor.set_correlation_id(None)


# Initialize logging on module import
setup_logging()


# Example usage and testing
if __name__ == "__main__":
    # Test logging setup
    logger = get_logger("test")
    
    # Test basic logging
    logger.info("Test log message", test_data="example")
    
    # Test performance logging
    with PerformanceLogger("test_operation") as perf:
        perf.add_metadata(test_param="value")
        import time
        time.sleep(0.1)  # Simulate work
    
    # Test audit logging
    audit_logger.log_claim_submitted("test-claim-123", "text", "test-source")
    
    # Test metrics logging
    metrics_logger.log_system_metrics({
        "claims_processed": 100,
        "average_processing_time": 2.5,
        "error_rate": 0.02
    })
    
    print("Logging test completed. Check logs directory for output.")