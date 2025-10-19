"""
Standardized interfaces for VeriSphere agents and components
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import asyncio
from datetime import datetime

from defame.core.models import (
    Claim, VerificationResult, Evidence, AgentCapability, 
    PipelineState, ProcessingMetrics
)
from config.globals import ClaimType, AgentType, Verdict


class BaseAgent(ABC):
    """Abstract base class for all verification agents"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize agent with configuration"""
        self.config = config
        self.agent_id = f"{self.__class__.__name__}_{id(self)}"
        self.agent_type = self._get_agent_type()
        self.is_healthy = True
        self.last_health_check = datetime.utcnow()
        self.processing_count = 0
        self.error_count = 0
        
    @abstractmethod
    def _get_agent_type(self) -> AgentType:
        """Return the agent type"""
        pass
    
    @abstractmethod
    async def verify_claim(self, claim: Claim, metadata: Optional[Dict] = None) -> VerificationResult:
        """
        Verify a claim and return structured results
        
        Args:
            claim: The claim to verify
            metadata: Additional context and parameters
            
        Returns:
            VerificationResult with confidence, evidence, and reasoning
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> AgentCapability:
        """Return agent capabilities and supported claim types"""
        pass
    
    async def health_check(self) -> bool:
        """
        Perform health check and return status
        
        Returns:
            True if agent is healthy, False otherwise
        """
        try:
            # Basic health check - can be overridden by subclasses
            self.last_health_check = datetime.utcnow()
            return self.is_healthy
        except Exception:
            self.is_healthy = False
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "processing_count": self.processing_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.processing_count, 1),
            "is_healthy": self.is_healthy,
            "last_health_check": self.last_health_check.isoformat()
        }
    
    async def _process_with_timeout(self, claim: Claim, timeout: float = 60.0) -> VerificationResult:
        """Process claim with timeout protection"""
        try:
            result = await asyncio.wait_for(
                self.verify_claim(claim), 
                timeout=timeout
            )
            self.processing_count += 1
            return result
        except asyncio.TimeoutError:
            self.error_count += 1
            return VerificationResult(
                claim_id=claim.id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                confidence=0.0,
                verdict=Verdict.INCONCLUSIVE,
                reasoning=f"Agent {self.agent_id} timed out after {timeout} seconds"
            )
        except Exception as e:
            self.error_count += 1
            return VerificationResult(
                claim_id=claim.id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                confidence=0.0,
                verdict=Verdict.INCONCLUSIVE,
                reasoning=f"Agent {self.agent_id} failed with error: {str(e)}"
            )


class BaseEvidenceTool(ABC):
    """Abstract base class for evidence retrieval tools"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize tool with configuration"""
        self.config = config
        self.tool_id = f"{self.__class__.__name__}_{id(self)}"
        self.is_available = True
        
    @abstractmethod
    async def gather_evidence(self, claim: Claim, query: str, **kwargs) -> List[Evidence]:
        """
        Gather evidence for a claim
        
        Args:
            claim: The claim being verified
            query: Search query or parameters
            **kwargs: Additional tool-specific parameters
            
        Returns:
            List of Evidence objects
        """
        pass
    
    @abstractmethod
    def get_tool_info(self) -> Dict[str, Any]:
        """Return tool information and capabilities"""
        pass
    
    async def health_check(self) -> bool:
        """Check if tool is available and working"""
        try:
            # Basic availability check
            return self.is_available
        except Exception:
            self.is_available = False
            return False


class BasePipelineStage(ABC):
    """Abstract base class for pipeline stages"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize pipeline stage"""
        self.config = config
        self.stage_id = f"{self.__class__.__name__}_{id(self)}"
        
    @abstractmethod
    async def execute(self, state: PipelineState) -> PipelineState:
        """
        Execute this pipeline stage
        
        Args:
            state: Current pipeline state
            
        Returns:
            Updated pipeline state
        """
        pass
    
    @abstractmethod
    def get_stage_info(self) -> Dict[str, Any]:
        """Return stage information"""
        pass


class BaseOrchestrator(ABC):
    """Abstract base class for orchestration engines"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize orchestrator"""
        self.config = config
        self.orchestrator_id = f"{self.__class__.__name__}_{id(self)}"
        
    @abstractmethod
    async def orchestrate_verification(self, claim: Claim) -> PipelineState:
        """
        Orchestrate the complete verification process
        
        Args:
            claim: Claim to verify
            
        Returns:
            Final pipeline state with results
        """
        pass
    
    @abstractmethod
    def select_agents(self, claim: Claim) -> List[BaseAgent]:
        """
        Select appropriate agents for a claim
        
        Args:
            claim: Claim to analyze
            
        Returns:
            List of selected agents
        """
        pass


class BaseReporter(ABC):
    """Abstract base class for report generators"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize reporter"""
        self.config = config
        
    @abstractmethod
    async def generate_report(self, state: PipelineState, format_type: str = "json") -> Dict[str, Any]:
        """
        Generate verification report
        
        Args:
            state: Pipeline state with results
            format_type: Output format (json, pdf, html)
            
        Returns:
            Generated report
        """
        pass


class BaseAuditor(ABC):
    """Abstract base class for audit trail systems"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize auditor"""
        self.config = config
        
    @abstractmethod
    async def log_action(self, claim_id: str, action: str, actor: str, details: Dict[str, Any]) -> str:
        """
        Log an action to the audit trail
        
        Args:
            claim_id: ID of the claim
            action: Action being performed
            actor: Who performed the action
            details: Additional details
            
        Returns:
            Audit entry ID or blockchain hash
        """
        pass
    
    @abstractmethod
    async def verify_audit_trail(self, claim_id: str) -> bool:
        """
        Verify the integrity of an audit trail
        
        Args:
            claim_id: ID of the claim to verify
            
        Returns:
            True if audit trail is valid
        """
        pass


class BaseNotifier(ABC):
    """Abstract base class for notification systems"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize notifier"""
        self.config = config
        
    @abstractmethod
    async def send_notification(self, message: str, recipients: List[str], priority: str = "normal") -> bool:
        """
        Send notification
        
        Args:
            message: Notification message
            recipients: List of recipients
            priority: Notification priority
            
        Returns:
            True if notification sent successfully
        """
        pass


# Agent Registry for dynamic agent discovery
class AgentRegistry:
    """Registry for managing available agents"""
    
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._capabilities: Dict[AgentType, AgentCapability] = {}
    
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent"""
        self._agents[agent.agent_id] = agent
        self._capabilities[agent.agent_type] = agent.get_capabilities()
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent"""
        if agent_id in self._agents:
            agent = self._agents[agent_id]
            del self._agents[agent_id]
            # Remove capability if no other agents of this type
            if not any(a.agent_type == agent.agent_type for a in self._agents.values()):
                self._capabilities.pop(agent.agent_type, None)
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID"""
        return self._agents.get(agent_id)
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[BaseAgent]:
        """Get all agents of a specific type"""
        return [agent for agent in self._agents.values() if agent.agent_type == agent_type]
    
    def get_agents_for_claim(self, claim: Claim) -> List[BaseAgent]:
        """Get agents that can handle a specific claim type"""
        suitable_agents = []
        for agent in self._agents.values():
            capability = agent.get_capabilities()
            if capability.can_handle(claim.claim_type):
                suitable_agents.append(agent)
        return suitable_agents
    
    def get_all_agents(self) -> List[BaseAgent]:
        """Get all registered agents"""
        return list(self._agents.values())
    
    def get_capabilities(self) -> Dict[AgentType, AgentCapability]:
        """Get all agent capabilities"""
        return self._capabilities.copy()
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Perform health check on all agents"""
        results = {}
        for agent_id, agent in self._agents.items():
            results[agent_id] = await agent.health_check()
        return results


# Global agent registry instance
agent_registry = AgentRegistry()