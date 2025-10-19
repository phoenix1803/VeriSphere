"""
Agent Manager for lifecycle management and coordination
"""
import asyncio
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
import json

from defame.core.interfaces import BaseAgent, agent_registry
from defame.core.models import Claim, VerificationResult, AgentCapability, ProcessingMetrics
from defame.agents.ml_agent import MLAgent
from defame.agents.wikipedia_agent import WikipediaAgent
from defame.agents.coherence_agent import CoherenceAgent
from defame.agents.webscrape_agent import WebScrapeAgent
from defame.utils.logger import get_logger, audit_logger, metrics_logger
from defame.utils.helpers import CircuitBreaker
from config.globals import AgentType, ClaimType, get_agent_config

logger = get_logger(__name__)


class AgentManager:
    """Manages agent lifecycle, health, and coordination"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agents: Dict[str, BaseAgent] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.agent_metrics: Dict[str, ProcessingMetrics] = {}
        self.health_check_interval = config.get('health_check_interval', 60)  # seconds
        self.max_concurrent_agents = config.get('max_concurrent_agents', 10)
        self.agent_timeout = config.get('agent_timeout', 60)  # seconds
        
        # Agent configurations
        self.agent_configs = {
            AgentType.ML_AGENT: config.get('ml_agent', {}),
            AgentType.WIKIPEDIA_AGENT: config.get('wikipedia_agent', {}),
            AgentType.COHERENCE_AGENT: config.get('coherence_agent', {}),
            AgentType.WEBSCRAPE_AGENT: config.get('webscrape_agent', {})
        }
        
        # Initialize agents
        self._initialize_agents()
        
        # Start health monitoring
        self._health_monitor_task = None
        self._start_health_monitoring()
    
    def _initialize_agents(self):
        """Initialize all configured agents"""
        try:
            # ML Agent
            if self.agent_configs.get(AgentType.ML_AGENT, {}).get('enabled', True):
                ml_config = {**get_agent_config(AgentType.ML_AGENT), **self.agent_configs[AgentType.ML_AGENT]}
                ml_agent = MLAgent(ml_config)
                self._register_agent(ml_agent)
            
            # Wikipedia Agent
            if self.agent_configs.get(AgentType.WIKIPEDIA_AGENT, {}).get('enabled', True):
                wiki_config = {**get_agent_config(AgentType.WIKIPEDIA_AGENT), **self.agent_configs[AgentType.WIKIPEDIA_AGENT]}
                wiki_agent = WikipediaAgent(wiki_config)
                self._register_agent(wiki_agent)
            
            # Coherence Agent
            if self.agent_configs.get(AgentType.COHERENCE_AGENT, {}).get('enabled', True):
                coherence_config = {**get_agent_config(AgentType.COHERENCE_AGENT), **self.agent_configs[AgentType.COHERENCE_AGENT]}
                coherence_agent = CoherenceAgent(coherence_config)
                self._register_agent(coherence_agent)
            
            # WebScrape Agent
            if self.agent_configs.get(AgentType.WEBSCRAPE_AGENT, {}).get('enabled', True):
                webscrape_config = {**get_agent_config(AgentType.WEBSCRAPE_AGENT), **self.agent_configs[AgentType.WEBSCRAPE_AGENT]}
                webscrape_agent = WebScrapeAgent(webscrape_config)
                self._register_agent(webscrape_agent)
            
            logger.info(f"Initialized {len(self.agents)} agents")
            
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
    
    def _register_agent(self, agent: BaseAgent):
        """Register an agent with the manager"""
        try:
            self.agents[agent.agent_id] = agent
            agent_registry.register_agent(agent)
            
            # Initialize circuit breaker
            self.circuit_breakers[agent.agent_id] = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=300.0  # 5 minutes
            )
            
            # Initialize metrics
            self.agent_metrics[agent.agent_id] = ProcessingMetrics(
                agent_performance={agent.agent_id: {
                    'total_requests': 0,
                    'successful_requests': 0,
                    'failed_requests': 0,
                    'average_response_time': 0.0
                }}
            )
            
            logger.info(f"Registered agent: {agent.agent_id} ({agent.agent_type.value})")
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent.agent_id}: {e}")
    
    def _start_health_monitoring(self):
        """Start background health monitoring"""
        async def health_monitor():
            while True:
                try:
                    await self._perform_health_checks()
                    await asyncio.sleep(self.health_check_interval)
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
                    await asyncio.sleep(self.health_check_interval)
        
        self._health_monitor_task = asyncio.create_task(health_monitor())
    
    async def _perform_health_checks(self):
        """Perform health checks on all agents"""
        try:
            health_results = {}
            
            for agent_id, agent in self.agents.items():
                try:
                    is_healthy = await agent.health_check()
                    health_results[agent_id] = is_healthy
                    
                    if not is_healthy:
                        logger.warning(f"Agent {agent_id} failed health check")
                        audit_logger.log_error(
                            claim_id="system",
                            error_type="agent_health_failure",
                            error_message=f"Agent {agent_id} failed health check",
                            component="agent_manager"
                        )
                
                except Exception as e:
                    health_results[agent_id] = False
                    logger.error(f"Health check failed for agent {agent_id}: {e}")
            
            # Log health metrics
            healthy_count = sum(1 for is_healthy in health_results.values() if is_healthy)
            metrics_logger.log_system_metrics({
                'healthy_agents': healthy_count,
                'total_agents': len(self.agents),
                'health_check_timestamp': datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Health check process failed: {e}")
    
    async def select_agents_for_claim(self, claim: Claim, metadata: Optional[Dict] = None) -> List[BaseAgent]:
        """Select appropriate agents for a claim"""
        try:
            suitable_agents = []
            
            # Get agents that can handle this claim type
            for agent in self.agents.values():
                if not agent.is_healthy:
                    continue
                
                capability = agent.get_capabilities()
                if capability.can_handle(claim.claim_type):
                    suitable_agents.append(agent)
            
            # Apply selection strategy based on metadata
            selection_strategy = metadata.get('agent_selection', 'all') if metadata else 'all'
            
            if selection_strategy == 'fast':
                # Select fastest agents
                suitable_agents.sort(key=lambda a: a.get_capabilities().processing_time_estimate)
                suitable_agents = suitable_agents[:2]
            elif selection_strategy == 'accurate':
                # Select most accurate agents (based on historical performance)
                suitable_agents = self._select_by_accuracy(suitable_agents)
            elif selection_strategy == 'balanced':
                # Balance speed and accuracy
                suitable_agents = self._select_balanced(suitable_agents)
            # 'all' strategy uses all suitable agents
            
            logger.info(
                f"Selected {len(suitable_agents)} agents for claim",
                claim_id=claim.id,
                claim_type=claim.claim_type.value,
                strategy=selection_strategy,
                selected_agents=[a.agent_type.value for a in suitable_agents]
            )
            
            return suitable_agents
            
        except Exception as e:
            logger.error(f"Agent selection failed: {e}")
            return []
    
    def _select_by_accuracy(self, agents: List[BaseAgent]) -> List[BaseAgent]:
        """Select agents based on historical accuracy"""
        # Sort by accuracy metrics (simplified - in production, use historical data)
        agent_accuracy = {}
        for agent in agents:
            metrics = self.agent_metrics.get(agent.agent_id)
            if metrics and metrics.agent_performance.get(agent.agent_id):
                perf = metrics.agent_performance[agent.agent_id]
                total = perf.get('total_requests', 1)
                successful = perf.get('successful_requests', 0)
                accuracy = successful / total if total > 0 else 0.5
                agent_accuracy[agent.agent_id] = accuracy
            else:
                agent_accuracy[agent.agent_id] = 0.5  # Default
        
        # Sort by accuracy and take top performers
        sorted_agents = sorted(agents, key=lambda a: agent_accuracy[a.agent_id], reverse=True)
        return sorted_agents[:3]  # Top 3 most accurate
    
    def _select_balanced(self, agents: List[BaseAgent]) -> List[BaseAgent]:
        """Select agents balancing speed and accuracy"""
        scored_agents = []
        
        for agent in agents:
            capability = agent.get_capabilities()
            metrics = self.agent_metrics.get(agent.agent_id)
            
            # Speed score (inverse of processing time)
            speed_score = 1.0 / max(capability.processing_time_estimate, 1.0)
            
            # Accuracy score
            accuracy_score = 0.5  # Default
            if metrics and metrics.agent_performance.get(agent.agent_id):
                perf = metrics.agent_performance[agent.agent_id]
                total = perf.get('total_requests', 1)
                successful = perf.get('successful_requests', 0)
                accuracy_score = successful / total if total > 0 else 0.5
            
            # Combined score (weighted)
            combined_score = (speed_score * 0.3) + (accuracy_score * 0.7)
            scored_agents.append((agent, combined_score))
        
        # Sort by combined score and take top performers
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        return [agent for agent, score in scored_agents[:3]]
    
    async def execute_agents(self, claim: Claim, agents: List[BaseAgent], metadata: Optional[Dict] = None) -> List[VerificationResult]:
        """Execute multiple agents concurrently with circuit breaker protection"""
        try:
            # Limit concurrent execution
            semaphore = asyncio.Semaphore(self.max_concurrent_agents)
            
            async def execute_agent_with_protection(agent: BaseAgent) -> Optional[VerificationResult]:
                async with semaphore:
                    circuit_breaker = self.circuit_breakers[agent.agent_id]
                    
                    try:
                        # Execute with circuit breaker protection
                        result = await circuit_breaker.call(
                            agent._process_with_timeout,
                            claim,
                            self.agent_timeout
                        )
                        
                        # Update metrics
                        await self._update_agent_metrics(agent.agent_id, True, result.processing_time)
                        
                        # Log successful execution
                        audit_logger.log_agent_completed(
                            claim.id,
                            agent.agent_id,
                            result.verdict.value,
                            result.confidence
                        )
                        
                        return result
                        
                    except Exception as e:
                        # Update metrics for failure
                        await self._update_agent_metrics(agent.agent_id, False, 0.0)
                        
                        # Log failure
                        audit_logger.log_error(
                            claim.id,
                            "agent_execution_failure",
                            str(e),
                            agent.agent_id
                        )
                        
                        logger.error(f"Agent {agent.agent_id} execution failed: {e}")
                        return None
            
            # Execute all agents concurrently
            tasks = [execute_agent_with_protection(agent) for agent in agents]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter successful results
            successful_results = []
            for result in results:
                if isinstance(result, VerificationResult):
                    successful_results.append(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Agent execution exception: {result}")
            
            logger.info(
                f"Agent execution completed",
                claim_id=claim.id,
                agents_executed=len(agents),
                successful_results=len(successful_results)
            )
            
            return successful_results
            
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return []
    
    async def _update_agent_metrics(self, agent_id: str, success: bool, processing_time: float):
        """Update agent performance metrics"""
        try:
            if agent_id not in self.agent_metrics:
                self.agent_metrics[agent_id] = ProcessingMetrics()
            
            metrics = self.agent_metrics[agent_id]
            
            if agent_id not in metrics.agent_performance:
                metrics.agent_performance[agent_id] = {
                    'total_requests': 0,
                    'successful_requests': 0,
                    'failed_requests': 0,
                    'average_response_time': 0.0
                }
            
            perf = metrics.agent_performance[agent_id]
            perf['total_requests'] += 1
            
            if success:
                perf['successful_requests'] += 1
                # Update average response time
                current_avg = perf['average_response_time']
                total_successful = perf['successful_requests']
                perf['average_response_time'] = ((current_avg * (total_successful - 1)) + processing_time) / total_successful
            else:
                perf['failed_requests'] += 1
            
            # Log metrics periodically
            if perf['total_requests'] % 10 == 0:  # Every 10 requests
                metrics_logger.log_agent_metrics(agent_id, self.agents[agent_id].agent_type.value, perf)
            
        except Exception as e:
            logger.warning(f"Failed to update metrics for agent {agent_id}: {e}")
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        status = {
            'total_agents': len(self.agents),
            'healthy_agents': 0,
            'agents': {}
        }
        
        for agent_id, agent in self.agents.items():
            agent_status = {
                'agent_type': agent.agent_type.value,
                'is_healthy': agent.is_healthy,
                'capabilities': agent.get_capabilities().to_dict() if hasattr(agent.get_capabilities(), 'to_dict') else str(agent.get_capabilities()),
                'metrics': agent.get_metrics(),
                'circuit_breaker_state': self.circuit_breakers[agent_id].state
            }
            
            if agent.is_healthy:
                status['healthy_agents'] += 1
            
            status['agents'][agent_id] = agent_status
        
        return status
    
    def get_system_metrics(self) -> ProcessingMetrics:
        """Get aggregated system metrics"""
        total_claims = 0
        total_processing_time = 0.0
        total_successful = 0
        
        for metrics in self.agent_metrics.values():
            for agent_perf in metrics.agent_performance.values():
                total_claims += agent_perf.get('total_requests', 0)
                total_successful += agent_perf.get('successful_requests', 0)
                total_processing_time += agent_perf.get('average_response_time', 0.0) * agent_perf.get('successful_requests', 0)
        
        avg_processing_time = total_processing_time / max(total_successful, 1)
        accuracy = total_successful / max(total_claims, 1)
        
        return ProcessingMetrics(
            claims_processed=total_claims,
            average_processing_time=avg_processing_time,
            accuracy_score=accuracy,
            agent_performance={agent_id: metrics.agent_performance for agent_id, metrics in self.agent_metrics.items()},
            error_rate=1.0 - accuracy
        )
    
    async def shutdown(self):
        """Shutdown agent manager and cleanup resources"""
        try:
            logger.info("Shutting down agent manager")
            
            # Cancel health monitoring
            if self._health_monitor_task:
                self._health_monitor_task.cancel()
                try:
                    await self._health_monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Cleanup agents
            for agent_id in list(self.agents.keys()):
                agent_registry.unregister_agent(agent_id)
            
            self.agents.clear()
            self.circuit_breakers.clear()
            self.agent_metrics.clear()
            
            logger.info("Agent manager shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during agent manager shutdown: {e}")


# Global agent manager instance (will be initialized by the application)
agent_manager: Optional[AgentManager] = None


def get_agent_manager() -> Optional[AgentManager]:
    """Get the global agent manager instance"""
    return agent_manager


def initialize_agent_manager(config: Dict[str, Any]) -> AgentManager:
    """Initialize the global agent manager"""
    global agent_manager
    agent_manager = AgentManager(config)
    return agent_manager