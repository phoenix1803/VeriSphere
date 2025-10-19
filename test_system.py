#!/usr/bin/env python3
"""
Simple test script to verify VeriSphere system functionality
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from defame.core.models import Claim
from defame.core.pipeline import initialize_pipeline_controller, get_pipeline_controller
from defame.core.agent_manager import initialize_agent_manager, get_agent_manager
from defame.core.mcp_orchestration import initialize_orchestrator
from defame.core.database import init_database
from defame.utils.logger import get_logger, setup_logging
from config.globals import ClaimType, Priority

# Setup logging
setup_logging()
logger = get_logger(__name__)


async def test_system():
    """Test basic system functionality"""
    try:
        print("🚀 Testing VeriSphere System")
        print("=" * 50)
        
        # Initialize database
        print("📊 Initializing database...")
        init_database()
        print("✅ Database initialized")
        
        # Initialize agent manager
        print("🤖 Initializing agents...")
        agent_manager = initialize_agent_manager({
            'ml_agent': {'enabled': True},
            'wikipedia_agent': {'enabled': True},
            'coherence_agent': {'enabled': True},
            'webscrape_agent': {'enabled': True}
        })
        print("✅ Agent manager initialized")
        
        # Check agent status
        agent_status = agent_manager.get_agent_status()
        print(f"   - Total agents: {agent_status['total_agents']}")
        print(f"   - Healthy agents: {agent_status['healthy_agents']}")
        
        # Initialize orchestrator
        print("🎭 Initializing orchestrator...")
        orchestrator = initialize_orchestrator({
            'consensus_threshold': 0.6,
            'min_agents_required': 2
        })
        print("✅ Orchestrator initialized")
        
        # Initialize pipeline
        print("⚡ Initializing pipeline...")
        pipeline_controller = initialize_pipeline_controller({
            'max_processing_time': 60,  # Shorter for testing
            'enable_checkpoints': False  # Disable for testing
        })
        print("✅ Pipeline controller initialized")
        
        # Test with a simple claim
        print("\n🔍 Testing claim verification...")
        test_claim = Claim(
            content="The Earth is round and orbits the Sun",
            claim_type=ClaimType.TEXT,
            priority=Priority.NORMAL,
            source="test_system"
        )
        
        print(f"   - Claim ID: {test_claim.id}")
        print(f"   - Content: {test_claim.content}")
        
        # Process the claim
        print("⏳ Processing claim...")
        result = await pipeline_controller.process_claim(test_claim)
        
        # Display results
        print("\n📋 Results:")
        print(f"   - Verdict: {result.overall_verdict.value}")
        print(f"   - Confidence: {result.overall_confidence:.2f}")
        print(f"   - Processing time: {result.processing_time:.2f}s")
        print(f"   - Agents used: {len(result.agent_results)}")
        print(f"   - Evidence pieces: {sum(len(r.evidence) for r in result.agent_results)}")
        print(f"   - Pipeline complete: {result.is_complete}")
        
        if result.errors:
            print(f"   - Errors: {len(result.errors)}")
            for error in result.errors[:3]:  # Show first 3 errors
                print(f"     • {error}")
        
        # Show agent results
        if result.agent_results:
            print("\n🤖 Agent Results:")
            for i, agent_result in enumerate(result.agent_results, 1):
                print(f"   {i}. {agent_result.agent_type.value.replace('_', ' ').title()}")
                print(f"      - Verdict: {agent_result.verdict.value}")
                print(f"      - Confidence: {agent_result.confidence:.2f}")
                print(f"      - Evidence: {len(agent_result.evidence)} pieces")
        
        print("\n✅ System test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function"""
    success = await test_system()
    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Test cancelled by user")
        sys.exit(130)