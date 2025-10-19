#!/usr/bin/env python3
"""
Main CLI entry point for VeriSphere - Single claim verification
"""
import asyncio
import sys
import argparse
import json
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from defame.core.models import Claim
from defame.core.pipeline import initialize_pipeline_controller, get_pipeline_controller
from defame.core.agent_manager import initialize_agent_manager
from defame.core.mcp_orchestration import initialize_orchestrator
from defame.core.database import init_database
from defame.utils.logger import get_logger, setup_logging
from config.globals import ClaimType, Priority, get_config

# Setup logging
setup_logging()
logger = get_logger(__name__)


async def initialize_system():
    """Initialize all system components"""
    try:
        config = get_config()
        
        # Initialize database
        init_database()
        
        # Initialize components
        agent_manager = initialize_agent_manager({
            'ml_agent': {'enabled': True, 'huggingface_api_key': config.secret_key},
            'wikipedia_agent': {'enabled': True},
            'coherence_agent': {'enabled': True},
            'webscrape_agent': {'enabled': True}
        })
        
        orchestrator = initialize_orchestrator({
            'consensus_threshold': 0.6,
            'min_agents_required': 2
        })
        
        pipeline_controller = initialize_pipeline_controller({
            'max_processing_time': 300,
            'enable_checkpoints': True
        })
        
        logger.info("System initialization completed")
        return True
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        return False


async def verify_claim(claim_text: str, claim_type: str = "text", priority: str = "normal", 
                      source: Optional[str] = None, output_format: str = "json") -> dict:
    """Verify a single claim"""
    try:
        # Create claim object
        claim = Claim(
            content=claim_text,
            claim_type=ClaimType(claim_type.lower()),
            priority=Priority(priority.lower()),
            source=source
        )
        
        logger.info(f"Starting verification for claim: {claim.id}")
        
        # Get pipeline controller
        pipeline = get_pipeline_controller()
        if not pipeline:
            raise Exception("Pipeline controller not initialized")
        
        # Process claim
        result = await pipeline.process_claim(claim)
        
        # Format output
        if output_format.lower() == "json":
            return format_json_output(claim, result)
        elif output_format.lower() == "text":
            return format_text_output(claim, result)
        else:
            return format_json_output(claim, result)
            
    except Exception as e:
        logger.error(f"Claim verification failed: {e}")
        return {
            "error": str(e),
            "claim_id": getattr(claim, 'id', 'unknown'),
            "status": "failed"
        }


def format_json_output(claim: Claim, result) -> dict:
    """Format output as JSON"""
    return {
        "claim_id": claim.id,
        "claim_content": str(claim.content)[:200] + "..." if len(str(claim.content)) > 200 else str(claim.content),
        "claim_type": claim.claim_type.value,
        "priority": claim.priority.value,
        "source": claim.source,
        "verification_result": {
            "verdict": result.overall_verdict.value,
            "confidence": round(result.overall_confidence, 3),
            "processing_time": round(result.processing_time, 2),
            "agents_used": len(result.agent_results),
            "evidence_count": sum(len(r.evidence) for r in result.agent_results),
            "pipeline_complete": result.is_complete
        },
        "agent_results": [
            {
                "agent_type": r.agent_type.value,
                "verdict": r.verdict.value,
                "confidence": round(r.confidence, 3),
                "processing_time": round(r.processing_time, 2),
                "evidence_count": len(r.evidence),
                "reasoning": r.reasoning[:200] + "..." if len(r.reasoning) > 200 else r.reasoning
            }
            for r in result.agent_results
        ],
        "errors": result.errors,
        "timestamp": result.start_time.isoformat()
    }


def format_text_output(claim: Claim, result) -> dict:
    """Format output as human-readable text"""
    output = []
    
    output.append("=" * 80)
    output.append("VERISPHERE MISINFORMATION DETECTION RESULTS")
    output.append("=" * 80)
    output.append("")
    
    output.append(f"Claim ID: {claim.id}")
    output.append(f"Claim Type: {claim.claim_type.value.upper()}")
    output.append(f"Priority: {claim.priority.value.upper()}")
    if claim.source:
        output.append(f"Source: {claim.source}")
    output.append("")
    
    output.append("CLAIM CONTENT:")
    output.append("-" * 40)
    output.append(str(claim.content))
    output.append("")
    
    output.append("VERIFICATION RESULT:")
    output.append("-" * 40)
    output.append(f"Verdict: {result.overall_verdict.value.upper()}")
    output.append(f"Confidence: {result.overall_confidence:.1%}")
    output.append(f"Processing Time: {result.processing_time:.2f} seconds")
    output.append(f"Agents Used: {len(result.agent_results)}")
    output.append(f"Evidence Pieces: {sum(len(r.evidence) for r in result.agent_results)}")
    output.append("")
    
    if result.agent_results:
        output.append("AGENT ANALYSIS:")
        output.append("-" * 40)
        for i, agent_result in enumerate(result.agent_results, 1):
            output.append(f"{i}. {agent_result.agent_type.value.replace('_', ' ').title()}")
            output.append(f"   Verdict: {agent_result.verdict.value}")
            output.append(f"   Confidence: {agent_result.confidence:.1%}")
            output.append(f"   Processing Time: {agent_result.processing_time:.2f}s")
            output.append(f"   Evidence: {len(agent_result.evidence)} pieces")
            if agent_result.reasoning:
                reasoning = agent_result.reasoning[:150] + "..." if len(agent_result.reasoning) > 150 else agent_result.reasoning
                output.append(f"   Reasoning: {reasoning}")
            output.append("")
    
    if result.errors:
        output.append("ERRORS:")
        output.append("-" * 40)
        for error in result.errors:
            output.append(f"• {error}")
        output.append("")
    
    output.append("=" * 80)
    
    return {"text_output": "\n".join(output)}


async def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="VeriSphere - Multi-Agent Misinformation Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run.py "The Earth is flat"
  python scripts/run.py "Breaking: Major earthquake hits California" --priority high
  python scripts/run.py "Climate change is a hoax" --output text --source "social_media"
  python scripts/run.py --file claim.txt --type text --priority critical
        """
    )
    
    # Claim input options
    claim_group = parser.add_mutually_exclusive_group(required=True)
    claim_group.add_argument("claim", nargs="?", help="Claim text to verify")
    claim_group.add_argument("--file", "-f", help="File containing claim text")
    
    # Claim properties
    parser.add_argument("--type", "-t", choices=["text", "image", "multimodal"], 
                       default="text", help="Type of claim (default: text)")
    parser.add_argument("--priority", "-p", choices=["low", "normal", "high", "critical"],
                       default="normal", help="Claim priority (default: normal)")
    parser.add_argument("--source", "-s", help="Source of the claim")
    
    # Output options
    parser.add_argument("--output", "-o", choices=["json", "text"], default="json",
                       help="Output format (default: json)")
    parser.add_argument("--save", help="Save results to file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Get claim text
    if args.claim:
        claim_text = args.claim
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                claim_text = f.read().strip()
        except Exception as e:
            print(f"Error reading file {args.file}: {e}", file=sys.stderr)
            return 1
    else:
        print("Error: No claim provided", file=sys.stderr)
        return 1
    
    if not claim_text:
        print("Error: Empty claim", file=sys.stderr)
        return 1
    
    try:
        # Initialize system
        print("Initializing VeriSphere system...", file=sys.stderr if not args.verbose else sys.stdout)
        if not await initialize_system():
            print("Error: System initialization failed", file=sys.stderr)
            return 1
        
        if args.verbose:
            print(f"Processing claim: {claim_text[:100]}{'...' if len(claim_text) > 100 else ''}")
        
        # Verify claim
        result = await verify_claim(
            claim_text=claim_text,
            claim_type=args.type,
            priority=args.priority,
            source=args.source,
            output_format=args.output
        )
        
        # Handle output
        if args.output == "text":
            output_text = result.get("text_output", str(result))
            print(output_text)
        else:
            output_text = json.dumps(result, indent=2, ensure_ascii=False)
            print(output_text)
        
        # Save to file if requested
        if args.save:
            try:
                with open(args.save, 'w', encoding='utf-8') as f:
                    if args.output == "text":
                        f.write(result.get("text_output", str(result)))
                    else:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                if args.verbose:
                    print(f"Results saved to {args.save}", file=sys.stderr)
            except Exception as e:
                print(f"Error saving to file: {e}", file=sys.stderr)
        
        # Return appropriate exit code
        if "error" in result:
            return 1
        elif result.get("verification_result", {}).get("verdict") == "false":
            return 2  # Misinformation detected
        else:
            return 0  # Success
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        sys.exit(130)