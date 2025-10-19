#!/usr/bin/env python3
"""
Batch processing script for VeriSphere using YAML configuration
"""
import asyncio
import sys
import yaml
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from defame.core.models import Claim
from defame.core.pipeline import initialize_pipeline_controller, get_pipeline_controller
from defame.core.agent_manager import initialize_agent_manager
from defame.core.mcp_orchestration import initialize_orchestrator
from defame.core.database import init_database
from defame.utils.logger import get_logger, setup_logging
from config.globals import ClaimType, Priority
from scripts.run import initialize_system, format_json_output

# Setup logging
setup_logging()
logger = get_logger(__name__)


async def process_batch_claims(config_file: str) -> Dict[str, Any]:
    """Process multiple claims from configuration file"""
    try:
        # Load configuration
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        claims_config = config.get('claims', [])
        if not claims_config:
            raise ValueError("No claims found in configuration file")
        
        # Initialize system
        logger.info("Initializing system for batch processing")
        if not await initialize_system():
            raise Exception("System initialization failed")
        
        # Process claims
        results = []
        pipeline = get_pipeline_controller()
        
        for i, claim_config in enumerate(claims_config, 1):
            try:
                logger.info(f"Processing claim {i}/{len(claims_config)}")
                
                # Create claim
                claim = Claim(
                    content=claim_config.get('content', ''),
                    claim_type=ClaimType(claim_config.get('type', 'text')),
                    priority=Priority(claim_config.get('priority', 'normal')),
                    source=claim_config.get('source'),
                    metadata=claim_config.get('metadata', {})
                )
                
                # Process claim
                result = await pipeline.process_claim(claim)
                
                # Format result
                formatted_result = format_json_output(claim, result)
                formatted_result['batch_index'] = i
                results.append(formatted_result)
                
                logger.info(f"Completed claim {i}: {result.overall_verdict.value}")
                
            except Exception as e:
                logger.error(f"Failed to process claim {i}: {e}")
                results.append({
                    'batch_index': i,
                    'error': str(e),
                    'claim_content': claim_config.get('content', '')[:100]
                })
        
        # Generate summary
        summary = {
            'total_claims': len(claims_config),
            'successful_claims': len([r for r in results if 'error' not in r]),
            'failed_claims': len([r for r in results if 'error' in r]),
            'verdicts': {},
            'processing_time': sum(r.get('verification_result', {}).get('processing_time', 0) 
                                 for r in results if 'error' not in r),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Count verdicts
        for result in results:
            if 'error' not in result:
                verdict = result.get('verification_result', {}).get('verdict', 'unknown')
                summary['verdicts'][verdict] = summary['verdicts'].get(verdict, 0) + 1
        
        return {
            'summary': summary,
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return {
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }


def create_sample_config(output_file: str):
    """Create a sample configuration file"""
    sample_config = {
        'claims': [
            {
                'content': 'The Earth is flat and NASA is hiding the truth',
                'type': 'text',
                'priority': 'normal',
                'source': 'social_media',
                'metadata': {
                    'category': 'science',
                    'tags': ['conspiracy', 'space']
                }
            },
            {
                'content': 'Vaccines cause autism in children',
                'type': 'text',
                'priority': 'high',
                'source': 'blog_post',
                'metadata': {
                    'category': 'health',
                    'tags': ['medical', 'misinformation']
                }
            },
            {
                'content': 'Climate change is a natural phenomenon and not caused by humans',
                'type': 'text',
                'priority': 'normal',
                'source': 'news_article',
                'metadata': {
                    'category': 'environment',
                    'tags': ['climate', 'science']
                }
            }
        ],
        'output': {
            'format': 'json',
            'save_individual': True,
            'save_summary': True
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(sample_config, f, default_flow_style=False, indent=2)
    
    print(f"Sample configuration created: {output_file}")


async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="VeriSphere Batch Processing - Process multiple claims from YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_config.py config/batch_claims.yaml
  python scripts/run_config.py config/test_claims.yaml --output results.json
  python scripts/run_config.py --create-sample sample_config.yaml
        """
    )
    
    parser.add_argument("config_file", nargs="?", help="YAML configuration file")
    parser.add_argument("--output", "-o", help="Output file for results")
    parser.add_argument("--create-sample", help="Create sample configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Create sample configuration
    if args.create_sample:
        create_sample_config(args.create_sample)
        return 0
    
    # Validate arguments
    if not args.config_file:
        print("Error: Configuration file required", file=sys.stderr)
        parser.print_help()
        return 1
    
    if not Path(args.config_file).exists():
        print(f"Error: Configuration file not found: {args.config_file}", file=sys.stderr)
        return 1
    
    try:
        # Process batch claims
        if args.verbose:
            print(f"Starting batch processing with config: {args.config_file}")
        
        results = await process_batch_claims(args.config_file)
        
        # Output results
        output_text = json.dumps(results, indent=2, ensure_ascii=False)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output_text)
            if args.verbose:
                print(f"Results saved to: {args.output}")
        else:
            print(output_text)
        
        # Print summary if verbose
        if args.verbose and 'summary' in results:
            summary = results['summary']
            print(f"\nBatch Processing Summary:", file=sys.stderr)
            print(f"Total Claims: {summary['total_claims']}", file=sys.stderr)
            print(f"Successful: {summary['successful_claims']}", file=sys.stderr)
            print(f"Failed: {summary['failed_claims']}", file=sys.stderr)
            print(f"Total Processing Time: {summary['processing_time']:.2f}s", file=sys.stderr)
            
            if summary['verdicts']:
                print("Verdicts:", file=sys.stderr)
                for verdict, count in summary['verdicts'].items():
                    print(f"  {verdict}: {count}", file=sys.stderr)
        
        # Return appropriate exit code
        if 'error' in results:
            return 1
        elif results.get('summary', {}).get('failed_claims', 0) > 0:
            return 2  # Some claims failed
        else:
            return 0  # Success
            
    except KeyboardInterrupt:
        print("\nBatch processing cancelled by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Batch processing error: {e}", file=sys.stderr)
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