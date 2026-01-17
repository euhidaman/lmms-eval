"""
Safe Evaluation Script for EmberVLM
Handles CUDA errors gracefully and ensures benchmarks complete successfully.
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


BENCHMARKS = [
    'mmmu_val',
    'mathvista_testmini', 
    'mmstar',
    'docvqa_test',
    'textvqa_val'
]


def run_benchmark_isolated(
    benchmark: str,
    checkpoint_path: str,
    output_dir: str,
    batch_size: int = 1
) -> Dict[str, float]:
    """
    Run a single benchmark in isolation with error handling.
    
    Returns:
        Dict with benchmark results, or empty dict if failed
    """
    logger.info(f"🔬 Running benchmark: {benchmark}")
    
    output_path = Path(output_dir) / benchmark
    output_path.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, '-m', 'lmms_eval',
        '--model', 'embervlm',
        '--tasks', benchmark,
        '--model_args', f'pretrained={checkpoint_path}',
        '--batch_size', str(batch_size),
        '--log_samples',
        '--output_path', str(output_path)
    ]
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        # Run in subprocess with timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout per benchmark
            env={**os.environ, 'CUDA_LAUNCH_BLOCKING': '0'}
        )
        
        # Parse results
        results_file = output_path / f'results_{benchmark}.json'
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
                logger.info(f"✅ {benchmark} completed: {results.get('results', {})}")
                return results.get('results', {})
        else:
            logger.warning(f"⚠️ {benchmark} completed but no results file found")
            # Check for any JSON files
            json_files = list(output_path.glob('*.json'))
            if json_files:
                with open(json_files[0]) as f:
                    results = json.load(f)
                    return results.get('results', {})
    
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {benchmark} timed out after 10 minutes")
    except Exception as e:
        logger.error(f"❌ {benchmark} failed with error: {e}")
    
    return {}


def safe_evaluate_all(
    checkpoint_path: str,
    output_dir: str,
    batch_size: int = 1
) -> Dict[str, Dict[str, float]]:
    """
    Run all benchmarks safely with individual error handling.
    
    Returns:
        Dict mapping benchmark names to their results
    """
    logger.info("=" * 80)
    logger.info("🚀 Starting Safe EmberVLM Evaluation")
    logger.info("=" * 80)
    
    all_results = {}
    
    for benchmark in BENCHMARKS:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Benchmark: {benchmark}")
        logger.info('=' * 80)
        
        results = run_benchmark_isolated(
            benchmark=benchmark,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            batch_size=batch_size
        )
        
        if results:
            all_results[benchmark] = results
            logger.info(f"✅ {benchmark}: {results}")
        else:
            logger.warning(f"⚠️ {benchmark}: No results obtained")
            all_results[benchmark] = {}
    
    # Save aggregated results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary_file = output_path / 'evaluation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n{'=' * 80}")
    logger.info("📊 Evaluation Summary")
    logger.info('=' * 80)
    
    for benchmark, results in all_results.items():
        if results:
            # Extract main metric
            main_metric = None
            if benchmark == 'mmmu_val':
                main_metric = results.get('mmmu_val', {}).get('acc', 0)
            elif benchmark == 'mathvista_testmini':
                main_metric = results.get('mathvista_testmini', {}).get('acc', 0)
            elif benchmark == 'mmstar':
                main_metric = results.get('mmstar', {}).get('average', 0)
            elif benchmark == 'docvqa_test':
                main_metric = results.get('docvqa_test', {}).get('anls', 0)
            elif benchmark == 'textvqa_val':
                main_metric = results.get('textvqa_val', {}).get('acc', 0)
            
            if main_metric is not None:
                logger.info(f"{benchmark:20s}: {main_metric:6.2f}%")
            else:
                logger.info(f"{benchmark:20s}: Results obtained but metric unclear")
        else:
            logger.info(f"{benchmark:20s}: Failed")
    
    logger.info('=' * 80)
    logger.info(f"Summary saved to: {summary_file}")
    
    return all_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Safe EmberVLM Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to EmberVLM checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation')
    parser.add_argument('--benchmark', type=str, default=None,
                        help='Single benchmark to run (default: all)')
    
    args = parser.parse_args()
    
    if args.benchmark:
        # Run single benchmark
        results = run_benchmark_isolated(
            benchmark=args.benchmark,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            batch_size=args.batch_size
        )
        print(json.dumps(results, indent=2))
    else:
        # Run all benchmarks
        all_results = safe_evaluate_all(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            batch_size=args.batch_size
        )
