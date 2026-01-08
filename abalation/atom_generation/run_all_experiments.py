"""
Main script to run the complete atom generation ablation study.

Orchestrates:
1. Atom generation for all model-dataset combinations
2. Metric evaluation (diversity and quality using GPT-5)
3. Result aggregation and ranking
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add current directory and parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(os.path.dirname(parent_dir))
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, project_root)

from experiments.generate_atoms import generate_all_atoms
from experiments.evaluate_metrics import evaluate_all_metrics
from experiments.aggregate_results import aggregate_results
from config import RESULTS_DIR, API_MODELS_TO_TEST, DATASETS


def main():
    """Run the complete ablation study."""
    parser = argparse.ArgumentParser(
        description="Run atom generation ablation study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip atom generation (use existing generated atoms)"
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Skip metric evaluation (use existing metrics)"
    )
    parser.add_argument(
        "--skip-aggregation",
        action="store_true",
        help="Skip result aggregation (use existing rankings)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=RESULTS_DIR,
        help="Results directory"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific models to test (default: all in config)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific datasets to test (default: all in config)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("ATOM GENERATION MODEL ABLATION STUDY")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results directory: {args.results_dir}")
    print("="*80)
    
    models = args.models or API_MODELS_TO_TEST
    datasets = args.datasets or DATASETS
    
    print(f"\nConfiguration:")
    print(f"  Models: {len(models)}")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Total combinations: {len(models) * len(datasets)}")
    
    # Step 1: Generate atoms
    if not args.skip_generation:
        print(f"\n{'='*80}")
        print("STEP 1: GENERATING ATOMS")
        print(f"{'='*80}")
        try:
            generate_all_atoms(
                models=models,
                datasets=datasets,
                results_dir=args.results_dir
            )
        except Exception as e:
            print(f"\n✗ Error in atom generation: {e}")
            print("  Continuing with existing atoms if available...")
    else:
        print("\n⏭ Skipping atom generation (using existing atoms)")
    
    # Step 2: Evaluate metrics (includes GPT-5 quality evaluation)
    if not args.skip_metrics:
        print(f"\n{'='*80}")
        print("STEP 2: EVALUATING METRICS (Diversity + Quality via GPT-5)")
        print(f"{'='*80}")
        try:
            evaluate_all_metrics(
                results_dir=args.results_dir,
                models=models,
                datasets=datasets
            )
        except Exception as e:
            print(f"\n✗ Error in metric evaluation: {e}")
            print("  Continuing with existing metrics if available...")
    else:
        print("\n⏭ Skipping metric evaluation (using existing metrics)")
    
    # Step 3: Aggregate results
    if not args.skip_aggregation:
        print(f"\n{'='*80}")
        print("STEP 3: AGGREGATING RESULTS")
        print(f"{'='*80}")
        try:
            df = aggregate_results(results_dir=args.results_dir)
            
            if not df.empty:
                print(f"\n{'='*80}")
                print("TOP 10 RESULTS (by combined score)")
                print(f"{'='*80}")
                top_10 = df.head(10)
                for idx, row in top_10.iterrows():
                    print(f"{idx+1}. {row['model_name']} ({row['model_family']}) × {row['dataset']}")
                    print(f"   Combined Score: {row['combined_score']:.4f}")
                    print(f"   Diversity: {row['diversity_uniqueness']:.3f} | "
                          f"Quality: {row['quality_coherence']:.2f} | "
                          f"Clarity: {row['quality_clarity']:.2f}")
                    print()
        except Exception as e:
            print(f"\n✗ Error in result aggregation: {e}")
    else:
        print("\n⏭ Skipping result aggregation (using existing rankings)")
    
    print(f"\n{'='*80}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*80}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to: {Path(args.results_dir).absolute()}")
    print(f"  - Generated atoms: {args.results_dir}/generated_atoms/")
    print(f"  - Metrics: {args.results_dir}/metrics/")
    print(f"  - Rankings: {args.results_dir}/final_rankings/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

