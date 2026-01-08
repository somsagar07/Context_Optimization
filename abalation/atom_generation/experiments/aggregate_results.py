"""
Aggregate results and generate rankings.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
atom_gen_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(os.path.dirname(current_dir))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, atom_gen_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, project_root)

from config import RESULTS_DIR, SCORE_WEIGHTS, API_MODELS_TO_TEST, DATASETS
from models import get_model_metadata


def load_metrics(results_dir: str = RESULTS_DIR) -> Dict:
    """Load all metrics from disk."""
    metrics_file = Path(results_dir) / "metrics" / "all_metrics.json"
    if not metrics_file.exists():
        return {}
    
    with open(metrics_file, 'r') as f:
        return json.load(f)


# Consensus evaluation removed - using only diversity and quality metrics


def normalize_score(value: float, min_val: float, max_val: float) -> float:
    """Normalize score to 0-1 range."""
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)


def compute_combined_score(row: Dict, weights: Dict = None) -> float:
    """Compute combined score from diversity and quality metrics only."""
    if weights is None:
        weights = SCORE_WEIGHTS
    
    # Diversity: already 0-1
    diversity_score = (
        row.get("diversity_uniqueness", 0) +
        row.get("diversity_coverage", 0) +
        row.get("diversity_semantic", 0)
    ) / 3.0
    
    # Quality: coherence/clarity are 1-10, specificity is 0-1
    quality_coherence_norm = row.get("quality_coherence", 0) / 10.0
    quality_specificity_norm = row.get("quality_specificity", 0)  # Already 0-1
    quality_clarity_norm = row.get("quality_clarity", 0) / 10.0
    quality_score = (quality_coherence_norm + quality_specificity_norm + quality_clarity_norm) / 3.0
    
    # Weighted combination (no consensus)
    combined = (
        weights["diversity"] * diversity_score +
        weights["quality"] * quality_score
    )
    
    return combined


def aggregate_results(results_dir: str = RESULTS_DIR) -> pd.DataFrame:
    """
    Aggregate all results into a comparison table.
    
    Args:
        results_dir: Results directory
        
    Returns:
        DataFrame with aggregated results
    """
    print(f"\n{'='*80}")
    print(f"AGGREGATING RESULTS")
    print(f"{'='*80}")
    
    # Load data
    print("Loading metrics...")
    metrics = load_metrics(results_dir)
    
    if not metrics:
        print("  ✗ No metrics found. Run evaluate_metrics.py first.")
        return pd.DataFrame()
    
    # Build rows
    rows = []
    
    for model_id, model_metrics in metrics.items():
        model_metadata = get_model_metadata(model_id)
        
        for dataset_name, dataset_metrics in model_metrics.items():
            row = {
                "model_id": model_id,
                "model_name": model_metadata["display_name"],
                "model_family": model_metadata["family"],
                "model_size": model_metadata["estimated_params"],
                "dataset": dataset_name,
            }
            
            # Diversity metrics
            diversity = dataset_metrics.get("diversity", {}).get("overall", {})
            row["diversity_uniqueness"] = diversity.get("uniqueness", 0)
            row["diversity_coverage"] = diversity.get("strategy_coverage", 0)
            row["diversity_semantic"] = diversity.get("semantic_diversity", 0)
            
            # Quality metrics
            quality = dataset_metrics.get("quality", {}).get("overall", {})
            row["quality_coherence"] = quality.get("coherence", 0)
            row["quality_specificity"] = quality.get("specificity", 0)
            row["quality_clarity"] = quality.get("clarity", 0)
            
            # Compute combined score (diversity + quality only)
            row["combined_score"] = compute_combined_score(row)
            
            rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    if df.empty:
        print("  ✗ No data to aggregate")
        return df
    
    # Save full table
    output_dir = Path(results_dir) / "final_rankings"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    csv_file = output_dir / "all_results.csv"
    df.to_csv(csv_file, index=False)
    print(f"  ✓ Saved full results to {csv_file}")
    
    # Generate rankings
    rankings = {}
    
    # By diversity
    df_diversity = df.copy()
    df_diversity["diversity_score"] = (
        df_diversity["diversity_uniqueness"] +
        df_diversity["diversity_coverage"] +
        df_diversity["diversity_semantic"]
    ) / 3.0
    df_diversity = df_diversity.sort_values("diversity_score", ascending=False)
    rankings["diversity"] = df_diversity
    df_diversity.to_csv(output_dir / "ranking_by_diversity.csv", index=False)
    
    # By quality
    df_quality = df.copy()
    df_quality["quality_score"] = (
        df_quality["quality_coherence"] / 10.0 +
        df_quality["quality_specificity"] +
        df_quality["quality_clarity"] / 10.0
    ) / 3.0
    df_quality = df_quality.sort_values("quality_score", ascending=False)
    rankings["quality"] = df_quality
    df_quality.to_csv(output_dir / "ranking_by_quality.csv", index=False)
    
    # By combined score (diversity + quality)
    df_combined = df.sort_values("combined_score", ascending=False)
    rankings["combined"] = df_combined
    df_combined.to_csv(output_dir / "ranking_by_combined.csv", index=False)
    
    print(f"  ✓ Generated ranking tables")
    
    # Create human evaluation format
    human_eval = []
    
    for _, row in df_combined.iterrows():
        human_eval.append({
            "model_id": row["model_id"],
            "model_name": row["model_name"],
            "model_family": row["model_family"],
            "dataset": row["dataset"],
            "scores": {
                "diversity": {
                    "uniqueness": row["diversity_uniqueness"],
                    "coverage": row["diversity_coverage"],
                    "semantic": row["diversity_semantic"],
                },
                "quality": {
                    "coherence": row["quality_coherence"],
                    "specificity": row["quality_specificity"],
                    "clarity": row["quality_clarity"],
                },
                "combined": row["combined_score"],
            },
        })
    
    human_eval_file = output_dir / "human_evaluation_format.json"
    with open(human_eval_file, 'w') as f:
        json.dump(human_eval, f, indent=2)
    
    print(f"  ✓ Saved human evaluation format to {human_eval_file}")
    
    print(f"\n{'='*80}")
    print(f"Aggregation complete!")
    print(f"{'='*80}")
    
    return df


if __name__ == "__main__":
    # Run aggregation
    df = aggregate_results()
    
    if not df.empty:
        print(f"\nTop 5 models by combined score:")
        print(df.head(5)[["model_name", "dataset", "combined_score"]])

