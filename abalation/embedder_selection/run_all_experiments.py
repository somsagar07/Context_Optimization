"""
Main script to run all embedding ablation experiments.
Generates comprehensive comparison table.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedders import create_embedder, EMBEDDER_CONFIGS
from config import EMBEDDERS_TO_TEST, TARGET_EMBEDDING_DIM, RESULTS_DIR, DATASETS, PROJECT_TO_SAME_DIM
from experiment_1_clustering import run_clustering_experiment
from experiment_2_classification import run_classification_experiment
from experiment_3_complexity import run_complexity_experiment
from experiment_5_decision_prediction import run_decision_prediction_experiment


def run_all_experiments_for_embedder(embedder_name: str, embedder_config: Dict, use_projection: bool = True) -> Dict:
    """Run all experiments for a single embedder."""
    projection_mode = "projected" if use_projection else "native"
    print(f"\n{'='*80}")
    print(f"Testing Embedder: {embedder_name} ({projection_mode} dimensions)")
    print(f"Description: {embedder_config.get('description', 'N/A')}")
    print(f"{'='*80}")
    
    all_results = {
        "embedder": embedder_name,
        "config": embedder_config,
        "projection_mode": projection_mode,
        "use_projection": use_projection,
    }
    
    start_time = time.time()
    
    try:
        # Create embedder
        embedder_type = embedder_config["type"]
        print(f"\nInitializing embedder: {embedder_type}...")
        target_dim = TARGET_EMBEDDING_DIM if use_projection else None
        embedder = create_embedder(embedder_type, target_dim=target_dim)
        actual_dim = embedder.get_dimension()
        print(f"Using {'projected' if use_projection else 'native'} dimension: {actual_dim}")
        
        # Run experiments
        print(f"\nRunning Experiment 1: Clustering...")
        try:
            results_clustering = run_clustering_experiment(embedder, embedder_name)
            all_results["clustering"] = results_clustering
        except Exception as e:
            print(f"  Error in clustering experiment: {e}")
            all_results["clustering"] = {"error": str(e)}
        
        print(f"\nRunning Experiment 2: Classification...")
        try:
            results_classification = run_classification_experiment(embedder, embedder_name)
            all_results["classification"] = results_classification
        except Exception as e:
            print(f"  Error in classification experiment: {e}")
            all_results["classification"] = {"error": str(e)}
        
        print(f"\nRunning Experiment 3: Complexity...")
        try:
            results_complexity = run_complexity_experiment(embedder, embedder_name)
            all_results["complexity"] = results_complexity
        except Exception as e:
            print(f"  Error in complexity experiment: {e}")
            all_results["complexity"] = {"error": str(e)}
        
        print(f"\nRunning Experiment 4: Decision Prediction...")
        try:
            results_decision = run_decision_prediction_experiment(embedder, embedder_name)
            all_results["decision_prediction"] = results_decision
        except Exception as e:
            print(f"  Error in decision prediction experiment: {e}")
            all_results["decision_prediction"] = {"error": str(e)}
        
        elapsed_time = time.time() - start_time
        all_results["total_time_seconds"] = elapsed_time
        
        print(f"\n✓ Completed all experiments for {embedder_name} in {elapsed_time:.1f}s")
        
    except Exception as e:
        print(f"\n✗ Failed to test {embedder_name}: {e}")
        all_results["error"] = str(e)
    
    return all_results


def aggregate_results(all_embedder_results: List[Dict], separate_by_mode: bool = True) -> pd.DataFrame:
    """Aggregate results into comparison table."""
    rows = []
    
    for result in all_embedder_results:
        embedder_name = result["embedder"]
        projection_mode = result.get("projection_mode", "unknown")
        
        if separate_by_mode:
            # Include mode in embedder name for separate rows
            display_name = f"{embedder_name} ({projection_mode})"
        else:
            display_name = embedder_name
        
        row = {
            "Embedder": display_name,
            "Base_Embedder": embedder_name,  # Original name without mode
            "Mode": projection_mode,
            "Description": result.get("config", {}).get("description", "N/A"),
        }
        
        # Clustering metrics
        if "clustering" in result and "error" not in result["clustering"]:
            clust = result["clustering"]
            row["Clustering_Silhouette"] = clust.get("silhouette_score", 0)
            row["Clustering_ARI"] = clust.get("adjusted_rand_index", 0)
            row["Clustering_Separation"] = clust.get("cluster_separation", 0)
        else:
            row["Clustering_Silhouette"] = None
            row["Clustering_ARI"] = None
            row["Clustering_Separation"] = None
        
        # Classification metrics
        if "classification" in result and "error" not in result["classification"]:
            classif = result["classification"]
            row["Classify_Dataset_Acc"] = classif.get("dataset_classification", {}).get("test_accuracy", 0)
            row["Classify_Tool_Acc"] = classif.get("tool_classification", {}).get("test_accuracy", 0)
            row["Classify_Avg_Acc"] = classif.get("average_accuracy", 0)
        else:
            row["Classify_Dataset_Acc"] = None
            row["Classify_Tool_Acc"] = None
            row["Classify_Avg_Acc"] = None
        
        # Complexity metrics
        if "complexity" in result and "error" not in result["complexity"]:
            compl = result["complexity"]
            row["Complexity_Spearman"] = compl.get("ranking", {}).get("spearman_test", 0)
            row["Complexity_R2"] = compl.get("regression", {}).get("test_r2", 0)
            row["Complexity_Score"] = compl.get("complexity_score", 0)
        else:
            row["Complexity_Spearman"] = None
            row["Complexity_R2"] = None
            row["Complexity_Score"] = None
        
        # Decision prediction metrics
        if "decision_prediction" in result and "error" not in result["decision_prediction"]:
            decision = result["decision_prediction"]
            row["Decision_Workflow_Acc"] = decision.get("workflow_prediction", {}).get("accuracy", 0)
            row["Decision_Tool_Acc"] = decision.get("tool_prediction", {}).get("accuracy", 0)
            row["Decision_Budget_Acc"] = decision.get("budget_prediction", {}).get("accuracy", 0)
            row["Decision_Combined_Acc"] = decision.get("combined_accuracy", 0)
            row["Decision_Score"] = decision.get("decision_score", 0)
        else:
            row["Decision_Workflow_Acc"] = None
            row["Decision_Tool_Acc"] = None
            row["Decision_Budget_Acc"] = None
            row["Decision_Combined_Acc"] = None
            row["Decision_Score"] = None
        
        # Time
        row["Time_Seconds"] = result.get("total_time_seconds", None)
        
        # Compute overall score (normalized average of key metrics)
        scores = []
        if row["Clustering_ARI"] is not None:
            scores.append(row["Clustering_ARI"])
        if row["Classify_Avg_Acc"] is not None:
            scores.append(row["Classify_Avg_Acc"])
        if row["Complexity_Spearman"] is not None:
            scores.append(row["Complexity_Spearman"])
        if row["Decision_Score"] is not None:
            scores.append(row["Decision_Score"])
        
        row["Overall_Score"] = sum(scores) / len(scores) if scores else None
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def generate_summary_table(df: pd.DataFrame) -> str:
    """Generate formatted summary table for display."""
    # Sort by overall score
    df_sorted = df.sort_values("Overall_Score", ascending=False, na_position='last')
    
    # Select key columns for summary
    summary_cols = ["Embedder", "Overall_Score"]
    
    # Add mode if present
    if "Mode" in df.columns:
        summary_cols.insert(1, "Mode")
    
    summary_cols.extend([
        "Clustering_ARI",
        "Classify_Avg_Acc",
        "Complexity_Spearman",
        "Decision_Score",
        "Time_Seconds"
    ])
    
    summary_df = df_sorted[summary_cols].copy()
    
    # Format numeric columns
    for col in summary_df.columns:
        if col not in ["Embedder", "Mode", "Base_Embedder"] and summary_df[col].dtype in ['float64', 'float32']:
            summary_df[col] = summary_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    
    return summary_df.to_string(index=False)


def main():
    """Run all experiments and generate comparison table."""
    print("="*80)
    print("Embedding Ablation Study")
    print("="*80)
    print(f"Testing BOTH modes:")
    print(f"  1. Projected to {TARGET_EMBEDDING_DIM}D (fair comparison, RL-ready)")
    print(f"  2. Native dimensions (no information loss, intrinsic quality)")
    print(f"Datasets: {', '.join(DATASETS)}")
    print(f"Embedders to test: {len(EMBEDDERS_TO_TEST)}")
    print("="*80)
    
    # Create results directory
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(exist_ok=True)
    
    # Run experiments for each embedder in BOTH modes
    all_results = []
    
    # Test both projection modes
    projection_modes = [True, False]  # First projected, then native
    
    for use_projection in projection_modes:
        mode_name = "projected" if use_projection else "native"
        print(f"\n{'#'*80}")
        print(f"# TESTING MODE: {mode_name.upper()} DIMENSIONS")
        print(f"{'#'*80}\n")
        
        for embedder_name in EMBEDDERS_TO_TEST:
            if embedder_name not in EMBEDDER_CONFIGS:
                print(f"\nWarning: {embedder_name} not found in EMBEDDER_CONFIGS, skipping...")
                continue
            
            embedder_config = EMBEDDER_CONFIGS[embedder_name]
            results = run_all_experiments_for_embedder(embedder_name, embedder_config, use_projection=use_projection)
            all_results.append(results)
            
            # Save intermediate results
            mode_suffix = "_projected" if use_projection else "_native"
            with open(results_dir / f"{embedder_name}{mode_suffix}_results.json", "w") as f:
                json.dump(results, f, indent=2)
    
    # Save all results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"all_results_{timestamp}.json"
    with open(results_dir / results_filename, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {results_dir / results_filename}")
    
    # Generate comparison table
    print(f"\n{'='*80}")
    print("Generating Comparison Table")
    print(f"{'='*80}")
    
    df = aggregate_results(all_results, separate_by_mode=True)
    
    # Save full table
    df.to_csv(results_dir / "comparison_table.csv", index=False)
    df.to_excel(results_dir / "comparison_table.xlsx", index=False)
    
    # Generate separate tables for each mode
    df_projected = df[df["Mode"] == "projected"].copy()
    df_native = df[df["Mode"] == "native"].copy()
    
    if not df_projected.empty:
        df_projected.to_csv(results_dir / "comparison_table_projected.csv", index=False)
        df_projected.to_excel(results_dir / "comparison_table_projected.xlsx", index=False)
    
    if not df_native.empty:
        df_native.to_csv(results_dir / "comparison_table_native.csv", index=False)
        df_native.to_excel(results_dir / "comparison_table_native.xlsx", index=False)
    
    # Generate summary for both modes
    print("\n" + "="*80)
    print("SUMMARY - Projected Dimensions Mode")
    print("="*80)
    if not df_projected.empty:
        summary_projected = generate_summary_table(df_projected)
        print(summary_projected)
    else:
        print("No projected results available")
    
    print("\n" + "="*80)
    print("SUMMARY - Native Dimensions Mode")
    print("="*80)
    if not df_native.empty:
        summary_native = generate_summary_table(df_native)
        print(summary_native)
    else:
        print("No native results available")
    
    # Combined summary showing both modes
    print("\n" + "="*80)
    print("COMBINED SUMMARY - All Results")
    print("="*80)
    summary_all = generate_summary_table(df)
    print(summary_all)
    
    # Save summary
    with open(results_dir / "summary.txt", "w") as f:
        f.write("Embedding Ablation Study - Summary\n")
        f.write("="*80 + "\n\n")
        f.write("PROJECTED MODE:\n")
        if not df_projected.empty:
            f.write(generate_summary_table(df_projected))
            f.write("\n\n")
        f.write("NATIVE MODE:\n")
        if not df_native.empty:
            f.write(generate_summary_table(df_native))
            f.write("\n\n")
        f.write("ALL RESULTS:\n")
        f.write(summary_all)
        f.write("\n\n")
        f.write("="*80 + "\n")
        f.write("Full results saved to:\n")
        f.write(f"  - {results_dir / 'comparison_table.csv'}\n")
        f.write(f"  - {results_dir / 'comparison_table.xlsx'}\n")
        f.write(f"  - {results_dir / results_filename}\n")
    
    print(f"\n✓ Results saved to {results_dir}/")
    print(f"  - comparison_table.csv (all results)")
    print(f"  - comparison_table_projected.csv (projected mode only)")
    print(f"  - comparison_table_native.csv (native mode only)")
    
    # Print recommendations for both modes
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    if not df_projected.empty:
        # Filter out rows with NaN Overall_Score
        df_projected_valid = df_projected[df_projected["Overall_Score"].notna()]
        if not df_projected_valid.empty:
            top_projected = df_projected_valid.loc[df_projected_valid["Overall_Score"].idxmax()]
            print(f"\nBest Embedder (Projected to {TARGET_EMBEDDING_DIM}D):")
            print(f"  Name: {top_projected['Base_Embedder']}")
            print(f"  Overall Score: {top_projected['Overall_Score']:.4f}")
            print(f"  → Use this for RL training (fixed dimension required)")
        else:
            print("\nNo valid results in projected mode")
    
    if not df_native.empty:
        # Filter out rows with NaN Overall_Score
        df_native_valid = df_native[df_native["Overall_Score"].notna()]
        if not df_native_valid.empty:
            top_native = df_native_valid.loc[df_native_valid["Overall_Score"].idxmax()]
            print(f"\nBest Embedder (Native Dimensions):")
            print(f"  Name: {top_native['Base_Embedder']}")
            print(f"  Overall Score: {top_native['Overall_Score']:.4f}")
            print(f"  → Intrinsic quality (no information loss)")
        else:
            print("\nNo valid results in native mode")
    
    # Overall best
    df_valid = df[df["Overall_Score"].notna()]
    if not df_valid.empty:
        top_overall = df_valid.loc[df_valid["Overall_Score"].idxmax()]
        print(f"\nOverall Best (across both modes):")
        print(f"  Name: {top_overall['Base_Embedder']} ({top_overall['Mode']} mode)")
        print(f"  Overall Score: {top_overall['Overall_Score']:.4f}")
        if not df_projected.empty and not df_native.empty:
            df_projected_valid = df_projected[df_projected["Overall_Score"].notna()]
            df_native_valid = df_native[df_native["Overall_Score"].notna()]
            if not df_projected_valid.empty and not df_native_valid.empty:
                top_proj = df_projected_valid.loc[df_projected_valid["Overall_Score"].idxmax()]
                top_nat = df_native_valid.loc[df_native_valid["Overall_Score"].idxmax()]
                print(f"\nComparison:")
                print(f"  Projected mode best: {top_proj['Base_Embedder']} ({top_proj['Overall_Score']:.4f})")
                print(f"  Native mode best: {top_nat['Base_Embedder']} ({top_nat['Overall_Score']:.4f})")
                print(f"\n  → If scores are similar, use projected mode for RL")
                print(f"  → If native mode is significantly better, consider using that embedder's native dimension")
    else:
        print("\nNo valid results available")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

