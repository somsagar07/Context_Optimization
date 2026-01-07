"""
Generate final table from summary files with mean ± std.
"""
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def parse_summary_file(filepath):
    """Parse a summary file and extract data."""
    data = []
    current_section = None
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Detect section headers
        if "PROJECTED MODE:" in line:
            current_section = "projected"
            continue
        elif "NATIVE MODE:" in line:
            current_section = "native"
            continue
        elif "ALL RESULTS:" in line:
            current_section = None  # Skip ALL RESULTS section
            continue
        
        # Skip header lines and empty lines
        if not line or "Embedder" in line and "Mode" in line:
            continue
        
        # Parse data lines
        if current_section and line:
            # Split by whitespace (handling variable spacing)
            parts = line.split()
            if len(parts) >= 8:
                try:
                    # Extract embedder name (everything before mode)
                    embedder_part = []
                    mode_idx = None
                    for j, part in enumerate(parts):
                        if part in ["projected", "native"]:
                            mode_idx = j
                            break
                        embedder_part.append(part)
                    
                    if mode_idx is None:
                        continue
                    
                    embedder_str = " ".join(embedder_part)
                    mode = parts[mode_idx]
                    
                    # Extract values (after mode)
                    values = parts[mode_idx + 1:]
                    if len(values) >= 6:
                        overall_score = float(values[0])
                        clustering_ari = float(values[1])
                        classify_avg_acc = float(values[2])
                        complexity_spearman = float(values[3])
                        decision_score = float(values[4])
                        time_seconds = float(values[5])
                        
                        # Extract base embedder name and dimension info
                        # Format: "sentence-t5-base (projected)" or "sentence-t5-base (native)"
                        match = re.match(r'(.+?)\s*\((.+?)\)', embedder_str)
                        if match:
                            base_embedder = match.group(1).strip()
                            dim_info = match.group(2).strip()
                        else:
                            base_embedder = embedder_str
                            dim_info = current_section
                        
                        data.append({
                            "Base_Embedder": base_embedder,
                            "Embedder": embedder_str,
                            "Mode": mode,
                            "Dim_Info": dim_info,
                            "Overall_Score": overall_score,
                            "Clustering_ARI": clustering_ari,
                            "Classify_Avg_Acc": classify_avg_acc,
                            "Complexity_Spearman": complexity_spearman,
                            "Decision_Score": decision_score,
                            "Time_Seconds": time_seconds
                        })
                except (ValueError, IndexError) as e:
                    continue
    
    return data

def get_embedder_dimension(embedder_name):
    """Get dimension info for embedder from config."""
    # This is a simplified version - you may need to adjust based on your embedder configs
    dim_map = {
        "all-MiniLM-L6-v2": "384D",
        "all-MiniLM-L12-v2": "384D",
        "all-mpnet-base-v2": "768D",
        "sentence-t5-base": "768D",
        "e5-base": "768D",
        "jina-clip-v2": "1024D",
        "flava-full": "768D",
        "siglip-base": "768D",
        "siglip-large": "1024D",
        "clip-base": "512D",
        "clip-large": "768D",
        "clip-base-patch16": "512D",
        "metaclip-h14": "1024D",
    }
    return dim_map.get(embedder_name, "N/A")

def generate_final_table(summary_files, output_file="final_results_table.csv"):
    """Generate final table with mean ± std from summary files."""
    all_data = []
    
    # Parse all summary files
    for summary_file in summary_files:
        data = parse_summary_file(summary_file)
        all_data.extend(data)
    
    if not all_data:
        print("No data found in summary files!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Group by Base_Embedder and Mode
    grouped = df.groupby(["Base_Embedder", "Mode"])
    
    results = []
    for (base_embedder, mode), group_df in grouped:
        # Get dimension info - for projected mode, always use 512D
        if mode == "projected":
            dim_info = "512D"
        else:
            dim_info = get_embedder_dimension(base_embedder)
        embedder_display = f"{base_embedder} ({dim_info})"
        
        # Compute mean ± std for each metric
        metrics = {
            "Clustering_ARI": group_df["Clustering_ARI"].values,
            "Classify_Avg_Acc": group_df["Classify_Avg_Acc"].values,
            "Complexity_Spearman": group_df["Complexity_Spearman"].values,
            "Decision_Score": group_df["Decision_Score"].values,
            "Overall_Score": group_df["Overall_Score"].values,
            "Time_Seconds": group_df["Time_Seconds"].values,
        }
        
        result_row = {
            "Embedder": embedder_display,
            "Mode": mode,
        }
        
        for metric_name, values in metrics.items():
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values)
                result_row[metric_name] = f"{mean_val:.4f} ± {std_val:.4f}"
            else:
                result_row[metric_name] = "N/A"
        
        results.append(result_row)
    
    # Create final DataFrame
    final_df = pd.DataFrame(results)
    
    # Reorder columns
    column_order = [
        "Embedder",
        "Mode",
        "Clustering_ARI",
        "Classify_Avg_Acc",
        "Complexity_Spearman",
        "Decision_Score",
        "Overall_Score",
        "Time_Seconds"
    ]
    final_df = final_df[column_order]
    
    # Sort by Overall_Score (descending)
    # Extract mean from Overall_Score for sorting
    def extract_mean(score_str):
        try:
            return float(score_str.split(" ± ")[0])
        except:
            return 0.0
    
    final_df["_sort_key"] = final_df["Overall_Score"].apply(extract_mean)
    final_df = final_df.sort_values("_sort_key", ascending=False)
    final_df = final_df.drop("_sort_key", axis=1)
    
    # Save to CSV
    output_path = Path("embedder_selection_results") / output_file
    final_df.to_csv(output_path, index=False)
    
    print(f"\nFinal table saved to: {output_path}")
    print("\n" + "="*100)
    print("FINAL RESULTS TABLE (Mean ± Std)")
    print("="*100)
    print(final_df.to_string(index=False))
    print("="*100)
    
    return final_df

def main():
    """Main function."""
    results_dir = Path("embedder_selection_results")
    
    # Find all summary files
    summary_files = sorted(results_dir.glob("summary_*.txt"))
    
    if not summary_files:
        print("No summary files found!")
        return
    
    print(f"Found {len(summary_files)} summary files:")
    for f in summary_files:
        print(f"  - {f}")
    
    # Generate final table
    final_df = generate_final_table(summary_files, output_file="final_results_table.csv")
    
    # Also save as Excel
    if final_df is not None:
        excel_path = results_dir / "final_results_table.xlsx"
        final_df.to_excel(excel_path, index=False)
        print(f"\nAlso saved to: {excel_path}")

if __name__ == "__main__":
    main()

