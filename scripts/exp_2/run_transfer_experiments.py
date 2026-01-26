"""
Policy Transfer Experiments for Table 2

Runs all transfer experiments needed for Table 2 (in-domain policy transfer):
- Computes dataset similarity (DS) via embedding cosine distance
- Evaluates source accuracy (S_T) on training dataset
- Evaluates zero-shot accuracy (S_N) on new dataset

Experiments:
  Reasoning:
    - GSM8K → DROP
    - GSM8K → MedQA
    - DROP → GSM8K
  
  Tool Use:
    - HotpotQA → GAIA
    - GAIA → HotpotQA
    - HotpotQA → MedQA

Usage:
    # Run all experiments
    python scripts/exp_2/run_transfer_experiments.py --all --api --api-model "google/gemini-2.5-flash-lite" --workers 8
    
    # Run specific experiment
    python scripts/exp_2/run_transfer_experiments.py \
        --source gsm8k --target drop \
        --source-structure models/ppo/gsm8k/structure_final.pt \
        --source-prompt models/ppo/gsm8k/prompt_final.pt \
        --api --api-model "google/gemini-2.5-flash-lite" --workers 8
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from configs import load_config
from scripts.transfer_policy import evaluate_zero_shot
from scripts.eval_hrl import evaluate, load_structure_policy, load_prompt_policy
import torch


# Define all transfer experiments for Table 2
TRANSFER_EXPERIMENTS = {
    "reasoning": [
        ("gsm8k", "drop"),
        ("gsm8k", "medqa"),
        ("drop", "gsm8k"),
    ],
    "tool_use": [
        ("hotpotqa", "gaia"),
        ("gaia", "hotpotqa"),
        ("hotpotqa", "medqa"),
    ],
}


def load_dataset_embeddings(dataset_name, split="train"):
    """Load precomputed embeddings for a dataset."""
    embeddings_cache_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "embeddings_cache"
    )
    
    # Try different split names
    possible_names = [
        f"{dataset_name}_{split}_embeddings.npz",
        f"{dataset_name}_train_embeddings.npz",
        f"{dataset_name}_validation_embeddings.npz",
        f"{dataset_name}_test_embeddings.npz",
    ]
    
    for filename in possible_names:
        filepath = os.path.join(embeddings_cache_dir, filename)
        if os.path.exists(filepath):
            print(f"  Loading embeddings from {filename}...")
            data = np.load(filepath, allow_pickle=True)
            embeddings = data["embeddings"]
            print(f"  Loaded {len(embeddings)} embeddings (shape: {embeddings.shape})")
            return embeddings
    
    raise FileNotFoundError(
        f"Could not find embeddings for {dataset_name} (split: {split}). "
        f"Tried: {possible_names}. "
        f"Please run scripts/precompute_embeddings.py first."
    )


def compute_dataset_similarity(source_dataset, target_dataset):
    """
    Compute dataset similarity (DS) via embedding cosine distance.
    
    Returns:
        float: Cosine similarity between average embeddings (0-1, higher = more similar)
    """
    print(f"\nComputing dataset similarity: {source_dataset} ↔ {target_dataset}")
    
    try:
        # Load embeddings
        source_embeddings = load_dataset_embeddings(source_dataset)
        target_embeddings = load_dataset_embeddings(target_dataset)
        
        # Compute average embedding for each dataset
        source_mean = np.mean(source_embeddings, axis=0, keepdims=True)
        target_mean = np.mean(target_embeddings, axis=0, keepdims=True)
        
        # Compute cosine similarity
        similarity = cosine_similarity(source_mean, target_mean)[0, 0]
        
        print(f"  Dataset similarity (cosine): {similarity:.3f}")
        return float(similarity)
        
    except Exception as e:
        print(f"  Warning: Could not compute similarity: {e}")
        print(f"  Returning 0.0 (unknown similarity)")
        return 0.0


def evaluate_source_accuracy(source_dataset, structure_path, prompt_path, cfg, 
                            num_episodes=50, use_api=False, api_model=None, hf_model=None,
                            num_workers=1):
    """
    Evaluate policy on source dataset to get S_T (training dataset accuracy).
    
    Returns:
        float: Accuracy on source dataset
    """
    print(f"\nEvaluating source accuracy on {source_dataset}...")
    
    # Set config to source dataset
    cfg.DATASET_NAME = source_dataset
    
    # Load policies
    device = "cuda" if torch.cuda.is_available() else "cpu"
    structure_policy, _ = load_structure_policy(structure_path, device)
    prompt_policy, _ = load_prompt_policy(prompt_path, device)
    
    # Evaluate
    if use_api and num_workers > 1:
        from scripts.eval_hrl import evaluate_parallel
        results = evaluate_parallel(
            structure_policy, prompt_policy, cfg,
            num_episodes=num_episodes,
            deterministic=True,
            use_api=use_api,
            api_model=api_model,
            hf_model=hf_model,
            temperature=1.0,
            num_workers=num_workers
        )
    else:
        results = evaluate(
            structure_policy, prompt_policy, cfg,
            num_episodes=num_episodes,
            deterministic=True,
            verbose=False,
            use_api=use_api,
            api_model=api_model,
            hf_model=hf_model,
            temperature=1.0
        )
    
    accuracy = results.get("accuracy", 0.0)
    print(f"  Source accuracy (S_T): {accuracy:.1%}")
    return accuracy


def run_single_transfer_experiment(source_dataset, target_dataset, structure_path, prompt_path,
                                  cfg, num_episodes=50, use_api=False, api_model=None, 
                                  hf_model=None, num_workers=1, compute_similarity=True):
    """
    Run a single transfer experiment.
    
    Returns:
        dict with DS, S_T, S_N
    """
    print("\n" + "="*70)
    print(f"TRANSFER EXPERIMENT: {source_dataset.upper()} → {target_dataset.upper()}")
    print("="*70)
    
    results = {}
    
    # 1. Compute dataset similarity (DS)
    if compute_similarity:
        results["DS"] = compute_dataset_similarity(source_dataset, target_dataset)
    else:
        results["DS"] = None
    
    # 2. Evaluate source accuracy (S_T)
    try:
        results["S_T"] = evaluate_source_accuracy(
            source_dataset, structure_path, prompt_path, cfg,
            num_episodes=num_episodes, use_api=use_api, api_model=api_model,
            hf_model=hf_model, num_workers=num_workers
        )
    except Exception as e:
        print(f"  Error evaluating source accuracy: {e}")
        results["S_T"] = None
    
    # 3. Evaluate zero-shot target accuracy (S_N)
    try:
        transfer_results = evaluate_zero_shot(
            structure_path, prompt_path, target_dataset, cfg,
            num_episodes=num_episodes, use_api=use_api, api_model=api_model,
            hf_model=hf_model, num_workers=num_workers
        )
        results["S_N"] = transfer_results.get("accuracy", 0.0)
        print(f"  Target accuracy (S_N): {results['S_N']:.1%}")
    except Exception as e:
        print(f"  Error evaluating target accuracy: {e}")
        results["S_N"] = None
    
    return results


def find_model_paths(dataset_name, model_dir="models/ppo"):
    """
    Find structure and prompt model paths for a dataset.
    
    Returns:
        tuple: (structure_path, prompt_path) or (None, None) if not found
    """
    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        model_dir,
        dataset_name
    )
    
    structure_path = os.path.join(base_dir, "structure_final.pt")
    prompt_path = os.path.join(base_dir, "prompt_final.pt")
    
    # Try alternative names
    if not os.path.exists(structure_path):
        alt_structure = os.path.join(base_dir, "structure.pt")
        if os.path.exists(alt_structure):
            structure_path = alt_structure
    
    if not os.path.exists(prompt_path):
        alt_prompt = os.path.join(base_dir, "prompt.pt")
        if os.path.exists(alt_prompt):
            prompt_path = alt_prompt
    
    if os.path.exists(structure_path) and os.path.exists(prompt_path):
        return structure_path, prompt_path
    
    return None, None


def run_all_experiments(cfg, num_episodes=50, use_api=False, api_model=None, 
                       hf_model=None, num_workers=1, output_dir="transfer_results"):
    """Run all transfer experiments defined in TRANSFER_EXPERIMENTS."""
    
    all_results = {}
    
    for capability, experiments in TRANSFER_EXPERIMENTS.items():
        print("\n" + "="*70)
        print(f"RUNNING {capability.upper()} EXPERIMENTS")
        print("="*70)
        
        capability_results = {}
        
        for source_dataset, target_dataset in experiments:
            # Find model paths
            structure_path, prompt_path = find_model_paths(source_dataset)
            
            if structure_path is None or prompt_path is None:
                print(f"\n⚠️  Skipping {source_dataset} → {target_dataset}: Models not found")
                print(f"   Expected: models/ppo/{source_dataset}/structure_final.pt")
                print(f"   Expected: models/ppo/{source_dataset}/prompt_final.pt")
                capability_results[f"{source_dataset}_to_{target_dataset}"] = {
                    "DS": None, "S_T": None, "S_N": None,
                    "error": "Models not found"
                }
                continue
            
            # Run experiment
            try:
                results = run_single_transfer_experiment(
                    source_dataset, target_dataset, structure_path, prompt_path,
                    cfg, num_episodes=num_episodes, use_api=use_api, 
                    api_model=api_model, hf_model=hf_model, num_workers=num_workers
                )
                capability_results[f"{source_dataset}_to_{target_dataset}"] = results
            except Exception as e:
                print(f"\n❌ Error in {source_dataset} → {target_dataset}: {e}")
                capability_results[f"{source_dataset}_to_{target_dataset}"] = {
                    "DS": None, "S_T": None, "S_N": None,
                    "error": str(e)
                }
        
        all_results[capability] = capability_results
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"transfer_experiments_{timestamp}.json")
    
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_file}")
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE (for LaTeX)")
    print("="*70)
    print_table_summary(all_results)
    
    return all_results, output_file


def print_table_summary(results):
    """Print results in a format suitable for LaTeX table."""
    print("\n\\begin{table}[t]")
    print("    \\centering")
    print("    \\small")
    print("    \\caption{In-domain policy transfer across datasets. \\textbf{DS}: dataset similarity measured via embedding cosine distance; \\textbf{$S_T$}: accuracy on training dataset; \\textbf{$S_N$}: zero-shot accuracy on new dataset. Policies trained on one dataset show limited transfer to related tasks, with performance degradation of $20$--$50\\%$.}")
    print("    \\label{tab:in_domain_transfer}")
    print("    \\setlength{\\tabcolsep}{6pt}")
    print("    \\begin{tabular}{l | l | c | c | c}")
    print("        \\toprule")
    print("        \\textbf{Capability} ")
    print("        & \\textbf{Train $\\rightarrow$ New} ")
    print("        & \\textbf{DS} ")
    print("        & \\textbf{$S_T$} ")
    print("        & \\textbf{$S_N$} \\\\")
    print("        \\midrule")
    
    # Reasoning experiments
    print("        \\multirow{3}{*}{Reasoning} ")
    reasoning_exps = TRANSFER_EXPERIMENTS["reasoning"]
    for i, (source, target) in enumerate(reasoning_exps):
        key = f"{source}_to_{target}"
        exp_results = results.get("reasoning", {}).get(key, {})
        ds = exp_results.get("DS")
        s_t = exp_results.get("S_T")
        s_n = exp_results.get("S_N")
        
        ds_str = f"{ds:.2f}" if ds is not None else "--"
        s_t_str = f"{s_t:.1%}" if s_t is not None else "--"
        s_n_str = f"{s_n:.1%}" if s_n is not None else "--"
        
        source_upper = source.upper() if source == "gsm8k" else source.capitalize()
        target_upper = target.upper() if target == "gsm8k" else target.capitalize()
        
        print(f"            & {source_upper} $\\rightarrow$ {target_upper}       & {ds_str} & {s_t_str} & {s_n_str} \\\\")
    
    print("        \\midrule")
    print("        \\multirow{3}{*}{\\makecell[l]{Tool Use}} ")
    tool_use_exps = TRANSFER_EXPERIMENTS["tool_use"]
    for i, (source, target) in enumerate(tool_use_exps):
        key = f"{source}_to_{target}"
        exp_results = results.get("tool_use", {}).get(key, {})
        ds = exp_results.get("DS")
        s_t = exp_results.get("S_T")
        s_n = exp_results.get("S_N")
        
        ds_str = f"{ds:.2f}" if ds is not None else "--"
        s_t_str = f"{s_t:.1%}" if s_t is not None else "--"
        s_n_str = f"{s_n:.1%}" if s_n is not None else "--"
        
        source_upper = source.capitalize()
        target_upper = target.upper() if target == "gaia" else target.capitalize()
        
        print(f"            & {source_upper} $\\rightarrow$ {target_upper}    & {ds_str} & {s_t_str} & {s_n_str} \\\\")
    
    print("        \\bottomrule")
    print("    \\end{tabular}")
    print("\\end{table}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run policy transfer experiments for Table 2",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Experiment selection
    parser.add_argument("--all", action="store_true",
                       help="Run all transfer experiments")
    parser.add_argument("--source", type=str, default=None,
                       help="Source dataset (for single experiment)")
    parser.add_argument("--target", type=str, default=None,
                       help="Target dataset (for single experiment)")
    
    # Model paths (for single experiment)
    parser.add_argument("--source-structure", type=str, default=None,
                       help="Path to source structure policy")
    parser.add_argument("--source-prompt", type=str, default=None,
                       help="Path to source prompt policy")
    
    # Evaluation
    parser.add_argument("--episodes", type=int, default=50,
                       help="Number of evaluation episodes per experiment")
    
    # LLM configuration
    parser.add_argument("--api", action="store_true",
                       help="Use OpenRouter API")
    parser.add_argument("--api-model", type=str, default=None,
                       help="OpenRouter model ID")
    parser.add_argument("--hf-model", type=str, default=None,
                       help="HuggingFace model name")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of parallel workers (API mode only)")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="transfer_results",
                       help="Directory to save results")
    parser.add_argument("--config", type=str, default="hierarchical",
                       help="Config to use")
    parser.add_argument("--skip-similarity", action="store_true",
                       help="Skip dataset similarity computation (faster)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate arguments
    if not args.all and (args.source is None or args.target is None):
        print("Error: Must specify --all or both --source and --target")
        return
    
    if not args.all and (args.source_structure is None or args.source_prompt is None):
        print("Error: For single experiment, must specify --source-structure and --source-prompt")
        return
    
    # Validate LLM configuration
    if args.api:
        if not args.api_model:
            print("Error: --api requires --api-model")
            return
    else:
        if not args.hf_model:
            print("Warning: Neither --api nor --hf-model specified. Will use config default.")
    
    # Load config
    cfg = load_config(args.config)
    
    if args.all:
        # Run all experiments
        results, output_file = run_all_experiments(
            cfg, num_episodes=args.episodes, use_api=args.api,
            api_model=args.api_model, hf_model=args.hf_model,
            num_workers=args.workers, output_dir=args.output_dir
        )
    else:
        # Run single experiment
        results = run_single_transfer_experiment(
            args.source, args.target, args.source_structure, args.source_prompt,
            cfg, num_episodes=args.episodes, use_api=args.api,
            api_model=args.api_model, hf_model=args.hf_model,
            num_workers=args.workers, compute_similarity=not args.skip_similarity
        )
        
        # Save single result
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            args.output_dir,
            f"transfer_{args.source}_to_{args.target}_{timestamp}.json"
        )
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

