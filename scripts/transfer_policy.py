"""
Zero-Shot Transfer Learning Evaluation Script

Evaluates cross-domain transfer of RL policies without fine-tuning.
Tests how well a policy trained on one dataset performs on another dataset.

Usage:
    # Zero-shot transfer: GSM8K -> MedQA (using HuggingFace model)
    python scripts/transfer_policy.py \
        --source-dataset gsm8k \
        --target-dataset medqa \
        --source-structure models/ppo/gsm8k/structure_final.pt \
        --source-prompt models/ppo/gsm8k/prompt_final.pt \
        --hf-model Qwen/Qwen2.5-7B-Instruct \
        --episodes 50
    
    # Zero-shot transfer: GSM8K -> MedQA (using OpenRouter API)
    python scripts/transfer_policy.py \
        --source-dataset gsm8k \
        --target-dataset medqa \
        --source-structure models/ppo/gsm8k/structure_final.pt \
        --source-prompt models/ppo/gsm8k/prompt_final.pt \
        --api \
        --api-model openai/gpt-4o \
        --workers 4 \
        --episodes 50
    
    Note: The RL policies (structure and prompt) need an LLM to execute workflows.
    Specify either --api with --api-model (for OpenRouter API) or --hf-model (for local HuggingFace).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from datetime import datetime
import torch

from configs import load_config
from scripts.eval_hrl import (
    load_structure_policy, load_prompt_policy,
    evaluate, evaluate_parallel
)


def evaluate_zero_shot(source_structure_path, source_prompt_path, target_dataset, 
                       cfg, num_episodes=50, use_api=False, api_model=None, hf_model=None,
                       num_workers=1):
    """
    Evaluate pre-trained policy on target domain without fine-tuning.
    
    Returns:
        dict with accuracy, avg_reward, avg_tools, avg_tokens, workflow_distribution
    """
    print("\n" + "="*70)
    print("ZERO-SHOT TRANSFER EVALUATION")
    print("="*70)
    print(f"  Source models: {source_structure_path}, {source_prompt_path}")
    print(f"  Target domain: {target_dataset}")
    print(f"  Episodes: {num_episodes}")
    if use_api:
        print(f"  LLM: API mode - {api_model}")
        if num_workers > 1:
            print(f"  Workers: {num_workers} (parallel)")
    else:
        model_display = hf_model if hf_model else "default (from config)"
        print(f"  LLM: HuggingFace - {model_display}")
    print("="*70)
    
    # Update config for target dataset BEFORE loading policies
    # This ensures environments use the correct dataset
    cfg.DATASET_NAME = target_dataset
    
    # Load pre-trained policies (handle edge case where CUDA detected but no devices)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = "cuda"
    else:
        device = "cpu"
    print(f"\nLoading pre-trained policies (device: {device})...")
    structure_policy, struct_algo = load_structure_policy(source_structure_path, device)
    prompt_policy, prompt_algo = load_prompt_policy(source_prompt_path, device)
    
    # Verify algorithms match
    if struct_algo != prompt_algo:
        print(f"  Warning: Algorithm mismatch (structure: {struct_algo}, prompt: {prompt_algo})")
    
    print(f"  Loaded policies trained with {struct_algo} algorithm")
    
    # Evaluate on target domain
    if use_api and num_workers > 1:
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
            verbose=True,
            use_api=use_api,
            api_model=api_model,
            hf_model=hf_model,
            temperature=1.0
        )
    
    return results


def save_transfer_results(results, source_dataset, target_dataset, 
                         output_dir="transfer_results"):
    """Save zero-shot transfer experiment results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"transfer_{source_dataset}_to_{target_dataset}_zero-shot_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Prepare summary
    summary = {
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "mode": "zero-shot",
        "timestamp": datetime.now().isoformat(),
        "accuracy": results.get("accuracy", 0.0),
        "avg_reward": results.get("avg_reward", 0.0),
        "avg_tools": results.get("avg_tools", 0.0),
        "avg_tokens": results.get("avg_tokens", 0.0),
        "workflow_distribution": results.get("workflow_distribution", {}),
    }
    
    # Full results
    output = {
        "summary": summary,
        "detailed_results": results
    }
    
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")
    return filepath


def parse_args():
    from utils import validate_dataset_name, get_dataset_help_text
    
    parser = argparse.ArgumentParser(
        description="Zero-shot transfer learning: evaluate cross-domain transfer without fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Source and target domains
    parser.add_argument("--source-dataset", type=validate_dataset_name, required=True,
                       help="Source domain dataset. " + get_dataset_help_text(include_tau2=True))
    parser.add_argument("--target-dataset", type=validate_dataset_name, required=True,
                       help="Target domain dataset. " + get_dataset_help_text(include_tau2=True))
    
    # Pre-trained model paths
    parser.add_argument("--source-structure", type=str, required=True,
                       help="Path to pre-trained structure policy from source domain")
    parser.add_argument("--source-prompt", type=str, required=True,
                       help="Path to pre-trained prompt policy from source domain")
    
    # Evaluation
    parser.add_argument("--episodes", type=int, default=50,
                       help="Number of evaluation episodes")
    
    # LLM configuration (required - RL policies need LLM to execute workflows)
    parser.add_argument("--api", action="store_true", default=False,
                       help="Use OpenRouter API instead of local HuggingFace model")
    parser.add_argument("--api-model", type=str, default=None,
                       help="OpenRouter model ID (e.g., 'openai/gpt-4o', 'anthropic/claude-3.5-sonnet'). Required if --api is used.")
    parser.add_argument("--hf-model", type=str, default=None,
                       help="HuggingFace model name (e.g., 'Qwen/Qwen2.5-7B-Instruct'). Required if --api is not used. Defaults to config default if not specified.")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of parallel workers for API evaluation (only used with --api)")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="transfer_results",
                       help="Directory to save results")
    parser.add_argument("--config", type=str, default="hierarchical",
                       help="Config to use")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate model file paths exist
    if not os.path.exists(args.source_structure):
        raise FileNotFoundError(
            f"Source structure policy not found: {args.source_structure}\n"
            f"Please provide a valid path to the pre-trained structure policy."
        )
    if not os.path.exists(args.source_prompt):
        raise FileNotFoundError(
            f"Source prompt policy not found: {args.source_prompt}\n"
            f"Please provide a valid path to the pre-trained prompt policy."
        )
    
    # Validate LLM configuration
    if args.api:
        if not args.api_model:
            raise ValueError(
                "When using --api, you must specify --api-model (e.g., 'openai/gpt-4o'). "
                "The RL policies need an LLM to execute workflows."
            )
        print(f"Using OpenRouter API with model: {args.api_model}")
        if args.hf_model:
            print(f"Warning: --hf-model specified but --api is used. Ignoring --hf-model.")
    else:
        # If not using API, use HF model (from arg or config default)
        if args.hf_model:
            print(f"Using HuggingFace model (from argument): {args.hf_model}")
        else:
            # Warn that neither API nor HF model was explicitly specified
            print("Warning: Neither --api nor --hf-model was specified. Falling back to config default...")
            # Try to get from config
            try:
                from configs.base import LLM_MODEL_NAME
                args.hf_model = LLM_MODEL_NAME
                print(f"Using HuggingFace model (fallback to config default): {args.hf_model}")
            except Exception as e:
                print(f"Warning: Config default not found: {e}")
                print("LLMWorker will attempt to use its default model.")
                # Set to None so LLMWorker can use its default
                args.hf_model = None
    
    # Validate source and target are different
    if args.source_dataset == args.target_dataset:
        print(f"Warning: Source and target datasets are the same ({args.source_dataset}).")
        print("This will test same-domain performance rather than cross-domain transfer.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Load config
    cfg = load_config(args.config)
    
    # Ensure prompt atoms exist for target dataset
    from prompts import library
    print(f"Checking prompt atoms for target dataset: {args.target_dataset}...")
    atoms_path = library._get_atoms_path(args.target_dataset)
    
    if not os.path.exists(atoms_path):
        print(f"  Atoms not found. Initializing temporary worker to generate them...")
        from agents_system.worker import OpenRouterWorker, LLMWorker
        try:
            temp_worker = OpenRouterWorker(model_name="openai/gpt-4o")
            library.load_or_create_atoms(args.target_dataset, worker=temp_worker)
            print("  Generation complete.")
            del temp_worker
        except Exception as e:
            print(f"  Error using OpenRouter: {e}")
            print("  Falling back to local model...")
            import gc
            temp_worker = LLMWorker(model_name="Qwen/Qwen2.5-7B-Instruct")
            library.load_or_create_atoms(args.target_dataset, worker=temp_worker)
            print("  Generation complete.")
            del temp_worker
            gc.collect()
            torch.cuda.empty_cache()
    else:
        print(f"  Found existing atoms at {atoms_path}. Loading...")
        library.load_or_create_atoms(args.target_dataset, worker=None)
    
    print(f"  Active Atoms: {library.NUM_ATOMS}")
    
    # Run zero-shot evaluation
    results = evaluate_zero_shot(
        args.source_structure, args.source_prompt,
        args.target_dataset, cfg,
        num_episodes=args.episodes,
        use_api=args.api,
        api_model=args.api_model,
        hf_model=args.hf_model,
        num_workers=args.workers
    )
    
    # Save results
    save_transfer_results(results, args.source_dataset, 
                        args.target_dataset, args.output_dir)
    
    print("\n" + "="*70)
    print("ZERO-SHOT TRANSFER EVALUATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
