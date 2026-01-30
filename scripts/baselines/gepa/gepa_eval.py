#!/usr/bin/env python3
"""
GEPA prompt evaluation script for testing optimized prompts.
Loads a saved GEPA prompt JSON file, extracts instructions, and evaluates using base model.
Supports multiple datasets: gsm8k, hotpotqa, gaia, medqa, aime25, mmlu, etc.
"""
from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
import numpy as np

# Load .env file FIRST (before any imports that might need env vars)
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    project_root = Path(__file__).parent.parent.parent.parent
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()  # Try current directory
except ImportError:
    # python-dotenv not installed, skip
    pass
except Exception as e:
    # Could not load .env file, continue anyway
    pass

# Add parent directories to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from utils.get_dataset import get_dataset_loader, validate_dataset_name, get_dataset_help_text
from agents_system import LLMWorker, OpenRouterWorker


def load_prompt_instructions(prompt_path: Path) -> str:
    """
    Load saved prompt JSON and extract signature.instructions.
    
    Args:
        prompt_path: Path to the saved prompt JSON file
        
    Returns:
        The instruction string from signature.instructions
    """
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    with open(prompt_path, 'r') as f:
        prompt_data = json.load(f)
    
    # Navigate to signature.instructions
    # Structure: {"cot.predict": {"signature": {"instructions": "..."}}}
    if "cot.predict" in prompt_data:
        signature = prompt_data["cot.predict"].get("signature", {})
        instructions = signature.get("instructions", "")
        if instructions:
            return instructions
    
    # Try alternative structure
    if "signature" in prompt_data:
        instructions = prompt_data["signature"].get("instructions", "")
        if instructions:
            return instructions
    
    raise ValueError(f"Could not find signature.instructions in prompt file: {prompt_path}")


def get_sample_from_dataset(dataset, sample_idx: int):
    """
    Get a sample from dataset by index.
    
    Args:
        dataset: Dataset object
        sample_idx: Index of the sample to retrieve
        
    Returns:
        Tuple of (question, answer)
    """
    dataset_name = getattr(dataset, 'name', '').lower()
    
    if hasattr(dataset, 'data'):
        # Standard datasets use .data
        sample = dataset.data[sample_idx]
        
        if dataset_name == 'gaia':
            # GAIA: 'Question', 'Final answer', optional file attachment
            question = sample['Question']
            answer = sample['Final answer']
            # Handle file attachments
            rel_path = sample.get('file_path', '')
            if rel_path and hasattr(dataset, 'data_dir'):
                full_path = os.path.join(dataset.data_dir, rel_path)
                question += f"\n\n[System Notification]\nFile Attachment: {full_path}\nYou can use your tools to read or process this file."
        elif dataset_name == 'medqa':
            # MedQA: nested 'data' dict with 'Question', 'Options', 'Correct Answer'
            data = sample['data']
            question = data['Question']
            options = data['Options']  # Dict: {'A': '...', 'B': '...', etc.}
            options_text = "\n".join([f"{k}: {v}" for k, v in sorted(options.items())])
            question = f"{question}\n\nOptions:\n{options_text}"
            answer = data['Correct Answer']
        elif dataset_name == 'aime25':
            # AIME25: 'problem', 'answer'
            question = sample.get('problem', '')
            answer = sample.get('answer', '')
        elif dataset_name.startswith('mmlu'):
            # MMLU: format options and map answer index to text
            if hasattr(dataset, "get_question") and hasattr(dataset, "get_answer"):
                question = dataset.get_question(sample_idx)
                answer = dataset.get_answer(sample_idx)
            else:
                question_text = sample.get('question', '')
                choices = sample.get('choices', [])
                if choices:
                    options_text = "\n".join([f"{chr(65+i)}: {choice}" for i, choice in enumerate(choices)])
                    question = f"{question_text}\n\nOptions:\n{options_text}"
                    answer_idx = sample.get('answer', None)
                    if isinstance(answer_idx, int) and 0 <= answer_idx < len(choices):
                        answer = choices[answer_idx]
                    else:
                        answer = sample.get('answer', '')
                else:
                    question = question_text
                    answer = sample.get('answer', '')
        elif dataset_name == 'drop':
            # DROP: passage + question, extracts answer from answers_spans
            passage = sample.get('passage', '')
            question_text = sample.get('question', '')
            question = f"{passage}\n\nQuestion: {question_text}"
            answer_spans = sample.get('answers_spans', {})
            if 'spans' in answer_spans and len(answer_spans['spans']) > 0:
                answer = answer_spans['spans'][0]
            else:
                answer = sample.get('answer', '')
        else:
            # Standard datasets (GSM8K, HotPotQA, etc.)
            question = sample.get('question', sample.get('problem', ''))
            answer = sample.get('answer', sample.get('Answer', sample.get('Final answer', '')))
    else:
        # Fallback: try to index directly
        sample = dataset[sample_idx]
        question = sample.get('question', sample.get('problem', ''))
        answer = sample.get('answer', '')
    
    return question, answer


def run_single_episode(ep, worker, dataset, instruction: str, sample_idx: int, max_tokens: int = 1024):
    """
    Run a single evaluation episode.
    
    Args:
        ep: Episode number (for logging)
        worker: LLMWorker or OpenRouterWorker instance
        dataset: Dataset instance
        instruction: Instruction prompt to prepend
        sample_idx: Dataset index to evaluate
        max_tokens: Maximum tokens for generation
        
    Returns:
        Dict with correct, tokens, and episode_log
    """
    # Get question and answer
    question, answer = get_sample_from_dataset(dataset, sample_idx)
    
    # Prepend instruction to question
    full_prompt = f"{instruction}\n\n{question}"
    
    # Debug: Print first episode to verify instruction is being added
    if ep == 0:
        print(f"\n[DEBUG] First episode prompt preview:")
        print(f"  Instruction length: {len(instruction)} characters")
        print(f"  Instruction preview: {instruction[:200]}...")
        print(f"  Question preview: {question[:200]}...")
        print(f"  Full prompt length: {len(full_prompt)} characters")
        print(f"  Full prompt preview:\n{full_prompt[:500]}...\n")
    
    # Generate answer using the model (no tools for prompt optimization)
    # This is prompt optimization, so we just use the base model with the instruction
    response = worker._generate(
        prompt=full_prompt,
        active_tools=["calculator", "web_search", "python", "ocr_reader"],
        max_tokens=max_tokens,
        prompt_suffix=None
    )
    
    prediction = response
    
    # Evaluate correctness
    correct = dataset.evaluate_correctness(prediction, answer)
    
    # Approximate token count
    total_tokens = int(len(prediction.split()) * 1.3)
    
    # Build episode log
    episode_log = {
        "episode": ep + 1,
        "correct": bool(correct),
        "tokens": int(total_tokens),
        "question": question,
        "prediction": prediction,
        "ground_truth": answer
    }
    
    # Add full prompt for first episode for debugging
    if ep == 0:
        episode_log["full_prompt"] = full_prompt
        episode_log["instruction"] = instruction
    
    return {
        "correct": bool(correct),
        "tokens": int(total_tokens),
        "episode_log": episode_log
    }


def evaluate_parallel(worker_class, dataset, instruction: str, num_episodes: int, 
                      use_api: bool = False, api_model: str = None, 
                      hf_model: str = None, num_workers: int = 4, max_tokens: int = 1024):
    """
    Parallel evaluation using ThreadPoolExecutor.
    
    Args:
        worker_class: Not used, kept for compatibility
        dataset: Dataset instance
        instruction: Instruction prompt to prepend
        num_episodes: Number of episodes to evaluate
        use_api: Whether to use API model
        api_model: API model name
        hf_model: HuggingFace model name
        num_workers: Number of parallel workers
        max_tokens: Maximum tokens for generation
        
    Returns:
        Dict with accuracy, avg_tokens, and episodes
    """
    dataset_size = len(dataset.data)
    
    if num_episodes > dataset_size:
        print(f"  Warning: Requested {num_episodes} episodes but only {dataset_size} samples. Evaluating all {dataset_size}.")
        num_episodes = dataset_size
    
    indices = list(range(num_episodes))
    print(f"  Will evaluate {num_episodes} unique datapoints")
    
    # Pre-create workers for each thread
    print(f"\nPre-creating {num_workers} workers...")
    workers = []
    for i in range(num_workers):
        print(f"  Creating worker {i+1}/{num_workers}...")
        if use_api:
            worker = OpenRouterWorker(model_name=api_model)
        else:
            worker = LLMWorker(model_name=hf_model)
        workers.append(worker)
    print(f"  Done! All {num_workers} workers ready.")
    
    # Lock for each worker to ensure thread-safe access
    worker_locks = [threading.Lock() for _ in range(num_workers)]
    
    def worker_fn(ep):
        """Worker function to run a single episode using a specific worker."""
        worker_idx = ep % num_workers
        worker = workers[worker_idx]
        sample_idx = indices[ep]
        
        # Lock this specific worker while using it
        with worker_locks[worker_idx]:
            return run_single_episode(
                ep, worker, dataset, instruction, sample_idx, max_tokens
            )
    
    results = {"correct": [], "tokens": []}
    episode_logs = []
    
    # Thread-safe counters for progress
    completed = [0]
    correct_count = [0]
    total_tokens = [0]
    results_lock = threading.Lock()
    
    print(f"Evaluating on {num_episodes} episodes with {num_workers} parallel workers...")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(worker_fn, ep): ep for ep in range(num_episodes)}
        
        # Process results as they complete with progress bar
        with tqdm(total=num_episodes, desc=f"Evaluating ({num_workers} workers)") as pbar:
            for future in as_completed(futures):
                ep = futures[future]
                try:
                    result = future.result()
                    
                    with results_lock:
                        results["correct"].append(result["correct"])
                        results["tokens"].append(result["tokens"])
                        episode_logs.append(result["episode_log"])
                        
                        completed[0] += 1
                        correct_count[0] += int(result["correct"])
                        total_tokens[0] += result["tokens"]
                        
                        # Update progress bar
                        acc = correct_count[0] / completed[0] * 100
                        avg_tokens = total_tokens[0] / completed[0]
                        pbar.set_postfix({"acc": f"{acc:.1f}%", "tokens": f"{avg_tokens:.0f}"})
                        pbar.update(1)
                        
                except Exception as e:
                    print(f"\nError in episode {ep}: {e}")
                    import traceback
                    traceback.print_exc()
                    pbar.update(1)
    
    # Sort episode logs by episode number
    episode_logs.sort(key=lambda x: x["episode"])
    
    # Summary
    accuracy = np.mean(results["correct"]) * 100
    avg_tokens = np.mean(results["tokens"])
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Accuracy:    {accuracy:.1f}%")
    print(f"  Avg Tokens:  {avg_tokens:.0f}")
    
    return {
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "episodes": episode_logs
    }


def evaluate(worker, dataset, instruction: str, num_episodes: int, max_tokens: int = 1024):
    """
    Sequential evaluation (for local HuggingFace models).
    
    Args:
        worker: LLMWorker or OpenRouterWorker instance
        dataset: Dataset instance
        instruction: Instruction prompt to prepend
        num_episodes: Number of episodes to evaluate
        max_tokens: Maximum tokens for generation
        
    Returns:
        Dict with accuracy, avg_tokens, and episodes
    """
    dataset_size = len(dataset.data)
    
    if num_episodes > dataset_size:
        print(f"  Warning: Requested {num_episodes} episodes but only {dataset_size} samples. Evaluating all {dataset_size}.")
        num_episodes = dataset_size
    
    results = {"correct": [], "tokens": []}
    episode_logs = []
    
    # Running stats for tqdm
    running_correct = 0
    running_tokens = 0
    
    print(f"\nEvaluating on {num_episodes} episodes...")
    
    pbar = tqdm(range(num_episodes), desc="Evaluating", leave=True)
    for ep in pbar:
        result = run_single_episode(
            ep, worker, dataset, instruction, sample_idx=ep, max_tokens=max_tokens
        )
        
        # Record results
        results["correct"].append(result["correct"])
        results["tokens"].append(result["tokens"])
        episode_logs.append(result["episode_log"])
        
        # Update running stats
        running_correct += int(result["correct"])
        running_tokens += result["tokens"]
        
        # Update tqdm with running accuracy and tokens
        pbar.set_postfix({
            "acc": f"{running_correct/(ep+1)*100:.1f}%",
            "tokens": f"{running_tokens/(ep+1):.0f}"
        })
    
    # Summary
    accuracy = np.mean(results["correct"]) * 100
    avg_tokens = np.mean(results["tokens"])
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Accuracy:    {accuracy:.1f}%")
    print(f"  Avg Tokens:  {avg_tokens:.0f}")
    
    return {
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "episodes": episode_logs
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved DSPy prompt using base model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=get_dataset_help_text()
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=None,
        help="Path to the saved prompt JSON file"
    )
    parser.add_argument(
        "--dataset",
        type=validate_dataset_name,
        default=None,
        help="Dataset name to evaluate on (default: inferred from prompt filename or 'gsm8k')"
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default=None,
        help="HuggingFace model name (e.g., 'Qwen/Qwen2.5-7B-Instruct'). Defaults to LLM_MODEL_NAME from config"
    )
    parser.add_argument(
        "--api",
        action="store_true",
        default=False,
        help="Use OpenRouter API instead of local HuggingFace models"
    )
    parser.add_argument(
        "--api-model",
        type=str,
        default=None,
        help="OpenRouter model ID (e.g., 'openai/gpt-4o', 'anthropic/claude-3.5-sonnet'). Defaults to OPENROUTER_MODEL env var"
    )
    parser.add_argument(
        "--n-eval",
        type=str,
        default="20",
        help="Number of evaluation examples (default: 20) or 'all' to evaluate on entire test set"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers for evaluation (default: 1, recommended: 4-8 for API)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max tokens for generation (default: 1024)"
    )
    parser.add_argument(
        "--use-test",
        action="store_true",
        default=True,
        help="Use test split for evaluation (default: True)"
    )
    parser.add_argument(
        "--use-train",
        action="store_true",
        default=False,
        help="Use train split for evaluation (overrides --use-test)"
    )

    args = parser.parse_args()
    
    # Validate prompt path
    if args.prompt_path is None:
        raise ValueError("--prompt_path is required")
    
    # Resolve path - try multiple locations
    prompt_path = Path(args.prompt_path)
    
    # If path starts with "gen_prompts/", it's likely relative to script directory
    script_dir = Path(__file__).parent
    if not prompt_path.is_absolute() and str(prompt_path).startswith("gen_prompts/"):
        # Try relative to script directory first
        potential_path = script_dir / prompt_path
        if potential_path.exists():
            prompt_path = potential_path
    
    # If not absolute and not found yet, try to find it in multiple locations
    if not prompt_path.is_absolute() or not prompt_path.exists():
        # List of potential base directories to try
        potential_bases = [
            Path.cwd(),  # Current working directory
            script_dir,  # Script directory
            script_dir.parent.parent.parent,  # Project root
        ]
        
        found = False
        if not prompt_path.exists():
            for base in potential_bases:
                potential_path = base / prompt_path
                if potential_path.exists():
                    prompt_path = potential_path
                    found = True
                    break
        
        if not found and prompt_path.exists():
            found = True
        
        if not found:
            # Last attempt: resolve to see the actual path we're looking for
            prompt_path = prompt_path.resolve()
    
    # Resolve to absolute path for clarity
    prompt_path = prompt_path.resolve()
    
    if not prompt_path.exists():
        # Provide helpful error message with suggestions
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent.parent
        example_path = script_dir / "gen_prompts" / "qwen_qwen-2_5-7b-instruct" / "gsm8k" / "gsm8k_qwen_qwen-2_5-7b-instruct_heavy.json"
        raise FileNotFoundError(
            f"Prompt file not found: {args.prompt_path}\n"
            f"Resolved to: {prompt_path}\n"
            f"Current working directory: {Path.cwd()}\n"
            f"Script directory: {script_dir}\n"
            f"Project root: {project_root}\n"
            f"Example path format: {example_path.relative_to(project_root) if example_path.exists() else 'N/A'}"
        )
    
    # Load instruction from prompt file
    print(f"Loading prompt instructions from {prompt_path}...")
    instruction = load_prompt_instructions(prompt_path)
    print(f"  Loaded instruction ({len(instruction)} characters)")
    print()
    
    # Infer dataset from prompt filename if not provided
    dataset_name = args.dataset
    if dataset_name is None:
        # Try to extract dataset name from prompt filename
        # Format: {dataset}_{model}_{mode}.json
        prompt_stem = prompt_path.stem
        parts = prompt_stem.split("_")
        if len(parts) >= 1:
            potential_dataset = parts[0]
            try:
                validate_dataset_name(potential_dataset)
                dataset_name = potential_dataset
                print(f"Inferred dataset from filename: {dataset_name}")
            except:
                dataset_name = "gsm8k"  # Default fallback
                print(f"Could not infer dataset from filename, using default: {dataset_name}")
        else:
            dataset_name = "gsm8k"
            print(f"Using default dataset: {dataset_name}")
    
    # Handle "all" for n_eval
    if args.n_eval.lower() == "all":
        n_eval = None  # Will evaluate on entire dataset
        print("Will evaluate on ENTIRE test set")
    else:
        n_eval = int(args.n_eval)
    
    # Determine model - try to infer from prompt filename if not provided
    if args.api:
        model_name = args.api_model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
        print(f"  API Mode:  {model_name}")
    else:
        from configs.base import LLM_MODEL_NAME
        model_name = args.hf_model or LLM_MODEL_NAME
        print(f"  HF Model:  {model_name}")
    
    # Try to infer model name from prompt filename if not explicitly provided
    # Format: {dataset}_{model}_{mode}.json
    if args.hf_model is None and not args.api:
        prompt_stem = prompt_path.stem
        parts = prompt_stem.split("_")
        if len(parts) >= 3:
            # Assume format: dataset_model_mode, extract model part
            potential_model = "_".join(parts[1:-1])  # Everything between dataset and mode
            # Convert back from safe format (replace _ with / for model names)
            # But keep it as is since we'll use it for directory matching
            inferred_model = potential_model.replace("_", "/")
            # Check if this looks like a valid model name
            if "/" in inferred_model or len(potential_model) > 5:
                model_name = inferred_model
                print(f"  Inferred model from filename: {model_name}")
    
    # Set up output directory structure: gen_prompts/{model_name}/{dataset_name}/
    # Same as train script
    model_name_safe = model_name.replace("/", "_").replace(":", "_").replace(".", "_")
    dataset_name_safe = dataset_name.replace("/", "_").replace(":", "_")
    output_dir = Path(__file__).parent / "gen_prompts" / model_name_safe / dataset_name_safe
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Evaluation configuration:")
    print(f"  Prompt path: {prompt_path}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Model: {model_name}")
    print(f"  Model type: {'API' if args.api else 'HuggingFace'}")
    print(f"  Evaluation examples: {args.n_eval}")
    print(f"  Split: {'test' if args.use_test and not args.use_train else 'train'}")
    print(f"  Workers: {args.num_workers}")
    print(f"  Output directory: {output_dir}")
    print()
    
    # Load evaluation dataset
    print(f"Loading {dataset_name} evaluation dataset...")
    eval_dataset = get_dataset_loader(
        dataset_name,
        is_eval=(args.use_test and not args.use_train)
    )
    
    # Get dataset size
    dataset_size = len(eval_dataset.data)
    
    # Determine number of samples
    if n_eval is None:
        n_eval = dataset_size
        print(f"Evaluating on ALL {n_eval} examples from test set")
    else:
        n_eval = min(n_eval, dataset_size)
        print(f"Evaluating on {n_eval} examples from test set")
    
    # Use parallel evaluation for API mode with multiple workers
    if args.api and args.num_workers > 1:
        results = evaluate_parallel(
            None,  # worker_class not needed
            eval_dataset,
            instruction,
            num_episodes=n_eval,
            use_api=args.api,
            api_model=args.api_model,
            hf_model=args.hf_model,
            num_workers=args.num_workers,
            max_tokens=args.max_tokens
        )
    else:
        # Sequential evaluation
        if args.api:
            worker = OpenRouterWorker(model_name=args.api_model)
        else:
            worker = LLMWorker(model_name=args.hf_model)
        
        results = evaluate(
            worker, eval_dataset, instruction,
            num_episodes=n_eval,
            max_tokens=args.max_tokens
        )
    
    # Save results in gen_prompts folder
    # Save in the same directory structure as train files: gen_prompts/{model_name}/{dataset_name}/eval_{timestamp}.json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_file = output_dir / f"eval_{timestamp}.json"
    
    # Build complete log with metadata
    log_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "method": "gepa_prompt_eval",
            "dataset": dataset_name,
            "num_episodes": n_eval,
            "model": model_name,
            "api_mode": args.api,
            "parallel_workers": args.num_workers if args.api else 1,
            "prompt_path": str(prompt_path),
            "instruction_length": len(instruction)
        },
        "summary": {
            "accuracy": results["accuracy"],
            "avg_tokens": results["avg_tokens"]
        },
        "episodes": results["episodes"]
    }
    
    with open(eval_file, "w") as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\n  Evaluation logs saved to: {eval_file}")


if __name__ == "__main__":
    main()