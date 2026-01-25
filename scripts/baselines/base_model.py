"""
Baseline Model Evaluation Script

Evaluates a base LLM (no special policies) on a dataset.
Similar to eval_hrl.py but without structure/prompt policies - just direct model inference.

Usage:
    # HuggingFace models (default)
    python scripts/baselines/base_model.py --dataset gsm8k
    python scripts/baselines/base_model.py --dataset gsm8k --episodes 50
    
    # Evaluate on all datapoints
    python scripts/baselines/base_model.py --dataset gsm8k --episodes all
    
    # API mode
    python scripts/baselines/base_model.py --dataset gsm8k --api --api-model openai/gpt-4o
    
    # API mode with parallel workers (faster evaluation)
    python scripts/baselines/base_model.py --dataset gsm8k --api --api-model openai/gpt-4o --workers 8
"""
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
import numpy as np
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

from configs import load_config
from agents_system import LLMWorker, OpenRouterWorker
from utils import get_dataset_loader, validate_dataset_name, get_dataset_help_text
from tools import ToolRegistry
import re


def run_single_episode(ep, worker, dataset, sample_idx=None):
    """Run a single evaluation episode. Used by both sequential and parallel evaluation.
    
    Args:
        ep: Episode number (for logging)
        worker: LLMWorker or OpenRouterWorker instance
        dataset: Dataset instance
        sample_idx: If provided, use this specific dataset index instead of random sampling.
                    This enables iterating through unique datapoints.
    """
    # Get question and answer
    if sample_idx is not None:
        # Directly access indexed sample instead of random sampling
        dataset_name = getattr(dataset, 'name', '').lower()
        
        if hasattr(dataset, 'tasks'):
            # Tau2 dataset uses .tasks
            sample = dataset.tasks[sample_idx]
            question = sample.get('question', '')
            answer = sample.get('answer', '')
        elif hasattr(dataset, 'data'):
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
            elif dataset_name == 'drop':
                # DROP has passage + question structure (same format as get_sample())
                passage = sample.get('passage', '')
                question_text = sample.get('question', '')
                # Format exactly like DROPDataset.get_sample() does
                question = f"{passage}\n\nQuestion: {question_text}"
                # Extract answer from answers_spans (same logic as get_sample())
                answer_spans = sample.get('answers_spans', {})
                if 'spans' in answer_spans and len(answer_spans['spans']) > 0:
                    # Use first answer (usually there's one primary answer)
                    answer = answer_spans['spans'][0]
                else:
                    # Fallback: try to find answer in other fields
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
            else:
                # GSM8K, HotPotQA, and other standard datasets: 'question', 'answer'
                question = sample.get('question', sample.get('problem', ''))
                answer = sample.get('answer', '')
        else:
            # Fallback: try to index directly
            sample = dataset[sample_idx]
            question = sample.get('question', sample.get('problem', ''))
            answer = sample.get('answer', '')
    else:
        # Original behavior - random sampling via get_sample()
        question, answer = dataset.get_sample()
    
    # Generate answer using the model with all tools enabled
    # Use a reasonable max_tokens for the answer
    max_tokens = 1024
    
    # All available tools
    all_tools = ["calculator", "web_search", "python", "ocr_reader"]
    
    # Initialize tool registry for executing tool calls
    tools = ToolRegistry()
    
    # Simple tool execution loop: generate -> parse tools -> execute -> feed back -> repeat
    max_iterations = 10  # Prevent infinite loops
    conversation_history = question
    full_response = ""
    total_tokens_used = 0
    
    for iteration in range(max_iterations):
        # Generate response with tools enabled
        response = worker._generate(
            prompt=conversation_history,
            active_tools=all_tools,  # Enable all tools
            max_tokens=max_tokens,
            prompt_suffix=None  # No special prompt modifications
        )
        
        full_response += response
        total_tokens_used += len(response.split()) * 1.3  # Approximate token count
        
        # Parse tool calls from response
        # Pattern: TOOL: <name> || QUERY: <query>
        tool_pattern = r"TOOL:\s*(\w+)\s*\|\|\s*QUERY:\s*(.*?)(?=\n\n|\nTOOL:|Final Answer:|$)"
        tool_matches = list(re.finditer(tool_pattern, response, re.DOTALL))
        
        if not tool_matches:
            # No more tool calls, check if we have a final answer
            if "Final Answer:" in response or iteration == max_iterations - 1:
                break
            # Continue to get final answer
            conversation_history += "\n\n" + response
            continue
        
        # Execute all tool calls found in this response
        tool_results = []
        for match in tool_matches:
            tool_name = match.group(1).strip().lower()
            tool_query = match.group(2).strip()
            
            if tool_name in all_tools:
                try:
                    tool_result = tools.execute(tool_name, tool_query)
                    tool_results.append(f"[System] Tool Output ({tool_name}): {tool_result}")
                except Exception as e:
                    tool_results.append(f"[System] Tool Error ({tool_name}): {str(e)}")
        
        # If we have tool results, append them to conversation and continue
        if tool_results:
            conversation_history += "\n\n" + response + "\n\n" + "\n".join(tool_results)
        else:
            # No valid tools executed, we're done
            break
    
    prediction = full_response
    
    # Evaluate correctness
    correct = dataset.evaluate_correctness(prediction, answer)
    
    # Use the token count from the loop (includes all iterations)
    total_tokens = int(total_tokens_used)
    
    # Build episode log
    episode_log = {
        "episode": ep + 1,
        "correct": bool(correct),
        "tokens": int(total_tokens),
        "question": question,
        "prediction": prediction,
        "ground_truth": answer
    }
    
    return {
        "correct": bool(correct),
        "tokens": int(total_tokens),
        "episode_log": episode_log
    }


def evaluate_parallel(worker_class, cfg, num_episodes=20, use_api=False, 
                      api_model=None, hf_model=None, num_workers=4):
    """Parallel evaluation using ThreadPoolExecutor for API mode.
    
    Iterates through unique datapoints (not random sampling) and splits work across workers.
    """
    # Load dataset ONCE and share across all workers
    print(f"\nLoading dataset once...")
    shared_dataset = get_dataset_loader(cfg.DATASET_NAME, is_eval=True)
    dataset_size = len(shared_dataset.tasks) if hasattr(shared_dataset, 'tasks') else len(shared_dataset.data)
    print(f"  Dataset loaded: {dataset_size} samples")
    
    # Create list of indices to evaluate (iterate through unique datapoints)
    if num_episodes > dataset_size:
        print(f"  Warning: Requested {num_episodes} episodes but only {dataset_size} samples. Evaluating all {dataset_size}.")
        num_episodes = dataset_size
    
    indices = list(range(num_episodes))  # [0, 1, 2, ..., num_episodes-1]
    print(f"  Will evaluate {num_episodes} unique datapoints")
    
    # Pre-create workers for each thread, sharing the dataset
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
        # Use modulo to assign episodes to workers round-robin
        worker_idx = ep % num_workers
        worker = workers[worker_idx]
        
        # Get the specific dataset index for this episode
        sample_idx = indices[ep]
        
        # Lock this specific worker while using it
        with worker_locks[worker_idx]:
            return run_single_episode(
                ep, worker, shared_dataset, sample_idx=sample_idx
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


def evaluate(worker, dataset, num_episodes=20):
    """Sequential evaluation (for local HuggingFace models).
    
    Iterates through unique datapoints (not random sampling).
    """
    # Get dataset size and cap num_episodes if needed
    dataset_size = len(dataset.tasks) if hasattr(dataset, 'tasks') else len(dataset.data)
    
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
            ep, worker, dataset, sample_idx=ep  # Iterate through datapoints sequentially
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate baseline model")
    parser.add_argument("--config", type=str, default="hierarchical")
    parser.add_argument("--episodes", type=str, default="20",
                       help="Number of episodes to evaluate, or 'all' for all datapoints")
    parser.add_argument("--dataset", type=validate_dataset_name, required=True, default=None,
                       help=get_dataset_help_text())
    
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log-file", type=str, default=None,
                       help="Path to save evaluation logs as JSON. If not provided, auto-generates filename in eval_logs/")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of parallel workers for API evaluation (default: 1, recommended: 4-8 for API)")
    
    # API configuration
    parser.add_argument("--api", action="store_true", default=False,
                       help="Use OpenRouter API instead of local HuggingFace models")
    parser.add_argument("--api-model", type=str, default=None,
                       help="OpenRouter model ID (e.g., 'openai/gpt-4o', 'anthropic/claude-3.5-sonnet'). Defaults to OPENROUTER_MODEL env var")
    parser.add_argument("--hf-model", type=str, default=None,
                       help="HuggingFace model name (e.g., 'Qwen/Qwen2.5-7B-Instruct'). Defaults to LLM_MODEL_NAME from config")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    cfg = load_config(args.config)
    if args.dataset:
        cfg.DATASET_NAME = args.dataset
    
    # Handle "all" episodes - evaluate on entire dataset
    if args.episodes.lower() == "all":
        dataset = get_dataset_loader(cfg.DATASET_NAME, is_eval=True)
        # Get dataset size - tau2 uses .tasks, others use .data
        if hasattr(dataset, 'tasks'):
            num_episodes = len(dataset.tasks)
        elif hasattr(dataset, 'data'):
            num_episodes = len(dataset.data)
        else:
            num_episodes = len(dataset)
        print(f"Evaluating on ALL {num_episodes} datapoints from {cfg.DATASET_NAME}")
    else:
        num_episodes = int(args.episodes)
    
    print("=" * 60)
    print("BASELINE MODEL EVALUATION")
    print("=" * 60)
    
    if args.api:
        model_name = args.api_model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
        print(f"  API Mode:  {model_name}")
        if args.workers > 1:
            print(f"  Workers:   {args.workers} (parallel)")
    else:
        from configs.base import LLM_MODEL_NAME
        model_name = args.hf_model or getattr(cfg, "LLM_MODEL_NAME", LLM_MODEL_NAME)
        print(f"  HF Model:  {model_name}")
        if args.workers > 1:
            print(f"  Warning: Parallel workers ignored for local HF models (GPU contention)")
            args.workers = 1
    print("=" * 60)
    
    # Use parallel evaluation for API mode with multiple workers
    if args.api and args.workers > 1:
        results = evaluate_parallel(
            None,  # worker_class not needed, we create workers inside
            cfg,
            num_episodes=num_episodes,
            use_api=args.api,
            api_model=args.api_model,
            hf_model=args.hf_model,
            num_workers=args.workers
        )
    else:
        # Sequential evaluation
        dataset = get_dataset_loader(cfg.DATASET_NAME, is_eval=True)
        
        if args.api:
            worker = OpenRouterWorker(model_name=args.api_model)
        else:
            worker = LLMWorker(model_name=args.hf_model)
        
        results = evaluate(
            worker, dataset,
            num_episodes=num_episodes
        )
    
    # Save logs
    log_file = args.log_file
    if log_file is None:
        # Auto-generate path: eval_logs/base_model/<dataset_name>/<model_name>/eval_<timestamp>.json
        # Extract just the model name (after '/') and replace '.' with '_'
        model_suffix = model_name.split("/")[-1].replace(".", "_").replace(":", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join("eval_logs", "base_model", cfg.DATASET_NAME, model_suffix)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"eval_{timestamp}.json")
    
    # Build complete log with metadata
    log_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "method": "base_model",
            "dataset": cfg.DATASET_NAME,
            "num_episodes": num_episodes,
            "model": model_name,
            "api_mode": args.api,
            "parallel_workers": args.workers if args.api else 1
        },
        "summary": {
            "accuracy": results["accuracy"],
            "avg_tokens": results["avg_tokens"]
        },
        "episodes": results["episodes"]
    }
    
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\n  Logs saved to: {log_file}")


if __name__ == "__main__":
    main()
