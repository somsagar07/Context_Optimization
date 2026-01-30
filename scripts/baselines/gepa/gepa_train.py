#!/usr/bin/env python3
"""
DSPy training script for optimizing question-answering models using MIPROv2.
Supports multiple datasets: gsm8k, hotpotqa, gaia, medqa, aime25, mmlu, etc.
Trains a model and saves the optimized version to a JSON file.
"""
from __future__ import annotations
import os
import sys
import random
import argparse
import shutil
import atexit
from pathlib import Path
from typing import List, Tuple

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

# Set up custom cache directory in the same folder as this script
SCRIPT_DIR = Path(__file__).parent.resolve()
CACHE_DIR = SCRIPT_DIR / ".dspy_cache"

# Clear cache BEFORE importing dspy
def setup_cache(clear_existing=True):
    """Set up and optionally clear the cache directory."""
    if clear_existing and CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return str(CACHE_DIR)

def cleanup_cache():
    """Clean up the cache directory."""
    if CACHE_DIR.exists():
        try:
            shutil.rmtree(CACHE_DIR)
            print(f"Cleaned up cache directory: {CACHE_DIR}")
        except Exception as e:
            print(f"Warning: Could not clean up cache directory: {e}")

# Set up cache
cache_path = setup_cache(clear_existing=True)
# Register cleanup on exit
atexit.register(cleanup_cache)

# Set cache environment variable before importing dspy
# DSPy uses DSP_CACHEDIR (not DSPY_CACHE_DIR)
os.environ["DSP_CACHEDIR"] = cache_path

import dspy
import json
from datetime import datetime
from tqdm import tqdm
from utils.get_dataset import get_dataset_loader, validate_dataset_name, get_dataset_help_text

# Also configure cache programmatically after import (more reliable)
dspy.configure_cache(disk_cache_dir=cache_path)

# Generic signature for question-answering tasks
class AnswerQuestion(dspy.Signature):
    """Answer the given question accurately and completely."""
    question: str = dspy.InputField()
    reasoning: str = dspy.OutputField(desc="step-by-step reasoning to arrive at the answer")
    answer: str = dspy.OutputField(desc="final answer")


# Generic Chain-of-Thought based solver module
class QuestionSolver(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cot = dspy.ChainOfThought(AnswerQuestion)

    def forward(self, question: str) -> dspy.Prediction:
        """Return a DSPy prediction with fields: reasoning, answer."""
        return self.cot(question=question)


def sample_from_dataset(dataset, n_samples: int, seed: int = 42):
    """
    Sample n_samples from a dataset using dataset's built-in methods.
    Uses deterministic indexing to ensure reproducibility.
    
    Args:
        dataset: Dataset object with get_sample() and evaluate_correctness() methods
        n_samples: Number of samples to get
        seed: Random seed for reproducibility
        
    Returns:
        List of (question, answer) tuples, formatted by dataset's get_sample() logic
    """
    # Get dataset size
    if hasattr(dataset, 'tasks'):
        total_size = len(dataset.tasks)
        data = dataset.tasks
    elif hasattr(dataset, 'data'):
        total_size = len(dataset.data)
        data = dataset.data
    else:
        raise ValueError(f"Dataset {type(dataset)} doesn't have .data or .tasks attribute")
    
    n_samples = min(n_samples, total_size)
    
    # Generate random indices for deterministic sampling
    rnd = random.Random(seed)
    indices = rnd.sample(range(total_size), k=n_samples)
    
    samples = []
    dataset_name = getattr(dataset, 'name', '').lower()
    
    for idx in indices:
        # Use dataset's built-in methods where available
        if dataset_name.startswith('mmlu') and hasattr(dataset, "get_question") and hasattr(dataset, "get_answer"):
            # MMLU has helper methods for indexed access
            question = dataset.get_question(idx)
            answer = dataset.get_answer(idx)
        else:
            # For other datasets, replicate get_sample() logic but with deterministic index
            sample = data[idx]
            
            if dataset_name == 'gaia':
                question = sample.get('Question', '')
                answer = sample.get('Final answer', '')
                # Handle file attachments (same as get_sample() does)
                rel_path = sample.get('file_path', '')
                if rel_path and hasattr(dataset, 'data_dir'):
                    import os
                    full_path = os.path.join(dataset.data_dir, rel_path)
                    question += f"\n\n[System Notification]\nFile Attachment: {full_path}\nYou can use your tools to read or process this file."
            elif dataset_name == 'medqa':
                # MedQA get_sample() formats question with options
                data_dict = sample.get('data', {})
                question = data_dict.get('Question', '')
                options = data_dict.get('Options', {})
                if options:
                    options_text = "\n".join([f"{k}: {v}" for k, v in sorted(options.items())])
                    question = f"{question}\n\nOptions:\n{options_text}"
                answer = data_dict.get('Correct Answer', '')
            elif dataset_name == 'aime25':
                question = sample.get('problem', '')
                answer = sample.get('answer', '')
            elif dataset_name == 'drop':
                # DROP get_sample() formats: passage + question, extracts answer from answers_spans
                passage = sample.get('passage', '')
                question_text = sample.get('question', '')
                # Format: passage + question (same as get_sample() does)
                question = f"{passage}\n\nQuestion: {question_text}"
                # Extract answer - DROP has 'answers_spans' with 'spans' (list of answer strings)
                answer_spans = sample.get('answers_spans', {})
                if 'spans' in answer_spans and len(answer_spans['spans']) > 0:
                    # Use first answer (usually there's one primary answer)
                    answer = answer_spans['spans'][0]
                else:
                    # Fallback: try to find answer in other fields
                    answer = sample.get('answer', '')
            else:
                # Standard datasets (GSM8K, HotPotQA, etc.) - use get_sample() format
                question = sample.get('question', sample.get('problem', ''))
                answer = sample.get('answer', sample.get('Answer', sample.get('Final answer', '')))
        
        samples.append((question, answer))
    
    return samples


def create_metric_fn(dataset):
    """
    Create a metric function that uses the dataset's evaluate_correctness method.
    
    Args:
        dataset: Dataset object with evaluate_correctness method
        
    Returns:
        Metric function for DSPy evaluation
    """
    def metric_fn(ex, pred, *args, **kwargs):
        """
        Metric function for DSPy evaluation.
        ex: dspy.Example with .answer attribute (ground truth)
        pred: dspy.Prediction with .answer attribute (prediction)
        """
        ground_truth = str(ex.answer)
        prediction = str(pred.answer)
        return dataset.evaluate_correctness(prediction, ground_truth)
    
    return metric_fn


def main():
    parser = argparse.ArgumentParser(
        description="Train DSPy model using MIPROv2 optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=get_dataset_help_text()
    )
    parser.add_argument(
        "--dataset",
        type=validate_dataset_name,
        default="gsm8k",
        help="Dataset name. " + get_dataset_help_text()
    )
    parser.add_argument(
        "--task-model",
        type=str,
        default=None,
        help="Task model identifier. For HuggingFace: model path (e.g., 'Qwen/Qwen2.5-7B-Instruct'). For API: model name (e.g., 'openai/gpt-4o', 'ollama_chat/qwen3:1.7b')"
    )
    parser.add_argument(
        "--task-api-base",
        type=str,
        default=None,
        help="API base URL for task model (e.g., 'http://localhost:11434' for Ollama, 'https://api.openai.com/v1' for OpenAI). Required for API models."
    )
    parser.add_argument(
        "--task-api-key",
        type=str,
        default=None,
        help="API key for task model (default: from OPENAI_API_KEY or OPENROUTER_API_KEY env vars)"
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        default=False,
        help="Use API model instead of HuggingFace model"
    )
    parser.add_argument(
        "--prompt-model",
        type=str,
        default=os.getenv("OPENAI_MODEL", "openai/gpt-5-nano"),
        help="Prompt model for optimization (default: OPENAI_MODEL from .env or openai/gpt-5-nano)"
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key (default: from OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=80,
        help="Number of training examples (default: 80)"
    )
    parser.add_argument(
        "--n-dev",
        type=int,
        default=80,
        help="Number of dev examples (default: 80)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset sampling (default: 42)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens for generation (default: 512)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperature for task model (default: 0.2)"
    )
    parser.add_argument(
        "--auto",
        type=str,
        default="heavy",
        choices=["light", "medium", "heavy"],
        help="GEPA auto mode (default: heavy)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base directory to save optimized model (default: dspy folder/{model_name})"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output filename (default: auto-generated as {dataset}_{model}_{mode}.json)"
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        default=False,
        help="Keep cache directory after training (default: False, cache is cleaned up)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to save training logs as JSON. If not provided, auto-generates filename in training_logs/"
    )

    args = parser.parse_args()
    
    # Update cleanup behavior based on argument
    if args.keep_cache:
        atexit.unregister(cleanup_cache)
        print(f"Cache will be kept at: {CACHE_DIR}")

    # Determine task model (default based on API flag)
    if args.task_model is None:
        if args.use_api:
            args.task_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
        else:
            from configs.base import LLM_MODEL_NAME
            args.task_model = LLM_MODEL_NAME

    # Set up output directory structure: gen_prompts/{model_name}/{dataset_name}/
    if args.output_dir is None:
        # Default: gen_prompts/{model_name}/{dataset_name}/
        model_name_safe = args.task_model.replace("/", "_").replace(":", "_").replace(".", "_")
        dataset_name_safe = args.dataset.replace("/", "_").replace(":", "_")
        output_dir = Path(__file__).parent / "gen_prompts" / model_name_safe / dataset_name_safe
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    if args.output_name is None:
        model_name_safe = args.task_model.replace("/", "_").replace(":", "_").replace(".", "_")
        dataset_name_safe = args.dataset.replace("/", "_").replace(":", "_")
        output_name = f"{dataset_name_safe}_{model_name_safe}_{args.auto}.json"
    else:
        output_name = args.output_name
        if not output_name.endswith(".json"):
            output_name += ".json"

    output_path = output_dir / output_name

    print(f"Training configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Task model: {args.task_model}")
    print(f"  Prompt model: {args.prompt_model}")
    print(f"  Training examples: {args.n_train}")
    print(f"  Dev examples: {args.n_dev}")
    print(f"  GEPA mode: {args.auto}")
    print(f"  Output path: {output_path}")
    print(f"  Cache directory: {CACHE_DIR}")
    print()

    # Set up generation kwargs
    GEN_KW = dict(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=0.95,
        do_sample=True
    )

    # Task model - support both HuggingFace and API models
    # Auto-detect API models: check for API keys or explicit --use-api flag
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    
    # Determine if this is an API model
    is_api_model = args.use_api
    if not is_api_model:
        # Auto-detect: if OpenRouter API key exists and model doesn't have a provider prefix, assume OpenRouter
        if openrouter_api_key and not any(args.task_model.startswith(prefix) for prefix in ["openai/", "openrouter/", "anthropic/", "ollama", "huggingface/"]):
            is_api_model = True
            print(f"  Auto-detected OpenRouter model (OPENROUTER_API_KEY found)")
    
    if is_api_model:
        # API model (OpenRouter, OpenAI, etc.)
        api_key = args.task_api_key or openrouter_api_key or openai_api_key
        model_name = args.task_model
        
        # Add openrouter/ prefix if OpenRouter API key is present and no prefix exists
        if openrouter_api_key and not any(model_name.startswith(prefix) for prefix in ["openai/", "openrouter/", "anthropic/", "ollama"]):
            model_name = f"openrouter/{model_name}"
            print(f"  Using OpenRouter model: {model_name}")
        
        # Set API base for OpenRouter if not specified
        if not args.task_api_base and model_name.startswith("openrouter/"):
            api_base = "https://openrouter.ai/api/v1"
        else:
            api_base = args.task_api_base
        
        if not api_key and not model_name.startswith("ollama"):
            raise ValueError(f"API key required for API model {model_name}. Set OPENAI_API_KEY or OPENROUTER_API_KEY env var, or use --task-api-key")
        
        task_lm = dspy.LM(
            model_name,
            api_base=api_base,
            api_key=api_key if api_key else "",
            max_tokens=GEN_KW["max_new_tokens"],
            temperature=GEN_KW["temperature"]
        )
    else:
        # HuggingFace model - DSPy requires "huggingface/" prefix
        hf_model_name = args.task_model
        if not hf_model_name.startswith("huggingface/"):
            hf_model_name = f"huggingface/{hf_model_name}"
        
        task_lm = dspy.LM(
            hf_model_name,
            max_tokens=GEN_KW["max_new_tokens"],
            temperature=GEN_KW["temperature"]
        )

    # Prompt model - OpenAI model for generating optimized instructions
    openai_api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(
            "OpenAI API key required. Set OPENAI_API_KEY env var or use --openai-api-key"
        )

    prompt_lm = dspy.LM(
        args.prompt_model,
        api_key=openai_api_key,
        max_tokens=16000,
        temperature=1.0
    )

    # Use task model by default
    dspy.configure(lm=task_lm)

    # Load datasets - IMPORTANT: use train split for optimization, test split for validation
    print(f"Loading {args.dataset} dataset...")
    print("  Using TRAIN split for optimization (not touching test set)")
    train_dataset = get_dataset_loader(args.dataset, is_eval=False)  # Train split
    
    print("  Using TEST split for validation during training")
    dev_dataset = get_dataset_loader(args.dataset, is_eval=True)  # Test split (for validation)
    
    print(f"Sampling {args.n_train} training examples from TRAIN split...")
    train_samples = sample_from_dataset(train_dataset, args.n_train, seed=args.seed)
    
    print(f"Sampling {args.n_dev} validation examples from TEST split...")
    dev_samples = sample_from_dataset(dev_dataset, args.n_dev, seed=args.seed + 1)

    # Create DSPy examples
    trainset = [
        dspy.Example(question=q, answer=a).with_inputs("question")
        for q, a in train_samples
    ]

    devset = [
        dspy.Example(question=q, answer=a).with_inputs("question")
        for q, a in dev_samples
    ]

    print(f"Loaded {len(trainset)} training examples and {len(devset)} dev examples")
    print()

    # Create metric function using dataset's evaluate_correctness
    metric_fn = create_metric_fn(dev_dataset)

    # Initialize solver with dataset-specific signature
    solver = QuestionSolver()

    # Evaluate baseline and collect detailed logs
    print("Evaluating baseline model...")
    print(f"  Evaluating {len(devset)} examples...")
    
    # Custom evaluation to collect detailed logs
    baseline_logs = []
    baseline_correct = []
    
    from tqdm import tqdm
    for i, example in enumerate(tqdm(devset, desc="Baseline evaluation")):
        try:
            pred = solver(question=example.question)
            correct = metric_fn(example, pred)
            # Ensure correct is a float
            correct_float = float(correct) if not isinstance(correct, bool) else (1.0 if correct else 0.0)
            baseline_correct.append(correct_float)
            
            baseline_logs.append({
                "episode": i + 1,
                "question": example.question,
                "ground_truth": example.answer,
                "prediction": pred.answer if hasattr(pred, 'answer') else str(pred),
                "reasoning": pred.reasoning if hasattr(pred, 'reasoning') else "",
                "correct": bool(correct_float > 0.5),
            })
        except Exception as e:
            print(f"Error evaluating example {i+1}: {e}")
            baseline_correct.append(0.0)
            baseline_logs.append({
                "episode": i + 1,
                "question": example.question,
                "ground_truth": example.answer,
                "prediction": "",
                "reasoning": "",
                "correct": False,
                "error": str(e)
            })
    
    baseline_acc = sum(baseline_correct) / len(baseline_correct) if baseline_correct else 0.0
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    print()

    from dspy.teleprompt import GEPA
    
    # Optimize with GEPA
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_dir = Path(__file__).parent / "gepa_logs"
    unique_log_dir = parent_dir / args.dataset / args.task_model.replace('/', '_') / timestamp
    unique_log_dir.mkdir(parents=True, exist_ok=True)
    
    print("Optimizing with GEPA...")
    print(f"  > Log Directory: {unique_log_dir} (Unique per run)")
    print()
    
    optimizer = GEPA(
        metric=metric_fn,
        auto=args.auto,                   # "light", "medium", or "heavy" optimization mode
        reflection_lm=prompt_lm,        # LLM for reflecting and proposing new instructions (GPT-5-nano)
        reflection_minibatch_size=3,    # Number of examples to reflect on per iteration
        num_threads=1,                  # Number of threads for parallel evaluation
        track_stats=True,               # Track optimization statistics
        skip_perfect_score=True,        # Skip examples with perfect scores
        log_dir=unique_log_dir,          # Directory to save optimization logs
    )

    optimized_solver = optimizer.compile(
        student=solver,
        trainset=trainset,
        valset=devset,
    )

    # Evaluate optimized model and collect detailed logs
    print("Evaluating optimized model...")
    print(f"  Evaluating {len(devset)} examples...")
    
    # Custom evaluation to collect detailed logs
    optimized_logs = []
    optimized_correct = []
    
    for i, example in enumerate(tqdm(devset, desc="Optimized evaluation")):
        try:
            pred = optimized_solver(question=example.question)
            correct = metric_fn(example, pred)
            # Ensure correct is a float
            correct_float = float(correct) if not isinstance(correct, bool) else (1.0 if correct else 0.0)
            optimized_correct.append(correct_float)
            
            optimized_logs.append({
                "episode": i + 1,
                "question": example.question,
                "ground_truth": example.answer,
                "prediction": pred.answer if hasattr(pred, 'answer') else str(pred),
                "reasoning": pred.reasoning if hasattr(pred, 'reasoning') else "",
                "correct": bool(correct_float > 0.5),
            })
        except Exception as e:
            print(f"Error evaluating example {i+1}: {e}")
            optimized_correct.append(0.0)
            optimized_logs.append({
                "episode": i + 1,
                "question": example.question,
                "ground_truth": example.answer,
                "prediction": "",
                "reasoning": "",
                "correct": False,
                "error": str(e)
            })
    
    optimized_acc = sum(optimized_correct) / len(optimized_correct) if optimized_correct else 0.0
    print(f"Post-optimization accuracy: {optimized_acc:.4f}")
    print()

    # Save optimized model
    print(f"Saving optimized model to {output_path}...")
    optimized_solver.save(str(output_path))
    print(f"Model saved successfully!")
    print(f"  Baseline accuracy: {baseline_acc:.4f}")
    print(f"  Optimized accuracy: {optimized_acc:.4f}")
    print(f"  Improvement: {optimized_acc - baseline_acc:+.4f}")
    
    # Save training logs in the same folder as the optimized model
    log_file = args.log_file
    if log_file is None:
        # Save in the same directory as the optimized model: gen_prompts/{model_name}/{dataset_name}/train_{timestamp}.json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = output_dir / f"train_{timestamp}.json"
    else:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Build complete log with metadata
    log_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "method": "dspy_gepa",
            "dataset": args.dataset,
            "task_model": args.task_model,
            "prompt_model": args.prompt_model,
            "n_train": args.n_train,
            "n_dev": args.n_dev,
            "gepa_mode": args.auto,
            "seed": args.seed,
            "model_path": str(output_path),
            "use_api": args.use_api
        },
        "summary": {
            "baseline_accuracy": float(baseline_acc),
            "optimized_accuracy": float(optimized_acc),
            "improvement": float(optimized_acc - baseline_acc)
        },
        "baseline_evaluation": {
            "accuracy": float(baseline_acc),
            "episodes": baseline_logs
        },
        "optimized_evaluation": {
            "accuracy": float(optimized_acc),
            "episodes": optimized_logs
        }
    }
    
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\n  Training logs saved to: {log_file}")


if __name__ == "__main__":
    main()
