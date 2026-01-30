import sys
import os
import asyncio
import argparse
import json
import logging
import shutil
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

# Add project root to path (similar to base_model.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Load .env file FIRST (before any imports that might need env vars)
try:
    from dotenv import load_dotenv
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()  # Try current directory
except ImportError:
    pass  # python-dotenv not installed
except Exception:
    pass  # Could not load .env file

# AutoGen Imports
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_core import CancellationToken
from autogen_core.model_context import BufferedChatCompletionContext

# Local Imports
from configs import load_config
from utils import get_dataset_loader, validate_dataset_name, get_dataset_help_text

# Configure Logging to suppress excessive AutoGen debug output
logging.getLogger("autogen_core").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def extract_token_usage(message):
    """Extract token usage from an AutoGen message or TaskResult."""
    prompt_tokens = 0
    completion_tokens = 0
    
    # 1. Check for models_usage (Common in TaskResult - list of RequestUsage)
    models_usage = getattr(message, 'models_usage', None)
    if models_usage:
        if isinstance(models_usage, list):
            for item in models_usage:
                prompt_tokens += getattr(item, 'prompt_tokens', 0) or 0
                completion_tokens += getattr(item, 'completion_tokens', 0) or 0
        else:
            prompt_tokens += getattr(models_usage, 'prompt_tokens', 0) or 0
            completion_tokens += getattr(models_usage, 'completion_tokens', 0) or 0
    
    # 2. Check usage attribute directly (for streaming CreateResult)
    usage = getattr(message, 'usage', None)
    if usage and usage != models_usage:  # Avoid double counting
        prompt_tokens += getattr(usage, 'prompt_tokens', 0) or 0
        completion_tokens += getattr(usage, 'completion_tokens', 0) or 0
    
    # 3. Check metadata fallback
    if hasattr(message, 'metadata') and message.metadata:
        meta = message.metadata
        if isinstance(meta, dict) and 'usage' in meta:
            u = meta['usage']
            if isinstance(u, dict):
                prompt_tokens += u.get('prompt_tokens', 0) or 0
                completion_tokens += u.get('completion_tokens', 0) or 0
            else:
                prompt_tokens += getattr(u, 'prompt_tokens', 0) or 0
                completion_tokens += getattr(u, 'completion_tokens', 0) or 0
    
    return prompt_tokens, completion_tokens


def get_client_usage(model_client):
    """Get current total usage from model client (if available)."""
    try:
        if hasattr(model_client, 'total_usage'):
            usage = model_client.total_usage()
            if usage:
                return (
                    getattr(usage, 'prompt_tokens', 0) or 0,
                    getattr(usage, 'completion_tokens', 0) or 0
                )
    except Exception:
        pass
    return 0, 0


def format_message_for_log(message):
    """Format an AutoGen message for logging."""
    msg_dict = {
        "type": type(message).__name__,
    }
    
    # Extract source/sender
    if hasattr(message, 'source'):
        msg_dict["source"] = str(message.source)
    elif hasattr(message, 'sender'):
        msg_dict["source"] = str(message.sender)
    
    # Extract content
    if hasattr(message, 'content'):
        content = message.content
        # Handle different content types
        if isinstance(content, str):
            msg_dict["content"] = content
        elif isinstance(content, list):
            # Multi-modal content
            msg_dict["content"] = [str(c) for c in content]
        else:
            msg_dict["content"] = str(content)
    
    # Extract token usage
    prompt_tokens, completion_tokens = extract_token_usage(message)
    if prompt_tokens or completion_tokens:
        msg_dict["tokens"] = {
            "prompt": prompt_tokens,
            "completion": completion_tokens
        }
    
    return msg_dict

async def run_autogen_episode(ep, dataset, sample_idx, model_client, semaphore=None, max_turns=3, 
                               model_config=None, context_buffer_size=10):
    """
    Runs a single episode using the MagenticOne team on OpenRouter.
    
    Args:
        ep: Episode number
        dataset: Dataset instance
        sample_idx: Index of the sample in the dataset
        model_client: OpenAI client for the model (used for sequential execution)
        semaphore: Optional asyncio.Semaphore for concurrency control
        max_turns: Maximum number of turns for the team conversation
        model_config: Dict with model configuration for creating per-episode clients (for concurrent execution)
        context_buffer_size: Number of messages to keep in conversation history (default: 10)
    """
    # Use semaphore if provided (for concurrent execution)
    if semaphore:
        async with semaphore:
            return await _run_episode_impl(ep, dataset, sample_idx, model_client, max_turns, 
                                          model_config, context_buffer_size)
    else:
        return await _run_episode_impl(ep, dataset, sample_idx, model_client, max_turns,
                                      model_config, context_buffer_size)


async def _run_episode_impl(ep, dataset, sample_idx, model_client, max_turns=3,
                            model_config=None, context_buffer_size=10):
    """Internal implementation of episode execution.
    
    Args:
        model_config: If provided, creates a fresh model client for this episode to isolate token counting.
        context_buffer_size: Number of messages to keep in conversation history (limits token usage).
    """
    # Create per-episode model client if config provided (for accurate concurrent token counting)
    episode_client = model_client
    if model_config:
        episode_client = OpenAIChatCompletionClient(**model_config)
    
    # 1. Extract Question/Answer (Reusing logic from base_model.py)
    dataset_name = getattr(dataset, 'name', '').lower()
    
    if hasattr(dataset, 'tasks'):
        sample = dataset.tasks[sample_idx]
        question = sample.get('question', '')
        answer = sample.get('answer', '')
    elif hasattr(dataset, 'data'):
        sample = dataset.data[sample_idx]
        if dataset_name == 'gaia':
            question = sample['Question']
            answer = sample['Final answer']
            file_path = sample.get('file_path')
            if file_path:
                full_path = os.path.join(dataset.data_dir, file_path)
                question += f"\n\n[Attachment] File available at: {full_path}"
        elif dataset_name == 'medqa':
            data = sample['data']
            question = data['Question']
            options = "\n".join([f"{k}: {v}" for k, v in sorted(data['Options'].items())])
            question = f"{question}\n\nOptions:\n{options}"
            answer = data['Correct Answer']
        elif dataset_name == 'aime25':
            question = sample.get('problem', '')
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
        # Fallback
        sample = dataset[sample_idx]
        question = sample.get('question', '')
        answer = sample.get('answer', '')

    # 2. Initialize MagenticOne Components
    # We create fresh agents for each episode to avoid context pollution
    
    # Coder: Handles Python & Calculations
    # Use unique work_dir per episode to avoid conflicts in parallel execution
    work_dir = f"coding_workspace/ep_{ep}"
    os.makedirs(work_dir, exist_ok=True)
    
    # 3. Run the Team and collect conversation + tokens
    prediction = ""
    total_prompt_tokens = 0
    total_completion_tokens = 0
    conversation_log = []
    
    # Track usage via episode_client.total_usage() (isolated per episode when model_config is provided)
    before_prompt, before_completion = get_client_usage(episode_client)
    
    try:
        # Initialize agents inside try block
        code_executor = LocalCommandLineCodeExecutor(work_dir=work_dir)
        
        # Create a buffered context to limit conversation history and reduce token usage
        # This prevents unbounded context growth that causes token count spikes
        # Note: Only AssistantAgent-derived agents (like MagenticOneCoderAgent) support model_context
        buffered_context = BufferedChatCompletionContext(buffer_size=context_buffer_size)
        
        coder = MagenticOneCoderAgent(
            "Coder",
            model_client=episode_client,
            code_executor=code_executor,
            model_context=buffered_context,  # Limit history for coder agent
        )

        # WebSurfer: Handles Search, Browsing, and OCR (Multimodal)
        # Note: Requires 'playwright install'
        # MultimodalWebSurfer does NOT support model_context parameter
        # Disable screenshots and OCR to reduce token usage (~1100 tokens per screenshot)
        surfer = MultimodalWebSurfer(
            "WebSurfer",
            model_client=episode_client,
            to_save_screenshots=False,  # Reduce token overhead
        )

        # The Team - MagenticOneGroupChat does NOT support model_context
        team = MagenticOneGroupChat(
            participants=[coder, surfer],
            model_client=episode_client,
            max_turns=max_turns,
        )
        
#         # Wrap question with clear instructions for better results
#         task_prompt = f"""{question}

# Think step by step."""
        
        # Use team.run() instead of run_stream() to get TaskResult with token usage
        # TaskResult contains models_usage with aggregated token counts
        result = await team.run(task=question)
        
        # Log conversation from result.messages
        if hasattr(result, 'messages') and result.messages:
            for msg in result.messages:
                msg_log = format_message_for_log(msg)
                conversation_log.append(msg_log)
        
        # Method 1: Extract token usage from the TaskResult (preferred - per-call usage)
        total_prompt_tokens, total_completion_tokens = extract_token_usage(result)
        
        # Method 2: Fallback - use episode_client.total_usage() difference
        # This is accurate when model_config is provided (per-episode client isolation)
        if total_prompt_tokens == 0 and total_completion_tokens == 0:
            after_prompt, after_completion = get_client_usage(episode_client)
            total_prompt_tokens = after_prompt - before_prompt
            total_completion_tokens = after_completion - before_completion
        
        # Get the final answer from the last message
        if hasattr(result, 'messages') and result.messages:
            last_msg = result.messages[-1]
            if hasattr(last_msg, 'content') and last_msg.content:
                prediction = str(last_msg.content)
            else:
                prediction = "[Error: No content in final message]"
        else:
            prediction = "[Error: No messages in result]"

    except Exception as e:
        print(f"Error in Episode {ep}: {e}")
        prediction = f"[Error: {str(e)}]"
        conversation_log.append({
            "type": "Error",
            "content": str(e)
        })
    
    finally:
        # Cleanup: Remove the coding workspace to avoid redundant files
        try:
            if os.path.exists(work_dir):
                shutil.rmtree(work_dir)
        except Exception as cleanup_error:
            pass  # Ignore cleanup errors

    # 4. Evaluate
    correct = dataset.evaluate_correctness(prediction, answer)
    
    total_tokens = total_prompt_tokens + total_completion_tokens
    
    # Log Entry with full conversation
    return {
        "episode": ep + 1,
        "correct": bool(correct),
        "question": question,
        "prediction": prediction,
        "ground_truth": answer,
        "tokens": {
            "prompt": total_prompt_tokens,
            "completion": total_completion_tokens,
            "total": total_tokens
        },
        "conversation": conversation_log,
        "num_turns": len([m for m in conversation_log if m.get("content")])
    }

async def main_async():
    parser = argparse.ArgumentParser(description="AutoGen MagenticOne Baseline")
    parser.add_argument("--dataset", type=validate_dataset_name, required=True, help=get_dataset_help_text())
    parser.add_argument("--episodes", type=str, default="20", help="Number of episodes or 'all'")
    parser.add_argument("--model", type=str, default="openai/gpt-4o", help="OpenRouter model ID")
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"), help="OpenRouter API Key")
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent workers (default: 1, recommended: 2-4)")
    parser.add_argument("--max-turns", type=int, default=5, help="Maximum turns per episode (default:5)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Model temperature (default: 0.0 for reproducibility)")
    parser.add_argument("--context-buffer", type=int, default=10, help="Max messages to keep in conversation history (default: 10, reduces token usage)")
    args = parser.parse_args()
    
    # Validate API key
    if not args.api_key:
        print("ERROR: No API key provided.")
        print("Set OPENROUTER_API_KEY environment variable or pass --api-key argument.")
        sys.exit(1)

    # 1. Config & Data
    cfg = load_config("hierarchical")  # Load default config to satisfy dependencies
    dataset = get_dataset_loader(args.dataset, is_eval=True)
    
    if args.episodes.lower() == "all":
        num_episodes = len(dataset.tasks) if hasattr(dataset, 'tasks') else len(dataset.data)
    else:
        num_episodes = int(args.episodes)

    print("=" * 60)
    print(f"AUTOGEN MAGENTIC-ONE BASELINE")
    print(f"Dataset: {args.dataset} ({num_episodes} samples)")
    print(f"Model:   {args.model} (OpenRouter)")
    print(f"Workers: {args.workers} (concurrent)")
    print(f"Max Turns: {args.max_turns}")
    print(f"Temperature: {args.temperature}")
    print(f"Context Buffer: {args.context_buffer} messages")
    print("=" * 60)

    # 2. Client Setup
    # Note: MagenticOne often works best with strong models (GPT-4o, Claude 3.5 Sonnet)
    # Store config for creating per-episode clients (isolates token counting in concurrent mode)
    model_config = {
        "model": args.model,
        "api_key": args.api_key,
        "base_url": "https://openrouter.ai/api/v1",
        "temperature": args.temperature,
        "max_tokens": 512,
        "model_capabilities": {
            "vision": False,
            "function_calling": True,
            "json_output": True,
        }
    }
    
    # Create shared client for sequential execution (or as fallback)
    model_client = OpenAIChatCompletionClient(**model_config)

    # 3. Execution Loop (with optional concurrency)
    results = []
    total_tokens = {"prompt": 0, "completion": 0, "total": 0}
    
    if args.workers > 1:
        # Concurrent execution using semaphore
        # Pass model_config to create per-episode clients for isolated token counting
        semaphore = asyncio.Semaphore(args.workers)
        
        # Create all tasks - each episode gets its own model client for accurate token accounting
        tasks = [
            run_autogen_episode(i, dataset, i, model_client, semaphore, args.max_turns,
                               model_config=model_config, context_buffer_size=args.context_buffer)
            for i in range(num_episodes)
        ]
        
        # Run with progress bar
        print(f"Running {num_episodes} episodes with {args.workers} concurrent workers...")
        print(f"(Using per-episode clients for accurate token counting)")
        results = []
        pbar = tqdm(asyncio.as_completed(tasks), total=num_episodes, desc=f"Evaluating ({args.workers} workers)")
        for coro in pbar:
            result = await coro
            results.append(result)
            # Update running accuracy
            correct_so_far = sum(1 for r in results if r['correct'])
            pbar.set_postfix({"acc": f"{(correct_so_far/len(results))*100:.1f}%"})
        
        # Sort results by episode number
        results.sort(key=lambda x: x["episode"])
    else:
        # Sequential execution - shared client is fine for sequential
        pbar = tqdm(range(num_episodes), desc="Evaluating")
        for i in pbar:
            result = await run_autogen_episode(i, dataset, i, model_client, max_turns=args.max_turns,
                                              context_buffer_size=args.context_buffer)
            results.append(result)
            
            correct_so_far = sum(1 for r in results if r['correct'])
            pbar.set_postfix({"acc": f"{(correct_so_far/len(results))*100:.1f}%"})
    
    # Calculate totals
    correct_count = sum(1 for r in results if r['correct'])
    for r in results:
        if isinstance(r.get('tokens'), dict):
            total_tokens["prompt"] += r["tokens"].get("prompt", 0)
            total_tokens["completion"] += r["tokens"].get("completion", 0)
            total_tokens["total"] += r["tokens"].get("total", 0)

    # 4. Save Logs
    if not args.log_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_suffix = args.model.replace("/", "_")
        log_dir = os.path.join("eval_logs", "autogen_baseline", args.dataset, model_suffix)
        os.makedirs(log_dir, exist_ok=True)
        args.log_file = os.path.join(log_dir, f"eval_{timestamp}.json")

    log_data = {
        "metadata": {
            "method": "autogen_magentic_one",
            "dataset": args.dataset,
            "model": args.model,
            "workers": args.workers,
            "max_turns": args.max_turns,
            "temperature": args.temperature,
            "context_buffer": args.context_buffer,
            "timestamp": datetime.now().isoformat()
        },
        "summary": {
            "accuracy": (correct_count / num_episodes) * 100,
            "total_episodes": num_episodes,
            "correct_count": correct_count,
            "total_tokens": total_tokens,
            "avg_tokens_per_episode": total_tokens["total"] / num_episodes if num_episodes > 0 else 0
        },
        "episodes": results
    }

    with open(args.log_file, "w") as f:
        json.dump(log_data, f, indent=2)
    
    # Cleanup: Remove the coding_workspace directory if empty
    try:
        coding_workspace = "coding_workspace"
        if os.path.exists(coding_workspace) and not os.listdir(coding_workspace):
            shutil.rmtree(coding_workspace)
    except Exception:
        pass  # Ignore cleanup errors
    
    # Print summary
    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"Accuracy: {(correct_count / num_episodes) * 100:.1f}% ({correct_count}/{num_episodes})")
    print(f"Total Tokens: {total_tokens['total']:,} (prompt: {total_tokens['prompt']:,}, completion: {total_tokens['completion']:,})")
    print(f"Avg Tokens/Episode: {total_tokens['total'] / num_episodes:,.0f}")
    print(f"\nLogs saved to: {args.log_file}")

if __name__ == "__main__":
    asyncio.run(main_async())