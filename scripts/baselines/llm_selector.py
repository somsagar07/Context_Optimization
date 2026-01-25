"""
LLM Selector Baseline - LLM decides workflow, tools, and budgets

This baseline uses an LLM to decide the optimal configuration for each question:
- Workflow type (9 options)
- Tools for each agent
- Token budgets

The LLM makes the decision via prompting (zero-shot or few-shot), then
the selected configuration is executed to answer the question.

This tests whether an LLM can reason about optimal configurations without RL training.

Usage:
    # API mode (recommended)
    python scripts/baselines/llm_selector.py --dataset gsm8k --api --api-model qwen/qwen-2.5-7b-instruct
    python scripts/baselines/llm_selector.py --dataset drop --api --api-model google/gemini-2.5-flash-lite --workers 8
    
    # Evaluate all datapoints
    python scripts/baselines/llm_selector.py --dataset gsm8k --api --api-model qwen/qwen-2.5-7b-instruct --episodes all
    
    # Local HuggingFace model
    python scripts/baselines/llm_selector.py --dataset gsm8k --hf-model Qwen/Qwen2.5-7B-Instruct
    
    # With few-shot examples
    python scripts/baselines/llm_selector.py --dataset gsm8k --api --api-model qwen/qwen-2.5-7b-instruct --few-shot
"""
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
import re
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

from configs import load_config
from agents_system import LLMWorker, OpenRouterWorker
from agents_system.workflows import get_workflow
from utils import get_dataset_loader, validate_dataset_name, get_dataset_help_text
from tools import ToolRegistry


# Available workflows
WORKFLOWS = [
    "Direct",
    "Reason+Answer",
    "Reason+Verify+Answer",
    "Routing",
    "Parallel-Sectioning",
    "Parallel-Voting",
    "Orchestrator-Workers",
    "Evaluator-Optimizer",
    "Autonomous-Agent"
]

WORKFLOW_MAP = {
    "direct": 0,
    "reason+answer": 1,
    "reason+verify+answer": 2,
    "routing": 3,
    "parallel-sectioning": 4,
    "parallel-voting": 5,
    "orchestrator-workers": 6,
    "evaluator-optimizer": 7,
    "autonomous-agent": 8
}

# Available tools
TOOLS = ["calculator", "web_search", "python", "ocr_reader"]

# Token budgets
BUDGETS = {"low": 0, "medium": 1, "mid": 1, "high": 2}
TOKEN_BUDGETS = {
    "reasoner": {0: 256, 1: 512, 2: 1024},
    "verifier": {0: 128, 1: 256, 2: 512},
    "answerer": {0: 64, 1: 128, 2: 256}
}

# System prompt for LLM selector
SELECTOR_SYSTEM_PROMPT = """You are an expert at selecting optimal configurations for AI agent systems.

Given a question, you must decide the best configuration to answer it:

WORKFLOWS (choose one):
1. Direct - Simple direct answer, no reasoning steps
2. Reason+Answer - Think step by step, then answer
3. Reason+Verify+Answer - Reason, verify the reasoning, then answer
4. Routing - Route to specialized sub-agents based on question type
5. Parallel-Sectioning - Split problem into parallel sections
6. Parallel-Voting - Multiple agents vote on answer
7. Orchestrator-Workers - Central orchestrator delegates to workers
8. Evaluator-Optimizer - Generate and iteratively improve answer
9. Autonomous-Agent - Self-directed agent with planning

TOOLS (choose any combination for each agent):
- calculator: For mathematical calculations
- web_search: For retrieving external information
- python: For code execution and complex computations
- ocr_reader: For reading text from images

BUDGETS (choose one per agent):
- Low: Quick, concise responses
- Medium: Balanced detail
- High: Comprehensive, detailed responses

OUTPUT FORMAT (must be valid JSON):
{
    "workflow": "<workflow name>",
    "agent1_tools": ["tool1", "tool2"],
    "agent1_budget": "<low/medium/high>",
    "agent2_tools": ["tool1"],
    "agent2_budget": "<low/medium/high>",
    "answerer_budget": "<low/medium/high>",
    "reasoning": "<brief explanation of your choice>"
}

IMPORTANT:
- For simple math: Use Direct or Reason+Answer with calculator
- For complex reasoning: Use Reason+Verify+Answer
- For factual questions: Consider web_search
- For code/computation: Use python tool
- For reading comprehension: Reason+Verify+Answer works well
- Match budget to question complexity"""

# Few-shot examples for different dataset types
FEW_SHOT_EXAMPLES = {
    "math": """
Example 1:
Question: "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

{
    "workflow": "Reason+Answer",
    "agent1_tools": ["calculator"],
    "agent1_budget": "medium",
    "agent2_tools": [],
    "agent2_budget": "low",
    "answerer_budget": "low",
    "reasoning": "Math word problem requiring step-by-step calculation. Calculator tool needed for arithmetic."
}
""",
    "reading": """
Example 1:
Question: "The quick brown fox jumps over the lazy dog. What animal jumps over the dog?"

{
    "workflow": "Direct",
    "agent1_tools": [],
    "agent1_budget": "low",
    "agent2_tools": [],
    "agent2_budget": "low",
    "answerer_budget": "low",
    "reasoning": "Simple reading comprehension with explicit answer in text."
}

Example 2:
Question: "[Long passage about World War 2]... Who was the main commander of the Allied forces in Europe?"

{
    "workflow": "Reason+Verify+Answer",
    "agent1_tools": [],
    "agent1_budget": "high",
    "agent2_tools": [],
    "agent2_budget": "medium",
    "answerer_budget": "medium",
    "reasoning": "Complex reading comprehension requiring careful analysis and verification."
}
""",
    "medical": """
Example 1:
Question: "A 45-year-old man presents with chest pain radiating to the left arm. ECG shows ST elevation. What is the most likely diagnosis? A) Angina B) MI C) Pericarditis D) Aortic dissection"

{
    "workflow": "Reason+Verify+Answer",
    "agent1_tools": [],
    "agent1_budget": "high",
    "agent2_tools": [],
    "agent2_budget": "high",
    "answerer_budget": "medium",
    "reasoning": "Medical diagnosis requiring careful clinical reasoning and verification of symptoms."
}
""",
    "general": """
Example 1:
Question: "What is the capital of France?"

{
    "workflow": "Direct",
    "agent1_tools": [],
    "agent1_budget": "low",
    "agent2_tools": [],
    "agent2_budget": "low",
    "answerer_budget": "low",
    "reasoning": "Simple factual question with direct answer."
}
"""
}


def get_few_shot_examples(dataset_name):
    """Get appropriate few-shot examples based on dataset."""
    dataset_name = dataset_name.lower()
    if dataset_name in ['gsm8k', 'aime25']:
        return FEW_SHOT_EXAMPLES["math"]
    elif dataset_name in ['drop', 'hotpotqa']:
        return FEW_SHOT_EXAMPLES["reading"]
    elif dataset_name == 'medqa':
        return FEW_SHOT_EXAMPLES["medical"]
    else:
        return FEW_SHOT_EXAMPLES["general"]


def parse_llm_config(response_text):
    """Parse LLM's JSON response into configuration dict."""
    # Try to extract JSON from response
    json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
    if not json_match:
        # Try to find JSON with nested braces
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    
    if json_match:
        try:
            config = json.loads(json_match.group())
            return config
        except json.JSONDecodeError:
            pass
    
    # Fallback: try to parse key-value patterns
    config = {
        "workflow": "Reason+Answer",
        "agent1_tools": [],
        "agent1_budget": "medium",
        "agent2_tools": [],
        "agent2_budget": "low",
        "answerer_budget": "low",
        "reasoning": "Fallback config - could not parse LLM response"
    }
    
    # Try to extract workflow
    workflow_match = re.search(r'workflow["\s:]+(["\']?)([^"\'}\n,]+)\1', response_text, re.IGNORECASE)
    if workflow_match:
        config["workflow"] = workflow_match.group(2).strip()
    
    return config


def config_to_action(config):
    """Convert LLM config dict to action tuple matching environment format."""
    # Workflow
    workflow_name = config.get("workflow", "Reason+Answer").lower().strip()
    workflow_idx = WORKFLOW_MAP.get(workflow_name, 1)  # Default to Reason+Answer
    
    # Agent1 tools (binary encoding)
    agent1_tools = config.get("agent1_tools", [])
    if isinstance(agent1_tools, str):
        agent1_tools = [agent1_tools]
    agent1_tools_idx = 0
    for i, tool in enumerate(TOOLS):
        if tool in [t.lower().strip() for t in agent1_tools]:
            agent1_tools_idx |= (1 << i)
    
    # Agent1 budget
    agent1_budget = config.get("agent1_budget", "medium").lower().strip()
    agent1_budget_idx = BUDGETS.get(agent1_budget, 1)
    
    # Agent2 tools
    agent2_tools = config.get("agent2_tools", [])
    if isinstance(agent2_tools, str):
        agent2_tools = [agent2_tools]
    agent2_tools_idx = 0
    for i, tool in enumerate(TOOLS):
        if tool in [t.lower().strip() for t in agent2_tools]:
            agent2_tools_idx |= (1 << i)
    
    # Agent2 budget
    agent2_budget = config.get("agent2_budget", "low").lower().strip()
    agent2_budget_idx = BUDGETS.get(agent2_budget, 0)
    
    # Answerer budget
    answerer_budget = config.get("answerer_budget", "low").lower().strip()
    answerer_budget_idx = BUDGETS.get(answerer_budget, 0)
    
    return (workflow_idx, agent1_tools_idx, agent1_budget_idx, 
            agent2_tools_idx, agent2_budget_idx, answerer_budget_idx)


def decode_tools(tool_idx):
    """Decode binary tool index to list of tool names."""
    tools = []
    for i, tool in enumerate(TOOLS):
        if tool_idx & (1 << i):
            tools.append(tool)
    return tools


def execute_workflow(worker, question, action, tools_registry):
    """Execute the selected workflow with the given configuration."""
    workflow_idx, agent1_tools_idx, agent1_budget_idx, agent2_tools_idx, agent2_budget_idx, answerer_budget_idx = action
    
    # Decode tools
    agent1_tools = decode_tools(agent1_tools_idx)
    agent2_tools = decode_tools(agent2_tools_idx)
    
    # Get token budgets
    reasoner_budget = TOKEN_BUDGETS["reasoner"][agent1_budget_idx]
    verifier_budget = TOKEN_BUDGETS["verifier"][agent2_budget_idx]
    answerer_budget = TOKEN_BUDGETS["answerer"][answerer_budget_idx]
    
    # Get workflow name for the workflow factory
    workflow_names = [
        "direct", "prompt_chaining", "prompt_chaining_with_verification",
        "routing", "parallel_sectioning", "parallel_voting",
        "orchestrator_workers", "evaluator_optimizer", "autonomous_agent"
    ]
    workflow_name = workflow_names[workflow_idx] if workflow_idx < len(workflow_names) else "prompt_chaining"
    
    try:
        # Get workflow instance
        workflow = get_workflow(workflow_name, worker)
        
        # Execute workflow with configuration
        response = workflow.execute(
            question=question,
            active_tools=agent1_tools,
            max_tokens=reasoner_budget,
            agent2_tools=agent2_tools if workflow_idx >= 2 else None,
            agent2_budget=verifier_budget if workflow_idx >= 2 else None,
            answerer_budget=answerer_budget
        )
        
        return response
        
    except Exception as e:
        # Fallback: direct generation
        response = worker._generate(
            prompt=question,
            active_tools=agent1_tools,
            max_tokens=reasoner_budget,
            prompt_suffix=None
        )
        return response


def run_single_episode(ep, selector_worker, executor_worker, dataset, sample_idx=None, 
                       use_few_shot=False, tools_registry=None):
    """Run a single LLM Selector evaluation episode.
    
    1. LLM (selector_worker) decides configuration
    2. Configuration is parsed and converted to action
    3. Workflow is executed with that configuration
    4. Answer is evaluated
    """
    # Get question and answer based on dataset type
    if sample_idx is not None:
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
                rel_path = sample.get('file_path', '')
                if rel_path and hasattr(dataset, 'data_dir'):
                    full_path = os.path.join(dataset.data_dir, rel_path)
                    question += f"\n\n[File Attachment: {full_path}]"
            elif dataset_name == 'medqa':
                data = sample['data']
                question = data['Question']
                options = data['Options']
                options_text = "\n".join([f"{k}: {v}" for k, v in sorted(options.items())])
                question = f"{question}\n\nOptions:\n{options_text}"
                answer = data['Correct Answer']
            elif dataset_name == 'aime25':
                question = sample.get('problem', '')
                answer = sample.get('answer', '')
            elif dataset_name == 'drop':
                passage = sample.get('passage', '')
                question_text = sample.get('question', '')
                question = f"{passage}\n\nQuestion: {question_text}"
                answer_spans = sample.get('answers_spans', {})
                if 'spans' in answer_spans and len(answer_spans['spans']) > 0:
                    answer = answer_spans['spans'][0]
                else:
                    answer = sample.get('answer', '')
            elif dataset_name.startswith('mmlu'):
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
                question = sample.get('question', sample.get('problem', ''))
                answer = sample.get('answer', '')
        else:
            sample = dataset[sample_idx]
            question = sample.get('question', sample.get('problem', ''))
            answer = sample.get('answer', '')
    else:
        question, answer = dataset.get_sample()
    
    # Step 1: Ask LLM to decide configuration
    selector_prompt = SELECTOR_SYSTEM_PROMPT
    if use_few_shot:
        dataset_name = getattr(dataset, 'name', 'general').lower()
        selector_prompt += "\n\nHere are some examples:\n" + get_few_shot_examples(dataset_name)
    
    selector_prompt += f"\n\nNow decide the configuration for this question:\n\nQuestion: {question}\n\nOutput your configuration as JSON:"
    
    # Generate configuration decision
    config_response = selector_worker._generate(
        prompt=selector_prompt,
        active_tools=[],
        max_tokens=512,
        prompt_suffix=None
    )
    
    # Parse configuration
    config = parse_llm_config(config_response)
    action = config_to_action(config)
    
    # Step 2: Execute workflow with selected configuration
    prediction = execute_workflow(executor_worker, question, action, tools_registry)
    
    # Estimate tokens (selector + executor)
    total_tokens = len(config_response.split()) * 1.3 + len(prediction.split()) * 1.3
    
    # Evaluate correctness
    correct = dataset.evaluate_correctness(prediction, answer)
    
    # Build episode log
    episode_log = {
        "episode": ep + 1,
        "correct": bool(correct),
        "tokens": int(total_tokens),
        "question": question,
        "prediction": prediction,
        "ground_truth": answer,
        "llm_config": config,
        "action": list(action),
        "workflow": WORKFLOWS[action[0]] if action[0] < len(WORKFLOWS) else "Unknown",
        "agent1_tools": decode_tools(action[1]),
        "agent2_tools": decode_tools(action[3]),
        "config_response": config_response
    }
    
    return {
        "correct": bool(correct),
        "tokens": int(total_tokens),
        "episode_log": episode_log
    }


def evaluate_parallel(cfg, num_episodes=20, use_api=False, api_model=None, 
                     hf_model=None, num_workers=4, use_few_shot=False):
    """Parallel LLM Selector evaluation."""
    
    print(f"\nLoading dataset...")
    shared_dataset = get_dataset_loader(cfg.DATASET_NAME, is_eval=True)
    dataset_size = len(shared_dataset.tasks) if hasattr(shared_dataset, 'tasks') else len(shared_dataset.data)
    print(f"  Dataset loaded: {dataset_size} samples")
    
    if num_episodes > dataset_size:
        print(f"  Warning: Requested {num_episodes} episodes but only {dataset_size} samples. Evaluating all {dataset_size}.")
        num_episodes = dataset_size
    
    indices = list(range(num_episodes))
    print(f"  Will evaluate {num_episodes} unique datapoints")
    
    # Pre-create workers (each thread gets its own selector and executor)
    print(f"\nPre-creating {num_workers} worker pairs...")
    selector_workers = []
    executor_workers = []
    tools_registries = []
    
    for i in range(num_workers):
        print(f"  Creating worker pair {i+1}/{num_workers}...")
        if use_api:
            selector_workers.append(OpenRouterWorker(model_name=api_model))
            executor_workers.append(OpenRouterWorker(model_name=api_model))
        else:
            selector_workers.append(LLMWorker(model_name=hf_model))
            executor_workers.append(LLMWorker(model_name=hf_model))
        tools_registries.append(ToolRegistry())
    print(f"  Done! All {num_workers} worker pairs ready.")
    
    worker_locks = [threading.Lock() for _ in range(num_workers)]
    
    def worker_fn(ep):
        worker_idx = ep % num_workers
        sample_idx = indices[ep]
        
        with worker_locks[worker_idx]:
            return run_single_episode(
                ep, 
                selector_workers[worker_idx], 
                executor_workers[worker_idx],
                shared_dataset, 
                sample_idx=sample_idx,
                use_few_shot=use_few_shot,
                tools_registry=tools_registries[worker_idx]
            )
    
    results = {"correct": [], "tokens": []}
    episode_logs = []
    
    completed = [0]
    correct_count = [0]
    total_tokens = [0]
    results_lock = threading.Lock()
    
    # Track workflow distribution
    workflow_counts = {w: 0 for w in WORKFLOWS}
    
    print(f"Evaluating on {num_episodes} episodes with {num_workers} parallel workers...")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(worker_fn, ep): ep for ep in range(num_episodes)}
        
        with tqdm(total=num_episodes, desc=f"LLM Selector ({num_workers} workers)") as pbar:
            for future in as_completed(futures):
                ep = futures[future]
                try:
                    result = future.result()
                    
                    with results_lock:
                        results["correct"].append(result["correct"])
                        results["tokens"].append(result["tokens"])
                        episode_logs.append(result["episode_log"])
                        
                        # Track workflow choice
                        workflow = result["episode_log"].get("workflow", "Unknown")
                        if workflow in workflow_counts:
                            workflow_counts[workflow] += 1
                        
                        completed[0] += 1
                        correct_count[0] += int(result["correct"])
                        total_tokens[0] += result["tokens"]
                        
                        acc = correct_count[0] / completed[0] * 100
                        avg_tokens = total_tokens[0] / completed[0]
                        pbar.set_postfix({"acc": f"{acc:.1f}%", "tokens": f"{avg_tokens:.0f}"})
                        pbar.update(1)
                        
                except Exception as e:
                    print(f"\nError in episode {ep}: {e}")
                    import traceback
                    traceback.print_exc()
                    pbar.update(1)
    
    episode_logs.sort(key=lambda x: x["episode"])
    
    accuracy = np.mean(results["correct"]) * 100
    avg_tokens = np.mean(results["tokens"])
    
    print("\n" + "=" * 60)
    print("LLM SELECTOR BASELINE RESULTS")
    print("=" * 60)
    print(f"  Accuracy:    {accuracy:.1f}%")
    print(f"  Avg Tokens:  {avg_tokens:.0f}")
    print("\n  Workflow Distribution:")
    for wf, count in sorted(workflow_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"    {wf}: {count} ({count/num_episodes*100:.1f}%)")
    
    return {
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "workflow_distribution": workflow_counts,
        "episodes": episode_logs
    }


def evaluate(selector_worker, executor_worker, dataset, num_episodes=20, 
             use_few_shot=False, tools_registry=None):
    """Sequential LLM Selector evaluation."""
    
    dataset_size = len(dataset.tasks) if hasattr(dataset, 'tasks') else len(dataset.data)
    
    if num_episodes > dataset_size:
        print(f"  Warning: Requested {num_episodes} episodes but only {dataset_size} samples. Evaluating all {dataset_size}.")
        num_episodes = dataset_size
    
    results = {"correct": [], "tokens": []}
    episode_logs = []
    workflow_counts = {w: 0 for w in WORKFLOWS}
    
    running_correct = 0
    running_tokens = 0
    
    print(f"\nEvaluating on {num_episodes} episodes...")
    
    pbar = tqdm(range(num_episodes), desc="LLM Selector", leave=True)
    for ep in pbar:
        result = run_single_episode(
            ep, selector_worker, executor_worker, dataset, 
            sample_idx=ep, use_few_shot=use_few_shot,
            tools_registry=tools_registry
        )
        
        results["correct"].append(result["correct"])
        results["tokens"].append(result["tokens"])
        episode_logs.append(result["episode_log"])
        
        # Track workflow choice
        workflow = result["episode_log"].get("workflow", "Unknown")
        if workflow in workflow_counts:
            workflow_counts[workflow] += 1
        
        running_correct += int(result["correct"])
        running_tokens += result["tokens"]
        
        pbar.set_postfix({
            "acc": f"{running_correct/(ep+1)*100:.1f}%",
            "tokens": f"{running_tokens/(ep+1):.0f}"
        })
    
    accuracy = np.mean(results["correct"]) * 100
    avg_tokens = np.mean(results["tokens"])
    
    print("\n" + "=" * 60)
    print("LLM SELECTOR BASELINE RESULTS")
    print("=" * 60)
    print(f"  Accuracy:    {accuracy:.1f}%")
    print(f"  Avg Tokens:  {avg_tokens:.0f}")
    print("\n  Workflow Distribution:")
    for wf, count in sorted(workflow_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"    {wf}: {count} ({count/num_episodes*100:.1f}%)")
    
    return {
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "workflow_distribution": workflow_counts,
        "episodes": episode_logs
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Selector Baseline - LLM decides workflow, tools, and budgets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # API mode with parallel workers (recommended)
  python scripts/baselines/llm_selector.py --dataset gsm8k --api --api-model qwen/qwen-2.5-7b-instruct --workers 8
  
  # With few-shot examples
  python scripts/baselines/llm_selector.py --dataset drop --api --api-model google/gemini-2.5-flash-lite --workers 8 --few-shot
  
  # Local HuggingFace model
  python scripts/baselines/llm_selector.py --dataset gsm8k --hf-model Qwen/Qwen2.5-7B-Instruct
        """
    )
    
    parser.add_argument("--config", type=str, default="hierarchical")
    parser.add_argument("--episodes", type=str, default="20",
                       help="Number of episodes to evaluate, or 'all' for all datapoints")
    parser.add_argument("--dataset", type=validate_dataset_name, required=True, default=None,
                       help=get_dataset_help_text())
    
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log-file", type=str, default=None,
                       help="Path to save evaluation logs as JSON. Auto-generates if not provided.")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of parallel workers for API evaluation (default: 1, recommended: 4-8 for API)")
    
    # Few-shot option
    parser.add_argument("--few-shot", action="store_true",
                       help="Include few-shot examples in the selector prompt")
    
    # API configuration
    parser.add_argument("--api", action="store_true", default=False,
                       help="Use OpenRouter API instead of local HuggingFace models")
    parser.add_argument("--api-model", type=str, default=None,
                       help="OpenRouter model ID (e.g., 'openai/gpt-4o', 'qwen/qwen-2.5-7b-instruct')")
    parser.add_argument("--hf-model", type=str, default=None,
                       help="HuggingFace model name (e.g., 'Qwen/Qwen2.5-7B-Instruct')")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    cfg = load_config(args.config)
    if args.dataset:
        cfg.DATASET_NAME = args.dataset
    
    # Handle "all" episodes
    if args.episodes.lower() == "all":
        dataset = get_dataset_loader(cfg.DATASET_NAME, is_eval=True)
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
    print("LLM SELECTOR BASELINE EVALUATION")
    print("=" * 60)
    print(f"  Method:     LLM Selector (LLM decides config, then executes)")
    print(f"  Dataset:    {cfg.DATASET_NAME}")
    print(f"  Episodes:   {num_episodes}")
    print(f"  Few-shot:   {'Yes' if args.few_shot else 'No'}")
    
    if args.api:
        model_name = args.api_model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
        print(f"  API Mode:   {model_name}")
        if args.workers > 1:
            print(f"  Workers:    {args.workers} (parallel)")
    else:
        from configs.base import LLM_MODEL_NAME
        model_name = args.hf_model or getattr(cfg, "LLM_MODEL_NAME", LLM_MODEL_NAME)
        print(f"  HF Model:   {model_name}")
        if args.workers > 1:
            print(f"  Warning: Parallel workers ignored for local HF models")
            args.workers = 1
    
    print("=" * 60)
    
    # Run evaluation
    if args.api and args.workers > 1:
        results = evaluate_parallel(
            cfg,
            num_episodes=num_episodes,
            use_api=args.api,
            api_model=args.api_model,
            hf_model=args.hf_model,
            num_workers=args.workers,
            use_few_shot=args.few_shot
        )
    else:
        dataset = get_dataset_loader(cfg.DATASET_NAME, is_eval=True)
        tools_registry = ToolRegistry()
        
        if args.api:
            selector_worker = OpenRouterWorker(model_name=args.api_model)
            executor_worker = OpenRouterWorker(model_name=args.api_model)
        else:
            selector_worker = LLMWorker(model_name=args.hf_model)
            executor_worker = LLMWorker(model_name=args.hf_model)
        
        results = evaluate(
            selector_worker, executor_worker, dataset,
            num_episodes=num_episodes,
            use_few_shot=args.few_shot,
            tools_registry=tools_registry
        )
    
    # Save logs
    log_file = args.log_file
    if log_file is None:
        model_suffix = model_name.split("/")[-1].replace(".", "_").replace(":", "_")
        few_shot_suffix = "_fewshot" if args.few_shot else ""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join("eval_logs", "llm_selector", cfg.DATASET_NAME, model_suffix)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"eval{few_shot_suffix}_{timestamp}.json")
    
    # Build complete log with metadata
    log_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "method": "llm_selector",
            "description": "LLM Selector baseline - LLM decides workflow, tools, budgets, then executes",
            "dataset": cfg.DATASET_NAME,
            "num_episodes": num_episodes,
            "model": model_name,
            "few_shot": args.few_shot,
            "api_mode": args.api,
            "parallel_workers": args.workers if args.api else 1
        },
        "summary": {
            "accuracy": results["accuracy"],
            "avg_tokens": results["avg_tokens"],
            "workflow_distribution": results["workflow_distribution"]
        },
        "episodes": results["episodes"]
    }
    
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\n  Logs saved to: {log_file}")
    
    # Print comparison note
    print("\n" + "=" * 60)
    print("COMPARISON NOTES")
    print("=" * 60)
    print("  LLM Selector: LLM decides config via prompting (no training)")
    print("  HRL:          Hierarchical RL learns optimal configs through training")
    print("  ")
    print("  For fair comparison, run:")
    print(f"    python scripts/baselines/base_model.py --dataset {cfg.DATASET_NAME} ...")
    print(f"    python scripts/eval_hrl.py --dataset {cfg.DATASET_NAME} ...")
    print("=" * 60)


if __name__ == "__main__":
    main()

