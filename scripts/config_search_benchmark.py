"""
Config & Search Benchmark Script

Benchmarks fixed configurations and search strategies (no RL policies).
Use eval_hrl.py for evaluating trained RL/SFT policies instead.

Current System Configuration:
- 9 Workflows: Direct, Reason+Ans, Reason+Verify+Ans, Routing, Parallel-Sectioning,
               Parallel-Voting, Orchestrator-Workers, Evaluator-Optimizer, Autonomous-Agent
- 4 Tools: calculator(1), web_search(2), python(4), ocr_reader(8) → 16 combinations
- 3 Budget levels: Low(0), Mid(1), High(2)
- Action Space: MultiDiscrete([9, 16, 3, 16, 3, 3])

Usage:
    # Run all baselines on full dataset with parallel workers
    python scripts/config_search_benchmark.py --dataset hotpotqa --episodes all --api --api-model "gemma" --workers 8
    
    # Run only Direct workflow baselines on full dataset
    python scripts/config_search_benchmark.py --baseline direct --dataset hotpotqa --episodes all --api --api-model "gemma" --workers 8
    
    # Run only search algorithms (Grid, Greedy, Best-First, Evolutionary)
    python scripts/config_search_benchmark.py --baseline search --dataset hotpotqa --episodes 50 --api --api-model "gemma"
    
    # Run a specific fixed config on full dataset
    python scripts/config_search_benchmark.py --baseline "Direct-Web-Mid" --dataset drop --episodes all --api --api-model "gemma" --workers 8
    
    # Run with limited episodes (useful for testing)
    python scripts/config_search_benchmark.py --baseline direct --dataset hotpotqa --episodes 50 --api --api-model "gemma"
    
    # List all available baselines
    python scripts/config_search_benchmark.py --list-baselines
"""

import os
import sys
# Add parent directory to path for imports when running from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import gc
import heapq
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from configs import load_config
from env import StructureEnv, PromptEnv
from utils import get_dataset_loader


# ============================================================
# WORKFLOW & TOOL DEFINITIONS (must match env/structure_env.py)
# ============================================================

WORKFLOW_NAMES = [
    "Direct",              # 0
    "Reason+Ans",          # 1
    "Reason+Verify+Ans",   # 2
    "Routing",             # 3
    "Parallel-Sectioning", # 4
    "Parallel-Voting",     # 5
    "Orchestrator-Workers",# 6
    "Evaluator-Optimizer", # 7
    "Autonomous-Agent",    # 8
]

# Workflows that use agent2 (verifier/second agent)
WORKFLOWS_WITH_AGENT2 = [2, 7]  # Reason+Verify+Ans, Evaluator-Optimizer

# Workflows that skip reasoner (only answerer)
WORKFLOWS_ANSWERER_ONLY = [0, 5]  # Direct, Parallel-Voting

BUDGET_NAMES = ["Low", "Mid", "High"]

# Tool encoding (binary): calculator=1, web_search=2, python=4, ocr_reader=8
TOOL_BITS = {
    "calculator": 1,
    "web_search": 2,
    "python": 4,
    "ocr_reader": 8,
}

TOOL_PRESETS = {
    "NoTools": 0,
    "Calc": 1,
    "Web": 2,
    "CalcWeb": 3,
    "Python": 4,
    "CalcPython": 5,
    "WebPython": 6,
    "CalcWebPython": 7,
    "OCR": 8,
    "AllTools": 15,
}


class SuppressOutput:
    """Context manager to suppress stdout/stderr prints."""
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        return self
    
    def __exit__(self, *args):
        sys.stdout = self._stdout
        sys.stderr = self._stderr


def decode_tools(idx: int) -> list:
    """Decode tool index to list of tools (binary encoding)."""
    tools = []
    if idx & 1: tools.append("calculator")
    if idx & 2: tools.append("web_search")
    if idx & 4: tools.append("python")
    if idx & 8: tools.append("ocr_reader")
    return tools if tools else ["none"]


def format_config(action):
    """Format action as readable configuration string."""
    workflow = WORKFLOW_NAMES[action[0]] if action[0] < len(WORKFLOW_NAMES) else f"Unknown({action[0]})"
    agent1_tools = decode_tools(action[1])
    agent1_budget = BUDGET_NAMES[action[2]] if action[2] < 3 else f"Unknown({action[2]})"
    agent2_tools = decode_tools(action[3])
    agent2_budget = BUDGET_NAMES[action[4]] if action[4] < 3 else f"Unknown({action[4]})"
    answerer_budget = BUDGET_NAMES[action[5]] if action[5] < 3 else f"Unknown({action[5]})"
    
    return f"{workflow}|A1:{'+'.join(agent1_tools)},{agent1_budget}|A2:{'+'.join(agent2_tools)},{agent2_budget}|Ans:{answerer_budget}"


def run_fixed_config_episode(structure_env, prompt_env, fixed_action):
    """Run episode with a fixed structure configuration."""
    struct_obs, _ = structure_env.reset()
    struct_action = np.array(fixed_action)
    _, _, _, _, struct_exec_info = structure_env.step(struct_action)
    
    prompt_env.set_structure(
        question=struct_exec_info["question"],
        answer=struct_exec_info["answer"],
        embedding=struct_exec_info["embedding"],
        structure=struct_exec_info["structure"]
    )
    
    # Use action 0 (DONE) for prompts - minimal prompt selection
    prompt_obs, _ = prompt_env.reset()
    done = False
    total_reward = 0.0
    prompt_steps = 0
    
    while not done:
        # Action 0 = DONE (skip prompt selection, use defaults)
        prompt_action = 0
        prompt_obs, reward, done, _, info = prompt_env.step(prompt_action)
        total_reward += reward
        prompt_steps += 1
    
    return {
        "correct": info.get("correct", False),
        "reward": total_reward,
        "workflow": struct_exec_info.get("workflow", WORKFLOW_NAMES[fixed_action[0]]),
        "steps_taken": info.get("steps_taken", 0),
        "tools_used": info.get("tools_used", 0),
        "total_tokens": info.get("total_tokens", 0),
        "decision_steps": 1 + prompt_steps,
        "config": format_config(struct_action),
        "final_answer": info.get("final_answer", ""),
        "ground_truth": info.get("ground_truth", ""),
    }


# ============================================================
# BASELINE CONFIGURATIONS
# Action format: [workflow, agent1_tools, agent1_budget, agent2_tools, agent2_budget, answerer_budget]
# ============================================================

# Direct workflow baselines (workflow=0) - only uses answerer
DIRECT_CONFIGS = {
    "Direct-NoTools-Low":    [0, 0, 0, 0, 0, 0],
    "Direct-NoTools-Mid":    [0, 0, 1, 0, 0, 1],
    "Direct-NoTools-High":   [0, 0, 2, 0, 0, 2],
    "Direct-Calc-Mid":       [0, 1, 1, 0, 0, 1],
    "Direct-Calc-High":      [0, 1, 2, 0, 0, 2],
    "Direct-Web-Mid":        [0, 2, 1, 0, 0, 1],
    "Direct-Web-High":       [0, 2, 2, 0, 0, 2],
    "Direct-CalcWeb-Mid":    [0, 3, 1, 0, 0, 1],
    "Direct-CalcWeb-High":   [0, 3, 2, 0, 0, 2],
    "Direct-Python-Mid":     [0, 4, 1, 0, 0, 1],
    "Direct-Python-High":    [0, 4, 2, 0, 0, 2],
    "Direct-AllTools-Mid":   [0, 15, 1, 0, 0, 1],
    "Direct-AllTools-High":  [0, 15, 2, 0, 0, 2],
}

# Reason+Answer workflow baselines (workflow=1) - uses reasoner + answerer
REASON_CONFIGS = {
    "Reason-NoTools-Mid":    [1, 0, 1, 0, 0, 1],
    "Reason-NoTools-High":   [1, 0, 2, 0, 0, 2],
    "Reason-Calc-Mid":       [1, 1, 1, 0, 0, 1],
    "Reason-Calc-High":      [1, 1, 2, 0, 0, 2],
    "Reason-Web-Mid":        [1, 2, 1, 0, 0, 1],
    "Reason-Web-High":       [1, 2, 2, 0, 0, 2],
    "Reason-CalcWeb-Mid":    [1, 3, 1, 0, 0, 1],
    "Reason-CalcWeb-High":   [1, 3, 2, 0, 0, 2],
    "Reason-Python-Mid":     [1, 4, 1, 0, 0, 1],
    "Reason-Python-High":    [1, 4, 2, 0, 0, 2],
    "Reason-AllTools-Mid":   [1, 15, 1, 0, 0, 1],
    "Reason-AllTools-High":  [1, 15, 2, 0, 0, 2],
}

# Reason+Verify+Answer workflow baselines (workflow=2) - uses reasoner + verifier + answerer
VERIFY_CONFIGS = {
    "Verify-NoTools-Mid":    [2, 0, 1, 0, 1, 1],
    "Verify-NoTools-High":   [2, 0, 2, 0, 2, 2],
    "Verify-Calc-Mid":       [2, 1, 1, 1, 1, 1],
    "Verify-Calc-High":      [2, 1, 2, 1, 2, 2],
    "Verify-Web-Mid":        [2, 2, 1, 2, 1, 1],
    "Verify-Web-High":       [2, 2, 2, 2, 2, 2],
    "Verify-CalcWeb-Mid":    [2, 3, 1, 3, 1, 1],
    "Verify-CalcWeb-High":   [2, 3, 2, 3, 2, 2],
    "Verify-AllTools-Mid":   [2, 15, 1, 15, 1, 1],
    "Verify-AllTools-High":  [2, 15, 2, 15, 2, 2],
}

# Routing workflow baselines (workflow=3)
ROUTING_CONFIGS = {
    "Routing-NoTools-Mid":   [3, 0, 1, 0, 0, 1],
    "Routing-NoTools-High":  [3, 0, 2, 0, 0, 2],
    "Routing-CalcWeb-Mid":   [3, 3, 1, 0, 0, 1],
    "Routing-CalcWeb-High":  [3, 3, 2, 0, 0, 2],
    "Routing-AllTools-High": [3, 15, 2, 0, 0, 2],
}

# Parallel-Sectioning workflow baselines (workflow=4)
PARALLEL_SECTION_CONFIGS = {
    "ParallelSec-NoTools-Mid":   [4, 0, 1, 0, 0, 1],
    "ParallelSec-NoTools-High":  [4, 0, 2, 0, 0, 2],
    "ParallelSec-CalcWeb-Mid":   [4, 3, 1, 0, 0, 1],
    "ParallelSec-CalcWeb-High":  [4, 3, 2, 0, 0, 2],
    "ParallelSec-AllTools-High": [4, 15, 2, 0, 0, 2],
}

# Parallel-Voting workflow baselines (workflow=5) - only uses answerer (multiple times)
PARALLEL_VOTE_CONFIGS = {
    "ParallelVote-NoTools-Mid":   [5, 0, 1, 0, 0, 1],
    "ParallelVote-NoTools-High":  [5, 0, 2, 0, 0, 2],
    "ParallelVote-CalcWeb-Mid":   [5, 3, 1, 0, 0, 1],
    "ParallelVote-CalcWeb-High":  [5, 3, 2, 0, 0, 2],
    "ParallelVote-AllTools-High": [5, 15, 2, 0, 0, 2],
}

# Orchestrator-Workers workflow baselines (workflow=6)
ORCHESTRATOR_CONFIGS = {
    "Orchestrator-NoTools-Mid":   [6, 0, 1, 0, 0, 1],
    "Orchestrator-NoTools-High":  [6, 0, 2, 0, 0, 2],
    "Orchestrator-CalcWeb-Mid":   [6, 3, 1, 0, 0, 1],
    "Orchestrator-CalcWeb-High":  [6, 3, 2, 0, 0, 2],
    "Orchestrator-AllTools-High": [6, 15, 2, 0, 0, 2],
}

# Evaluator-Optimizer workflow baselines (workflow=7) - uses agent2 (evaluator)
EVAL_OPT_CONFIGS = {
    "EvalOpt-NoTools-Mid":   [7, 0, 1, 0, 1, 1],
    "EvalOpt-NoTools-High":  [7, 0, 2, 0, 2, 2],
    "EvalOpt-CalcWeb-Mid":   [7, 3, 1, 3, 1, 1],
    "EvalOpt-CalcWeb-High":  [7, 3, 2, 3, 2, 2],
    "EvalOpt-AllTools-High": [7, 15, 2, 15, 2, 2],
}

# Autonomous-Agent workflow baselines (workflow=8)
AUTONOMOUS_CONFIGS = {
    "Autonomous-NoTools-Mid":   [8, 0, 1, 0, 0, 1],
    "Autonomous-NoTools-High":  [8, 0, 2, 0, 0, 2],
    "Autonomous-CalcWeb-Mid":   [8, 3, 1, 0, 0, 1],
    "Autonomous-CalcWeb-High":  [8, 3, 2, 0, 0, 2],
    "Autonomous-AllTools-High": [8, 15, 2, 0, 0, 2],
}

# All baselines combined
ALL_BASELINE_CONFIGS = {
    **DIRECT_CONFIGS, 
    **REASON_CONFIGS, 
    **VERIFY_CONFIGS,
    **ROUTING_CONFIGS,
    **PARALLEL_SECTION_CONFIGS,
    **PARALLEL_VOTE_CONFIGS,
    **ORCHESTRATOR_CONFIGS,
    **EVAL_OPT_CONFIGS,
    **AUTONOMOUS_CONFIGS,
}

# Grouped by workflow for easy selection
BASELINE_GROUPS = {
    "direct": DIRECT_CONFIGS,
    "reason": REASON_CONFIGS,
    "verify": VERIFY_CONFIGS,
    "routing": ROUTING_CONFIGS,
    "parallel_section": PARALLEL_SECTION_CONFIGS,
    "parallel_vote": PARALLEL_VOTE_CONFIGS,
    "orchestrator": ORCHESTRATOR_CONFIGS,
    "eval_opt": EVAL_OPT_CONFIGS,
    "autonomous": AUTONOMOUS_CONFIGS,
}


# ============================================================
# SEARCH ALGORITHMS
# ============================================================

def generate_grid_search_configs(max_configs=50):
    """Generate configurations for grid search over the full action space.
    
    Standard grid search: systematically explores the action space in order.
    No prioritization or heuristics - just enumerates configurations.
    Action space: [workflow(0-8), agent1_tools(0-15), agent1_budget(0-2), 
                   agent2_tools(0-15), agent2_budget(0-2), answerer_budget(0-2)]
    """
    configs = []
    
    # Enumerate in order: workflow -> tools -> budgets
    # This is a naive grid search without any optimization
    for wf in range(9):  # All 9 workflows
        for r_tools in range(16):  # All 16 tool combinations
            for r_budget in range(3):  # All 3 budget levels
                if wf in WORKFLOWS_WITH_AGENT2:  # Workflows that use agent2
                    for v_tools in range(16):
                        for v_budget in range(3):
                            for a_budget in range(3):
                                configs.append([wf, r_tools, r_budget, v_tools, v_budget, a_budget])
                                if len(configs) >= max_configs:
                                    return configs
                else:
                    for a_budget in range(3):
                        configs.append([wf, r_tools, r_budget, 0, 0, a_budget])
                        if len(configs) >= max_configs:
                            return configs
    return configs


def generate_greedy_configs():
    """Generate configurations for greedy search in sequential order.
    
    Standard greedy: tries configurations in a fixed sequential order.
    No heuristic prioritization - just enumerates workflow by workflow.
    This is a naive baseline without any "smart" ordering.
    """
    configs = []
    
    # Sequential order: workflow 0 -> 8, tools 0 -> 15, budget 0 -> 2
    # No prioritization - just enumerate in natural order
    for wf in range(9):  # 0, 1, 2, ... 8
        for tools in range(16):  # 0, 1, 2, ... 15
            for budget in range(3):  # 0, 1, 2
                if wf in WORKFLOWS_WITH_AGENT2:
                    configs.append([wf, tools, budget, tools, budget, budget])
                else:
                    configs.append([wf, tools, budget, 0, 0, budget])
    
    return configs[:50]


def generate_neighbors(config):
    """Generate neighboring configurations (naive: no expert knowledge).
    
    Simply generates neighbors by changing each dimension by +/- 1.
    No special handling for different action types.
    Action space bounds: [9, 16, 3, 16, 3, 3]
    """
    import random
    action_bounds = [9, 16, 3, 16, 3, 3]
    neighbors = []
    
    for i in range(len(config)):
        # Generate neighbors by +1 and -1 for each dimension
        for delta in [-1, 1]:
            new_val = config[i] + delta
            if 0 <= new_val < action_bounds[i] and new_val != config[i]:
                neighbor = config.copy()
                neighbor[i] = new_val
                neighbors.append(neighbor)
    
    # Shuffle to avoid bias toward any dimension
    random.shuffle(neighbors)
    return neighbors[:10]


def run_search_algorithm(structure_env, prompt_env, method, num_episodes=50):
    """Run a search algorithm and return results with progress tracking (sequential)."""
    import random
    results_list = []
    correct_count = 0
    total_tokens = 0
    
    method_names = {
        "grid": "Grid Search",
        "greedy": "Greedy Search", 
        "best_first": "Best-First Search",
        "evolutionary": "Evolutionary Search"
    }
    method_name = method_names.get(method, method)
    
    def update_stats(result):
        nonlocal correct_count, total_tokens
        if result["correct"]:
            correct_count += 1
        total_tokens += result.get("total_tokens", 0)
        acc = correct_count / len(results_list) * 100 if results_list else 0
        avg_tok = total_tokens / len(results_list) if results_list else 0
        return acc, avg_tok
    
    if method == "grid":
        configs = generate_grid_search_configs(max_configs=num_episodes)
        pbar = tqdm(enumerate(configs[:num_episodes]), total=min(len(configs), num_episodes), 
                    desc=f"  {method_name}")
        for i, config in pbar:
            result = run_fixed_config_episode(structure_env, prompt_env, config)
            results_list.append({"strategy": method_name, "episode": i, **result})
            acc, avg_tok = update_stats(result)
            pbar.set_postfix({"acc": f"{acc:.1f}%", "tokens": f"{avg_tok:.0f}"})
            
    elif method == "greedy":
        configs = generate_greedy_configs()
        pbar = tqdm(enumerate(configs[:num_episodes]), total=min(len(configs), num_episodes),
                    desc=f"  {method_name}")
        for i, config in pbar:
            result = run_fixed_config_episode(structure_env, prompt_env, config)
            results_list.append({"strategy": method_name, "episode": i, **result})
            acc, avg_tok = update_stats(result)
            pbar.set_postfix({"acc": f"{acc:.1f}%", "tokens": f"{avg_tok:.0f}"})
            
    elif method == "best_first":
        # Naive best-first: no expert heuristic, just exploration order
        # Uses a simple counter as priority (FIFO-like exploration)
        # No domain knowledge about which configs are "better"
        
        pq = []
        visited = set()
        counter = [0]  # Simple counter for insertion order (no heuristic)
        
        # Start with random initial configs (no expert knowledge)
        random.seed(42)  # Reproducible
        initial_configs = [
            [random.randint(0, 8), random.randint(0, 15), random.randint(0, 2),
             random.randint(0, 15), random.randint(0, 2), random.randint(0, 2)]
            for _ in range(5)
        ]
        for config in initial_configs:
            heapq.heappush(pq, (counter[0], tuple(config)))
            counter[0] += 1
        
        pbar = tqdm(total=num_episodes, desc=f"  {method_name}")
        while len(results_list) < num_episodes and pq:
            _, config_tuple = heapq.heappop(pq)
            if config_tuple in visited:
                continue
            visited.add(config_tuple)
            config = list(config_tuple)
            result = run_fixed_config_episode(structure_env, prompt_env, config)
            results_list.append({"strategy": method_name, "episode": len(results_list), **result})
            acc, avg_tok = update_stats(result)
            pbar.update(1)
            pbar.set_postfix({"acc": f"{acc:.1f}%", "tokens": f"{avg_tok:.0f}", "queue": len(pq)})
            for neighbor in generate_neighbors(config):
                if tuple(neighbor) not in visited:
                    heapq.heappush(pq, (counter[0], tuple(neighbor)))
                    counter[0] += 1
        pbar.close()
                    
    elif method == "evolutionary":
        pop_size = 20
        generation = 0
        population = [[random.randint(0, 8), random.randint(0, 15), random.randint(0, 2),
                       random.randint(0, 15), random.randint(0, 2), random.randint(0, 2)] 
                      for _ in range(pop_size)]
        population_scores = {}
        
        pbar = tqdm(total=num_episodes, desc=f"  {method_name}")
        
        # Initial population evaluation
        for config in population:
            if len(results_list) >= num_episodes:
                break
            config_key = tuple(config)
            if config_key not in population_scores:
                result = run_fixed_config_episode(structure_env, prompt_env, config)
                population_scores[config_key] = result["reward"]
                results_list.append({"strategy": method_name, "episode": len(results_list), **result})
                acc, avg_tok = update_stats(result)
                pbar.update(1)
                pbar.set_postfix({"acc": f"{acc:.1f}%", "tokens": f"{avg_tok:.0f}", "gen": generation})
        
        # Evolution loop
        while len(results_list) < num_episodes:
            generation += 1
            sorted_pop = sorted(population, key=lambda c: population_scores.get(tuple(c), -10), reverse=True)
            elite = sorted_pop[:pop_size // 2]
            new_population = elite.copy()
            
            while len(new_population) < pop_size:
                if len(elite) >= 2:
                    p1, p2 = random.sample(elite, 2)
                    child = [p1[i] if random.random() < 0.5 else p2[i] for i in range(6)]
                else:
                    child = elite[0].copy() if elite else population[0].copy()
                # Mutate
                for i in range(6):
                    if random.random() < 0.3:
                        if i == 0: child[i] = random.randint(0, 8)
                        elif i in [1, 3]: child[i] = random.randint(0, 15)
                        else: child[i] = random.randint(0, 2)
                new_population.append(child)
            
            for config in new_population[len(elite):]:
                if len(results_list) >= num_episodes:
                    break
                config_key = tuple(config)
                if config_key not in population_scores:
                    result = run_fixed_config_episode(structure_env, prompt_env, config)
                    population_scores[config_key] = result["reward"]
                    results_list.append({"strategy": method_name, "episode": len(results_list), **result})
                    acc, avg_tok = update_stats(result)
                    pbar.update(1)
                    pbar.set_postfix({"acc": f"{acc:.1f}%", "tokens": f"{avg_tok:.0f}", "gen": generation})
            population = new_population
        pbar.close()
    
    return results_list


# ============================================================
# MAIN BENCHMARK FUNCTION
# ============================================================

def run_baseline_benchmark(cfg, baseline, num_episodes=50, verbose=True, 
                           num_workers=1, use_api=False, api_model=None, hf_model=None):
    """Run benchmark for specified baseline(s).
    
    Args:
        cfg: Configuration object
        baseline: Baseline name or group to run
        num_episodes: Number of episodes (or 'all' resolved to int)
        verbose: Print per-episode progress
        num_workers: Number of parallel workers
        use_api: Use OpenRouter API
        api_model: API model name
        hf_model: HuggingFace model name
    """
    print(f"\n{'='*70}")
    print(f"BASELINE BENCHMARK")
    print(f"{'='*70}")
    print(f"  Dataset:   {cfg.DATASET_NAME}")
    print(f"  Baseline:  {baseline}")
    print(f"  Episodes:  {num_episodes}")
    print(f"  Workers:   {num_workers}")
    print(f"{'='*70}\n")
    
    results = []
    detailed_results = []
    
    # Individual search methods that can be selected
    INDIVIDUAL_SEARCH_METHODS = ["grid", "greedy", "best_first", "evolutionary"]
    
    # Determine which configs to run
    if baseline == "all":
        configs_to_run = ALL_BASELINE_CONFIGS
        search_methods = ["grid", "greedy", "best_first", "evolutionary"]
    elif baseline == "search":
        configs_to_run = {}
        search_methods = ["grid", "greedy", "best_first", "evolutionary"]
    elif baseline in INDIVIDUAL_SEARCH_METHODS:
        # Run only the specified search method
        configs_to_run = {}
        search_methods = [baseline]
    elif baseline in BASELINE_GROUPS:
        configs_to_run = BASELINE_GROUPS[baseline]
        search_methods = []
    elif baseline in ALL_BASELINE_CONFIGS:
        configs_to_run = {baseline: ALL_BASELINE_CONFIGS[baseline]}
        search_methods = []
    else:
        print(f"Unknown baseline: {baseline}")
        print("Use --list-baselines to see available options.")
        print("\nAvailable search methods: grid, greedy, best_first, evolutionary")
        return None, None
    
    # Run fixed configurations
    for config_name, fixed_action in configs_to_run.items():
        print(f"\n[{config_name}]")
        start_time = time.time()
        
        if num_workers > 1 and use_api:
            # Parallel evaluation
            config_results = run_config_parallel(
                cfg, fixed_action, config_name, num_episodes, num_workers,
                use_api, api_model, hf_model
            )
            detailed_results.extend(config_results)
            
            accuracies = [1 if r["correct"] else 0 for r in config_results]
            rewards = [r["reward"] for r in config_results]
            tokens = [r["total_tokens"] for r in config_results]
        else:
            # Sequential evaluation
            print("Initializing environments...", end=" ", flush=True)
            with SuppressOutput():
                structure_env = StructureEnv(cfg, use_api=use_api, api_model=api_model, hf_model=hf_model)
                prompt_env = PromptEnv(cfg, use_api=use_api, api_model=api_model, hf_model=hf_model)
            print("Done!")
            
            accuracies = []
            rewards = []
            tokens = []
            
            pbar = tqdm(range(num_episodes), desc=f"  {config_name}", leave=True)
            for ep in pbar:
                result = run_fixed_config_episode(structure_env, prompt_env, fixed_action)
                
                accuracies.append(1 if result["correct"] else 0)
                rewards.append(result["reward"])
                tokens.append(result["total_tokens"])
                
                detailed_results.append({
                    "strategy": config_name,
                    "search_method": "Fixed Config",
                    "episode": ep,
                    **result
                })
                
                # Update progress bar
                acc_so_far = np.mean(accuracies)
                pbar.set_postfix({"acc": f"{acc_so_far:.1%}", "tokens": f"{np.mean(tokens):.0f}"})
                
                if (ep + 1) % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
        
        elapsed = time.time() - start_time
        avg_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        avg_reward = np.mean(rewards)
        avg_tokens = np.mean(tokens)
        
        results.append({
            "Strategy": config_name,
            "Accuracy": f"{avg_acc:.1%}",
            "Std": f"{std_acc:.3f}",
            "Avg Reward": f"{avg_reward:.3f}",
            "Avg Tokens": f"{avg_tokens:.0f}",
            "Time (s)": f"{elapsed:.1f}",
        })
        
        print(f"  Final: Accuracy={avg_acc:.1%} +/- {std_acc:.3f}, Reward={avg_reward:.3f}, Tokens={avg_tokens:.0f}, Time={elapsed:.1f}s")
    
    # Run search algorithms
    if search_methods:
        # For parallel methods (grid/greedy), we'll create workers
        # For sequential methods (best_first/evolutionary), we use single env
        parallel_methods = ["grid", "greedy"]  # These can run in parallel
        sequential_methods = ["best_first", "evolutionary"]  # These require sequential execution
        
        # Check if we have any sequential methods
        has_sequential = any(m in search_methods for m in sequential_methods)
        if has_sequential:
            print("\nInitializing environment for sequential search algorithms...", end=" ", flush=True)
            with SuppressOutput():
                structure_env = StructureEnv(cfg, use_api=use_api, api_model=api_model, hf_model=hf_model)
                prompt_env = PromptEnv(cfg, use_api=use_api, api_model=api_model, hf_model=hf_model)
            print("Done!")
        
    for method in search_methods:
        method_name = {"grid": "Grid Search", "greedy": "Greedy Search", 
                       "best_first": "Best-First Search", "evolutionary": "Evolutionary Search"}[method]
        print(f"\n[{method_name}] - {num_episodes} episodes")
        start_time = time.time()
        
        # Use parallel execution for grid/greedy when workers > 1 and API mode
        if method in ["grid", "greedy"] and num_workers > 1 and use_api:
            search_results = run_search_parallel(
                cfg, method, num_episodes, num_workers,
                use_api, api_model, hf_model
            )
        else:
            # Sequential execution
            if method in ["grid", "greedy"] and not has_sequential:
                # Need to create env for sequential grid/greedy
                print("  Initializing environment...", end=" ", flush=True)
                with SuppressOutput():
                    structure_env = StructureEnv(cfg, use_api=use_api, api_model=api_model, hf_model=hf_model)
                    prompt_env = PromptEnv(cfg, use_api=use_api, api_model=api_model, hf_model=hf_model)
                print("Done!")
            search_results = run_search_algorithm(structure_env, prompt_env, method, num_episodes)
        
        detailed_results.extend(search_results)
        
        if search_results:
            accuracies = [1 if r["correct"] else 0 for r in search_results]
            rewards = [r["reward"] for r in search_results]
            tokens = [r["total_tokens"] for r in search_results]
            
            elapsed = time.time() - start_time
            avg_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            avg_reward = np.mean(rewards)
            avg_tokens = np.mean(tokens)
            
            print(f"  ✓ Complete: Accuracy={avg_acc:.1%} ± {std_acc:.3f}, Tokens={avg_tokens:.0f}, Time={elapsed:.1f}s")
            
            results.append({
                "Strategy": method_name,
                "Accuracy": f"{avg_acc:.1%}",
                "Std": f"{std_acc:.3f}",
                "Avg Reward": f"{avg_reward:.3f}",
                "Avg Tokens": f"{avg_tokens:.0f}",
                "Time (s)": f"{elapsed:.1f}",
            })
    
    # Print results table
    if results:
        df = pd.DataFrame(results)
        df["Accuracy_Num"] = df["Accuracy"].str.rstrip("%").astype(float) / 100
        df = df.sort_values("Accuracy_Num", ascending=False)
        df = df.drop("Accuracy_Num", axis=1)
        
        print(f"\n{'='*70}")
        print("RESULTS SUMMARY (sorted by accuracy)")
        print("="*70)
        print(df.to_string(index=False))
        print("="*70)
        
        # Save results
        os.makedirs("results", exist_ok=True)
        timestamp = int(time.time())
        output_path = f"results/benchmark_{cfg.DATASET_NAME}_{baseline}_{timestamp}.csv"
        pd.DataFrame(detailed_results).to_csv(output_path, index=False)
        print(f"\nDetailed results saved to: {output_path}")
        
        return df, pd.DataFrame(detailed_results)
    
    return None, None


def run_config_parallel(cfg, fixed_action, config_name, num_episodes, num_workers,
                        use_api, api_model, hf_model):
    """Run a single configuration with parallel workers.
    
    Each worker has its own environment pair to avoid thread conflicts.
    """
    print(f"  Pre-creating {num_workers} worker pairs...")
    
    # Create worker environments
    workers = []
    workers_lock = threading.Lock()
    
    for i in range(num_workers):
        print(f"    Creating worker pair {i+1}/{num_workers}...", end="\r")
        with SuppressOutput():
            struct_env = StructureEnv(cfg, use_api=use_api, api_model=api_model, hf_model=hf_model)
            prompt_env = PromptEnv(cfg, use_api=use_api, api_model=api_model, hf_model=hf_model)
        workers.append((struct_env, prompt_env))
    print(f"    Done! All {num_workers} worker pairs ready.              ")
    
    # Pre-initialize embedders to avoid race conditions during parallel execution
    print(f"    Pre-initializing embedders...", end="\r")
    with SuppressOutput():
        for struct_env, prompt_env in workers:
            for env in [struct_env, prompt_env]:
                if hasattr(env, 'worker') and hasattr(env.worker, 'embedder'):
                    embedder = env.worker.embedder
                    if hasattr(embedder, '_init_embedder') and not getattr(embedder, '_initialized', False):
                        try:
                            embedder._init_embedder()
                        except Exception:
                            pass  # Suppress errors during pre-init
    print(f"    Pre-initialized embedders.                              ")
    
    # Thread-local worker assignment
    worker_assignment = {}
    worker_idx = [0]  # Use list to allow modification in nested function
    
    def get_worker():
        thread_id = threading.current_thread().ident
        with workers_lock:
            if thread_id not in worker_assignment:
                worker_assignment[thread_id] = workers[worker_idx[0] % len(workers)]
                worker_idx[0] += 1
            return worker_assignment[thread_id]
    
    def worker_fn(ep_idx):
        struct_env, prompt_env = get_worker()
        result = run_fixed_config_episode(struct_env, prompt_env, fixed_action)
        return {
            "strategy": config_name,
            "search_method": "Fixed Config",
            "episode": ep_idx,
            **result
        }
    
    results = []
    correct_count = 0
    total_tokens = 0
    
    print(f"  Evaluating {num_episodes} episodes with {num_workers} parallel workers...")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(worker_fn, ep): ep for ep in range(num_episodes)}
        
        with tqdm(total=num_episodes, desc=f"  {config_name} ({num_workers} workers)") as pbar:
            for future in as_completed(futures):
                ep = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result["correct"]:
                        correct_count += 1
                    total_tokens += result.get("total_tokens", 0)
                    
                    # Update progress bar
                    acc = correct_count / len(results) * 100
                    avg_tok = total_tokens / len(results)
                    pbar.set_postfix({"acc": f"{acc:.1f}%", "tokens": f"{avg_tok:.0f}"})
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\n  Error on episode {ep}: {e}")
                    results.append({
                        "strategy": config_name,
                        "search_method": "Fixed Config",
                        "episode": ep,
                        "correct": False,
                        "reward": -1.0,
                        "workflow": "Error",
                        "total_tokens": 0,
                        "error": str(e)
                    })
                    pbar.update(1)
    
    # Sort by episode for consistency
    results.sort(key=lambda x: x["episode"])
    return results


def run_search_parallel(cfg, method, num_episodes, num_workers,
                        use_api, api_model, hf_model):
    """Run grid or greedy search with parallel workers.
    
    Each worker has its own environment pair to avoid thread conflicts.
    Only works for grid and greedy search (independent per config).
    """
    method_names = {
        "grid": "Grid Search",
        "greedy": "Greedy Search"
    }
    method_name = method_names.get(method, method)
    
    # Generate configs based on method
    if method == "grid":
        configs = generate_grid_search_configs(max_configs=num_episodes)[:num_episodes]
    elif method == "greedy":
        configs = generate_greedy_configs()[:num_episodes]
    else:
        raise ValueError(f"Parallel search not supported for method: {method}")
    
    actual_episodes = len(configs)
    print(f"  Pre-creating {num_workers} worker pairs...")
    
    # Create worker environments
    workers = []
    workers_lock = threading.Lock()
    
    for i in range(num_workers):
        print(f"    Creating worker pair {i+1}/{num_workers}...", end="\r")
        with SuppressOutput():
            struct_env = StructureEnv(cfg, use_api=use_api, api_model=api_model, hf_model=hf_model)
            prompt_env = PromptEnv(cfg, use_api=use_api, api_model=api_model, hf_model=hf_model)
        workers.append((struct_env, prompt_env))
    print(f"    Done! All {num_workers} worker pairs ready.              ")
    
    # Pre-initialize embedders to avoid race conditions during parallel execution
    print(f"    Pre-initializing embedders...", end="\r")
    with SuppressOutput():
        for struct_env, prompt_env in workers:
            for env in [struct_env, prompt_env]:
                if hasattr(env, 'worker') and hasattr(env.worker, 'embedder'):
                    embedder = env.worker.embedder
                    if hasattr(embedder, '_init_embedder') and not getattr(embedder, '_initialized', False):
                        try:
                            embedder._init_embedder()
                        except Exception:
                            pass
    print(f"    Pre-initialized embedders.                              ")
    
    # Thread-local worker assignment
    worker_assignment = {}
    worker_idx = [0]
    
    def get_worker():
        thread_id = threading.current_thread().ident
        with workers_lock:
            if thread_id not in worker_assignment:
                worker_assignment[thread_id] = workers[worker_idx[0] % len(workers)]
                worker_idx[0] += 1
            return worker_assignment[thread_id]
    
    def worker_fn(idx):
        config = configs[idx]
        struct_env, prompt_env = get_worker()
        result = run_fixed_config_episode(struct_env, prompt_env, config)
        return {
            "strategy": method_name,
            "episode": idx,
            "config": format_config(config),
            **result
        }
    
    results = []
    correct_count = 0
    total_tokens = 0
    
    print(f"  Evaluating {actual_episodes} configs with {num_workers} parallel workers...")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(worker_fn, i): i for i in range(actual_episodes)}
        
        with tqdm(total=actual_episodes, desc=f"  {method_name} ({num_workers} workers)") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result["correct"]:
                        correct_count += 1
                    total_tokens += result.get("total_tokens", 0)
                    
                    acc = correct_count / len(results) * 100
                    avg_tok = total_tokens / len(results)
                    pbar.set_postfix({"acc": f"{acc:.1f}%", "tokens": f"{avg_tok:.0f}"})
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\n  Error on config {idx}: {e}")
                    results.append({
                        "strategy": method_name,
                        "episode": idx,
                        "correct": False,
                        "reward": -1.0,
                        "workflow": "Error",
                        "total_tokens": 0,
                        "error": str(e)
                    })
                    pbar.update(1)
    
    # Sort by episode for consistency
    results.sort(key=lambda x: x["episode"])
    return results


def list_baselines():
    """Print all available baseline configurations."""
    print("\n" + "="*70)
    print("AVAILABLE BASELINES")
    print("="*70)
    
    print("\nBaseline Groups (use with --baseline):")
    print("  all            - Run all fixed configs + search algorithms")
    print("  search         - Run all search algorithms (grid, greedy, best_first, evolutionary)")
    print("")
    print("Individual Search Methods:")
    print("  grid           - Grid Search only (parallel supported)")
    print("  greedy         - Greedy Search only (parallel supported)")
    print("  best_first     - Best-First Search only (sequential)")
    print("  evolutionary   - Evolutionary Search only (sequential)")
    print("")
    print("Workflow Groups:")
    print("  direct         - Direct workflow (workflow=0)")
    print("  reason         - Reason+Answer workflow (workflow=1)")
    print("  verify         - Reason+Verify+Answer workflow (workflow=2)")
    print("  routing        - Routing workflow (workflow=3)")
    print("  parallel_section - Parallel-Sectioning workflow (workflow=4)")
    print("  parallel_vote  - Parallel-Voting workflow (workflow=5)")
    print("  orchestrator   - Orchestrator-Workers workflow (workflow=6)")
    print("  eval_opt       - Evaluator-Optimizer workflow (workflow=7)")
    print("  autonomous     - Autonomous-Agent workflow (workflow=8)")
    
    print("\n" + "-"*70)
    print("WORKFLOW DESCRIPTIONS:")
    print("-"*70)
    for i, name in enumerate(WORKFLOW_NAMES):
        uses_agent2 = "Yes" if i in WORKFLOWS_WITH_AGENT2 else "No"
        answerer_only = "Yes" if i in WORKFLOWS_ANSWERER_ONLY else "No"
        print(f"  {i}: {name:<25} uses_agent2={uses_agent2:<3}  answerer_only={answerer_only}")
    
    print("\n" + "-"*70)
    print("TOOL ENCODING (binary):")
    print("-"*70)
    print("  0  = NoTools")
    print("  1  = calculator")
    print("  2  = web_search")
    print("  3  = calculator + web_search")
    print("  4  = python")
    print("  7  = calculator + web_search + python")
    print("  8  = ocr_reader")
    print("  15 = AllTools (calculator + web_search + python + ocr_reader)")
    
    print("\n" + "-"*70)
    print("INDIVIDUAL CONFIGS:")
    print("-"*70)
    
    for group_name, configs in BASELINE_GROUPS.items():
        print(f"\n  [{group_name}]")
        for name, config in configs.items():
            wf = WORKFLOW_NAMES[config[0]]
            tools = decode_tools(config[1])
            budget = BUDGET_NAMES[config[2]]
            print(f"    {name:<30} wf={wf:<20} tools={'+'.join(tools):<15} budget={budget}")
    
    print("\n" + "="*70)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline Benchmark Script (no RL policies)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all baselines on full dataset
  python scripts/config_search_benchmark.py --dataset hotpotqa --episodes all --api --api-model "gemma" --workers 8

  # Run only Direct workflow baselines with 50 episodes
  python scripts/config_search_benchmark.py --baseline direct --dataset hotpotqa --episodes 50 --api --api-model "gemma"

  # Run only Autonomous workflow baselines on full dataset
  python scripts/config_search_benchmark.py --baseline autonomous --dataset hotpotqa --episodes all --api --api-model "gemma" --workers 8

  # Run a specific config
  python scripts/config_search_benchmark.py --baseline "Direct-Web-Mid" --dataset hotpotqa --episodes all --api --api-model "gemma" --workers 8

  # Run all search algorithms
  python scripts/config_search_benchmark.py --baseline search --dataset hotpotqa --episodes 50 --api --api-model "gemma"

  # Run only Grid Search (parallel supported)
  python scripts/config_search_benchmark.py --baseline grid --dataset drop --episodes all --api --api-model "qwen/qwen-2.5-7b-instruct" --workers 8

  # Run only Greedy Search (parallel supported)
  python scripts/config_search_benchmark.py --baseline greedy --dataset drop --episodes all --api --api-model "qwen/qwen-2.5-7b-instruct" --workers 8

  # Run only Best-First Search (sequential)
  python scripts/config_search_benchmark.py --baseline best_first --dataset drop --episodes 100 --api --api-model "qwen/qwen-2.5-7b-instruct"

  # Run only Evolutionary Search (sequential)
  python scripts/config_search_benchmark.py --baseline evolutionary --dataset drop --episodes 100 --api --api-model "qwen/qwen-2.5-7b-instruct"

  # List available baselines
  python scripts/config_search_benchmark.py --list-baselines
"""
    )
    
    parser.add_argument("--baseline", type=str, default="all",
                       help="Baseline to run: all, search, grid, greedy, best_first, evolutionary, direct, reason, verify, routing, parallel_section, parallel_vote, orchestrator, eval_opt, autonomous, or specific config name")
    from utils import validate_dataset_name, get_dataset_help_text
    
    parser.add_argument("--dataset", type=validate_dataset_name, default="hotpotqa",
                       help=get_dataset_help_text())
    parser.add_argument("--episodes", type=str, default="50",
                       help="Number of episodes per configuration, or 'all' for full dataset")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of parallel workers (requires --api)")
    parser.add_argument("--list-baselines", action="store_true",
                       help="List all available baseline configurations and exit")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")
    
    # API configuration
    parser.add_argument("--api", action="store_true",
                       help="Use OpenRouter API instead of local HuggingFace models")
    parser.add_argument("--api-model", type=str, default=None,
                       help="OpenRouter model ID (e.g., 'gemma', 'gpt-4o')")
    parser.add_argument("--hf-model", type=str, default=None,
                       help="HuggingFace model name for local inference")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.list_baselines:
        list_baselines()
        return
    
    # Load config
    cfg = load_config("hierarchical")
    cfg.DATASET_NAME = args.dataset
    
    # Handle "all" episodes - get dataset size
    if args.episodes.lower() == "all":
        dataset = get_dataset_loader(cfg.DATASET_NAME, is_eval=True)
        if hasattr(dataset, 'tasks'):
            num_episodes = len(dataset.tasks)
        elif hasattr(dataset, 'data'):
            num_episodes = len(dataset.data)
        else:
            num_episodes = len(dataset)
        print(f"\nEvaluating on ALL {num_episodes} datapoints from {cfg.DATASET_NAME}")
    else:
        num_episodes = int(args.episodes)
    
    # Validate workers
    if args.workers > 1 and not args.api:
        print("Warning: Parallel workers require --api mode. Using workers=1.")
        args.workers = 1
    
    # Print mode info
    if args.api:
        model_name = args.api_model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
        print(f"\nMode: API (OpenRouter)")
        print(f"Model: {model_name}")
        if args.workers > 1:
            print(f"Workers: {args.workers} (parallel)")
    else:
        model_name = args.hf_model or "default"
        print(f"\nMode: HuggingFace (local)")
        print(f"Model: {model_name}")
    
    # Run benchmark
    run_baseline_benchmark(
        cfg=cfg,
        baseline=args.baseline,
        num_episodes=num_episodes,
        verbose=not args.quiet,
        num_workers=args.workers,
        use_api=args.api,
        api_model=args.api_model,
        hf_model=args.hf_model
    )


if __name__ == "__main__":
    main()
