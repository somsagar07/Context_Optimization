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
    # Run all baselines on hotpotqa
    python scripts/config_search_benchmark.py --dataset hotpotqa --episodes 50 --api --api-model "gemma"
    
    # Run only Direct workflow baselines
    python scripts/config_search_benchmark.py --baseline direct --dataset hotpotqa --episodes 50 --api --api-model "gemma"
    
    # Run only search algorithms (Grid, Greedy, Best-First, Evolutionary)
    python scripts/config_search_benchmark.py --baseline search --dataset hotpotqa --episodes 50 --api --api-model "gemma"
    
    # Run a specific fixed config
    python scripts/config_search_benchmark.py --baseline "Direct-Web-Mid" --dataset hotpotqa --episodes 50 --api --api-model "gemma"
    
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
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

import numpy as np
import pandas as pd
import torch

from configs import load_config
from env import StructureEnv, PromptEnv


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
    """Generate configurations for grid search over key dimensions."""
    configs = []
    workflows = list(range(9))  # All 9 workflows
    tool_sets = [0, 1, 2, 3, 15]  # None, Calc, Web, Calc+Web, All
    budgets = [0, 1, 2]
    
    for wf in workflows:
        for r_tools in tool_sets:
            for r_budget in budgets:
                if wf in WORKFLOWS_WITH_AGENT2:  # Workflows that use agent2
                    for v_tools in tool_sets:
                        for v_budget in budgets:
                            for a_budget in budgets:
                                configs.append([wf, r_tools, r_budget, v_tools, v_budget, a_budget])
                                if len(configs) >= max_configs:
                                    return configs
                else:
                    for a_budget in budgets:
                        configs.append([wf, r_tools, r_budget, 0, 0, a_budget])
                        if len(configs) >= max_configs:
                            return configs
    return configs


def generate_greedy_configs():
    """Generate configurations ordered by expected quality (heuristic)."""
    configs = []
    
    # Priority: More complex workflows with tools, then simpler ones
    # Complex workflows first
    for wf in [8, 7, 6, 2, 4, 3, 1, 5, 0]:  # Autonomous -> Direct
        for tools in [15, 3, 2, 1, 0]:  # All -> None
            for budget in [2, 1, 0]:  # High -> Low
                if wf in WORKFLOWS_WITH_AGENT2:
                    configs.append([wf, tools, budget, tools, budget, budget])
                else:
                    configs.append([wf, tools, budget, 0, 0, budget])
    
    return configs[:50]


def generate_neighbors(config):
    """Generate neighboring configurations."""
    neighbors = []
    for i in range(len(config)):
        if i == 0:  # workflow
            for val in range(9):
                if val != config[i]:
                    neighbor = config.copy()
                    neighbor[i] = val
                    neighbors.append(neighbor)
        elif i in [1, 3]:  # tools
            for tool_bit in [1, 2, 4, 8]:
                neighbor = config.copy()
                neighbor[i] = int(config[i]) ^ tool_bit
                neighbors.append(neighbor)
        else:  # budgets
            for val in [0, 1, 2]:
                if val != config[i]:
                    neighbor = config.copy()
                    neighbor[i] = val
                    neighbors.append(neighbor)
    return neighbors[:10]


def run_search_algorithm(structure_env, prompt_env, method, num_episodes=50):
    """Run a search algorithm and return results."""
    import random
    results_list = []
    
    if method == "grid":
        configs = generate_grid_search_configs(max_configs=num_episodes)
        for i, config in enumerate(configs[:num_episodes]):
            result = run_fixed_config_episode(structure_env, prompt_env, config)
            results_list.append({"strategy": "Grid Search", "episode": i, **result})
            
    elif method == "greedy":
        configs = generate_greedy_configs()
        for i, config in enumerate(configs[:num_episodes]):
            result = run_fixed_config_episode(structure_env, prompt_env, config)
            results_list.append({"strategy": "Greedy Search", "episode": i, **result})
            
    elif method == "best_first":
        def heuristic(config):
            # Prefer complex workflows with tools
            score = config[0] * 1.5  # workflow complexity
            score += bin(config[1]).count('1') * 2  # agent1 tools count
            score += bin(config[3]).count('1')  # agent2 tools count
            score += config[2] + config[4] + config[5]  # budgets
            return score
        
        pq = []
        visited = set()
        # Start with promising configs across different workflows
        initial_configs = [
            [8, 15, 2, 0, 0, 2],   # Autonomous, all tools, high
            [7, 15, 2, 15, 2, 2],  # EvalOpt, all tools, high
            [2, 15, 2, 15, 2, 2],  # Verify, all tools, high
            [1, 15, 2, 0, 0, 2],   # Reason, all tools, high
            [0, 3, 2, 0, 0, 2],    # Direct, calc+web, high
        ]
        for config in initial_configs:
            heapq.heappush(pq, (-heuristic(config), tuple(config)))
        
        while len(results_list) < num_episodes and pq:
            _, config_tuple = heapq.heappop(pq)
            if config_tuple in visited:
                continue
            visited.add(config_tuple)
            config = list(config_tuple)
            result = run_fixed_config_episode(structure_env, prompt_env, config)
            results_list.append({"strategy": "Best-First Search", "episode": len(results_list), **result})
            for neighbor in generate_neighbors(config):
                if tuple(neighbor) not in visited:
                    heapq.heappush(pq, (-heuristic(neighbor), tuple(neighbor)))
                    
    elif method == "evolutionary":
        pop_size = 20
        population = [[random.randint(0, 8), random.randint(0, 15), random.randint(0, 2),
                       random.randint(0, 15), random.randint(0, 2), random.randint(0, 2)] 
                      for _ in range(pop_size)]
        population_scores = {}
        
        for config in population:
            if len(results_list) >= num_episodes:
                break
            config_key = tuple(config)
            if config_key not in population_scores:
                result = run_fixed_config_episode(structure_env, prompt_env, config)
                population_scores[config_key] = result["reward"]
                results_list.append({"strategy": "Evolutionary Search", "episode": len(results_list), **result})
        
        while len(results_list) < num_episodes:
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
                    results_list.append({"strategy": "Evolutionary Search", "episode": len(results_list), **result})
            population = new_population
    
    return results_list


# ============================================================
# MAIN BENCHMARK FUNCTION
# ============================================================

def run_baseline_benchmark(cfg, baseline, num_episodes=50, verbose=True):
    """Run benchmark for specified baseline(s)."""
    print(f"\n{'='*70}")
    print(f"BASELINE BENCHMARK")
    print(f"{'='*70}")
    print(f"  Dataset:   {cfg.DATASET_NAME}")
    print(f"  Baseline:  {baseline}")
    print(f"  Episodes:  {num_episodes}")
    print(f"{'='*70}\n")
    
    results = []
    detailed_results = []
    
    # Initialize environments
    print("Initializing environments...", end=" ", flush=True)
    with SuppressOutput():
        structure_env = StructureEnv(cfg)
        prompt_env = PromptEnv(cfg)
    print("Done!\n")
    
    # Determine which configs to run
    if baseline == "all":
        configs_to_run = ALL_BASELINE_CONFIGS
        search_methods = ["grid", "greedy", "best_first", "evolutionary"]
    elif baseline == "search":
        configs_to_run = {}
        search_methods = ["grid", "greedy", "best_first", "evolutionary"]
    elif baseline in BASELINE_GROUPS:
        configs_to_run = BASELINE_GROUPS[baseline]
        search_methods = []
    elif baseline in ALL_BASELINE_CONFIGS:
        configs_to_run = {baseline: ALL_BASELINE_CONFIGS[baseline]}
        search_methods = []
    else:
        print(f"Unknown baseline: {baseline}")
        print("Use --list-baselines to see available options.")
        return None, None
    
    # Run fixed configurations
    for config_name, fixed_action in configs_to_run.items():
        print(f"[{config_name}]")
        start_time = time.time()
        
        accuracies = []
        rewards = []
        tokens = []
        
        for ep in range(num_episodes):
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
            
            if verbose:
                acc_so_far = np.mean(accuracies)
                status = "Y" if result["correct"] else "X"
                print(f"\r  {ep+1}/{num_episodes} | Acc: {acc_so_far:.1%} | Last: {status}", end="", flush=True)
            
            if (ep + 1) % 10 == 0 and torch.cuda.is_available():
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
        
        print(f"\r  Final: Accuracy={avg_acc:.1%} +/- {std_acc:.3f}, Reward={avg_reward:.3f}, Tokens={avg_tokens:.0f}")
    
    # Run search algorithms
    for method in search_methods:
        method_name = {"grid": "Grid Search", "greedy": "Greedy Search", 
                       "best_first": "Best-First Search", "evolutionary": "Evolutionary Search"}[method]
        print(f"\n[{method_name}]")
        start_time = time.time()
        
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
            
            results.append({
                "Strategy": method_name,
                "Accuracy": f"{avg_acc:.1%}",
                "Std": f"{std_acc:.3f}",
                "Avg Reward": f"{avg_reward:.3f}",
                "Avg Tokens": f"{avg_tokens:.0f}",
                "Time (s)": f"{elapsed:.1f}",
            })
            
            print(f"  Final: Accuracy={avg_acc:.1%} +/- {std_acc:.3f}, Reward={avg_reward:.3f}, Tokens={avg_tokens:.0f}")
    
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
        timestamp = int(time.time())
        output_path = f"benchmark_{cfg.DATASET_NAME}_{baseline}_{timestamp}.csv"
        pd.DataFrame(detailed_results).to_csv(output_path, index=False)
        print(f"\nDetailed results saved to: {output_path}")
        
        return df, pd.DataFrame(detailed_results)
    
    return None, None


def list_baselines():
    """Print all available baseline configurations."""
    print("\n" + "="*70)
    print("AVAILABLE BASELINES")
    print("="*70)
    
    print("\nBaseline Groups (use with --baseline):")
    print("  all            - Run all fixed configs + search algorithms")
    print("  search         - Run only search algorithms")
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
  # Run all baselines
  python scripts/config_search_benchmark.py --dataset hotpotqa --episodes 50 --api --api-model "gemma"

  # Run only Direct workflow baselines
  python scripts/config_search_benchmark.py --baseline direct --dataset hotpotqa --episodes 50 --api --api-model "gemma"

  # Run only Autonomous workflow baselines
  python scripts/config_search_benchmark.py --baseline autonomous --dataset hotpotqa --episodes 50 --api --api-model "gemma"

  # Run a specific config
  python scripts/config_search_benchmark.py --baseline "Direct-Web-Mid" --dataset hotpotqa --episodes 50 --api --api-model "gemma"

  # Run only search algorithms
  python scripts/config_search_benchmark.py --baseline search --dataset hotpotqa --episodes 50 --api --api-model "gemma"

  # List available baselines
  python scripts/config_search_benchmark.py --list-baselines
"""
    )
    
    parser.add_argument("--baseline", type=str, default="all",
                       help="Baseline to run: all, search, direct, reason, verify, routing, parallel_section, parallel_vote, orchestrator, eval_opt, autonomous, or specific config name")
    from utils import validate_dataset_name, get_dataset_help_text
    
    parser.add_argument("--dataset", type=validate_dataset_name, default="hotpotqa",
                       help=get_dataset_help_text(include_tau2=False))
    parser.add_argument("--episodes", type=int, default=50,
                       help="Number of episodes per configuration")
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
    
    # Print mode info
    if args.api:
        model_name = args.api_model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
        print(f"\nMode: API (OpenRouter)")
        print(f"Model: {model_name}")
    else:
        model_name = args.hf_model or "default"
        print(f"\nMode: HuggingFace (local)")
        print(f"Model: {model_name}")
    
    # Patch environment initialization to use API mode
    original_init = StructureEnv.__init__
    def patched_struct_init(self, cfg, is_eval=False, use_api=False, api_model=None, hf_model=None):
        original_init(self, cfg, is_eval=is_eval, use_api=args.api, api_model=args.api_model, hf_model=args.hf_model)
    StructureEnv.__init__ = patched_struct_init
    
    original_prompt_init = PromptEnv.__init__
    def patched_prompt_init(self, cfg, is_eval=False, use_api=False, api_model=None, hf_model=None):
        original_prompt_init(self, cfg, is_eval=is_eval, use_api=args.api, api_model=args.api_model, hf_model=args.hf_model)
    PromptEnv.__init__ = patched_prompt_init
    
    # Run benchmark
    run_baseline_benchmark(
        cfg=cfg,
        baseline=args.baseline,
        num_episodes=args.episodes,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
