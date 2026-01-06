"""
Benchmark Script

Supports:
- Hierarchical RL models (structure + prompt policies, .pt files)
- SFT post-trained models (structure + prompt policies, .pt files)
- Single-step RL models (Stable-Baselines3, .zip files)
- Multi-step RL models (Stable-Baselines3, .zip files)
- Multiple search algorithms (Grid, Greedy, Best-First, Evolutionary)
- Fixed baseline configurations
- Random baselines

Usage:
    # Hierarchical/SFT models with all search algorithms
    python benchmark.py --structure-model models/structure_policy.pt --prompt-model models/prompt_policy.pt --episodes 50
    
    # Include single-step and multi-step models for comparison
    python benchmark.py --structure-model models/structure_policy.pt --prompt-model models/prompt_policy.pt \
                        --single-step-model models/controller_single_step.zip \
                        --multi-step-model models/controller_multi_step.zip
    
    # Just search algorithms (no learned models)
    python benchmark.py --episodes 50
    
    # Specific dataset
    python benchmark.py --dataset gsm8k --episodes 50
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import gc
import heapq
import sys
import time
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.distributions import Categorical

from configs import load_config
from env import GeneralAgentEnv, MultiStepAgentEnv, StructureEnv, PromptEnv


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


class PolicyNetwork(nn.Module):
    """Simple policy network for discrete actions."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, has_value: bool = False):
        super().__init__()
        self.has_value = has_value
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        if has_value:
            self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        features = self.network(x)
        action_logits = self.action_head(features)
        if self.has_value:
            value = self.value_head(features)
            return action_logits, value
        return action_logits, None
    
    def get_action(self, obs, deterministic=False):
        """Get action from observation."""
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        device = next(self.parameters()).device
        obs = obs.to(device)
            
        action_logits, value = self.forward(obs)
        probs = torch.softmax(action_logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = Categorical(probs)
            action = dist.sample()
        
        return action.item()


class MultiDiscretePolicy(nn.Module):
    """Policy network for MultiDiscrete action space [3, 8, 3, 8, 3, 3]."""
    
    def __init__(self, obs_dim: int, action_dims: list, hidden_dim: int = 256, has_value: bool = False):
        super().__init__()
        self.action_dims = action_dims
        self.has_value = has_value
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.action_heads = nn.ModuleList([
            nn.Linear(hidden_dim, dim) for dim in action_dims
        ])
        if has_value:
            self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        features = self.network(x)
        action_logits = [head(features) for head in self.action_heads]
        if self.has_value:
            value = self.value_head(features)
            return action_logits, value
        return action_logits, None
    
    def get_action(self, obs, deterministic=False):
        """Get MultiDiscrete action from observation."""
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        device = next(self.parameters()).device
        obs = obs.to(device)
            
        action_logits_list, value = self.forward(obs)
        
        actions = []
        for logits in action_logits_list:
            probs = torch.softmax(logits, dim=-1)
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                dist = Categorical(probs)
                action = dist.sample()
            actions.append(action.item())
        
        return np.array(actions)


def load_hierarchical_models(structure_path: str, prompt_path: str, device="cpu"):
    """Load hierarchical structure and prompt policies."""
    # Load structure policy
    struct_checkpoint = torch.load(structure_path, map_location=device, weights_only=False)
    struct_obs_dim = struct_checkpoint["obs_dim"]
    struct_action_dims = struct_checkpoint["action_dims"]
    struct_algo = struct_checkpoint.get("algorithm", "PPO")
    struct_has_value = "PPO" in struct_algo
    
    structure_policy = MultiDiscretePolicy(struct_obs_dim, struct_action_dims, has_value=struct_has_value).to(device)
    structure_policy.load_state_dict(struct_checkpoint["model_state_dict"])
    structure_policy.eval()
    
    # Load prompt policy
    prompt_checkpoint = torch.load(prompt_path, map_location=device, weights_only=False)
    prompt_obs_dim = prompt_checkpoint["obs_dim"]
    prompt_action_dim = prompt_checkpoint["action_dim"]
    prompt_algo = prompt_checkpoint.get("algorithm", "PPO")
    prompt_has_value = "PPO" in prompt_algo
    
    prompt_policy = PolicyNetwork(prompt_obs_dim, prompt_action_dim, has_value=prompt_has_value).to(device)
    prompt_policy.load_state_dict(prompt_checkpoint["model_state_dict"])
    prompt_policy.eval()
    
    # Clear cache after loading
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return structure_policy, prompt_policy


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
    workflow = ["Direct", "Reason+Ans", "Reason+Verify+Ans"][action[0]]
    reasoner_tools = decode_tools(action[1])
    reasoner_budget = ["Low", "Mid", "High"][action[2]]
    verifier_tools = decode_tools(action[3])
    verifier_budget = ["Low", "Mid", "High"][action[4]]
    answerer_budget = ["Low", "Mid", "High"][action[5]]
    
    return f"{workflow}|R:{'+'.join(reasoner_tools)},{reasoner_budget}|V:{'+'.join(verifier_tools)},{verifier_budget}|A:{answerer_budget}"


def run_hierarchical_episode(structure_policy, prompt_policy, structure_env, prompt_env, deterministic=True):
    """Run a single hierarchical episode using provided environments."""
    # Step 1: Structure decision
    struct_obs, struct_info = structure_env.reset()
    with torch.no_grad():
        struct_action = structure_policy.get_action(struct_obs, deterministic=deterministic)
    _, _, _, _, struct_exec_info = structure_env.step(struct_action)
    
    # Step 2: Set structure in prompt env
    prompt_env.set_structure(
        question=struct_exec_info["question"],
        answer=struct_exec_info["answer"],
        embedding=struct_exec_info["embedding"],
        structure=struct_exec_info["structure"]
    )
    
    # Step 3: Sequential prompt selection
    prompt_obs, _ = prompt_env.reset()
    done = False
    total_reward = 0.0
    prompt_steps = 0
    
    while not done:
        with torch.no_grad():
            prompt_action = prompt_policy.get_action(prompt_obs, deterministic=deterministic)
        prompt_obs, reward, done, _, info = prompt_env.step(prompt_action)
        total_reward += reward
        prompt_steps += 1
    
    return {
        "correct": info.get("correct", False),
        "reward": total_reward,
        "workflow": struct_exec_info["workflow"],
        "steps_taken": info.get("steps_taken", 0),
        "tools_used": info.get("tools_used", 0),
        "total_tokens": info.get("total_tokens", 0),
        "decision_steps": 1 + prompt_steps,
        "reasoner_tools": info.get("reasoner_tools", []),
        "verifier_tools": info.get("verifier_tools", []),
        "reasoner_prompts": info.get("reasoner_prompts", []),
        "verifier_prompts": info.get("verifier_prompts", []),
        "answerer_prompts": info.get("answerer_prompts", []),
        "reasoner_budget": info.get("reasoner_budget", "N/A"),
        "verifier_budget": info.get("verifier_budget", "N/A"),
        "answerer_budget": info.get("answerer_budget", "N/A"),
        "config": format_config(struct_action),
    }


def run_random_hierarchical_episode(structure_env, prompt_env):
    """Run a single hierarchical episode with random actions."""
    # Random structure
    struct_obs, _ = structure_env.reset()
    struct_action = structure_env.action_space.sample()
    _, _, _, _, struct_exec_info = structure_env.step(struct_action)
    
    # Set structure
    prompt_env.set_structure(
        question=struct_exec_info["question"],
        answer=struct_exec_info["answer"],
        embedding=struct_exec_info["embedding"],
        structure=struct_exec_info["structure"]
    )
    
    # Random prompts
    prompt_obs, _ = prompt_env.reset()
    done = False
    total_reward = 0.0
    prompt_steps = 0
    
    while not done:
        prompt_action = prompt_env.action_space.sample()
        prompt_obs, reward, done, _, info = prompt_env.step(prompt_action)
        total_reward += reward
        prompt_steps += 1
    
    return {
        "correct": info.get("correct", False),
        "reward": total_reward,
        "workflow": struct_exec_info["workflow"],
        "steps_taken": info.get("steps_taken", 0),
        "tools_used": info.get("tools_used", 0),
        "total_tokens": info.get("total_tokens", 0),
        "decision_steps": 1 + prompt_steps,
        "reasoner_tools": info.get("reasoner_tools", []),
        "verifier_tools": info.get("verifier_tools", []),
        "reasoner_prompts": info.get("reasoner_prompts", []),
        "verifier_prompts": info.get("verifier_prompts", []),
        "answerer_prompts": info.get("answerer_prompts", []),
        "reasoner_budget": info.get("reasoner_budget", "N/A"),
        "verifier_budget": info.get("verifier_budget", "N/A"),
        "answerer_budget": info.get("answerer_budget", "N/A"),
        "config": format_config(struct_action),
    }


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
    
    # Random prompts (or could be deterministic)
    prompt_obs, _ = prompt_env.reset()
    done = False
    total_reward = 0.0
    prompt_steps = 0
    
    while not done:
        prompt_action = prompt_env.action_space.sample()
        prompt_obs, reward, done, _, info = prompt_env.step(prompt_action)
        total_reward += reward
        prompt_steps += 1
    
    return {
        "correct": info.get("correct", False),
        "reward": total_reward,
        "workflow": struct_exec_info["workflow"],
        "steps_taken": info.get("steps_taken", 0),
        "tools_used": info.get("tools_used", 0),
        "total_tokens": info.get("total_tokens", 0),
        "decision_steps": 1 + prompt_steps,
        "reasoner_tools": info.get("reasoner_tools", []),
        "verifier_tools": info.get("verifier_tools", []),
        "reasoner_prompts": info.get("reasoner_prompts", []),
        "verifier_prompts": info.get("verifier_prompts", []),
        "answerer_prompts": info.get("answerer_prompts", []),
        "reasoner_budget": info.get("reasoner_budget", "N/A"),
        "verifier_budget": info.get("verifier_budget", "N/A"),
        "answerer_budget": info.get("answerer_budget", "N/A"),
        "config": format_config(struct_action),
    }


def run_single_step_episode(env, model, deterministic=True):
    """Run a single-step episode."""
    obs = env.reset()
    if model is not None:
        action, _ = model.predict(obs, deterministic=deterministic)
    else:
        action = env.action_space.sample()
    
    # Format action for vectorized env
    if not isinstance(action, np.ndarray):
        action = np.array(action)
    if action.ndim == 0:
        action_batch = np.array([action.item()])
    elif action.ndim == 1:
        action_batch = action.reshape(1, -1)
    else:
        action_batch = action
    
    obs, rewards, dones, infos = env.step(action_batch)
    
    return {
        "correct": infos[0].get("correct", False),
        "reward": float(rewards[0]),
        "workflow": infos[0].get("workflow", ""),
        "steps_taken": infos[0].get("steps_taken", 0),
        "tools_used": infos[0].get("tools_used", 0),
        "total_tokens": infos[0].get("total_tokens", 0),
        "decision_steps": 1,
        "config": "Single-Step",
    }


def run_multi_step_episode(env, model, deterministic=True):
    """Run a multi-step episode."""
    obs = env.reset()
    done = False
    total_reward = 0.0
    decision_steps = 0
    
    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=deterministic)
        else:
            action = env.action_space.sample()
        
        # Format action
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        if action.ndim == 0:
            action_batch = np.array([action.item()])
        else:
            action_batch = action.reshape(1, -1) if action.ndim == 1 else action
        
        obs, rewards, dones, infos = env.step(action_batch)
        total_reward += float(rewards[0])
        decision_steps += 1
        done = dones[0]
    
    return {
        "correct": infos[0].get("correct", False),
        "reward": total_reward,
        "workflow": infos[0].get("workflow", ""),
        "steps_taken": infos[0].get("steps_taken", 0),
        "tools_used": infos[0].get("tools_used", 0),
        "total_tokens": infos[0].get("total_tokens", 0),
        "decision_steps": decision_steps,
        "config": "Multi-Step",
    }


# Define baseline configurations
BASELINE_CONFIGS = {
    "Direct-Low": [0, 0, 0, 0, 0, 0],
    "Direct-Mid": [0, 0, 1, 0, 1, 1],
    "Direct-High": [0, 0, 2, 0, 2, 2],
    "Direct-Calc-Low": [0, 1, 0, 0, 0, 0],
    "Direct-Calc-Mid": [0, 1, 1, 0, 1, 1],
    "Direct-Calc-High": [0, 1, 2, 0, 2, 2],
    "Reason-Mid": [1, 0, 1, 0, 1, 1],
    "Reason-High": [1, 0, 2, 0, 2, 2],
    "Reason-Calc-Mid": [1, 1, 1, 0, 1, 1],
    "Reason-Calc-High": [1, 1, 2, 0, 2, 2],
    "Reason-AllTools-Mid": [1, 7, 1, 0, 1, 1],
    "Reason-AllTools-High": [1, 7, 2, 0, 2, 2],
    "Full-Mid": [2, 0, 1, 0, 1, 1],
    "Full-High": [2, 0, 2, 0, 2, 2],
    "Full-Calc-Mid": [2, 1, 1, 1, 1, 1],
    "Full-Calc-High": [2, 1, 2, 1, 2, 2],
    "Full-AllTools-Mid": [2, 7, 1, 7, 1, 1],
    "Full-AllTools-High": [2, 7, 2, 7, 2, 2],
}


def generate_grid_search_configs(max_configs=50):
    """Generate configurations for grid search over key dimensions."""
    configs = []
    
    # Key dimensions to search
    workflows = [0, 1, 2]  # Direct, Reason+Ans, Reason+Verify+Ans
    tool_sets = [0, 1, 3, 7, 15]  # For all tools
    budgets = [0, 1, 2]  # Low, Mid, High
    
    # Grid search: workflow × tool_sets × budgets
    for wf in workflows:
        for r_tools in tool_sets:
            for r_budget in budgets:
                # For verifier, only relevant if workflow is 2
                if wf == 2:
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
    
    # Priority order: More complex workflows first, then tools, then budgets
    # Heuristic: More complex = better, but also more expensive
    
    # High priority: Full workflows with tools
    for tools in [15, 7, 1, 3, 0]:  # All, Calc, Calc+Web, None
        for budget in [1, 2, 0]:  # Mid, High, Low
            configs.append([2, tools, budget, tools, budget, budget])  # Full workflow
    
    # Medium priority: Reason+Ans with tools
    for tools in [15, 7, 1, 3, 0]:
        for budget in [1, 2, 0]:
            configs.append([1, tools, budget, 0, 0, budget])
    
    # Lower priority: Direct (simpler)
    for tools in [1, 0]:
        for budget in [1, 2, 0]:
            configs.append([0, tools, budget, 0, 0, budget])
    
    return configs[:50]  # Limit to top 50


def generate_evolutionary_population(pop_size=20):
    """Generate initial population for evolutionary search."""
    import random
    population = []
    
    # Start with diverse configurations
    for _ in range(pop_size):
        config = [
            random.randint(0, 2),  # workflow
            random.randint(0, 15),  # reasoner_tools
            random.randint(0, 2),  # reasoner_budget
            random.randint(0, 15),  # verifier_tools
            random.randint(0, 2),  # verifier_budget
            random.randint(0, 2),  # answerer_budget
        ]
        population.append(config)
    
    return population


def mutate_config(config, mutation_rate=0.3):
    """Mutate a configuration."""
    import random
    new_config = config.copy()
    
    for i in range(len(new_config)):
        if random.random() < mutation_rate:
            if i == 0:  # workflow
                new_config[i] = random.randint(0, 2)
            elif i in [1, 3]:  # tools
                new_config[i] = random.randint(0, 15)
            else:  # budgets
                new_config[i] = random.randint(0, 2)
    
    return new_config


def crossover_config(config1, config2):
    """Crossover two configurations."""
    import random
    child = []
    for i in range(len(config1)):
        child.append(config1[i] if random.random() < 0.5 else config2[i])
    return child


def run_evolutionary_search(structure_env, prompt_env, num_episodes=50, pop_size=20):
    """Run evolutionary search for configurations."""
    results_list = []
    
    # Initialize population
    population = generate_evolutionary_population(pop_size)
    population_scores = {}  # Cache scores
    
    # Evaluate initial population
    for config in population:
        config_key = tuple(config)
        if config_key not in population_scores:
            result = run_fixed_config_episode(structure_env, prompt_env, config)
            population_scores[config_key] = result["reward"]
            results_list.append({
                "strategy": "Evolutionary Search",
                "search_method": "Evolutionary",
                "episode": len(results_list),
                **result
            })
            if len(results_list) >= num_episodes:
                break
    
    # Evolutionary loop
    generation = 0
    while len(results_list) < num_episodes:
        generation += 1
        
        # Select top performers
        sorted_pop = sorted(population, key=lambda c: population_scores.get(tuple(c), -10), reverse=True)
        elite = sorted_pop[:pop_size // 2]
        
        # Create new generation
        new_population = elite.copy()
        while len(new_population) < pop_size:
            if len(elite) >= 2:
                parent1, parent2 = np.random.choice(len(elite), 2, replace=False)
                child = crossover_config(elite[parent1], elite[parent2])
                child = mutate_config(child)
            else:
                child = mutate_config(elite[0] if elite else population[0])
            new_population.append(child)
        
        # Evaluate new configurations
        for config in new_population[len(elite):]:
            if len(results_list) >= num_episodes:
                break
            config_key = tuple(config)
            if config_key not in population_scores:
                result = run_fixed_config_episode(structure_env, prompt_env, config)
                population_scores[config_key] = result["reward"]
                results_list.append({
                    "strategy": "Evolutionary Search",
                    "search_method": "Evolutionary",
                    "episode": len(results_list),
                    **result
                })
        
        population = new_population
    
    return results_list


def run_grid_search(structure_env, prompt_env, num_episodes=50):
    """Run grid search over configurations."""
    configs = generate_grid_search_configs(max_configs=num_episodes)
    results_list = []
    
    for i, config in enumerate(configs[:num_episodes]):
        result = run_fixed_config_episode(structure_env, prompt_env, config)
        results_list.append({
            "strategy": "Grid Search",
            "search_method": "Grid Search",
            "episode": i,
            **result
        })
    
    return results_list


def run_greedy_search(structure_env, prompt_env, num_episodes=50):
    """Run greedy search (try best configs first)."""
    configs = generate_greedy_configs()
    results_list = []
    
    for i, config in enumerate(configs[:num_episodes]):
        result = run_fixed_config_episode(structure_env, prompt_env, config)
        results_list.append({
            "strategy": "Greedy Search",
            "search_method": "Greedy",
            "episode": i,
            **result
        })
    
    return results_list


def run_best_first_search(structure_env, prompt_env, num_episodes=50):
    """Run best-first search with priority queue."""
    # Priority queue: (-reward, config) - negative for max heap
    pq = []
    visited = set()
    results_list = []
    
    # Heuristic function: estimate config quality
    def heuristic(config):
        score = 0
        # Prefer more complex workflows
        score += config[0] * 2  # workflow
        # Prefer having tools
        score += bin(config[1]).count('1')  # reasoner tools count
        score += bin(config[3]).count('1')  # verifier tools count
        # Prefer mid/high budgets
        score += config[2] + config[4] + config[5]  # budgets
        return score
    
    # Start with some promising configs
    initial_configs = [
        [2, 15, 2, 15, 2, 2],  # Full, all tools, high
        [2, 1, 2, 1, 2, 2],  # Full, calc, high
        [1, 15, 2, 0, 0, 2],  # Reason, all tools, high
        [1, 1, 2, 0, 0, 2],  # Reason, calc, high
    ]
    
    for config in initial_configs:
        h = heuristic(config)
        heapq.heappush(pq, (-h, tuple(config)))
    
    while len(results_list) < num_episodes and pq:
        _, config_tuple = heapq.heappop(pq)
        
        if config_tuple in visited:
            continue
        
        visited.add(config_tuple)
        config = list(config_tuple)
        
        result = run_fixed_config_episode(structure_env, prompt_env, config)
        reward = result["reward"]
        
        results_list.append({
            "strategy": "Best-First Search",
            "search_method": "Best-First",
            "episode": len(results_list),
            **result
        })
        
        # Add neighbors to queue
        for neighbor in generate_neighbors(config):
            neighbor_tuple = tuple(neighbor)
            if neighbor_tuple not in visited:
                h = heuristic(neighbor)
                heapq.heappush(pq, (-h, neighbor_tuple))
    
    # Fill remaining with random if needed
    while len(results_list) < num_episodes:
        config = structure_env.action_space.sample()
        result = run_fixed_config_episode(structure_env, prompt_env, config)
        results_list.append({
            "strategy": "Best-First Search",
            "search_method": "Best-First",
            "episode": len(results_list),
            **result
        })
    
    return results_list


def generate_neighbors(config):
    """Generate neighboring configurations."""
    neighbors = []
    
    # Mutate each dimension slightly
    for i in range(len(config)):
        if i == 0:  # workflow
            for val in [0, 1, 2]:
                if val != config[i]:
                    neighbor = config.copy()
                    neighbor[i] = val
                    neighbors.append(neighbor)
        elif i in [1, 3]:  # tools
            # Try adding/removing one tool (bitwise XOR)
            for tool_bit in [1, 2, 4, 8]:
                neighbor = config.copy()
                neighbor[i] = int(config[i]) ^ tool_bit  # Toggle bit
                neighbors.append(neighbor)
        else:  # budgets
            for val in [0, 1, 2]:
                if val != config[i]:
                    neighbor = config.copy()
                    neighbor[i] = val
                    neighbors.append(neighbor)
    
    return neighbors[:10]  # Limit neighbors


def run_benchmark(cfg, num_episodes=50, 
                 structure_model_path=None, prompt_model_path=None,
                 single_step_model_path=None, multi_step_model_path=None,
                 include_search_algorithms=True, include_random=True, use_cpu=False):
    """Run comprehensive benchmark with multiple search methods and configurations."""
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE BENCHMARK")
    print(f"{'='*80}")
    print(f"  Dataset:     {cfg.DATASET_NAME}")
    print(f"  Episodes:     {num_episodes} per strategy")
    print(f"{'='*80}\n")
    
    results = []
    detailed_results = []
    
    # Use CPU if requested or if CUDA not available
    if use_cpu or not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"
    
    print(f"Using device: {device}\n")
    
    # Clear cache at start
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create shared environments ONCE (suppress loading messages)
    print("Initializing environments...", end=" ", flush=True)
    with SuppressOutput():
        structure_env = StructureEnv(cfg)
        prompt_env = PromptEnv(cfg)
    print("Done!\n")
    
    strategy_num = 1
    
    # ========================================
    # 1. Hierarchical/SFT Model (if provided)
    # ========================================
    if structure_model_path and prompt_model_path:
        print(f"[{strategy_num}] Hierarchical Model (Learned)")
        print(f"    Structure: {structure_model_path}")
        print(f"    Prompt:    {prompt_model_path}")
        strategy_num += 1
        
        try:
            structure_policy, prompt_policy = load_hierarchical_models(
                structure_model_path, prompt_model_path, device
            )
            print(f"    Models loaded successfully! (device: {device})\n")
            
            start_time = time.time()
            accuracies = []
            rewards = []
            tokens = []
            workflows = []
            decision_steps = []
            tools_used = []
            configs = []
            
            for ep in range(num_episodes):
                result = run_hierarchical_episode(
                    structure_policy, prompt_policy, structure_env, prompt_env, deterministic=True
                )
                
                accuracies.append(1 if result["correct"] else 0)
                rewards.append(result["reward"])
                tokens.append(result["total_tokens"])
                workflows.append(result["workflow"])
                decision_steps.append(result["decision_steps"])
                tools_used.append(result["tools_used"])
                configs.append(result["config"])
                
                detailed_results.append({
                    "strategy": "Hierarchical (Learned)",
                    "search_method": "Learned Policy",
                    "episode": ep,
                    **result
                })
                
                # Progress indicator
                acc_so_far = np.mean(accuracies)
                status = "✓" if result["correct"] else "✗"
                print(f"\r    Progress: {ep+1}/{num_episodes} | Running Acc: {acc_so_far:.1%} | Last: {status}", end="", flush=True)
                
                # Clear cache every 10 episodes
                if (ep + 1) % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
            
            elapsed = time.time() - start_time
            avg_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            avg_reward = np.mean(rewards)
            avg_tokens = np.mean(tokens)
            avg_decisions = np.mean(decision_steps)
            avg_tools = np.mean(tools_used)
            
            results.append({
                "Strategy": "Hierarchical (Learned)",
                "Search Method": "Learned Policy",
                "Accuracy": f"{avg_acc:.1%}",
                "Accuracy (Std)": f"{std_acc:.3f}",
                "Avg Reward": f"{avg_reward:.3f}",
                "Avg Tokens": f"{avg_tokens:.0f}",
                "Avg Decisions": f"{avg_decisions:.1f}",
                "Avg Tools": f"{avg_tools:.2f}",
                "Time (s)": f"{elapsed:.1f}",
                "Episodes": num_episodes
            })
            
            print(f"\n    Final: Accuracy={avg_acc:.1%}±{std_acc:.3f}, Reward={avg_reward:.3f}, Tokens={avg_tokens:.0f}\n")
            
        except Exception as e:
            print(f"\n    Error loading models: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================
    # 2. Single-Step Model (if provided)
    # ========================================
    if single_step_model_path:
        print(f"[{strategy_num}] Single-Step Model (Learned)")
        print(f"    Model: {single_step_model_path}")
        strategy_num += 1
        
        try:
            with SuppressOutput():
                single_step_model = PPO.load(single_step_model_path, device='cpu')
                single_cfg = load_config("single_step")
                single_env = DummyVecEnv([lambda: GeneralAgentEnv(single_cfg)])
            
            start_time = time.time()
            accuracies = []
            rewards = []
            tokens = []
            workflows = []
            decision_steps = []
            tools_used = []
            
            for ep in range(num_episodes):
                result = run_single_step_episode(single_env, single_step_model, deterministic=True)
                
                accuracies.append(1 if result["correct"] else 0)
                rewards.append(result["reward"])
                tokens.append(result["total_tokens"])
                workflows.append(result["workflow"])
                decision_steps.append(result["decision_steps"])
                tools_used.append(result["tools_used"])
                
                detailed_results.append({
                    "strategy": "Single-Step (Learned)",
                    "search_method": "Learned Policy",
                    "episode": ep,
                    **result
                })
                
                # Progress indicator
                acc_so_far = np.mean(accuracies)
                status = "✓" if result["correct"] else "✗"
                print(f"\r    Progress: {ep+1}/{num_episodes} | Running Acc: {acc_so_far:.1%} | Last: {status}", end="", flush=True)
            
            elapsed = time.time() - start_time
            avg_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            avg_reward = np.mean(rewards)
            avg_tokens = np.mean(tokens)
            avg_decisions = np.mean(decision_steps)
            avg_tools = np.mean(tools_used)
            
            results.append({
                "Strategy": "Single-Step (Learned)",
                "Search Method": "Learned Policy",
                "Accuracy": f"{avg_acc:.1%}",
                "Accuracy (Std)": f"{std_acc:.3f}",
                "Avg Reward": f"{avg_reward:.3f}",
                "Avg Tokens": f"{avg_tokens:.0f}",
                "Avg Decisions": f"{avg_decisions:.1f}",
                "Avg Tools": f"{avg_tools:.2f}",
                "Time (s)": f"{elapsed:.1f}",
                "Episodes": num_episodes
            })
            
            print(f"\n    Final: Accuracy={avg_acc:.1%}±{std_acc:.3f}, Reward={avg_reward:.3f}, Tokens={avg_tokens:.0f}\n")
            
        except Exception as e:
            print(f"\n    Error loading single-step model: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================
    # 3. Multi-Step Model (if provided)
    # ========================================
    if multi_step_model_path:
        print(f"[{strategy_num}] Multi-Step Model (Learned)")
        print(f"    Model: {multi_step_model_path}")
        strategy_num += 1
        
        try:
            with SuppressOutput():
                multi_step_model = PPO.load(multi_step_model_path, device='cpu')
                multi_cfg = load_config("multi_step")
                multi_env = DummyVecEnv([lambda: MultiStepAgentEnv(multi_cfg)])
            
            start_time = time.time()
            accuracies = []
            rewards = []
            tokens = []
            workflows = []
            decision_steps = []
            tools_used = []
            
            for ep in range(num_episodes):
                result = run_multi_step_episode(multi_env, multi_step_model, deterministic=True)
                
                accuracies.append(1 if result["correct"] else 0)
                rewards.append(result["reward"])
                tokens.append(result["total_tokens"])
                workflows.append(result["workflow"])
                decision_steps.append(result["decision_steps"])
                tools_used.append(result["tools_used"])
                
                detailed_results.append({
                    "strategy": "Multi-Step (Learned)",
                    "search_method": "Learned Policy",
                    "episode": ep,
                    **result
                })
                
                # Progress indicator
                acc_so_far = np.mean(accuracies)
                status = "✓" if result["correct"] else "✗"
                print(f"\r    Progress: {ep+1}/{num_episodes} | Running Acc: {acc_so_far:.1%} | Last: {status}", end="", flush=True)
            
            elapsed = time.time() - start_time
            avg_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            avg_reward = np.mean(rewards)
            avg_tokens = np.mean(tokens)
            avg_decisions = np.mean(decision_steps)
            avg_tools = np.mean(tools_used)
            
            results.append({
                "Strategy": "Multi-Step (Learned)",
                "Search Method": "Learned Policy",
                "Accuracy": f"{avg_acc:.1%}",
                "Accuracy (Std)": f"{std_acc:.3f}",
                "Avg Reward": f"{avg_reward:.3f}",
                "Avg Tokens": f"{avg_tokens:.0f}",
                "Avg Decisions": f"{avg_decisions:.1f}",
                "Avg Tools": f"{avg_tools:.2f}",
                "Time (s)": f"{elapsed:.1f}",
                "Episodes": num_episodes
            })
            
            print(f"\n    Final: Accuracy={avg_acc:.1%}±{std_acc:.3f}, Reward={avg_reward:.3f}, Tokens={avg_tokens:.0f}\n")
            
        except Exception as e:
            print(f"\n    Error loading multi-step model: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================
    # 4. Random Search (Hierarchical)
    # ========================================
    if include_random:
        print(f"[{strategy_num}] Random Search (Hierarchical)")
        strategy_num += 1
        start_time = time.time()
        
        accuracies = []
        rewards = []
        tokens = []
        workflows = []
        decision_steps = []
        tools_used = []
        configs = []
        
        for ep in range(num_episodes):
            result = run_random_hierarchical_episode(structure_env, prompt_env)
            
            accuracies.append(1 if result["correct"] else 0)
            rewards.append(result["reward"])
            tokens.append(result["total_tokens"])
            workflows.append(result["workflow"])
            decision_steps.append(result["decision_steps"])
            tools_used.append(result["tools_used"])
            configs.append(result["config"])
            
            detailed_results.append({
                "strategy": "Random Search",
                "search_method": "Random",
                "episode": ep,
                **result
            })
            
            acc_so_far = np.mean(accuracies)
            status = "✓" if result["correct"] else "✗"
            print(f"\r    Progress: {ep+1}/{num_episodes} | Running Acc: {acc_so_far:.1%} | Last: {status}", end="", flush=True)
            
            if (ep + 1) % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        elapsed = time.time() - start_time
        avg_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        avg_reward = np.mean(rewards)
        avg_tokens = np.mean(tokens)
        avg_decisions = np.mean(decision_steps)
        avg_tools = np.mean(tools_used)
        
        results.append({
            "Strategy": "Random Search",
            "Search Method": "Random",
            "Accuracy": f"{avg_acc:.1%}",
            "Accuracy (Std)": f"{std_acc:.3f}",
            "Avg Reward": f"{avg_reward:.3f}",
            "Avg Tokens": f"{avg_tokens:.0f}",
            "Avg Decisions": f"{avg_decisions:.1f}",
            "Avg Tools": f"{avg_tools:.2f}",
            "Time (s)": f"{elapsed:.1f}",
            "Episodes": num_episodes
        })
        
        print(f"\n    Final: Accuracy={avg_acc:.1%}±{std_acc:.3f}, Reward={avg_reward:.3f}, Tokens={avg_tokens:.0f}\n")
    
    # ========================================
    # Search Algorithms (if enabled)
    # ========================================
    if include_search_algorithms:
        # Grid Search
        print(f"[{strategy_num}] Grid Search")
        strategy_num += 1
        start_time = time.time()
        
        grid_results = run_grid_search(structure_env, prompt_env, num_episodes)
        detailed_results.extend(grid_results)
        
        if grid_results:
            accuracies = [1 if r["correct"] else 0 for r in grid_results]
            rewards = [r["reward"] for r in grid_results]
            tokens = [r["total_tokens"] for r in grid_results]
            decision_steps = [r["decision_steps"] for r in grid_results]
            tools_used = [r["tools_used"] for r in grid_results]
            
            elapsed = time.time() - start_time
            avg_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            avg_reward = np.mean(rewards)
            avg_tokens = np.mean(tokens)
            avg_decisions = np.mean(decision_steps)
            avg_tools = np.mean(tools_used)
            
            results.append({
                "Strategy": "Grid Search",
                "Search Method": "Grid Search",
                "Accuracy": f"{avg_acc:.1%}",
                "Accuracy (Std)": f"{std_acc:.3f}",
                "Avg Reward": f"{avg_reward:.3f}",
                "Avg Tokens": f"{avg_tokens:.0f}",
                "Avg Decisions": f"{avg_decisions:.1f}",
                "Avg Tools": f"{avg_tools:.2f}",
                "Time (s)": f"{elapsed:.1f}",
                "Episodes": len(grid_results)
            })
            
            print(f"    Final: Accuracy={avg_acc:.1%}±{std_acc:.3f}, Reward={avg_reward:.3f}, Tokens={avg_tokens:.0f}\n")
        
        # Greedy Search
        print(f"[{strategy_num}] Greedy Search")
        strategy_num += 1
        start_time = time.time()
        
        greedy_results = run_greedy_search(structure_env, prompt_env, num_episodes)
        detailed_results.extend(greedy_results)
        
        if greedy_results:
            accuracies = [1 if r["correct"] else 0 for r in greedy_results]
            rewards = [r["reward"] for r in greedy_results]
            tokens = [r["total_tokens"] for r in greedy_results]
            decision_steps = [r["decision_steps"] for r in greedy_results]
            tools_used = [r["tools_used"] for r in greedy_results]
            
            elapsed = time.time() - start_time
            avg_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            avg_reward = np.mean(rewards)
            avg_tokens = np.mean(tokens)
            avg_decisions = np.mean(decision_steps)
            avg_tools = np.mean(tools_used)
            
            results.append({
                "Strategy": "Greedy Search",
                "Search Method": "Greedy",
                "Accuracy": f"{avg_acc:.1%}",
                "Accuracy (Std)": f"{std_acc:.3f}",
                "Avg Reward": f"{avg_reward:.3f}",
                "Avg Tokens": f"{avg_tokens:.0f}",
                "Avg Decisions": f"{avg_decisions:.1f}",
                "Avg Tools": f"{avg_tools:.2f}",
                "Time (s)": f"{elapsed:.1f}",
                "Episodes": len(greedy_results)
            })
            
            print(f"    Final: Accuracy={avg_acc:.1%}±{std_acc:.3f}, Reward={avg_reward:.3f}, Tokens={avg_tokens:.0f}\n")
        
        # Best-First Search
        print(f"[{strategy_num}] Best-First Search")
        strategy_num += 1
        start_time = time.time()
        
        bestfirst_results = run_best_first_search(structure_env, prompt_env, num_episodes)
        detailed_results.extend(bestfirst_results)
        
        if bestfirst_results:
            accuracies = [1 if r["correct"] else 0 for r in bestfirst_results]
            rewards = [r["reward"] for r in bestfirst_results]
            tokens = [r["total_tokens"] for r in bestfirst_results]
            decision_steps = [r["decision_steps"] for r in bestfirst_results]
            tools_used = [r["tools_used"] for r in bestfirst_results]
            
            elapsed = time.time() - start_time
            avg_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            avg_reward = np.mean(rewards)
            avg_tokens = np.mean(tokens)
            avg_decisions = np.mean(decision_steps)
            avg_tools = np.mean(tools_used)
            
            results.append({
                "Strategy": "Best-First Search",
                "Search Method": "Best-First",
                "Accuracy": f"{avg_acc:.1%}",
                "Accuracy (Std)": f"{std_acc:.3f}",
                "Avg Reward": f"{avg_reward:.3f}",
                "Avg Tokens": f"{avg_tokens:.0f}",
                "Avg Decisions": f"{avg_decisions:.1f}",
                "Avg Tools": f"{avg_tools:.2f}",
                "Time (s)": f"{elapsed:.1f}",
                "Episodes": len(bestfirst_results)
            })
            
            print(f"    Final: Accuracy={avg_acc:.1%}±{std_acc:.3f}, Reward={avg_reward:.3f}, Tokens={avg_tokens:.0f}\n")
        
        # Evolutionary Search
        print(f"[{strategy_num}] Evolutionary Search")
        strategy_num += 1
        start_time = time.time()
        
        evo_results = run_evolutionary_search(structure_env, prompt_env, num_episodes, pop_size=20)
        detailed_results.extend(evo_results)
        
        if evo_results:
            accuracies = [1 if r["correct"] else 0 for r in evo_results]
            rewards = [r["reward"] for r in evo_results]
            tokens = [r["total_tokens"] for r in evo_results]
            decision_steps = [r["decision_steps"] for r in evo_results]
            tools_used = [r["tools_used"] for r in evo_results]
            
            elapsed = time.time() - start_time
            avg_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            avg_reward = np.mean(rewards)
            avg_tokens = np.mean(tokens)
            avg_decisions = np.mean(decision_steps)
            avg_tools = np.mean(tools_used)
            
            results.append({
                "Strategy": "Evolutionary Search",
                "Search Method": "Evolutionary",
                "Accuracy": f"{avg_acc:.1%}",
                "Accuracy (Std)": f"{std_acc:.3f}",
                "Avg Reward": f"{avg_reward:.3f}",
                "Avg Tokens": f"{avg_tokens:.0f}",
                "Avg Decisions": f"{avg_decisions:.1f}",
                "Avg Tools": f"{avg_tools:.2f}",
                "Time (s)": f"{elapsed:.1f}",
                "Episodes": len(evo_results)
            })
            
            print(f"    Final: Accuracy={avg_acc:.1%}±{std_acc:.3f}, Reward={avg_reward:.3f}, Tokens={avg_tokens:.0f}\n")
        
        # Fixed Baseline Configurations
        print(f"[{strategy_num}] Fixed Baseline Configurations ({len(BASELINE_CONFIGS)} configs)")
        
        for config_name, fixed_action in BASELINE_CONFIGS.items():
            print(f"    Testing: {config_name}")
            start_time = time.time()
            
            accuracies = []
            rewards = []
            tokens = []
            workflows = []
            decision_steps = []
            tools_used = []
            
            for ep in range(num_episodes):
                result = run_fixed_config_episode(structure_env, prompt_env, fixed_action)
                
                accuracies.append(1 if result["correct"] else 0)
                rewards.append(result["reward"])
                tokens.append(result["total_tokens"])
                workflows.append(result["workflow"])
                decision_steps.append(result["decision_steps"])
                tools_used.append(result["tools_used"])
                
                detailed_results.append({
                    "strategy": config_name,
                    "search_method": "Fixed Config",
                    "episode": ep,
                    **result
                })
                
                if (ep + 1) % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
            
            elapsed = time.time() - start_time
            avg_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            avg_reward = np.mean(rewards)
            avg_tokens = np.mean(tokens)
            avg_decisions = np.mean(decision_steps)
            avg_tools = np.mean(tools_used)
            
            results.append({
                "Strategy": config_name,
                "Search Method": "Fixed Config",
                "Accuracy": f"{avg_acc:.1%}",
                "Accuracy (Std)": f"{std_acc:.3f}",
                "Avg Reward": f"{avg_reward:.3f}",
                "Avg Tokens": f"{avg_tokens:.0f}",
                "Avg Decisions": f"{avg_decisions:.1f}",
                "Avg Tools": f"{avg_tools:.2f}",
                "Time (s)": f"{elapsed:.1f}",
                "Episodes": num_episodes
            })
            
            print(f"      Accuracy={avg_acc:.1%}±{std_acc:.3f}, Reward={avg_reward:.3f}, Tokens={avg_tokens:.0f}")
        
        print()
    
    # ========================================
    # Print Results Table
    # ========================================
    if results:
        df = pd.DataFrame(results)
        
        # Sort by accuracy (descending)
        df["Accuracy_Num"] = df["Accuracy"].str.rstrip("%").astype(float) / 100
        df = df.sort_values("Accuracy_Num", ascending=False)
        df = df.drop("Accuracy_Num", axis=1)
        
        print("\n" + "="*80)
        print("COMPREHENSIVE BENCHMARK RESULTS")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        # Additional statistics
        print("\n" + "="*80)
        print("ADDITIONAL STATISTICS")
        print("="*80)
        
        if detailed_results:
            detailed_df = pd.DataFrame(detailed_results)
            
            # Workflow distribution
            print("\nWorkflow Distribution by Strategy:")
            workflow_dist = detailed_df.groupby(["strategy", "workflow"]).size().unstack(fill_value=0)
            print(workflow_dist)
            
            # Tool usage statistics
            print("\nAverage Tools Used by Strategy:")
            tools_stats = detailed_df.groupby("strategy")["tools_used"].agg(["mean", "std", "min", "max"])
            print(tools_stats)
            
            # Token usage statistics
            print("\nAverage Tokens Used by Strategy:")
            tokens_stats = detailed_df.groupby("strategy")["total_tokens"].agg(["mean", "std", "min", "max"])
            print(tokens_stats)
            
            # Save detailed results
            output_path = f"benchmark_results_{cfg.DATASET_NAME}_{int(time.time())}.csv"
            detailed_df.to_csv(output_path, index=False)
            print(f"\nDetailed results saved to: {output_path}")
            
            # Save summary table
            summary_path = f"benchmark_summary_{cfg.DATASET_NAME}_{int(time.time())}.csv"
            df.to_csv(summary_path, index=False)
            print(f"Summary table saved to: {summary_path}")
            
            return df, detailed_df
    
    return None, None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Comprehensive Benchmark Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", type=str, default="hierarchical",
        help="Configuration to use (hierarchical, single_step, multi_step)"
    )
    parser.add_argument(
        "--structure-model", type=str, default=None,
        help="Path to structure policy model (.pt file) for hierarchical models"
    )
    parser.add_argument(
        "--prompt-model", type=str, default=None,
        help="Path to prompt policy model (.pt file) for hierarchical models"
    )
    parser.add_argument(
        "--single-step-model", type=str, default=None,
        help="Path to single-step model (.zip file) for comparison"
    )
    parser.add_argument(
        "--multi-step-model", type=str, default=None,
        help="Path to multi-step model (.zip file) for comparison"
    )
    parser.add_argument(
        "--episodes", type=int, default=50,
        help="Number of episodes per strategy"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=["gsm8k", "hotpotqa", "gaia", "medqa", "aime25"],
        help="Override dataset"
    )
    parser.add_argument(
        "--no-search", action="store_true",
        help="Skip search algorithms (only run learned models and random)"
    )
    parser.add_argument(
        "--no-random", action="store_true",
        help="Skip random baseline"
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU usage (useful to avoid GPU memory issues)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    cfg = load_config(args.config)
    if args.dataset:
        cfg.DATASET_NAME = args.dataset
    
    # Check if model files exist
    if args.structure_model and not os.path.exists(args.structure_model):
        print(f"Warning: Structure model not found at {args.structure_model}")
        print("Skipping hierarchical model benchmark...")
        args.structure_model = None
        args.prompt_model = None
    
    if args.prompt_model and not os.path.exists(args.prompt_model):
        print(f"Warning: Prompt model not found at {args.prompt_model}")
        print("Skipping hierarchical model benchmark...")
        args.structure_model = None
        args.prompt_model = None
    
    if args.single_step_model and not os.path.exists(args.single_step_model):
        print(f"Warning: Single-step model not found at {args.single_step_model}")
        args.single_step_model = None
    
    if args.multi_step_model and not os.path.exists(args.multi_step_model):
        print(f"Warning: Multi-step model not found at {args.multi_step_model}")
        args.multi_step_model = None
    
    # Run benchmark
    run_benchmark(
        cfg=cfg,
        num_episodes=args.episodes,
        structure_model_path=args.structure_model,
        prompt_model_path=args.prompt_model,
        single_step_model_path=args.single_step_model,
        multi_step_model_path=args.multi_step_model,
        include_search_algorithms=not args.no_search,
        include_random=not args.no_random,
        use_cpu=args.cpu
    )


if __name__ == "__main__":
    main()
