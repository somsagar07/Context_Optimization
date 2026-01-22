"""
Unified Evaluation Script for Hierarchical RL

Evaluates trained structure and prompt policies.

Usage:
    # HuggingFace models (default)
    python scripts/eval_hrl.py --structure-model models/grpo_models/structure.pt --prompt-model models/grpo_models/prompt.pt --dataset gsm8k
    python scripts/eval_hrl.py --structure-model models/ppo_models/structure.pt --prompt-model models/ppo_models/prompt.pt --dataset gsm8k --episodes 50
    
    # Evaluate on all datapoints
    python scripts/eval_hrl.py --structure-model models/ppo_models/structure.pt --prompt-model models/ppo_models/prompt.pt --dataset gsm8k --episodes all
    
    # API mode (must match training configuration)
    python scripts/eval_hrl.py --structure-model models/ppo_models/structure.pt --prompt-model models/ppo_models/prompt.pt --dataset gsm8k --api --api-model openai/gpt-4o
"""
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.distributions import Categorical

from configs import load_config
from env.structure_env import StructureEnv
from env.prompt_env import PromptEnv


# Human-readable mappings
WORKFLOW_NAMES = [
    "Direct", "Reason+Ans", "Reason+Verify+Ans",
    "Routing", "Parallel-Sectioning", "Parallel-Voting",
    "Orchestrator-Workers", "Evaluator-Optimizer", "Autonomous-Agent"
]
BUDGET_NAMES = ["Low", "Mid", "High"]


# Replace lines 41-47
def decode_tools(idx, structure_env=None):
    """Decode tool index to list of tool names. Supports both standard and tau2 tools."""
    if structure_env and structure_env.is_tau2 and structure_env.tau2_tools:
        # Use tau2 tool registry
        return structure_env.tau2_tools.decode_tool_index(idx)
    else:
        # Standard tool decoding (4 tools: calculator, web_search, python, ocr_reader)
        tools = []
        if idx & 1: tools.append("calculator")
        if idx & 2: tools.append("web_search")
        if idx & 4: tools.append("python")
        if idx & 8: tools.append("ocr_reader")
        return tools if tools else ["none"]

# POLICY NETWORKS 
class PromptPolicy(nn.Module):
    """Load either PPO or GRPO prompt policy."""
    
    def __init__(self, obs_dim, action_dim, has_value=False):
        super().__init__()
        self.has_value = has_value
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.action_head = nn.Linear(256, action_dim)
        if has_value:
            self.value_head = nn.Linear(256, 1)
    
    def get_action(self, obs, deterministic=True, temperature=1.0):
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.to(next(self.parameters()).device)
        
        features = self.network(obs)
        logits = self.action_head(features)
        # Apply temperature scaling (lower = sharper, higher = more uniform)
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        
        if deterministic:
            return torch.argmax(probs, dim=-1).item()
        return Categorical(probs).sample().item()


class StructurePolicy(nn.Module):
    """Load either PPO or GRPO structure policy."""
    
    def __init__(self, obs_dim, action_dims, has_value=False):
        super().__init__()
        self.has_value = has_value
        self.action_dims = action_dims
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.action_heads = nn.ModuleList([nn.Linear(256, d) for d in action_dims])
        if has_value:
            self.value_head = nn.Linear(256, 1)
    
    def get_action(self, obs, deterministic=True, temperature=1.0):
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.to(next(self.parameters()).device)
        
        features = self.network(obs)
        actions = []
        for head in self.action_heads:
            logits = head(features)
            # Apply temperature scaling (lower = sharper, higher = more uniform/diverse)
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            if deterministic:
                actions.append(torch.argmax(probs, dim=-1).item())
            else:
                actions.append(Categorical(probs).sample().item())
        return np.array(actions, dtype=np.int64)


def load_structure_policy(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    obs_dim = checkpoint["obs_dim"]
    action_dims = checkpoint["action_dims"]
    algo = checkpoint.get("algorithm", "PPO")  # Default to PPO if missing
    has_value = "PPO" in algo 
    
    policy = StructurePolicy(obs_dim, action_dims, has_value).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    
    print(f"  Loaded structure policy ({algo})")
    return policy, algo


def load_prompt_policy(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    obs_dim = checkpoint["obs_dim"]
    action_dim = checkpoint["action_dim"]
    algo = checkpoint.get("algorithm", "PPO")  # Default to PPO if missing
    has_value = "PPO" in algo
    
    policy = PromptPolicy(obs_dim, action_dim, has_value).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    
    print(f"  Loaded prompt policy ({algo})")
    return policy, algo



# EVALUATION
def evaluate(structure_policy, prompt_policy, cfg, num_episodes=20, 
             deterministic=True, verbose=True, use_api=False, api_model=None, hf_model=None,
             temperature=1.0):
    """Evaluate dual policy system.
    
    Args:
        temperature: Softmax temperature for action sampling. 
                     <1.0 = sharper (more deterministic)
                     >1.0 = flatter (more diverse/exploratory)
    """
    structure_env = StructureEnv(cfg, is_eval=True, use_api=use_api, api_model=api_model, hf_model=hf_model)
    prompt_env = PromptEnv(cfg, is_eval=True, use_api=use_api, api_model=api_model, hf_model=hf_model)
    
    results = {"correct": [], "workflows": [], "tokens": [], "rewards": [], "tools": []}
    episode_logs = []  # Detailed per-episode logs
    
    # Running stats for tqdm
    running_correct = 0
    running_reward = 0.0
    
    print(f"\nEvaluating on {num_episodes} episodes...")
    
    pbar = tqdm(range(num_episodes), desc="Evaluating", leave=True)
    for ep in pbar:
        # Structure decision
        struct_obs, struct_info = structure_env.reset()
        
        with torch.no_grad():
            struct_action = structure_policy.get_action(struct_obs, deterministic, temperature)
        
        # Ensure action is numpy array
        if not isinstance(struct_action, np.ndarray):
            struct_action = np.array(struct_action, dtype=np.int64)
        
        _, _, _, _, struct_exec_info = structure_env.step(struct_action)
        
        # Setup prompt env
        prompt_env.set_structure(
            question=struct_exec_info["question"],
            answer=struct_exec_info["answer"],
            embedding=struct_exec_info["embedding"],
            structure=struct_exec_info["structure"],
            task=struct_exec_info.get("task")
        )
        
        # Prompt rollout
        prompt_obs, _ = prompt_env.reset()
        accumulated_reward = 0.0
        done = False
        
        while not done:
            with torch.no_grad():
                action = prompt_policy.get_action(prompt_obs, deterministic, temperature)
            prompt_obs, step_reward, done, _, info = prompt_env.step(action)
            accumulated_reward += step_reward
        
        # Compute final reward (matching training logic from base.py)
        correct = info.get("correct", False)
        reward_scale = 1.0  # Default reward scale
        tool_bonus = -0.05  # Default tool bonus (negative = penalty)
        
        # Final reward calculation (matches BaseTrainer.run_episode)
        final_reward = (1.0 if correct else 0.0) * 5.0 * reward_scale
        final_reward += accumulated_reward
        final_reward -= info.get("steps_taken", 1) * cfg.COST_PER_STEP
        final_reward += info.get("tools_used", 0) * tool_bonus
        
        max_tokens = 2048 + 1024 + 512  # reasoner_high + verifier_high + answerer_high
        final_reward -= (info.get("total_tokens", 256) / max_tokens) * cfg.COST_TOKEN_BUDGET
        
        # Decode tools for logging
        agent1_tools = decode_tools(struct_exec_info["structure"]["agent1_tools_idx"], structure_env)
        agent2_tools = decode_tools(struct_exec_info["structure"]["agent2_tools_idx"], structure_env)
        
        # Record aggregated results
        results["correct"].append(correct)
        results["workflows"].append(struct_exec_info["workflow"])
        results["tokens"].append(info.get("total_tokens", 0))
        results["rewards"].append(final_reward)
        results["tools"].append(info.get("tools_used", 0))
        
        # Record detailed episode log
        episode_logs.append({
            "episode": ep + 1,
            "correct": correct,
            "workflow": struct_exec_info["workflow"],
            "reward": float(final_reward),
            "tools_used": info.get("tools_used", 0),
            "tools_available": {
                "agent1": agent1_tools,
                "agent2": agent2_tools
            },
            "tokens": info.get("total_tokens", 0),
            "steps": info.get("steps_taken", 1),
            "question": struct_info["question"],
            "prediction": info.get("final_answer", ""),  # final_answer from prompt_env
            "ground_truth": info.get("ground_truth", "")
        })
        
        # Update running stats
        running_correct += int(correct)
        running_reward += final_reward
        
        # Update tqdm with running accuracy and reward
        pbar.set_postfix({
            "acc": f"{running_correct/(ep+1)*100:.1f}%",
            "reward": f"{running_reward/(ep+1):.2f}"
        })
    
    # Summary
    accuracy = np.mean(results["correct"]) * 100
    avg_reward = np.mean(results["rewards"])
    avg_tools = np.mean(results["tools"])
    avg_tokens = np.mean(results["tokens"])
    
    workflow_counts = defaultdict(int)
    for w in results["workflows"]:
        workflow_counts[w] += 1
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Accuracy:    {accuracy:.1f}%")
    print(f"  Avg Reward:  {avg_reward:.3f}")
    print(f"  Avg Tools:   {avg_tools:.2f}")
    print(f"  Avg Tokens:  {avg_tokens:.0f}")
    print(f"\n  Workflows:")
    for w, c in sorted(workflow_counts.items()):
        print(f"    {w}: {c} ({c/num_episodes*100:.1f}%)")
    
    return {
        "accuracy": accuracy, 
        "avg_reward": avg_reward, 
        "avg_tools": avg_tools,
        "avg_tokens": avg_tokens,
        "workflow_distribution": dict(workflow_counts),
        "episodes": episode_logs
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate hierarchical RL")
    parser.add_argument("--config", type=str, default="hierarchical")
    parser.add_argument("--structure-model", type=str, required=True)
    parser.add_argument("--prompt-model", type=str, required=True)
    parser.add_argument("--episodes", type=str, default="20",
                       help="Number of episodes to evaluate, or 'all' for all datapoints")
    parser.add_argument("--dataset", type=str, required=True, default=None, choices=["gsm8k", "hotpotqa", "gaia", "medqa", "aime25", "tau2_airline", "tau2_retail", "tau2_telecom"])

    parser.add_argument("--stochastic", action="store_true", help="Sample from policy distribution instead of argmax")
    parser.add_argument("--temperature", type=float, default=1.0, 
                       help="Softmax temperature: <1.0=sharper (deterministic), >1.0=flatter (diverse)")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log-file", type=str, default=None,
                       help="Path to save evaluation logs as JSON. If not provided, auto-generates filename in eval_logs/")
    
    # API configuration (must match training configuration)
    parser.add_argument("--api", action="store_true", default=False,
                       help="Use OpenRouter API instead of local HuggingFace models (must match training mode)")
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
        from utils.get_dataset import get_dataset_loader
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
    print("HIERARCHICAL RL EVALUATION")
    print("=" * 60)
    
    print(f"  Structure: {args.structure_model}")
    print(f"  Prompt:    {args.prompt_model}")
    print(f"  Mode:      {'Stochastic' if args.stochastic else 'Deterministic'} (temp={args.temperature})")
    if args.api:
        model_name = args.api_model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
        print(f"  API Mode:  {model_name}")
    else:
        from configs.base import LLM_MODEL_NAME
        model_name = args.hf_model or getattr(cfg, "LLM_MODEL_NAME", LLM_MODEL_NAME)
        print(f"  HF Model:  {model_name}")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading models (device: {device})...")
    
    structure_policy, struct_algo = load_structure_policy(args.structure_model, device)
    prompt_policy, prompt_algo = load_prompt_policy(args.prompt_model, device)
    
    # Use structure policy algorithm (should match prompt policy)
    method = struct_algo.lower()  # e.g., "ppo" or "grpo"
    
    results = evaluate(
        structure_policy, prompt_policy, cfg,
        num_episodes=num_episodes,
        deterministic=not args.stochastic,
        verbose=not args.quiet,
        use_api=args.api,
        api_model=args.api_model,
        hf_model=args.hf_model,
        temperature=args.temperature
    )
    
    # Save logs
    log_file = args.log_file
    if log_file is None:
        # Auto-generate path: eval_logs/<method>/<dataset_name>/<model_name>/eval_<timestamp>.json
        # Extract just the model name (after '/') and replace '.' with '_'
        model_suffix = model_name.split("/")[-1].replace(".", "_").replace(":", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join("eval_logs", method, cfg.DATASET_NAME, model_suffix)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"eval_{timestamp}.json")
    
    # Build complete log with metadata
    log_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "dataset": cfg.DATASET_NAME,
            "num_episodes": num_episodes,
            "structure_model": args.structure_model,
            "prompt_model": args.prompt_model,
            "model": model_name,
            "api_mode": args.api,
            "deterministic": not args.stochastic,
            "temperature": args.temperature
        },
        "summary": {
            "accuracy": results["accuracy"],
            "avg_reward": results["avg_reward"],
            "avg_tools": results["avg_tools"],
            "avg_tokens": results["avg_tokens"],
            "workflow_distribution": results["workflow_distribution"]
        },
        "episodes": results["episodes"]
    }
    
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\n  Logs saved to: {log_file}")


if __name__ == "__main__":
    main()