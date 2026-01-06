"""
Unified Evaluation Script for Hierarchical RL

Evaluates trained structure and prompt policies.

Usage:
    python scripts/eval_hrl.py --structure-model models/grpo_models/structure.pt --prompt-model models/grpo_models/prompt.pt
    python scripts/eval_hrl.py --structure-model models/ppo_models/structure.pt --prompt-model models/ppo_models/prompt.pt --episodes 50
"""
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from collections import defaultdict

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


def decode_tools(idx):
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
    
    def get_action(self, obs, deterministic=True):
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.to(next(self.parameters()).device)
        
        features = self.network(obs)
        logits = self.action_head(features)
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
    
    def get_action(self, obs, deterministic=True):
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.to(next(self.parameters()).device)
        
        features = self.network(obs)
        actions = []
        for head in self.action_heads:
            logits = head(features)
            probs = torch.softmax(logits, dim=-1)
            if deterministic:
                actions.append(torch.argmax(probs, dim=-1).item())
            else:
                actions.append(Categorical(probs).sample().item())
        return np.array(actions)


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
    return policy


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
    return policy



# EVALUATION
def evaluate(structure_policy, prompt_policy, cfg, num_episodes=20, 
             deterministic=True, verbose=True):
    """Evaluate dual policy system."""
    structure_env = StructureEnv(cfg, is_eval=True)
    prompt_env = PromptEnv(cfg, is_eval=True)
    
    results = {"correct": [], "workflows": [], "tokens": [], "rewards": [], "tools": []}
    
    print(f"\nEvaluating on {num_episodes} episodes...")
    print("-" * 60)
    
    for ep in range(num_episodes):
        # Structure decision
        struct_obs, struct_info = structure_env.reset()
        
        with torch.no_grad():
            struct_action = structure_policy.get_action(struct_obs, deterministic)
        
        _, _, _, _, struct_exec_info = structure_env.step(struct_action)
        
        # Setup prompt env
        prompt_env.set_structure(
            question=struct_exec_info["question"],
            answer=struct_exec_info["answer"],
            embedding=struct_exec_info["embedding"],
            structure=struct_exec_info["structure"]
        )
        
        # Prompt rollout
        prompt_obs, _ = prompt_env.reset()
        total_reward = 0.0
        done = False
        
        while not done:
            with torch.no_grad():
                action = prompt_policy.get_action(prompt_obs, deterministic)
            prompt_obs, reward, done, _, info = prompt_env.step(action)
            total_reward += reward
        
        # Record
        correct = info.get("correct", False)
        results["correct"].append(correct)
        results["workflows"].append(struct_exec_info["workflow"])
        results["tokens"].append(info.get("total_tokens", 0))
        results["rewards"].append(total_reward)
        results["tools"].append(info.get("tools_used", 0))
        
        if verbose:
            status = "✓" if correct else "✗"
            q = struct_info["question"][:50] + "..." if len(struct_info["question"]) > 50 else struct_info["question"]
            print(f"  [{ep+1:3d}] {status} | {struct_exec_info['workflow']:20s} | "
                  f"Tools: {info.get('tools_used', 0)} | Reward: {total_reward:.2f}")
    
    # Summary
    accuracy = np.mean(results["correct"]) * 100
    avg_reward = np.mean(results["rewards"])
    avg_tools = np.mean(results["tools"])
    
    workflow_counts = defaultdict(int)
    for w in results["workflows"]:
        workflow_counts[w] += 1
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Accuracy:    {accuracy:.1f}%")
    print(f"  Avg Reward:  {avg_reward:.3f}")
    print(f"  Avg Tools:   {avg_tools:.2f}")
    print(f"\n  Workflows:")
    for w, c in sorted(workflow_counts.items()):
        print(f"    {w}: {c} ({c/num_episodes*100:.1f}%)")
    
    return {"accuracy": accuracy, "avg_reward": avg_reward, "avg_tools": avg_tools}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate hierarchical RL")
    parser.add_argument("--config", type=str, default="hierarchical")
    parser.add_argument("--structure-model", type=str, required=True)
    parser.add_argument("--prompt-model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--dataset", type=str, default=None, choices=["gsm8k", "hotpotqa", "gaia", "medqa", "aime25"])
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    
    cfg = load_config(args.config)
    if args.dataset:
        cfg.DATASET_NAME = args.dataset
    
    print("=" * 60)
    print("HIERARCHICAL RL EVALUATION")
    print("=" * 60)
    
    print(f"  Structure: {args.structure_model}")
    print(f"  Prompt:    {args.prompt_model}")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading models (device: {device})...")
    
    structure_policy = load_structure_policy(args.structure_model, device)
    prompt_policy = load_prompt_policy(args.prompt_model, device)
    
    results = evaluate(
        structure_policy, prompt_policy, cfg,
        num_episodes=args.episodes,
        deterministic=not args.stochastic,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()

