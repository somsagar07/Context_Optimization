"""
Benchmark script for Hierarchical (Dual-Policy) models.

Evaluates:
- Hierarchical models (structure + prompt policies from train_dual.py)
- Random configurations
- Optionally: Single-step and multi-step models for comparison

Usage:
    # Evaluate hierarchical model + random baseline
    python benchmark_h.py --structure-model models/structure_policy.pt --prompt-model models/prompt_policy.pt
    
    # Include single-step and multi-step models for comparison
    python benchmark_h.py --structure-model models/structure_policy.pt --prompt-model models/prompt_policy.pt \
                          --single-step-model models/controller_single_step.zip \
                          --multi-step-model models/controller_multi_step.zip
    
    # Just random baseline
    python benchmark_h.py --random-only
    
    # Find latest models automatically
    python benchmark_h.py
"""
import argparse
import glob
import gc
import os
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
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        features = self.network(x)
        action_logits = self.action_head(features)
        value = self.value_head(features)
        return action_logits, value
    
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
    
    def __init__(self, obs_dim: int, action_dims: list, hidden_dim: int = 256):
        super().__init__()
        self.action_dims = action_dims
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.action_heads = nn.ModuleList([
            nn.Linear(hidden_dim, dim) for dim in action_dims
        ])
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        features = self.network(x)
        action_logits = [head(features) for head in self.action_heads]
        value = self.value_head(features)
        return action_logits, value
    
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
    structure_policy = MultiDiscretePolicy(struct_obs_dim, struct_action_dims).to(device)
    structure_policy.load_state_dict(struct_checkpoint["model_state_dict"])
    structure_policy.eval()
    
    # Load prompt policy
    prompt_checkpoint = torch.load(prompt_path, map_location=device, weights_only=False)
    prompt_obs_dim = prompt_checkpoint["obs_dim"]
    prompt_action_dim = prompt_checkpoint["action_dim"]
    prompt_policy = PolicyNetwork(prompt_obs_dim, prompt_action_dim).to(device)
    prompt_policy.load_state_dict(prompt_checkpoint["model_state_dict"])
    prompt_policy.eval()
    
    # Clear cache after loading
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return structure_policy, prompt_policy


def get_latest_hierarchical_models():
    """Find latest hierarchical model files."""
    models_dir = "models"
    
    struct_pattern = os.path.join(models_dir, "structure_policy_*.pt")
    prompt_pattern = os.path.join(models_dir, "prompt_policy_*.pt")
    
    struct_files = glob.glob(struct_pattern)
    prompt_files = glob.glob(prompt_pattern)
    
    struct_path = None
    prompt_path = None
    
    if struct_files:
        struct_path = max(struct_files, key=os.path.getctime)
    if prompt_files:
        prompt_path = max(prompt_files, key=os.path.getctime)
    
    return struct_path, prompt_path


def get_latest_sb3_model(env_mode: str):
    """Find latest stable-baselines3 model for single-step or multi-step."""
    models_dir = "models"
    pattern = os.path.join(models_dir, f"controller_{env_mode}_*.zip")
    files = glob.glob(pattern)
    if files:
        return max(files, key=os.path.getctime)
    return None


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
    }


def run_random_hierarchical_episode(structure_env, prompt_env):
    """Run a single hierarchical episode with random actions using provided environments."""
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
    }


def run_benchmark(cfg, num_episodes=30, 
                 structure_model_path=None, prompt_model_path=None,
                 single_step_model_path=None, multi_step_model_path=None,
                 include_random=True, use_cpu=False):
    """Run comprehensive benchmark."""
    print(f"\n{'='*70}")
    print(f"BENCHMARK: Hierarchical Models")
    print(f"{'='*70}")
    print(f"  Dataset:     {cfg.DATASET_NAME}")
    print(f"  Episodes:    {num_episodes}")
    print(f"{'='*70}\n")
    
    results = []
    detailed_results = []
    
    # Use CPU if requested or if CUDA not available
    if use_cpu or not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"
    
    print(f"Using device: {device}")
    
    # Clear cache at start
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create shared environments ONCE (suppress loading messages)
    print("Initializing environments...", end=" ", flush=True)
    with SuppressOutput():
        structure_env = StructureEnv(cfg)
        prompt_env = PromptEnv(cfg)
    print("Done!")
    
    # ========================================
    # 1. Hierarchical Model (if provided)
    # ========================================
    if structure_model_path and prompt_model_path:
        print(f"\nLoading hierarchical models...")
        print(f"  Structure: {structure_model_path}")
        print(f"  Prompt:    {prompt_model_path}")
        
        try:
            structure_policy, prompt_policy = load_hierarchical_models(
                structure_model_path, prompt_model_path, device
            )
            print(f"  Models loaded successfully! (device: {device})")
            
            print(f"\n[Hierarchical (Learned)]")
            start_time = time.time()
            
            accuracies = []
            rewards = []
            tokens = []
            workflows = []
            decision_steps = []
            
            for ep in range(num_episodes):
                result = run_hierarchical_episode(
                    structure_policy, prompt_policy, structure_env, prompt_env, deterministic=True
                )
                
                accuracies.append(1 if result["correct"] else 0)
                rewards.append(result["reward"])
                tokens.append(result["total_tokens"])
                workflows.append(result["workflow"])
                decision_steps.append(result["decision_steps"])
                
                detailed_results.append({
                    "strategy": "Hierarchical (Learned)",
                    "episode": ep,
                    **result
                })
                
                # Progress indicator
                acc_so_far = np.mean(accuracies)
                status = "✓" if result["correct"] else "✗"
                print(f"\r  Progress: {ep+1}/{num_episodes} | Running Acc: {acc_so_far:.1%} | Last: {status}", end="", flush=True)
                
                # Clear cache every 10 episodes to prevent accumulation
                if (ep + 1) % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
            
            elapsed = time.time() - start_time
            avg_acc = np.mean(accuracies)
            avg_reward = np.mean(rewards)
            avg_tokens = np.mean(tokens)
            avg_decisions = np.mean(decision_steps)
            
            results.append({
                "Strategy": "Hierarchical (Learned)",
                "Accuracy": f"{avg_acc:.1%}",
                "Avg Reward": f"{avg_reward:.3f}",
                "Avg Tokens": f"{avg_tokens:.0f}",
                "Avg Decisions": f"{avg_decisions:.1f}",
                "Time (s)": f"{elapsed:.1f}"
            })
            
            print(f"\n  Final: Accuracy={avg_acc:.1%}, Reward={avg_reward:.3f}, Tokens={avg_tokens:.0f}")
            
        except Exception as e:
            print(f"\nError loading hierarchical models: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================
    # 2. Random Hierarchical Baseline
    # ========================================
    if include_random:
        print(f"\n[Random (Hierarchical)]")
        start_time = time.time()
        
        accuracies = []
        rewards = []
        tokens = []
        workflows = []
        decision_steps = []
        
        for ep in range(num_episodes):
            result = run_random_hierarchical_episode(structure_env, prompt_env)
            
            accuracies.append(1 if result["correct"] else 0)
            rewards.append(result["reward"])
            tokens.append(result["total_tokens"])
            workflows.append(result["workflow"])
            decision_steps.append(result["decision_steps"])
            
            detailed_results.append({
                "strategy": "Random (Hierarchical)",
                "episode": ep,
                **result
            })
            
            # Progress indicator
            acc_so_far = np.mean(accuracies)
            status = "✓" if result["correct"] else "✗"
            print(f"\r  Progress: {ep+1}/{num_episodes} | Running Acc: {acc_so_far:.1%} | Last: {status}", end="", flush=True)
            
            # Clear cache periodically
            if (ep + 1) % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        elapsed = time.time() - start_time
        avg_acc = np.mean(accuracies)
        avg_reward = np.mean(rewards)
        avg_tokens = np.mean(tokens)
        avg_decisions = np.mean(decision_steps)
        
        results.append({
            "Strategy": "Random (Hierarchical)",
            "Accuracy": f"{avg_acc:.1%}",
            "Avg Reward": f"{avg_reward:.3f}",
            "Avg Tokens": f"{avg_tokens:.0f}",
            "Avg Decisions": f"{avg_decisions:.1f}",
            "Time (s)": f"{elapsed:.1f}"
        })
        
        print(f"\n  Final: Accuracy={avg_acc:.1%}, Reward={avg_reward:.3f}, Tokens={avg_tokens:.0f}")
    
    # ========================================
    # 3. Single-Step Model (optional)
    # ========================================
    if single_step_model_path:
        print(f"\nLoading single-step model: {single_step_model_path}")
        try:
            with SuppressOutput():
                single_step_model = PPO.load(single_step_model_path, device='cpu')
                single_cfg = load_config("single_step")
                single_env = DummyVecEnv([lambda: GeneralAgentEnv(single_cfg)])
            
            print(f"\n[Single-Step (Learned)]")
            start_time = time.time()
            
            accuracies = []
            rewards = []
            tokens = []
            workflows = []
            decision_steps = []
            
            for ep in range(num_episodes):
                result = run_single_step_episode(single_env, single_step_model, deterministic=True)
                
                accuracies.append(1 if result["correct"] else 0)
                rewards.append(result["reward"])
                tokens.append(result["total_tokens"])
                workflows.append(result["workflow"])
                decision_steps.append(result["decision_steps"])
                
                detailed_results.append({
                    "strategy": "Single-Step (Learned)",
                    "episode": ep,
                    **result
                })
                
                # Progress indicator
                acc_so_far = np.mean(accuracies)
                status = "✓" if result["correct"] else "✗"
                print(f"\r  Progress: {ep+1}/{num_episodes} | Running Acc: {acc_so_far:.1%} | Last: {status}", end="", flush=True)
            
            elapsed = time.time() - start_time
            avg_acc = np.mean(accuracies)
            avg_reward = np.mean(rewards)
            avg_tokens = np.mean(tokens)
            avg_decisions = np.mean(decision_steps)
            
            results.append({
                "Strategy": "Single-Step (Learned)",
                "Accuracy": f"{avg_acc:.1%}",
                "Avg Reward": f"{avg_reward:.3f}",
                "Avg Tokens": f"{avg_tokens:.0f}",
                "Avg Decisions": f"{avg_decisions:.1f}",
                "Time (s)": f"{elapsed:.1f}"
            })
            
            print(f"\n  Final: Accuracy={avg_acc:.1%}, Reward={avg_reward:.3f}, Tokens={avg_tokens:.0f}")
            
        except Exception as e:
            print(f"\nError loading single-step model: {e}")
    
    # ========================================
    # 4. Multi-Step Model (optional)
    # ========================================
    if multi_step_model_path:
        print(f"\nLoading multi-step model: {multi_step_model_path}")
        try:
            with SuppressOutput():
                multi_step_model = PPO.load(multi_step_model_path, device='cpu')
                multi_cfg = load_config("multi_step")
                multi_env = DummyVecEnv([lambda: MultiStepAgentEnv(multi_cfg)])
            
            print(f"\n[Multi-Step (Learned)]")
            start_time = time.time()
            
            accuracies = []
            rewards = []
            tokens = []
            workflows = []
            decision_steps = []
            
            for ep in range(num_episodes):
                result = run_multi_step_episode(multi_env, multi_step_model, deterministic=True)
                
                accuracies.append(1 if result["correct"] else 0)
                rewards.append(result["reward"])
                tokens.append(result["total_tokens"])
                workflows.append(result["workflow"])
                decision_steps.append(result["decision_steps"])
                
                detailed_results.append({
                    "strategy": "Multi-Step (Learned)",
                    "episode": ep,
                    **result
                })
                
                # Progress indicator
                acc_so_far = np.mean(accuracies)
                status = "✓" if result["correct"] else "✗"
                print(f"\r  Progress: {ep+1}/{num_episodes} | Running Acc: {acc_so_far:.1%} | Last: {status}", end="", flush=True)
            
            elapsed = time.time() - start_time
            avg_acc = np.mean(accuracies)
            avg_reward = np.mean(rewards)
            avg_tokens = np.mean(tokens)
            avg_decisions = np.mean(decision_steps)
            
            results.append({
                "Strategy": "Multi-Step (Learned)",
                "Accuracy": f"{avg_acc:.1%}",
                "Avg Reward": f"{avg_reward:.3f}",
                "Avg Tokens": f"{avg_tokens:.0f}",
                "Avg Decisions": f"{avg_decisions:.1f}",
                "Time (s)": f"{elapsed:.1f}"
            })
            
            print(f"\n  Final: Accuracy={avg_acc:.1%}, Reward={avg_reward:.3f}, Tokens={avg_tokens:.0f}")
            
        except Exception as e:
            print(f"\nError loading multi-step model: {e}")
    
    # ========================================
    # Print Results
    # ========================================
    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(df.to_string(index=False))
        print("="*70)
        
        # Workflow distribution
        if detailed_results:
            detailed_df = pd.DataFrame(detailed_results)
            print("\nWorkflow Distribution:")
            workflow_dist = detailed_df.groupby(["strategy", "workflow"]).size().unstack(fill_value=0)
            print(workflow_dist)
            
            # Save detailed results
            output_path = f"benchmark_results_hierarchical_{cfg.DATASET_NAME}_{int(time.time())}.csv"
            detailed_df.to_csv(output_path, index=False)
            print(f"\nDetailed results saved to: {output_path}")
            
            return df, detailed_df
    
    return None, None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Hierarchical Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", type=str, default="hierarchical",
        help="Configuration to use"
    )
    parser.add_argument(
        "--structure-model", type=str, default=None,
        help="Path to structure policy model (.pt file). If not provided, searches for latest."
    )
    parser.add_argument(
        "--prompt-model", type=str, default=None,
        help="Path to prompt policy model (.pt file). If not provided, searches for latest."
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
        "--episodes", type=int, default=30,
        help="Number of episodes per strategy"
    )
    parser.add_argument(
        "--random-only", action="store_true",
        help="Only run random baseline (no models needed)"
    )
    parser.add_argument(
        "--no-random", action="store_true",
        help="Skip random baseline"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=["gsm8k", "hotpotqa"],
        help="Override dataset"
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
    
    # Find models if not provided
    if not args.random_only:
        if not args.structure_model or not args.prompt_model:
            struct_path, prompt_path = get_latest_hierarchical_models()
            if struct_path and prompt_path:
                args.structure_model = struct_path
                args.prompt_model = prompt_path
                print(f"Found latest models:")
                print(f"  Structure: {struct_path}")
                print(f"  Prompt:    {prompt_path}")
            elif not args.structure_model and not args.prompt_model:
                print("Warning: No hierarchical models found. Running random baseline only.")
                args.random_only = True
    
    # Find single-step/multi-step models if not provided but requested
    if args.single_step_model is None:
        single_step_path = get_latest_sb3_model("single_step")
        if single_step_path:
            args.single_step_model = single_step_path
            print(f"Found single-step model: {single_step_path}")
    
    if args.multi_step_model is None:
        multi_step_path = get_latest_sb3_model("multi_step")
        if multi_step_path:
            args.multi_step_model = multi_step_path
            print(f"Found multi-step model: {multi_step_path}")
    
    # Run benchmark
    run_benchmark(
        cfg=cfg,
        num_episodes=args.episodes,
        structure_model_path=args.structure_model if not args.random_only else None,
        prompt_model_path=args.prompt_model if not args.random_only else None,
        single_step_model_path=args.single_step_model,
        multi_step_model_path=args.multi_step_model,
        include_random=not args.no_random,
        use_cpu=args.cpu
    )


if __name__ == "__main__":
    main()

