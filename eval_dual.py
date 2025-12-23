"""
Evaluation Script for Dual-Policy Hierarchical RL

Evaluates the trained structure and prompt policies together.
Also supports comparison with random baselines.

Usage:
    python eval_dual.py --structure-model models/structure_policy.pt --prompt-model models/prompt_policy.pt
    python eval_dual.py --structure-model models/structure_policy.pt --prompt-model models/prompt_policy.pt --episodes 20
"""
import argparse
import time
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
from torch.distributions import Categorical

from configs import load_config
from env.structure_env import StructureEnv
from env.prompt_env import PromptEnv


# Prompt atom lookup for human-readable output
PROMPT_ATOMS = {
    "reasoner": [
        "(DONE)",
        "Think step by step",
        "Break down the problem",
        "Identify the key quantities",
        "Write out your reasoning",
        "Check your arithmetic",
        "Consider edge cases",
    ],
    "verifier": [
        "(DONE)",
        "Verify the solution step-by-step",
        "Check if the answer makes sense",
        "Look for calculation errors",
        "Confirm the logic is sound",
        "Double-check the final answer",
        "Validate units and quantities",
    ],
    "answerer": [
        "(DONE)",
        "Based on the analysis, the answer is",
        "Therefore, the final answer is",
        "Summarizing the solution",
        "The result is",
    ],
}

BUDGET_NAMES = ["Low (64 tokens)", "Mid (256 tokens)", "High (512 tokens)"]
WORKFLOW_NAMES = ["Direct", "Reason+Ans", "Reason+Verify+Ans"]


def get_prompt_names(prompt_indices, agent_type):
    """Convert prompt indices to human-readable names."""
    atoms = PROMPT_ATOMS.get(agent_type, PROMPT_ATOMS["answerer"])
    names = []
    for idx in prompt_indices:
        if 0 < idx < len(atoms):  # Skip 0 (DONE)
            names.append(atoms[idx])
        elif idx == 0:
            continue  # DONE action, don't show
        else:
            names.append(f"Prompt #{idx}")
    return names if names else ["(default)"]


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
        
        # Move to same device as model
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
    
    def get_action(self, obs, deterministic=False, return_probs=False):
        """Get MultiDiscrete action from observation."""
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        # Move to same device as model
        device = next(self.parameters()).device
        obs = obs.to(device)
            
        action_logits_list, value = self.forward(obs)
        
        actions = []
        probs_list = []
        for logits in action_logits_list:
            probs = torch.softmax(logits, dim=-1)
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                dist = Categorical(probs)
                action = dist.sample()
            actions.append(action.item())
            if return_probs:
                probs_list.append(probs.cpu().numpy()[0])
        
        if return_probs:
            return np.array(actions), probs_list
        return np.array(actions)


def load_structure_policy(model_path: str, device="cpu"):
    """Load trained structure policy."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    obs_dim = checkpoint["obs_dim"]
    action_dims = checkpoint["action_dims"]
    
    policy = MultiDiscretePolicy(obs_dim, action_dims).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    
    return policy


def load_prompt_policy(model_path: str, device="cpu"):
    """Load trained prompt policy."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    obs_dim = checkpoint["obs_dim"]
    action_dim = checkpoint["action_dim"]
    
    policy = PolicyNetwork(obs_dim, action_dim).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    
    return policy


def evaluate_dual_policy(structure_policy, prompt_policy, cfg, 
                         num_episodes: int = 10, deterministic: bool = True,
                         verbose: bool = True):
    """
    Evaluate the dual policy system.
    
    Returns:
        Dict of metrics
    """
    structure_env = StructureEnv(cfg)
    prompt_env = PromptEnv(cfg)
    
    results = {
        "correct": [],
        "workflows": [],
        "total_tokens": [],
        "rewards": [],
        "prompt_steps": [],
    }
    
    print(f"\nEvaluating Dual Policy on {num_episodes} episodes...")
    if deterministic:
        print("  Note: Using deterministic mode (argmax). Structure SHOULD vary if policy learned to differentiate questions.")
        print("  If structure is always same, policy may have collapsed to prefer one workflow regardless of question.")
    print("-" * 60)
    
    # Store embeddings and actions to check diversity
    observed_embeddings = []
    observed_actions = []
    
    for ep in range(num_episodes):
        # Step 1: Get structure decision
        struct_obs, struct_info = structure_env.reset()
        
        # Store embedding for diversity check
        embedding_part = struct_obs[:struct_obs.shape[0]-8]  # Exclude stats
        observed_embeddings.append(embedding_part[:10])  # Store first 10 dims for comparison
        q_preview = struct_info["question"][:60] + "..." if len(struct_info["question"]) > 60 else struct_info["question"]
        
        with torch.no_grad():
            # Always get probabilities to check if policy is actually differentiating
            struct_action, struct_probs = structure_policy.get_action(
                struct_obs, deterministic=deterministic, return_probs=True
            )
            observed_actions.append(struct_action.copy())
            
            # Show probability analysis for first 3 episodes (debug info)
            if verbose and ep < 3:
                workflow_probs = struct_probs[0]
                print(f"\n  [Debug: Episode {ep+1} Probability Distribution]")
                print(f"    Workflow probs: Direct={workflow_probs[0]:.2%}, Reason+Ans={workflow_probs[1]:.2%}, Reason+Verify+Ans={workflow_probs[2]:.2%}")
                print(f"    → Selected: {WORKFLOW_NAMES[struct_action[0]]}")
        
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
        prompt_step_count = 0
        
        while not done:
            with torch.no_grad():
                prompt_action = prompt_policy.get_action(prompt_obs, deterministic=deterministic)
            prompt_obs, reward, done, _, info = prompt_env.step(prompt_action)
            total_reward += reward
            prompt_step_count += 1
        
        # Record results
        correct = info.get("correct", False)
        results["correct"].append(correct)
        results["workflows"].append(struct_exec_info["workflow"])
        results["total_tokens"].append(info.get("total_tokens", 0))
        results["rewards"].append(total_reward)
        results["prompt_steps"].append(prompt_step_count)
        
        if verbose:
            status = "✓" if correct else "✗"
            structure = struct_exec_info["structure"]
            workflow_name = struct_exec_info['workflow']
            
            # Get prompt details with human-readable names
            reasoner_prompts = info.get("reasoner_prompts", [])
            verifier_prompts = info.get("verifier_prompts", [])
            answerer_prompts = info.get("answerer_prompts", [])
            
            print(f"\n{'='*70}")
            print(f"EPISODE {ep+1}: {status} {'CORRECT' if correct else 'WRONG'}")
            print(f"{'='*70}")
            print(f"  Question: {q_preview}")
            
            print(f"\n  📋 STRUCTURE POLICY DECISION:")
            print(f"     Workflow: {workflow_name}")
            print(f"     Reasoner Budget: {BUDGET_NAMES[structure['reasoner_budget_idx']]}")
            print(f"     Verifier Budget: {BUDGET_NAMES[structure['verifier_budget_idx']]}")
            print(f"     Answerer Budget: {BUDGET_NAMES[structure['answerer_budget_idx']]}")
            
            print(f"\n  📝 PROMPT POLICY DECISIONS:")
            r_prompt_names = get_prompt_names(reasoner_prompts, 'reasoner')
            v_prompt_names = get_prompt_names(verifier_prompts, 'verifier')
            a_prompt_names = get_prompt_names(answerer_prompts, 'answerer')
            
            if reasoner_prompts:
                print(f"     Reasoner: {r_prompt_names}")
            else:
                print(f"     Reasoner: (none - DONE immediately)")
            if workflow_name in ["Reason+Verify+Ans"]:
                if verifier_prompts:
                    print(f"     Verifier: {v_prompt_names}")
                else:
                    print(f"     Verifier: (none - DONE immediately)")
            if answerer_prompts:
                print(f"     Answerer: {a_prompt_names}")
            else:
                print(f"     Answerer: (none - DONE immediately)")
            
            print(f"\n  📊 RESULTS:")
            print(f"     Total Tokens: {info.get('total_tokens', 0)}")
            print(f"     Reward: {total_reward:.3f}")
    
    # Compute summary
    accuracy = np.mean(results["correct"])
    avg_reward = np.mean(results["rewards"])
    avg_tokens = np.mean(results["total_tokens"])
    avg_prompt_steps = np.mean(results["prompt_steps"])
    
    # Workflow distribution
    workflow_counts = defaultdict(int)
    for w in results["workflows"]:
        workflow_counts[w] += 1
    
    # Check diversity of actions
    if len(observed_actions) > 1:
        actions_array = np.array(observed_actions)
        unique_workflows = len(np.unique(actions_array[:, 0]))
        unique_actions = len(set(tuple(a) for a in observed_actions))
        
        # Check embedding diversity
        embeddings_array = np.array(observed_embeddings)
        embedding_variance = np.var(embeddings_array, axis=0).mean()
    
    print("-" * 60)
    print(f"RESULTS:")
    print(f"  Accuracy:         {accuracy*100:.1f}%")
    print(f"  Avg Reward:       {avg_reward:.3f}")
    print(f"  Avg Tokens:       {avg_tokens:.1f}")
    print(f"  Avg Prompt Steps: {avg_prompt_steps:.1f}")
    print(f"\n  Workflow Distribution:")
    for wf, count in sorted(workflow_counts.items()):
        print(f"    {wf}: {count} ({count/num_episodes*100:.1f}%)")
    
    # Show diversity analysis
    if len(observed_actions) > 1:
        print(f"\n  Diversity Analysis:")
        print(f"    Unique workflows chosen: {unique_workflows} out of {len(observed_actions)} episodes")
        print(f"    Unique full action combinations: {unique_actions} out of {len(observed_actions)} episodes")
        print(f"    Avg embedding variance (first 10 dims): {embedding_variance:.6f}")
        if unique_workflows == 1:
            print(f"    ⚠️  WARNING: Policy always chooses same workflow - may have collapsed during training")
        if embedding_variance < 0.001:
            print(f"    ⚠️  WARNING: Very low embedding variance - questions may be too similar")
    
    return {
        "accuracy": accuracy,
        "avg_reward": avg_reward,
        "avg_tokens": avg_tokens,
        "avg_prompt_steps": avg_prompt_steps,
        "workflow_distribution": dict(workflow_counts),
        "raw_results": results,
    }


def evaluate_random_baseline(cfg, num_episodes: int = 10, verbose: bool = True):
    """
    Evaluate with random actions for comparison.
    """
    structure_env = StructureEnv(cfg)
    prompt_env = PromptEnv(cfg)
    
    results = {
        "correct": [],
        "workflows": [],
        "total_tokens": [],
        "rewards": [],
    }
    
    print(f"\nEvaluating Random Baseline on {num_episodes} episodes...")
    print("-" * 60)
    
    for ep in range(num_episodes):
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
        
        while not done:
            prompt_action = prompt_env.action_space.sample()
            prompt_obs, reward, done, _, info = prompt_env.step(prompt_action)
            total_reward += reward
        
        results["correct"].append(info.get("correct", False))
        results["workflows"].append(struct_exec_info["workflow"])
        results["total_tokens"].append(info.get("total_tokens", 0))
        results["rewards"].append(total_reward)
        
        if verbose:
            status = "✓" if info.get("correct") else "✗"
            print(f"  Episode {ep+1:3d}: {status} | Workflow: {struct_exec_info['workflow']:20s} | "
                  f"Tokens: {info.get('total_tokens', 0):4d} | Reward: {total_reward:.3f}")
    
    accuracy = np.mean(results["correct"])
    avg_reward = np.mean(results["rewards"])
    avg_tokens = np.mean(results["total_tokens"])
    
    workflow_counts = defaultdict(int)
    for w in results["workflows"]:
        workflow_counts[w] += 1
    
    print("-" * 60)
    print(f"RANDOM BASELINE RESULTS:")
    print(f"  Accuracy:   {accuracy*100:.1f}%")
    print(f"  Avg Reward: {avg_reward:.3f}")
    print(f"  Avg Tokens: {avg_tokens:.1f}")
    print(f"\n  Workflow Distribution:")
    for wf, count in sorted(workflow_counts.items()):
        print(f"    {wf}: {count} ({count/num_episodes*100:.1f}%)")
    
    return {
        "accuracy": accuracy,
        "avg_reward": avg_reward,
        "avg_tokens": avg_tokens,
        "workflow_distribution": dict(workflow_counts),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate dual-policy hierarchical RL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", type=str, default="hierarchical",
        help="Configuration to use"
    )
    parser.add_argument(
        "--structure-model", type=str, default=None,
        help="Path to trained structure policy (.pt file)"
    )
    parser.add_argument(
        "--prompt-model", type=str, default=None,
        help="Path to trained prompt policy (.pt file)"
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Use stochastic actions instead of deterministic"
    )
    parser.add_argument(
        "--random-only", action="store_true",
        help="Only evaluate random baseline (no model needed)"
    )
    parser.add_argument(
        "--compare-random", action="store_true",
        help="Also evaluate random baseline for comparison"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=["gsm8k", "hotpotqa"],
        help="Override dataset"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Apply overrides
    if args.dataset:
        cfg.DATASET_NAME = args.dataset
    
    print("=" * 60)
    print("DUAL POLICY EVALUATION")
    print("=" * 60)
    print(f"  Config:          {args.config}")
    print(f"  Dataset:         {cfg.DATASET_NAME}")
    
    if args.random_only:
        print(f"  Mode:            Random Baseline Only")
        print(f"  Episodes:        {args.episodes}")
        print("=" * 60)
        
        evaluate_random_baseline(cfg, num_episodes=args.episodes)
        return
    
    if not args.structure_model or not args.prompt_model:
        print("\nError: --structure-model and --prompt-model required unless --random-only")
        return
    
    print(f"  Structure Model: {args.structure_model}")
    print(f"  Prompt Model:    {args.prompt_model}")
    print(f"  Episodes:        {args.episodes}")
    print(f"  Deterministic:   {not args.stochastic}")
    print("=" * 60)
    
    # Load models
    print("\nLoading models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    structure_policy = load_structure_policy(args.structure_model, device)
    prompt_policy = load_prompt_policy(args.prompt_model, device)
    print(f"  Models loaded successfully! (device: {device})")
    
    # Evaluate dual policy
    dual_results = evaluate_dual_policy(
        structure_policy, prompt_policy, cfg,
        num_episodes=args.episodes,
        deterministic=not args.stochastic
    )
    
    # Optionally compare with random
    if args.compare_random:
        random_results = evaluate_random_baseline(cfg, num_episodes=args.episodes)
        
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Metric':<20} {'Dual Policy':>15} {'Random':>15}")
        print("-" * 50)
        print(f"{'Accuracy':<20} {dual_results['accuracy']*100:>14.1f}% {random_results['accuracy']*100:>14.1f}%")
        print(f"{'Avg Reward':<20} {dual_results['avg_reward']:>15.3f} {random_results['avg_reward']:>15.3f}")
        print(f"{'Avg Tokens':<20} {dual_results['avg_tokens']:>15.1f} {random_results['avg_tokens']:>15.1f}")


if __name__ == "__main__":
    main()
