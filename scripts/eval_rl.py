"""
Evaluation script for RL learned controller.
Supports both single-step and multi-step environments.

Usage:
    python scripts/eval_rl.py --config multi_step                    # Use latest model
    python scripts/eval_rl.py --config single_step --model path/to/model
    python scripts/eval_rl.py --config multi_step --episodes 100
"""
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob
import time

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from configs import load_config
from env import GeneralAgentEnv, MultiStepAgentEnv


def get_latest_model():
    """Find the latest model in the models directory."""
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    list_of_files = glob.glob(os.path.join(models_dir, "*.zip"))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    # Return path without extension as PPO.load expects
    return os.path.splitext(latest_file)[0]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate RL Controller",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="multi_step",
        choices=["single_step", "multi_step"],
        help="Configuration to use (must match how model was trained)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model (without .zip). If not provided, uses latest model."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["gsm8k", "hotpotqa", "gaia", "medqa", "aime25"],
        help="Override dataset from config"
    )
    return parser.parse_args()


def run_eval(cfg, model_path: str = None, num_episodes: int = 30, dataset_override: str = None):
    """
    Run evaluation for RL controller.
    
    Args:
        cfg: Configuration module
        model_path: Path to trained model (without .zip extension)
        num_episodes: Number of episodes
        dataset_override: Optional dataset override
    """
    dataset_name = dataset_override or cfg.DATASET_NAME
    
    # Resolve model path
    if model_path is None:
        model_path = get_latest_model()
        if model_path is None:
            print("Error: No model found in models/ directory.")
            return None
            
    print(f"\n{'='*70}")
    print(f"EVALUATION: RL Learned Controller")
    print(f"{'='*70}")
    print(f"  Config:     {cfg.ENV_MODE}")
    print(f"  Dataset:    {dataset_name}")
    print(f"  Episodes:   {num_episodes}")
    print(f"  Model:      {os.path.basename(model_path)}")
    print(f"{'='*70}\n")
    
    # Setup environment based on mode
    if cfg.ENV_MODE == "multi_step":
        print("Using MultiStepAgentEnv")
        env = DummyVecEnv([lambda: MultiStepAgentEnv(cfg)])
    else:
        print("Using GeneralAgentEnv")
        env = DummyVecEnv([lambda: GeneralAgentEnv(cfg)])
    
    # Load RL model
    real_path = model_path if model_path.endswith('.zip') else model_path + ".zip"
    if not os.path.exists(real_path):
        print(f"Error: Model file not found: {real_path}")
        return None
        
    print(f"Loading model: {real_path}")
    rl_model = PPO.load(model_path, device='cpu')
    
    # Results container
    detailed_results = []
    
    print("\nRunning evaluation...", flush=True)
    start_time = time.time()
    
    accuracies = []
    steps_list = []
    tools_list = []
    tokens_list = []
    workflows = []
    rewards_list = []
    decision_steps_list = []  # Track multi-step decision count
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        decision_steps = 0
        
        # Multi-step loop: keep stepping until episode terminates
        while not done:
            action, _ = rl_model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            
            episode_reward += float(rewards[0])
            decision_steps += 1
            done = dones[0]
        
        # Extract final info (only available when episode ends)
        info = infos[0]
        
        accuracies.append(1 if info["correct"] else 0)
        steps_list.append(info["steps_taken"])  # LLM calls
        tools_list.append(info["tools_used"])
        tokens_list.append(info["total_tokens"])
        workflows.append(info["workflow"])
        rewards_list.append(episode_reward)
        decision_steps_list.append(decision_steps)
        
        # Store detailed result
        detailed_results.append({
            "episode": ep,
            "correct": info["correct"],
            "workflow": info["workflow"],
            "llm_steps": info["steps_taken"],
            "decision_steps": decision_steps,  # RL decisions made
            "tools": info["tools_used"],
            "tokens": info["total_tokens"],
            "reward": episode_reward,
            "reasoner_tools": str(info.get("reasoner_tools", [])),
            "verifier_tools": str(info.get("verifier_tools", [])),
            "reasoner_budget": info.get("reasoner_budget", "N/A"),
            "verifier_budget": info.get("verifier_budget", "N/A"),
            "answerer_budget": info.get("answerer_budget", "N/A"),
        })
        
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{num_episodes} - Running Acc: {np.mean(accuracies):.1%}")
    
    elapsed = time.time() - start_time
    
    # Calculate metrics
    avg_acc = np.mean(accuracies)
    avg_steps = np.mean(steps_list)
    avg_decision_steps = np.mean(decision_steps_list)
    avg_tools = np.mean(tools_list)
    avg_tokens = np.mean(tokens_list)
    avg_reward = np.mean(rewards_list)
    
    # Print results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"  Accuracy:           {avg_acc:.1%} ({sum(accuracies)}/{num_episodes})")
    print(f"  Avg Reward:         {avg_reward:.3f}")
    print(f"  Avg LLM Steps:      {avg_steps:.2f}")
    print(f"  Avg Decision Steps: {avg_decision_steps:.2f}")
    print(f"  Avg Tools Used:     {avg_tools:.2f}")
    print(f"  Avg Tokens:         {avg_tokens:.0f}")
    print(f"  Time:               {elapsed:.1f}s")
    print("="*70)
    
    # Workflow distribution
    detailed_df = pd.DataFrame(detailed_results)
    print("\nWorkflow Distribution:")
    print(detailed_df["workflow"].value_counts().to_string())
    
    # Save detailed results
    output_path = f"eval_results_{cfg.ENV_MODE}_{dataset_name}_{int(time.time())}.csv"
    detailed_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")
    
    return detailed_df


def analyze_results(detailed_df: pd.DataFrame):
    """Analyze evaluation results."""
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    # Statistics
    print("\nStatistics:")
    print(f"  Correct:  {detailed_df['correct'].sum()} / {len(detailed_df)}")
    print(f"  Accuracy: {detailed_df['correct'].mean():.1%}")
    print(f"  Reward:   {detailed_df['reward'].mean():.3f} ± {detailed_df['reward'].std():.3f}")
    print(f"  Tokens:   {detailed_df['tokens'].mean():.0f} ± {detailed_df['tokens'].std():.0f}")
    
    # Workflow distribution
    print("\nWorkflow Distribution:")
    workflow_dist = detailed_df["workflow"].value_counts()
    for workflow, count in workflow_dist.items():
        pct = count / len(detailed_df) * 100
        print(f"  {workflow}: {count} ({pct:.1f}%)")
    
    # Accuracy by workflow
    print("\nAccuracy by Workflow:")
    for workflow in detailed_df["workflow"].unique():
        subset = detailed_df[detailed_df["workflow"] == workflow]
        acc = subset["correct"].mean()
        print(f"  {workflow}: {acc:.1%} (n={len(subset)})")


if __name__ == "__main__":
    args = parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    print(f"Loaded config: {args.config}")
    
    # Run evaluation
    detailed_df = run_eval(
        cfg=cfg,
        model_path=args.model,
        num_episodes=args.episodes,
        dataset_override=args.dataset
    )
    
    if detailed_df is not None:
        analyze_results(detailed_df)
