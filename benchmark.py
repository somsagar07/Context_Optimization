"""
Benchmark script to compare the trained RL controller against baseline strategies.
"""

import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from env import GeneralAgentEnv
import config
import os
import time

def run_benchmark(model_path: str = None, num_episodes: int = 30):
    """
    Run benchmark comparing RL controller vs baseline strategies.
    
    Args:
        model_path: Path to trained model (without .zip extension)
        num_episodes: Number of episodes per strategy
    """
    print(f"\n{'='*70}")
    print(f"BENCHMARKING: {num_episodes} Episodes per Strategy")
    print(f"Dataset: {config.DATASET_NAME}")
    print(f"{'='*70}\n")
    
    # Setup environment
    env = DummyVecEnv([lambda: GeneralAgentEnv()])
    
    # Define baseline strategies
    # Action format: [workflow, r_tools, r_budget, v_tools, v_budget, a_budget]
    # workflow: 0=Direct, 1=Reason+Ans, 2=Full
    # tools: 0=None, 1=Calc, 2=Search, 3=C+S, 4=Python, 5=C+P, 6=S+P, 7=All
    # budget: 0=Low, 1=Mid, 2=High
    
    strategies = {}
    
    # Load RL model FIRST if provided
    if model_path and os.path.exists(model_path + ".zip"):
        print(f"Loading model: {model_path}")
        rl_model = PPO.load(model_path, device='cpu')
        strategies["RL (Learned)"] = lambda obs: rl_model.predict(obs, deterministic=True)[0][0]
    elif model_path and os.path.exists(model_path):
        print(f"Loading model: {model_path}")
        rl_model = PPO.load(model_path, device='cpu')
        strategies["RL (Learned)"] = lambda obs: rl_model.predict(obs, deterministic=True)[0][0]
    elif model_path:
        print(f"Warning: Model not found at {model_path} or {model_path}.zip")
    
    # Baseline configurations
    configs = [
        [0, 0, 0, 0, 0, 0],  # Direct, no tools, all low
        [0, 0, 2, 0, 2, 2],  # Direct, no tools, all high
        [1, 0, 0, 0, 0, 0],  # Reason+Ans, no tools, all low
        [1, 0, 1, 0, 1, 1],  # Reason+Ans, no tools, all mid
        [1, 0, 2, 0, 2, 2],  # Reason+Ans, no tools, all high
        [1, 1, 1, 0, 1, 1],  # Reason+Ans, calc, mid
        [1, 2, 1, 0, 1, 1],  # Reason+Ans, search, mid
        [1, 4, 1, 0, 1, 1],  # Reason+Ans, python, mid
        [1, 7, 2, 0, 2, 2],  # Reason+Ans, all tools, high
        [2, 0, 0, 0, 0, 0],  # Full, no tools, all low
        [2, 0, 1, 0, 1, 1],  # Full, no tools, all mid
        [2, 1, 1, 1, 1, 1],  # Full, calc both, mid
        [2, 7, 2, 7, 2, 2],  # Full, all tools, all high
    ]
    
    for cfg in configs:
        name = f"[{cfg[0]},{cfg[1]},{cfg[2]},{cfg[3]},{cfg[4]},{cfg[5]}]"
        strategies[name] = lambda obs, c=cfg: np.array(c)
    
    # Random baseline
    strategies["Random"] = lambda obs: env.action_space.sample()
    
    # Results container
    results = []
    detailed_results = []

    for name, policy_fn in strategies.items():
        print(f"Testing: {name}...", end=" ", flush=True)
        start_time = time.time()
        
        accuracies = []
        steps_list = []
        tools_list = []
        tokens_list = []
        workflows = []
        
        obs = env.reset()
        
        for ep in range(num_episodes):
            action = policy_fn(obs)
            
            # Ensure action is proper format
            if isinstance(action, np.ndarray) and action.ndim == 0:
                action = action.item()
            if not isinstance(action, np.ndarray):
                action = np.array(action)
            
            obs, rewards, dones, infos = env.step([action])
            
            info = infos[0]
            accuracies.append(1 if info["correct"] else 0)
            steps_list.append(info["steps_taken"])
            tools_list.append(info["tools_used"])
            tokens_list.append(info["total_tokens"])
            workflows.append(info["workflow"])
            
            # Store detailed result
            detailed_results.append({
                "strategy": name,
                "episode": ep,
                "correct": info["correct"],
                "workflow": info["workflow"],
                "steps": info["steps_taken"],
                "tools": info["tools_used"],
                "tokens": info["total_tokens"],
                "reward": float(rewards[0])
            })
            
            if dones[0]:
                obs = env.reset()
        
        elapsed = time.time() - start_time
        
        # Calculate metrics
        avg_acc = np.mean(accuracies)
        avg_steps = np.mean(steps_list)
        avg_tools = np.mean(tools_list)
        avg_tokens = np.mean(tokens_list)
        
        results.append({
            "Strategy": name,
            "Accuracy": f"{avg_acc:.1%}",
            "Avg Steps": f"{avg_steps:.2f}",
            "Avg Tools": f"{avg_tools:.2f}",
            "Avg Tokens": f"{avg_tokens:.0f}",
            "Time (s)": f"{elapsed:.1f}"
        })
        
        print(f"Accuracy: {avg_acc:.1%}")

    # Print results table
    df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)
    
    # Save detailed results
    detailed_df = pd.DataFrame(detailed_results)
    output_path = f"benchmark_results_{config.DATASET_NAME}_{int(time.time())}.csv"
    detailed_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")
    
    return df, detailed_df


def analyze_results(detailed_df: pd.DataFrame):
    """Analyze benchmark results."""
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    # Group by strategy
    summary = detailed_df.groupby("strategy").agg({
        "correct": ["mean", "std", "sum"],
        "tokens": ["mean", "std"],
        "reward": ["mean", "std"]
    }).round(3)
    
    print("\nPer-Strategy Statistics:")
    print(summary)
    
    # Workflow distribution per strategy
    print("\nWorkflow Distribution:")
    workflow_dist = detailed_df.groupby(["strategy", "workflow"]).size().unstack(fill_value=0)
    print(workflow_dist)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark RL Controller")
    parser.add_argument("--model", type=str, default="models/controller_gsm8k.zip",
                        help="Path to trained model (without .zip)")
    parser.add_argument("--episodes", type=int, default=30,
                        help="Number of episodes per strategy")
    args = parser.parse_args()
    
    df, detailed_df = run_benchmark(
        model_path=args.model,
        num_episodes=args.episodes
    )
    
    analyze_results(detailed_df)

