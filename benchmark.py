import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
from env import GeneralAgentEnv
from stable_baselines3 import PPO
import config
import os

def run_benchmark(num_episodes=30):
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {num_episodes} Episodes per Strategy")
    print(f"{'='*60}\n")
    
    # Setup single environment
    env = DummyVecEnv([lambda: GeneralAgentEnv()])
    
    # Define Strategies
    strategies = {
        "Minimal (Direct)": lambda obs: np.array([0, 0, 0, 0]), # Direct, No Tools, Low Budget
        "Maximal (Full Chain)": lambda obs: np.array([2, 3, 3, 1]), # R->V->A, All Tools, High Budget
        "Meta-Controller (RL)": None # Will load model
    }
    
    # Load RL Model
    model_path = f"models/controller_{config.DATASET_NAME}"
    if os.path.exists(model_path + ".zip"):
        rl_model = PPO.load(model_path, device='cpu')
        # predict returns (action, state). Action is (1, 4). We want the vector.
        strategies["Meta-Controller (RL)"] = lambda obs: rl_model.predict(obs, deterministic=True)[0][0]
    else:
        print("Warning: RL Model not found. Skipping Meta-Controller.")
        del strategies["Meta-Controller (RL)"]

    # Results Container
    results = []

    for name, policy_fn in strategies.items():
        print(f"Testing Strategy: {name}...")
        
        accuracies = []
        steps = []
        tool_counts = []
        high_budget_counts = []
        
        # Reset env for fair comparison (we can't easily force same seed across loop without deeper hacks, 
        # but 10 random samples is decent for a quick check)
        obs = env.reset()
        
        for _ in range(num_episodes):
            action = policy_fn(obs)
            obs, rewards, dones, infos = env.step([action])
            
            info = infos[0]
            accuracies.append(1 if info["correct"] else 0)
            steps.append(info["steps_taken"])
            tool_counts.append(info["tools_loaded"])
            high_budget_counts.append(1 if info["budget"] == "High" else 0)
            
        avg_acc = np.mean(accuracies)
        avg_step = np.mean(steps)
        avg_tools = np.mean(tool_counts)
        high_budget_pct = np.mean(high_budget_counts)
        
        results.append({
            "Strategy": name,
            "Accuracy": f"{avg_acc:.1%}",
            "Avg Steps": f"{avg_step:.2f}",
            "Avg Tools": f"{avg_tools:.2f}",
            "High Budget": f"{high_budget_pct:.1%}"
        })

    # Print Table
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60 + "\n")

if __name__ == "__main__":
    # Optional: Enable this to debug exact location of CUDA errors if they persist
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    run_benchmark(num_episodes=5)

