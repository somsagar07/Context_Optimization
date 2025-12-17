import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import GeneralAgentEnv
import config
import os
import numpy as np

def run_evaluation(num_episodes=10):
    print(f"Starting Evaluation on Dataset: {config.DATASET_NAME}")
    
    # 1. Setup Env
    env = DummyVecEnv([lambda: GeneralAgentEnv()])
    
    # 2. Load Model
    model_path = f"models/controller_{config.DATASET_NAME}"
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: Model file {model_path}.zip not found. Train first!")
        return

    # Force CPU for stability during inference too
    model = PPO.load(model_path, device='cpu')
    print(f"Loaded model from {model_path}")

    # 3. Evaluation Loop
    total_rewards = []
    accuracies = []
    steps_history = []
    workflow_counts = {"Direct": 0, "Reason+Ans": 0, "Reason+Verify+Ans": 0}
    
    obs = env.reset()
    
    print("-" * 100)
    print(f"{'Query Preview':<30} | {'Workflow':<18} | {'Steps':<5} | {'Tools':<5} | {'Correct'}")
    print("-" * 100)

    for i in range(num_episodes):
        # Predict action
        action, _states = model.predict(obs, deterministic=True)
        
        # Execute step
        obs, rewards, dones, infos = env.step(action)
        
        # Log metrics
        info = infos[0] # Single env
        total_rewards.append(rewards[0])
        accuracies.append(1 if info["correct"] else 0)
        steps_history.append(info["steps_taken"])
        
        wf_name = info["workflow"]
        if wf_name in workflow_counts:
            workflow_counts[wf_name] += 1
            
        # Print concise row
        query_preview = info["query"][:27] + "..."
        tools_num = info["tools_loaded"]
        correct_str = "YES" if info["correct"] else "NO"
        
        print(f"{query_preview:<30} | {wf_name:<18} | {info['steps_taken']:<5} | {tools_num:<5} | {correct_str}")

    print("-" * 100)
    print("Evaluation Summary:")
    print(f"Average Accuracy: {np.mean(accuracies):.2%}")
    print(f"Average Reward:   {np.mean(total_rewards):.4f}")
    print(f"Avg Steps Taken:  {np.mean(steps_history):.2f}")
    print("Workflow Distribution:")
    for wf, count in workflow_counts.items():
        print(f"  {wf}: {count}")

if __name__ == "__main__":
    # Optional: Enable this to debug exact location of CUDA errors if they persist
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    run_evaluation()

