import json
import os
import numpy as np
from tqdm import tqdm
from env import GeneralAgentEnv
import config

def collect_data(num_samples=1000, output_path="/data/ssagar6/offline_dataset.json"):
    print(f"Starting Offline Data Collection: {num_samples} samples")
    print(f"Saving to: {output_path}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    env = GeneralAgentEnv()
    data = []
    
    # We will use a random policy to explore the space
    # But we can bias it slightly to avoid completely useless actions (like Verifier with no tools)
    
    for i in tqdm(range(num_samples)):
        obs, _ = env.reset()
        
        # Random Action from the MultiDiscrete space
        action = env.action_space.sample()
        
        # Execute
        # We need to manually handle the step because we are not using VecEnv here
        # env.step expects the action directly
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Record Entry
        entry = {
            "question": info["query"],
            "action": action.tolist(), # Convert numpy/array to list
            "reward": float(reward),
            "correct": bool(info["correct"]),
            "steps": int(info["steps_taken"]),
            "budget": info["budget"],
            "workflow": info["workflow"],
            "tools_loaded": int(info["tools_loaded"])
        }
        data.append(entry)
        
        # Periodically save (every 100)
        if (i + 1) % 100 == 0:
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
                
    # Final Save
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Done! Collected {len(data)} samples.")

if __name__ == "__main__":
    # Force CPU to avoid crashes during long collection
    # config.DEVICE = 'cpu' 
    collect_data(num_samples=50) # Start small to test

