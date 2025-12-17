# train.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import GeneralAgentEnv
from callbacks import TrainingMetricsCallback
import config
import os
import torch

def run_training():
    if not os.path.exists("models"):
        os.makedirs("models")

    print(f"Starting Training on Dataset: {config.DATASET_NAME}")
    
    # 1. Setup Env (The LLM inside here will naturally use the GPU defined in config.py)
    env = DummyVecEnv([lambda: GeneralAgentEnv()])

    # 2. Setup Agent
    # FIX: Force PPO to run on CPU to avoid "Device-side assert" conflicts 
    # and save VRAM for the LLM.
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=config.LEARNING_RATE,
        n_steps=config.N_STEPS,     # <--- Added to force short debugging loops
        batch_size=config.BATCH_SIZE,
        device='cpu'  # <--- THIS IS THE CRITICAL FIX FOR PPO STABILITY
    )

    # 3. Train
    print("Starting Learn loop...")
    callback = TrainingMetricsCallback(verbose=1)
    model.learn(total_timesteps=config.TOTAL_TIMESTEPS, progress_bar=True, callback=callback)
    
    # 4. Save
    save_path = f"models/controller_{config.DATASET_NAME}"
    model.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    # Optional: Enable this to debug exact location of CUDA errors if they persist
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    run_training()