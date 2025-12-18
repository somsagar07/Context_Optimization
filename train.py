from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import GeneralAgentEnv
from utils.callbacks import TrainingMetricsCallback
import config
import os
import time

def run_training():
    """Main training loop for the RL agent."""
    
    # Create models directory
    if not os.path.exists("models"):
        os.makedirs("models")

    print(f"=" * 60)
    print(f"Starting Training")
    print(f"Dataset: {config.DATASET_NAME}")
    print(f"Timesteps: {config.TOTAL_TIMESTEPS}")
    print(f"=" * 60)
    
    # Setup Environment
    env = DummyVecEnv([lambda: GeneralAgentEnv()])

    # Setup PPO Agent
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=config.LEARNING_RATE,
        n_steps=config.N_STEPS,    
        batch_size=config.BATCH_SIZE,
        gamma=config.GAMMA,
    )

    # Train
    print("\nStarting training loop...")
    callback = TrainingMetricsCallback(verbose=1)
    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS, 
        progress_bar=True, 
        callback=callback
    )
    
    # Save
    timestamp = int(time.time())
    save_path = f"models/controller_{config.DATASET_NAME}_{timestamp}"
    model.save(save_path)
    print(f"\nModel saved to {save_path}")
    
    return model

if __name__ == "__main__":
    run_training()

