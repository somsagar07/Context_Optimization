"""
Training script for RL-based LLM agent configuration optimization.

Usage:
    python train.py --config multi_step    # Multi-step environment (recommended)
    python train.py --config single_step   # Single-step environment (baseline)
    python train.py --config multi_step --timesteps 50000  # Override timesteps
"""
import argparse
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from configs import load_config
from env import GeneralAgentEnv, MultiStepAgentEnv
from utils.callbacks import TrainingMetricsCallback


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RL agent for LLM configuration optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="multi_step",
        choices=["single_step", "multi_step"],
        help="Configuration to use (single_step or multi_step)"
    )
    parser.add_argument(
        "--timesteps", 
        type=int, 
        default=None,
        help="Override total timesteps from config"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default=None,
        choices=["gsm8k", "hotpotqa"],
        help="Override dataset from config"
    )
    return parser.parse_args()


def run_training(cfg, timesteps_override=None, dataset_override=None):
    """
    Main training loop for the RL agent.
    
    Args:
        cfg: Configuration module (from configs/)
        timesteps_override: Optional override for total timesteps
        dataset_override: Optional override for dataset name
    """
    # Apply overrides
    total_timesteps = timesteps_override or cfg.TOTAL_TIMESTEPS
    dataset_name = dataset_override or cfg.DATASET_NAME
    
    # Create models directory
    if not os.path.exists("models"):
        os.makedirs("models")

    print(f"=" * 60)
    print(f"Starting Training")
    print(f"=" * 60)
    print(f"  Config:       {cfg.ENV_MODE}")
    print(f"  Dataset:      {dataset_name}")
    print(f"  Timesteps:    {total_timesteps}")
    print(f"  Learning Rate: {cfg.LEARNING_RATE}")
    print(f"  Gamma:        {cfg.GAMMA}")
    print(f"  N_Steps:      {cfg.N_STEPS}")
    print(f"  Batch Size:   {cfg.BATCH_SIZE}")
    print(f"=" * 60)
    
    # Setup Environment based on mode
    if cfg.ENV_MODE == "multi_step":
        print("\nUsing MultiStepAgentEnv (sequential decisions, proper credit assignment)")
        env = DummyVecEnv([lambda: MultiStepAgentEnv(cfg)])
    else:
        print("\nUsing GeneralAgentEnv (single-step, all actions at once)")
        env = DummyVecEnv([lambda: GeneralAgentEnv(cfg)])

    # Setup PPO Agent
    ent_coef = getattr(cfg, 'ENT_COEF', 0.01)  # Use config value or default
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        ent_coef=ent_coef,
        learning_rate=cfg.LEARNING_RATE,
        n_steps=cfg.N_STEPS,    
        batch_size=cfg.BATCH_SIZE,
        gamma=cfg.GAMMA,
    )

    # Train
    print("\nStarting training loop...")
    save_every = getattr(cfg, 'SAVE_EVERY_EPISODES', 500)
    callback = TrainingMetricsCallback(verbose=1, save_every_episodes=save_every)
    model.learn(
        total_timesteps=total_timesteps, 
        progress_bar=True, 
        callback=callback
    )
    
    # Save with descriptive name
    timestamp = int(time.time())
    save_path = f"models/controller_{cfg.ENV_MODE}_{dataset_name}_{total_timesteps}_{timestamp}"
    model.save(save_path)
    print(f"\nModel saved to {save_path}.zip")
    
    return model


if __name__ == "__main__":
    args = parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    print(f"Loaded config: {args.config}")
    
    # Run training
    run_training(
        cfg=cfg,
        timesteps_override=args.timesteps,
        dataset_override=args.dataset
    )
