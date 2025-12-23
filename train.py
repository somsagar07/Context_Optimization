"""
Training script for RL-based LLM agent configuration optimization.

Usage:
    python train.py --config single_step       # Single-step environment (baseline)
    python train.py --config multi_step        # Multi-step environment
    python train.py --config hierarchical      # Hierarchical dual-policy (recommended)
    
For hierarchical mode, this script calls train_dual.py which trains
two separate policy networks (structure + prompts).
"""
import argparse
import os
import sys
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
        default="hierarchical",
        choices=["single_step", "multi_step", "hierarchical"],
        help="Configuration to use"
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
    # Hierarchical-specific args
    parser.add_argument(
        "--joint", 
        action="store_true",
        help="(hierarchical only) Use alternating joint training"
    )
    parser.add_argument(
        "--joint-iterations", 
        type=int, 
        default=3,
        help="(hierarchical only) Number of joint training iterations"
    )
    return parser.parse_args()


def run_single_policy_training(cfg, timesteps_override=None, dataset_override=None):
    """
    Training for single-step and multi-step environments.
    Uses a single PPO policy.
    """
    total_timesteps = timesteps_override or cfg.TOTAL_TIMESTEPS
    dataset_name = dataset_override or cfg.DATASET_NAME
    
    os.makedirs("models", exist_ok=True)

    print(f"=" * 60)
    print(f"Starting Training (Single Policy)")
    print(f"=" * 60)
    print(f"  Config:        {cfg.ENV_MODE}")
    print(f"  Dataset:       {dataset_name}")
    print(f"  Timesteps:     {total_timesteps}")
    print(f"  Learning Rate: {cfg.LEARNING_RATE}")
    print(f"  Gamma:         {cfg.GAMMA}")
    print(f"  N_Steps:       {cfg.N_STEPS}")
    print(f"  Batch Size:    {cfg.BATCH_SIZE}")
    print(f"=" * 60)
    
    # Setup Environment
    if cfg.ENV_MODE == "multi_step":
        print("\nUsing MultiStepAgentEnv (sequential decisions)")
        env = DummyVecEnv([lambda: MultiStepAgentEnv(cfg)])
    else:
        print("\nUsing GeneralAgentEnv (single-step, all actions at once)")
        env = DummyVecEnv([lambda: GeneralAgentEnv(cfg)])

    # Setup PPO Agent
    ent_coef = getattr(cfg, 'ENT_COEF', 0.01)
    
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
    
    # Save
    timestamp = int(time.time())
    save_path = f"models/controller_{cfg.ENV_MODE}_{dataset_name}_{total_timesteps}_{timestamp}"
    model.save(save_path)
    print(f"\nModel saved to {save_path}.zip")
    
    return model


def run_dual_policy_training(cfg, args):
    """
    Training for hierarchical environment.
    Uses two separate PPO policies (structure + prompts).
    Delegates to train_dual.py logic.
    """
    # Import dual policy training functions
    from train_dual import train_structure_policy, train_prompt_policy, train_joint
    
    os.makedirs("models", exist_ok=True)
    
    timestamp = int(time.time())
    dataset = args.dataset or cfg.DATASET_NAME
    structure_path = f"models/structure_policy_{dataset}_{timestamp}"
    prompt_path = f"models/prompt_policy_{dataset}_{timestamp}"
    
    if args.joint:
        # Alternating joint training
        train_joint(
            cfg, structure_path, prompt_path,
            structure_timesteps=args.timesteps,
            prompt_timesteps=args.timesteps,
            iterations=args.joint_iterations
        )
    else:
        # Sequential training (structure first, then prompt)
        print("\n" + "=" * 60)
        print("HIERARCHICAL TRAINING (Dual Policy)")
        print("  Phase 1: Train structure policy (workflow, tools, budgets)")
        print("  Phase 2: Train prompt policy (sequential prompts)")
        print("=" * 60)
        
        # Phase 1: Structure policy
        structure_policy = train_structure_policy(
            cfg, structure_path,
            timesteps=args.timesteps
        )
        
        # Phase 2: Prompt policy
        prompt_policy = train_prompt_policy(
            cfg, structure_policy, prompt_path,
            timesteps=args.timesteps
        )
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"  Structure Policy: {structure_path}.zip")
        print(f"  Prompt Policy:    {prompt_path}.zip")
        print("\nTo evaluate, run:")
        print(f"  python eval_dual.py --structure-model {structure_path} --prompt-model {prompt_path}")


if __name__ == "__main__":
    args = parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    print(f"Loaded config: {args.config}")
    
    # Apply dataset override to config
    if args.dataset:
        cfg.DATASET_NAME = args.dataset
    
    # Run appropriate training
    if cfg.ENV_MODE == "hierarchical":
        run_dual_policy_training(cfg, args)
    else:
        run_single_policy_training(
            cfg=cfg,
            timesteps_override=args.timesteps,
            dataset_override=args.dataset
        )
