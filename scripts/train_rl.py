#!/usr/bin/env python3
"""
Train Flat RL Policies (GeneralAgentEnv or MultiStepAgentEnv)

Trains RL policies that select workflow structure, tools, and token budgets.
Supports both API-based LLMs (OpenRouter) and local HuggingFace models.

With --learn-prompts (multi_step only), the agent also learns which prompt atoms
to use for each agent type (reasoner, verifier, answerer) as additional sequential
steps. This is still a single policy (not hierarchical like HRL) but adds prompt
optimization capability.

Action space (single_step - NO prompt learning):
    [9 workflows, 16 agent1_tools, 3 agent1_budget, 16 agent2_tools, 3 agent2_budget, 3 answerer_budget]
    = 62,208 combinations selected all at once

Action space (multi_step - without --learn-prompts):
    Sequential decisions: workflow → reasoner config → verifier config → answerer budget
    Smaller per-step action spaces (max 48) with temporal credit assignment

Action space (multi_step WITH --learn-prompts):
    Sequential decisions: workflow → reasoner config → verifier config → answerer budget
                         → reasoner prompt → verifier prompt → answerer prompt
    Adds 3 additional steps for prompt selection (0=none, 1-N=specific prompt atom)

Usage:
    # Train single-step RL on GSM8K with default HuggingFace model
    python scripts/train_rl.py --config single_step --dataset gsm8k --episodes 15000

    # Train multi-step RL on MedQA
    python scripts/train_rl.py --config multi_step --dataset medqa --episodes 15000

    # Train multi-step with PROMPT LEARNING (learns which prompts to use)
    python scripts/train_rl.py --config multi_step --dataset gsm8k --learn-prompts --episodes 15000

    # Train with specific HuggingFace model
    python scripts/train_rl.py --config single_step --dataset gsm8k --hf-model Qwen/Qwen2.5-7B-Instruct --episodes 15000

    # Train with OpenRouter API (WARNING: can be expensive!)
    python scripts/train_rl.py --config single_step --dataset gsm8k --api --api-model openai/gpt-4o --episodes 1000
"""
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import re
import time
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from configs import load_config
from env import GeneralAgentEnv, MultiStepAgentEnv


def _sanitize_folder_name(model_name: str) -> str:
    """Sanitize model name for use as folder name."""
    if not model_name:
        return "default"
    # Get last part of path (e.g., "openai/gpt-4o" -> "gpt-4o")
    model_name = model_name.split('/')[-1]
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', model_name)
    sanitized = sanitized.replace(' ', '_')
    sanitized = sanitized.replace('.', '_')
    sanitized = sanitized.strip('. ')
    sanitized = re.sub(r'_+', '_', sanitized)
    return sanitized if sanitized else "default"


def _get_model_display_name(use_api: bool, api_model: str, hf_model: str) -> str:
    """Get a display name for the model being used."""
    if use_api and api_model:
        return api_model
    elif hf_model:
        return hf_model
    else:
        return "default"


class TrainingCallback(BaseCallback):
    """Callback for logging training progress."""
    
    def __init__(self, log_interval=100, verbose=1):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_correct = []
        self.start_time = None
        
    def _on_training_start(self):
        self.start_time = time.time()
        print("\n" + "="*70)
        print("TRAINING STARTED")
        print("="*70)
        
    def _on_step(self):
        # Check if episode finished
        if self.locals.get("dones") is not None and any(self.locals["dones"]):
            infos = self.locals.get("infos", [])
            for info in infos:
                if "correct" in info:
                    self.episode_correct.append(1 if info["correct"] else 0)
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])
        
        # Log progress
        if self.num_timesteps % self.log_interval == 0 and len(self.episode_correct) > 0:
            elapsed = time.time() - self.start_time
            recent_acc = np.mean(self.episode_correct[-100:]) if len(self.episode_correct) >= 100 else np.mean(self.episode_correct)
            recent_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards) if self.episode_rewards else 0
            
            print(f"  Step {self.num_timesteps:>6} | "
                  f"Episodes: {len(self.episode_correct):>4} | "
                  f"Acc: {recent_acc:.1%} | "
                  f"Reward: {recent_reward:.2f} | "
                  f"Time: {elapsed:.0f}s")
        
        return True
    
    def _on_training_end(self):
        elapsed = time.time() - self.start_time
        final_acc = np.mean(self.episode_correct[-100:]) if len(self.episode_correct) >= 100 else np.mean(self.episode_correct) if self.episode_correct else 0
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        print(f"  Total timesteps:  {self.num_timesteps}")
        print(f"  Total episodes:   {len(self.episode_correct)}")
        print(f"  Final accuracy:   {final_acc:.1%}")
        print(f"  Training time:    {elapsed:.1f}s")
        print("="*70)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train flat RL policies for LLM agent configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Environment configuration
    parser.add_argument("--config", type=str, default="single_step",
                       choices=["single_step", "multi_step"],
                       help="Environment configuration (single_step or multi_step)")
    
    # Dataset (required)
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["gsm8k", "hotpotqa", "gaia", "medqa", "aime25"],
                       help="Dataset to train on")
    
    # Training parameters
    parser.add_argument("--episodes", type=int, default=15000,
                       help="Total training timesteps")
    parser.add_argument("--n-steps", type=int, default=None,
                       help="Steps per rollout (uses config default if not specified)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size (uses config default if not specified)")
    parser.add_argument("--learning-rate", type=float, default=None,
                       help="Learning rate (uses config default if not specified)")
    parser.add_argument("--gamma", type=float, default=None,
                       help="Discount factor (uses config default if not specified)")
    parser.add_argument("--ent-coef", type=float, default=None,
                       help="Entropy coefficient (uses config default if not specified)")
    
    # LLM configuration (required for workflow execution)
    parser.add_argument("--api", action="store_true", default=False,
                       help="Use OpenRouter API instead of local HuggingFace model. "
                            "WARNING: API training can be expensive!")
    parser.add_argument("--api-model", type=str, default=None,
                       help="OpenRouter model ID (e.g., 'openai/gpt-4o'). Required if --api is used.")
    parser.add_argument("--hf-model", type=str, default=None,
                       help="HuggingFace model name (e.g., 'Qwen/Qwen2.5-7B-Instruct'). "
                            "Uses config default if not specified.")
    
    # Prompt learning (multi_step only)
    parser.add_argument("--learn-prompts", action="store_true", default=False,
                       help="Enable prompt learning (multi_step only). Adds additional "
                            "sequential steps to select prompt atoms for each agent.")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="models/flat_rl",
                       help="Directory to save trained models")
    parser.add_argument("--log-dir", type=str, default="logs/flat_rl",
                       help="Directory to save training logs")
    parser.add_argument("--save-interval", type=int, default=5000,
                       help="Save model every N timesteps")
    parser.add_argument("--log-interval", type=int, default=500,
                       help="Log progress every N timesteps")
    
    return parser.parse_args()


def make_env(cfg, use_api, api_model, hf_model, is_eval=False, learn_prompts=False):
    """Create environment based on config, wrapped with Monitor for episode stats."""
    def _init():
        if cfg.ENV_MODE == "multi_step":
            env = MultiStepAgentEnv(
                cfg=cfg, 
                is_eval=is_eval,
                use_api=use_api, 
                api_model=api_model, 
                hf_model=hf_model,
                learn_prompts=learn_prompts
            )
        else:
            env = GeneralAgentEnv(
                cfg=cfg, 
                is_eval=is_eval,
                use_api=use_api, 
                api_model=api_model, 
                hf_model=hf_model
            )
        # Wrap with Monitor to track episode stats (reward, length)
        return Monitor(env)
    return _init


def main():
    args = parse_args()
    
    # Validate LLM configuration
    if args.api:
        if not args.api_model:
            raise ValueError(
                "When using --api, you must specify --api-model (e.g., 'openai/gpt-4o'). "
                "The RL policy needs an LLM to execute workflows."
            )
        print(f"Using OpenRouter API with model: {args.api_model}")
        print("WARNING: API training can be expensive! Consider using --hf-model instead.")
    else:
        if args.hf_model:
            print(f"Using HuggingFace model: {args.hf_model}")
        else:
            print("Using default HuggingFace model from config")
    
    # Validate learn_prompts (only works with multi_step)
    if args.learn_prompts and args.config != "multi_step":
        raise ValueError(
            "--learn-prompts is only supported with --config multi_step. "
            "single_step environment does not support prompt learning."
        )
    
    # Load configuration
    cfg = load_config(args.config)
    cfg.DATASET_NAME = args.dataset  # Required, always set
    
    print(f"\nLoaded config: {args.config}")
    print(f"  ENV_MODE:    {cfg.ENV_MODE}")
    print(f"  DATASET:     {cfg.DATASET_NAME}")
    if args.learn_prompts:
        print(f"  PROMPTS:     Learning enabled (adds prompt selection steps)")
    
    # Get training hyperparameters (from args or config defaults)
    total_timesteps = args.episodes
    n_steps = args.n_steps or getattr(cfg, 'N_STEPS', 2048)
    batch_size = args.batch_size or getattr(cfg, 'BATCH_SIZE', 64)
    learning_rate = args.learning_rate or getattr(cfg, 'LEARNING_RATE', 3e-4)
    gamma = args.gamma if args.gamma is not None else getattr(cfg, 'GAMMA', 0.99)
    ent_coef = args.ent_coef if args.ent_coef is not None else getattr(cfg, 'ENT_COEF', 0.01)
    
    print(f"\nTraining hyperparameters:")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  N_STEPS:         {n_steps}")
    print(f"  BATCH_SIZE:      {batch_size}")
    print(f"  LEARNING_RATE:   {learning_rate}")
    print(f"  GAMMA:           {gamma}")
    print(f"  ENT_COEF:        {ent_coef}")
    
    # Get model display name and create subfolder structure (like HRL)
    model_display_name = _get_model_display_name(args.api, args.api_model, args.hf_model)
    model_folder = _sanitize_folder_name(model_display_name)
    
    # Create output directories with model subfolder (models/flat_rl/{model_folder}/)
    model_output_dir = os.path.join(args.output_dir, model_folder)
    log_output_dir = os.path.join(args.log_dir, model_folder)
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(log_output_dir, exist_ok=True)
    
    print(f"\nOutput directories:")
    print(f"  Models: {model_output_dir}")
    print(f"  Logs:   {log_output_dir}")
    
    # Create environment
    print(f"\nCreating environment: {cfg.ENV_MODE}...")
    env = DummyVecEnv([make_env(cfg, args.api, args.api_model, args.hf_model, learn_prompts=args.learn_prompts)])
    
    # Create PPO model
    print("Initializing PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gamma=gamma,
        ent_coef=ent_coef,
        verbose=0,
        device="auto"
    )
    
    # Create callback
    callback = TrainingCallback(log_interval=args.log_interval)
    
    # Train
    print(f"\nStarting training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_suffix = "_prompts" if args.learn_prompts else ""
    model_filename = f"flat_rl_{cfg.ENV_MODE}{prompt_suffix}_{cfg.DATASET_NAME}_{timestamp}"
    model_path = os.path.join(model_output_dir, model_filename)
    model.save(model_path)
    print(f"\nFinal model saved to: {model_path}.zip")
    
    # Save training log
    log_data = {
        # Algorithm and model info (like HRL train.py)
        "algorithm": "PPO",
        "model_name": model_display_name,
        # Config and dataset
        "config": args.config,
        "env_mode": cfg.ENV_MODE,
        "dataset": cfg.DATASET_NAME,
        "learn_prompts": args.learn_prompts,
        # Training hyperparameters
        "timesteps": total_timesteps,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "ent_coef": ent_coef,
        # LLM configuration
        "use_api": args.api,
        "api_model": args.api_model,
        "hf_model": args.hf_model,
        # Output paths
        "model_path": model_path + ".zip",
        "timestamp": timestamp,
        # Training results
        "total_episodes": len(callback.episode_correct),
        "final_accuracy": float(np.mean(callback.episode_correct[-100:])) if callback.episode_correct else 0.0,
        "episode_correct": callback.episode_correct,
        "episode_rewards": callback.episode_rewards,
    }
    
    import json
    log_filename = f"train_log_{cfg.ENV_MODE}{prompt_suffix}_{cfg.DATASET_NAME}_{timestamp}.json"
    log_path = os.path.join(log_output_dir, log_filename)
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    print(f"Training log saved to: {log_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"  Algorithm:     PPO")
    print(f"  Model:         {model_display_name}")
    print(f"  Config:        {args.config}")
    print(f"  Dataset:       {cfg.DATASET_NAME}")
    print(f"  Timesteps:     {total_timesteps}")
    print(f"  Episodes:      {len(callback.episode_correct)}")
    print(f"  Final Acc:     {log_data['final_accuracy']:.1%}")
    print(f"  Model saved:   {model_path}.zip")
    print(f"  Log saved:     {log_path}")
    print("="*70)
    
    print("\nTo evaluate the trained model, run:")
    print(f"  python scripts/eval_rl.py --config {args.config} --model {model_path}")


if __name__ == "__main__":
    main()

