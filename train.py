"""
Unified Training Script for Hierarchical RL

Supports both PPO and GRPO algorithms:
- PPO: Proximal Policy Optimization (with value function)
- GRPO: Group Relative Policy Optimization (critic-free, better for sparse rewards)

Usage:
    python train.py --algorithm grpo --episodes 20000
    python train.py --algorithm ppo --episodes 20000
    python train.py --algorithm grpo --episodes 20000 --entropy-coef 0.08 --tool-bonus 0.15
    python train.py --algorithm grpo --episodes 20000 --pretrain-structure models/sft_posttrained/structure_policy_sft.pt --pretrain-prompt models/sft_posttrained/prompt_policy_sft.pt
"""
import argparse
import os
import sys
from datetime import datetime


# Load .env file FIRST (before any imports that might need env vars)
try:
    from dotenv import load_dotenv
    # Try to load from project root
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        # Try current directory
        load_dotenv()
except ImportError:
    # python-dotenv not installed, skip
    pass
except Exception:
    # Failed to load, continue anyway
    pass

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# ---------------------------------------------------------------------------
# Output hygiene: keep training output focused on training-loop info.
# - CO_QUIET silences our worker/embedder one-time init prints.
# - tau2's loguru spam (Setting observation, Sending message, etc.) and noisy
#   third-party loggers get clamped to ERROR. The trainer's own prints
#   (episode count, reward, loss, eval bar) still come through normally.
# - Per-episode gymnasium "Overriding environment ... already in registry" warning
#   suppressed since we register the tau2 env on every gym.make.
# Useful errors (CUDA OOM, HTTP 500s, etc.) still surface via stderr.
# ---------------------------------------------------------------------------
os.environ.setdefault("CO_QUIET", "1")

import logging
import warnings
try:
    from loguru import logger as _loguru_logger
    _loguru_logger.remove()
except Exception:
    pass
for _noisy in ("LiteLLM", "litellm", "httpx", "openai", "urllib3", "tau2"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=r"Overriding environment .* already in registry")

from configs import load_config
from algorithms import Algorithm, PPOTrainer, GRPOTrainer
from agents_system.worker import LLMWorker, OpenRouterWorker
from prompts import library
from utils import validate_dataset_name, get_dataset_help_text


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train hierarchical RL with PPO or GRPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Algorithm selection
    parser.add_argument(
        "--algorithm", type=str, default="ppo",
        choices=["ppo", "grpo"],
        help="RL algorithm: ppo (with value function) or grpo (critic-free)"
    )
    
    # Basic training args
    parser.add_argument("--config", type=str, default="hierarchical", help="Config to use")
    parser.add_argument("--episodes", type=int, default=20000, help="Number of episodes")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--dataset", type=validate_dataset_name, default=None,
                       help=get_dataset_help_text())
    
    # Algorithm hyperparameters
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="Clipping parameter")
    parser.add_argument("--batch-epochs", "--epochs", dest="batch_epochs", type=int, default=4,
                       help="Number of PPO/GRPO gradient-update passes over each collected rollout batch (not dataset epochs). Default: 4")
    parser.add_argument("--entropy-coef", type=float, default=0.05, help="Entropy coefficient")
    parser.add_argument("--struct-entropy-coef", type=float, default=None, help="Structure entropy")
    parser.add_argument("--prompt-entropy-coef", type=float, default=None, help="Prompt entropy")
    
    # Learning rate (useful when continuing from pretrained models)
    parser.add_argument("--struct-lr", type=float, default=None, help="Structure learning rate (overrides config, recommended: 1e-4 for pretrained)")
    parser.add_argument("--prompt-lr", type=float, default=None, help="Prompt learning rate (overrides config, recommended: 5e-5 for pretrained)")
    
    # GRPO-specific
    parser.add_argument("--kl-coef", type=float, default=0.0, help="KL regularization (GRPO only)")
    parser.add_argument("--ref-update-every", type=int, default=1000, help="Reference policy update freq")
    
    # Reward tuning
    parser.add_argument("--reward-scale", type=float, default=1.0, help="Scale correctness reward")
    parser.add_argument("--tool-bonus", type=float, default=0.1, help="Bonus per tool used (positive encourages tool usage, default: 0.1)")
    
    # Action masking
    parser.add_argument("--mask", action="store_true", default=False,
                       help="Enable action masking to reduce invalid action space (masks agent2 for workflows 0,1,5)")
    
    # API configuration
    parser.add_argument("--api", action="store_true", default=False,
                       help="Use OpenRouter API instead of local HuggingFace models")
    parser.add_argument("--api-model", type=str, default=None,
                       help="OpenRouter model ID (e.g., 'openai/gpt-4o', 'anthropic/claude-3.5-sonnet'). Defaults to OPENROUTER_MODEL env var")
    
    # HuggingFace model configuration
    parser.add_argument("--hf-model", type=str, default=None,
                       help="HuggingFace model name (e.g., 'Qwen/Qwen2.5-7B-Instruct'). Defaults to LLM_MODEL_NAME from config")
    
    # Prompt generation model (for generating prompt atoms)
    parser.add_argument("--prompt-gen-model", type=str, default="anthropic/claude-opus-4.7",
                       help="Model to use for prompt generation (OpenRouter model ID). Defaults to 'anthropic/claude-opus-4.7' (used by AtomGeneratorV2).")
    
    # Logging
    parser.add_argument("--log-every", type=int, default=50, help="Log frequency")
    parser.add_argument("--save-every", type=int, default=2000, help="Checkpoint frequency")
    parser.add_argument("--save-log-every", type=int, default=100, help="Save log file frequency (default: 100)")
    
    # Parallel execution (only for API mode)
    parser.add_argument("--num-workers", type=int, default=1,
                       help="Number of parallel workers for API mode training (default: 1, recommended: 4-8 for API). Only used with --api flag.")

    # Tau2-specific knobs (only used for tau2_* datasets)
    parser.add_argument("--tau2-max-turns", type=int, default=12,
                       help="Per-episode dialog turn cap for tau2 datasets. Lower = faster (default 12; try 6-8 for fast iteration). Each turn costs ~workflow_depth+1 LLM calls.")
    parser.add_argument("--tau2-w-action", type=float, default=None,
                       help="Stage-A weight for action-component pass fraction (default 2.0). Max contribution to reward when action-pass = 100%%.")
    parser.add_argument("--tau2-w-communicate", type=float, default=None,
                       help="Stage-A weight for communicate-component pass fraction (default 1.0).")
    parser.add_argument("--tau2-w-env", type=float, default=None,
                       help="Stage-A weight for env (DB+env_assertions) pass fraction (default 2.0).")
    parser.add_argument("--tau2-w-nl", type=float, default=None,
                       help="Stage-A weight for NL-assertion pass fraction (default 1.0).")
    parser.add_argument("--tau2-completion-bonus", type=float, default=None,
                       help="Big bonus added when ALL required components fully pass (default 5.0). This dominates the gradient on perfect trajectories.")
    parser.add_argument("--use-shaped-rewards", dest="use_shaped_rewards",
                       action="store_true", default=True,
                       help="(tau2 only, default ON) Use ShapedTau2Env for per-turn dense shaping rewards (action_progress, tool_error, tool_dup, tokens). Sums into the terminal reward.")
    parser.add_argument("--no-shaped-rewards", dest="use_shaped_rewards",
                       action="store_false",
                       help="Disable per-turn shaping rewards. Reward is paper-faithful tau2 reward only.")
    parser.add_argument("--shaping-mode", choices=["training", "eval", "off"],
                       default="training",
                       help="training (default): full shaping incl. oracle action_progress. eval: zero out shaping (paper-faithful). off: equivalent to --no-shaped-rewards.")
    parser.add_argument("--per-turn-config", action="store_true", default=True,
                       help="(tau2 only, default ON) Re-run structure+prompt policies at every dialog turn instead of once per episode. Disable with --no-per-turn-config.")
    parser.add_argument("--no-per-turn-config", dest="per_turn_config", action="store_false",
                       help="(tau2 only) Use single-config mode: pick workflow/tools/atoms once per episode.")
    
    # Pretrained models (e.g., from SFT)
    parser.add_argument("--pretrain-structure", type=str, default=None,
                       help="Path to pretrained structure policy (e.g., from SFT)")
    parser.add_argument("--pretrain-prompt", type=str, default=None,
                       help="Path to pretrained prompt policy (e.g., from SFT)")

    # Resume from a previous run (preserves episode counter + appends to same log file)
    parser.add_argument("--resume-from", type=str, default=None,
                       help="Path to a previous run's training-log JSON. Restores episode_count, "
                            "correct_count, total_reward, recent rolling stats, and reuses the same "
                            "log file so new episodes APPEND. Combine with --pretrain-structure/--pretrain-prompt "
                            "to also restore the policy weights. Optimizer state is NOT restored "
                            "(PPO/GRPO are on-policy; momentum from the prior run is intentionally dropped).")
    
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    cfg = load_config(args.config)
    if args.dataset:
        cfg.DATASET_NAME = args.dataset

    # Tau2 max-turns is read by PromptEnv from cfg; CLI takes precedence.
    cfg.TAU2_MAX_TURNS = int(args.tau2_max_turns)
    # Shaping toggles likewise read from cfg.
    cfg.USE_SHAPED_REWARDS = bool(args.use_shaped_rewards)
    cfg.SHAPING_MODE = str(args.shaping_mode)
    cfg.SHAPING_CFG = None  # let ShapedTau2Env defaults apply; tunable later if needed
    cfg.PER_TURN_CONFIG = bool(args.per_turn_config)
        
    # Update Prompt Atoms based on dataset
    print(f"Checking prompt atoms for dataset: {cfg.DATASET_NAME}...")
    atoms_path = library._get_atoms_path(cfg.DATASET_NAME)
    
    # 1. Check if atoms exist (Fast path)
    if os.path.exists(atoms_path):
        print(f"  Found existing atoms at {atoms_path}. Loading...")
        library.load_or_create_atoms(cfg.DATASET_NAME, worker=None)
    
    else:
        print(f"  Atoms not found. Initializing temporary worker to generate them...")
        
        # Use GPT via OpenRouter for better prompt generation
        print(f"  Using {args.prompt_gen_model} for prompt generation...")
        try:
            temp_worker = OpenRouterWorker(model_name=args.prompt_gen_model)
            library.load_or_create_atoms(cfg.DATASET_NAME, worker=temp_worker)
            print("  Generation complete.")
            del temp_worker
        except Exception as e:
            print(f"  Error using OpenRouter for prompt generation: {e}")
            print("  Falling back to local Qwen model...")
            import gc
            import torch
            temp_worker = LLMWorker(model_name="Qwen/Qwen2.5-7B-Instruct")
            library.load_or_create_atoms(cfg.DATASET_NAME, worker=temp_worker)
            print("  Generation complete. Freeing memory for training...")
            del temp_worker
            gc.collect()
            torch.cuda.empty_cache()
    
    print(f"  Active Atoms: {library.NUM_ATOMS}")
    
    # Create trainer
    if args.algorithm == "ppo":
        trainer = PPOTrainer(cfg, use_action_masking=args.mask, use_api=args.api, api_model=args.api_model, hf_model=args.hf_model)
    else:
        trainer = GRPOTrainer(cfg, use_action_masking=args.mask, use_api=args.api, api_model=args.api_model, hf_model=args.hf_model)
    
    # Load pretrained models if provided (e.g., from SFT)
    if args.pretrain_structure or args.pretrain_prompt:
        trainer.load_pretrained(args.pretrain_structure, args.pretrain_prompt, reset_optimizers=True)
        # If using pretrained models, optionally use lower learning rates for fine-tuning
        if args.struct_lr is None and args.pretrain_structure:
            # Suggest lower LR when continuing from pretrained
            print("  Tip: Consider using --struct-lr 1e-4 for gentler fine-tuning from pretrained models")
        if args.prompt_lr is None and args.pretrain_prompt:
            print("  Tip: Consider using --prompt-lr 5e-5 for gentler fine-tuning from pretrained models")

    # Resume cumulative stats + log file from a previous run if requested.
    # Must come AFTER load_pretrained so we don't overwrite reset state.
    if args.resume_from:
        trainer.load_training_log(args.resume_from)
    
    # Override learning rates if specified
    if args.struct_lr is not None or args.prompt_lr is not None:
        trainer._init_optimizers(struct_lr=args.struct_lr, prompt_lr=args.prompt_lr)
        if args.struct_lr:
            print(f"  Using structure LR: {args.struct_lr}")
        if args.prompt_lr:
            print(f"  Using prompt LR: {args.prompt_lr}")
    
    # Entropy coefficients
    struct_ent = args.struct_entropy_coef or args.entropy_coef
    prompt_ent = args.prompt_entropy_coef or args.entropy_coef
    
    # Validate num_workers usage
    if args.num_workers > 1 and not args.api:
        print(f"\n⚠ Warning: --num-workers={args.num_workers} specified but --api flag not set.")
        print("  Parallel workers are only supported for API mode. Using sequential training.")
        args.num_workers = 1
    
    # Train
    trainer.train(
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        log_every=args.log_every,
        save_every=args.save_every,
        save_log_every=args.save_log_every,
        num_workers=args.num_workers,
        # Algorithm params
        gamma=cfg.PROMPT_GAMMA,
        clip_epsilon=args.clip_epsilon,
        epochs=args.batch_epochs,
        struct_entropy_coef=struct_ent,
        prompt_entropy_coef=prompt_ent,
        # GRPO-specific
        kl_coef=args.kl_coef,
        ref_update_every=args.ref_update_every,
        # Reward
        reward_scale=args.reward_scale,
        tool_bonus=args.tool_bonus,
        # Tau2 reward weights (None = keep trainer default)
        tau2_w_action=args.tau2_w_action,
        tau2_w_communicate=args.tau2_w_communicate,
        tau2_w_env=args.tau2_w_env,
        tau2_w_nl=args.tau2_w_nl,
        tau2_completion_bonus=args.tau2_completion_bonus,
    )
    
    # Save final
    struct_path, prompt_path = trainer.save_models("_final")
    
    print(f"\nTo evaluate:")
    print(f"  python scripts/eval_hrl.py --structure-model {struct_path} --prompt-model {prompt_path} ")


if __name__ == "__main__":
    main()
