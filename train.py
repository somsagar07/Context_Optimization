"""
Unified Training Script for Hierarchical RL

Supports both PPO and GRPO algorithms:
- PPO: Proximal Policy Optimization (with value function)
- GRPO: Group Relative Policy Optimization (critic-free, better for sparse rewards)

Usage:
    python train.py --algorithm grpo --episodes 20000
    python train.py --algorithm ppo --episodes 20000
    python train.py --algorithm grpo --episodes 20000 --entropy-coef 0.08 --tool-bonus 0.02
    python train.py --algorithm grpo --episodes 20000 --pretrain-structure models/sft_posttrained/structure_policy_sft.pt --pretrain-prompt models/sft_posttrained/prompt_policy_sft.pt
"""
import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from configs import load_config
from algorithms import Algorithm, PPOTrainer, GRPOTrainer
from agents_system.worker import LLMWorker
from prompts import library


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
    parser.add_argument("--dataset", type=str, default=None, choices=["gsm8k", "hotpotqa", "gaia"])
    
    # Algorithm hyperparameters
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="Clipping parameter")
    parser.add_argument("--epochs", type=int, default=4, help="Update epochs per batch")
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
    parser.add_argument("--tool-bonus", type=float, default=0.0, help="Bonus per tool (+ or -)")
    
    # Action masking
    parser.add_argument("--mask", action="store_true", default=False,
                       help="Enable action masking to reduce invalid action space (masks agent2 for workflows 0,1,5)")
    
    # API configuration
    parser.add_argument("--api", action="store_true", default=False,
                       help="Use OpenRouter API instead of local HuggingFace models")
    parser.add_argument("--api-model", type=str, default=None,
                       help="OpenRouter model ID (e.g., 'openai/gpt-4o', 'anthropic/claude-3.5-sonnet'). Defaults to OPENROUTER_MODEL env var")
    
    # Logging
    parser.add_argument("--log-every", type=int, default=50, help="Log frequency")
    parser.add_argument("--save-every", type=int, default=2000, help="Checkpoint frequency")
    
    # Pretrained models (e.g., from SFT)
    parser.add_argument("--pretrain-structure", type=str, default=None,
                       help="Path to pretrained structure policy (e.g., from SFT)")
    parser.add_argument("--pretrain-prompt", type=str, default=None,
                       help="Path to pretrained prompt policy (e.g., from SFT)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    cfg = load_config(args.config)
    if args.dataset:
        cfg.DATASET_NAME = args.dataset
        
    # Update Prompt Atoms based on dataset
    print(f"Checking prompt atoms for dataset: {cfg.DATASET_NAME}...")
    atoms_path = library._get_atoms_path(cfg.DATASET_NAME)
    
    # 1. Check if atoms exist (Fast path)
    if os.path.exists(atoms_path):
        print(f"  Found existing atoms at {atoms_path}. Loading...")
        library.load_or_create_atoms(cfg.DATASET_NAME, worker=None)
    
    else:
        print(f"  Atoms not found. Initializing temporary worker to generate them...")
        
        # Create temp worker
        import gc
        import torch
        
        # Loading bigger model for better atom generation
        temp_worker = LLMWorker(model_name="Qwen/Qwen2.5-7B-Instruct") 
        library.load_or_create_atoms(cfg.DATASET_NAME, worker=temp_worker)
        
        # CRITICAL: Free memory immediately
        print("  Generation complete. Freeing memory for training...")
        del temp_worker
        gc.collect()
        torch.cuda.empty_cache()
    
    print(f"  Active Atoms: {library.NUM_ATOMS}")
    
    # Create model directory
    os.makedirs(f"models/{args.algorithm}_models", exist_ok=True)
    
    # Create trainer
    if args.algorithm == "ppo":
        trainer = PPOTrainer(cfg, use_action_masking=args.mask, use_api=args.api, api_model=args.api_model)
    else:
        trainer = GRPOTrainer(cfg, use_action_masking=args.mask, use_api=args.api, api_model=args.api_model)
    
    # Load pretrained models if provided (e.g., from SFT)
    if args.pretrain_structure or args.pretrain_prompt:
        trainer.load_pretrained(args.pretrain_structure, args.pretrain_prompt, reset_optimizers=True)
        # If using pretrained models, optionally use lower learning rates for fine-tuning
        if args.struct_lr is None and args.pretrain_structure:
            # Suggest lower LR when continuing from pretrained
            print("  Tip: Consider using --struct-lr 1e-4 for gentler fine-tuning from pretrained models")
        if args.prompt_lr is None and args.pretrain_prompt:
            print("  Tip: Consider using --prompt-lr 5e-5 for gentler fine-tuning from pretrained models")
    
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
    
    # Train
    trainer.train(
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        log_every=args.log_every,
        save_every=args.save_every,
        # Algorithm params
        gamma=cfg.PROMPT_GAMMA,
        clip_epsilon=args.clip_epsilon,
        epochs=args.epochs,
        struct_entropy_coef=struct_ent,
        prompt_entropy_coef=prompt_ent,
        # GRPO-specific
        kl_coef=args.kl_coef,
        ref_update_every=args.ref_update_every,
        # Reward
        reward_scale=args.reward_scale,
        tool_bonus=args.tool_bonus,
    )
    
    # Save final
    struct_path, prompt_path = trainer.save_models("_final")
    
    print(f"\nTo evaluate:")
    print(f"  python eval.py --structure-model {struct_path} --prompt-model {prompt_path}")


if __name__ == "__main__":
    main()
