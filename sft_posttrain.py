"""
Supervised Fine-Tuning (SFT) Post-Training Script

Refines RL-trained models by training on high-quality correct episodes from RL logs.
This helps the model internalize correct behaviors and can improve accuracy by 2-5%.

Usage:
    # PPO Workflow
    python sft_posttrain.py \
        --rl-log logs/training_log_ppo_gaia_1767177868.json \
        --structure-model models/ppo_models/structure_policy_gaia_1767177868_final.pt \
        --prompt-model models/ppo_models/prompt_policy_gaia_1767177868_final.pt \
        --algorithm ppo --epochs 3

    # GRPO Workflow
    python sft_posttrain.py \
        --rl-log logs/training_log_grpo_gaia_1767177421.json \
        --structure-model models/grpo_models/structure_policy_gaia_1767177421_final.pt \
        --prompt-model models/grpo_models/prompt_policy_gaia_1767177421_final.pt \
        --algorithm grpo --epochs 3

    # After RL training
    python sft_posttrain.py --rl-log logs/training_log_grpo_gsm8k_1766872133.json --rl-model-dir models/grpo_models --epochs 3
    python eval.py --structure-model models/sft_posttrained/structure_policy_sft.pt --prompt-model models/sft_posttrained/prompt_policy_sft.pt
"""
import argparse
import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from glob import glob
import time

from configs import load_config
from algorithms.base import MultiDiscretePolicyGRPO, PolicyNetworkGRPO
from algorithms.ppo import MultiDiscretePolicyPPO, PolicyNetworkPPO
from env.structure_env import StructureEnv
from env.prompt_env import PromptEnv


def encode_tools(tools_list):
    """Encode tool list to bitmask index."""
    tool_mapping = {
        "calculator": 1,
        "web_search": 2,
        "python": 4,
        "ocr_reader": 8,
    }
    idx = 0
    for tool in tools_list:
        if tool in tool_mapping:
            idx |= tool_mapping[tool]
    return idx


def decode_budget(budget_str):
    """Convert budget string to index."""
    mapping = {"Low": 0, "Mid": 1, "High": 2, "N/A": 0}
    return mapping.get(budget_str, 0)


def decode_workflow(workflow_str):
    """Convert workflow string to index."""
    mapping = {"Direct": 0, "Reason+Ans": 1, "Reason+Verify+Ans": 2}
    return mapping.get(workflow_str, 1)


def load_correct_episodes_from_log(log_path, min_reward=4.0):
    """
    Load correct episodes from RL training log.
    
    Args:
        log_path: Path to training log JSON
        min_reward: Minimum reward threshold (filters high-quality examples)
    
    Returns:
        List of correct episode dictionaries
    """
    print(f"Loading log from: {log_path}")
    with open(log_path, 'r') as f:
        log_data = json.load(f)
    
    correct_episodes = []
    for episode in log_data.get("episodes", []):
        if episode.get("correct", False) and episode.get("reward", 0) >= min_reward:
            correct_episodes.append(episode)
    
    print(f"Loaded {len(correct_episodes)} high-quality correct episodes (min_reward={min_reward})")
    print(f"  Total episodes in log: {len(log_data.get('episodes', []))}")
    print(f"  Accuracy in log: {log_data.get('accuracy', 0):.1%}")
    return correct_episodes


def train_structure_policy_sft(structure_policy, episodes, structure_env, epochs=3, lr=1e-4, device="cuda"):
    """Train structure policy on correct RL episodes (one episode at a time)."""
    print(f"\n=== Training Structure Policy (Post-RL SFT) ===")
    
    optimizer = optim.Adam(structure_policy.parameters(), lr=lr)
    structure_policy.train()
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_updates = 0
        
        # Shuffle episodes for each epoch
        shuffled_episodes = episodes.copy()
        random.shuffle(shuffled_episodes)
        
        # Process one episode at a time
        for episode in tqdm(shuffled_episodes, desc=f"Epoch {epoch+1}/{epochs}"):
            question = episode.get("question", "")
            if not question:
                continue
            
            # Get observation
            structure_env.current_q = question
            structure_env.current_a = ""
            structure_env.question_embedding = structure_env.worker.get_embedding(question)
            obs = structure_env._get_observation()
            
            # Decode target action
            workflow = decode_workflow(episode.get("workflow", "Reason+Ans"))
            reasoner_tools = encode_tools(episode.get("reasoner_tools", []))
            reasoner_budget = decode_budget(episode.get("reasoner_budget", "Mid"))
            verifier_tools = encode_tools(episode.get("verifier_tools", []))
            verifier_budget = decode_budget(episode.get("verifier_budget", "Mid"))
            answerer_budget = decode_budget(episode.get("answerer_budget", "Mid"))
            
            target_action = np.array([
                workflow,
                reasoner_tools,
                reasoner_budget,
                verifier_tools,
                verifier_budget,
                answerer_budget,
            ])
            
            # Convert to tensors (single observation with batch dimension)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)  # [1, obs_dim]
            
            # Forward pass
            policy_output = structure_policy(obs_tensor)
            
            # If PPO, output is (logits_list, value). If GRPO, output is logits_list.
            if isinstance(policy_output, tuple):
                action_logits_list = policy_output[0]
            else:
                action_logits_list = policy_output  # List of [1, action_dim]
            
            # Compute loss for each action dimension
            total_loss = 0.0
            for i, logits in enumerate(action_logits_list):
                target = torch.LongTensor([target_action[i]]).to(device)
                total_loss += criterion(logits, target)
            
            # Average loss across action dimensions
            loss = total_loss / len(action_logits_list)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(structure_policy.parameters(), 0.5)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_updates += 1
        
        avg_loss = epoch_loss / num_updates if num_updates > 0 else 0.0
        losses.append(avg_loss)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
    
    structure_policy.eval()
    return losses


def train_prompt_policy_sft(prompt_policy, episodes, prompt_env, epochs=3, lr=1e-5, device="cuda"):
    """
    Train prompt policy on correct RL episodes (one step at a time).
    
    Uses lower default learning rate (1e-5) because:
    - Prompt policy processes ~76K steps per epoch (vs 11K for structure)
    - Fine-tuning requires gentler updates to avoid destabilizing pretrained weights
    - Sequential nature means errors compound across steps
    """
    print(f"\n=== Training Prompt Policy (Post-RL SFT, LR={lr:.2e}) ===")
    
    optimizer = optim.Adam(prompt_policy.parameters(), lr=lr)
    # Add learning rate scheduler to reduce LR if loss plateaus/increases
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
    )
    
    prompt_policy.train()
    criterion = nn.CrossEntropyLoss()
    
    # Get action dimension for validation
    action_dim = None
    if hasattr(prompt_policy, 'action_head'):
        action_dim = prompt_policy.action_head.out_features
    elif hasattr(prompt_policy, 'net') and hasattr(prompt_policy.net[-1], 'out_features'):
        action_dim = prompt_policy.net[-1].out_features
    
    if action_dim:
        print(f"  Action dimension: {action_dim}")
    
    losses = []
    invalid_targets = 0
    nan_losses = 0
    nan_grads = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_updates = 0
        
        # Shuffle episodes for each epoch
        shuffled_episodes = episodes.copy()
        random.shuffle(shuffled_episodes)
        
        # Process one episode at a time
        for episode in tqdm(shuffled_episodes, desc=f"Epoch {epoch+1}/{epochs}"):
            question = episode.get("question", "")
            if not question:
                continue
            
            # Decode structure from episode
            workflow = decode_workflow(episode.get("workflow", "Reason+Ans"))
            reasoner_tools = encode_tools(episode.get("reasoner_tools", []))
            reasoner_budget = decode_budget(episode.get("reasoner_budget", "Mid"))
            verifier_tools = encode_tools(episode.get("verifier_tools", []))
            verifier_budget = decode_budget(episode.get("verifier_budget", "Mid"))
            answerer_budget = decode_budget(episode.get("answerer_budget", "Mid"))
            
            # Set up prompt environment
            prompt_env.set_structure(
                question=question,
                answer="",
                embedding=prompt_env.worker.get_embedding(question),
                structure={
                    "workflow_depth": workflow,
                    "reasoner_tools_idx": reasoner_tools,
                    "reasoner_budget_idx": reasoner_budget,
                    "verifier_tools_idx": verifier_tools,
                    "verifier_budget_idx": verifier_budget,
                    "answerer_budget_idx": answerer_budget,
                }
            )
            
            # IMPORTANT: Reset prompt environment state for this episode
            if workflow == 0:
                prompt_env.prompt_stage = prompt_env.PROMPT_STAGE_ANSWERER
            else:
                prompt_env.prompt_stage = prompt_env.PROMPT_STAGE_REASONER
            prompt_env.prompt_step = 0
            prompt_env.selected_prompts = {"reasoner": [], "verifier": [], "answerer": []}
            
            # Get target prompts from episode
            reasoner_prompts = episode.get("reasoner_prompts", [])
            verifier_prompts = episode.get("verifier_prompts", [])
            answerer_prompts = episode.get("answerer_prompts", [])
            
            # Helper function to train on a single prompt step
            def train_prompt_step(prompt_idx, stage_name, max_valid_idx):
                """Train on a single prompt selection step with validation."""
                nonlocal epoch_loss, num_updates, invalid_targets, nan_losses, nan_grads
                
                # Validate prompt index
                if action_dim and (prompt_idx < 0 or prompt_idx >= action_dim):
                    invalid_targets += 1
                    if invalid_targets <= 5:  # Only print first few warnings
                        print(f"Warning: Invalid {stage_name} prompt_idx {prompt_idx} (action_dim={action_dim}), skipping")
                    return
                
                # Also validate against stage-specific max (reasoner: 6, verifier: 5, answerer: 4)
                if prompt_idx < 1 or prompt_idx > max_valid_idx:
                    invalid_targets += 1
                    if invalid_targets <= 5:
                        print(f"Warning: {stage_name} prompt_idx {prompt_idx} out of range [1, {max_valid_idx}], skipping")
                    return
                
                obs = prompt_env._get_observation()
                
                # Convert to tensor
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                target = torch.LongTensor([prompt_idx]).to(device)
                
                # Forward pass
                policy_output = prompt_policy(obs_tensor)
                
                # If PPO, output is (logits, value). If GRPO, output is logits.
                if isinstance(policy_output, tuple):
                    action_logits = policy_output[0]
                else:
                    action_logits = policy_output
                
                # Validate logits shape
                if action_dim and action_logits.shape[1] != action_dim:
                    if invalid_targets <= 5:
                        print(f"Warning: Logits shape {action_logits.shape[1]} != action_dim {action_dim}")
                    return
                
                # Compute loss
                loss = criterion(action_logits, target)
                
                # Check for NaN or Inf loss
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_losses += 1
                    if nan_losses <= 5:
                        print(f"Warning: Invalid loss detected: {loss.item()}, skipping")
                    return
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Check for NaN gradients
                has_nan_grad = False
                for param in prompt_policy.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    nan_grads += 1
                    if nan_grads <= 5:
                        print("Warning: NaN gradients detected, skipping update")
                    optimizer.zero_grad()
                    return
                
                torch.nn.utils.clip_grad_norm_(prompt_policy.parameters(), 0.5)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_updates += 1
            
            # Collect and train on reasoner prompt steps
            if workflow >= 1 and reasoner_prompts:
                prompt_env.prompt_stage = prompt_env.PROMPT_STAGE_REASONER
                prompt_env.prompt_step = 0
                for prompt_idx in reasoner_prompts:
                    train_prompt_step(prompt_idx, "reasoner", max_valid_idx=6)
                    # Update environment state for next observation
                    prompt_env.selected_prompts["reasoner"].append(prompt_idx)
                    prompt_env.prompt_step += 1
            
            # Collect and train on verifier prompt steps
            if workflow == 2 and verifier_prompts:
                prompt_env.prompt_stage = prompt_env.PROMPT_STAGE_VERIFIER
                prompt_env.prompt_step = 0
                for prompt_idx in verifier_prompts:
                    train_prompt_step(prompt_idx, "verifier", max_valid_idx=5)
                    # Update environment state
                    prompt_env.selected_prompts["verifier"].append(prompt_idx)
                    prompt_env.prompt_step += 1
            
            # Collect and train on answerer prompt steps
            if answerer_prompts:
                prompt_env.prompt_stage = prompt_env.PROMPT_STAGE_ANSWERER
                prompt_env.prompt_step = 0
                for prompt_idx in answerer_prompts:
                    train_prompt_step(prompt_idx, "answerer", max_valid_idx=4)
                    # Update environment state
                    prompt_env.selected_prompts["answerer"].append(prompt_idx)
                    prompt_env.prompt_step += 1
        
        avg_loss = epoch_loss / num_updates if num_updates > 0 else 0.0
        losses.append(avg_loss)
        
        # Update learning rate scheduler
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print warnings if any issues detected
        if invalid_targets > 0:
            print(f"  Warning: Skipped {invalid_targets} invalid targets")
        if nan_losses > 0:
            print(f"  Warning: Skipped {nan_losses} NaN/Inf losses")
        if nan_grads > 0:
            print(f"  Warning: Skipped {nan_grads} updates due to NaN gradients")
        
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f} (processed {num_updates} prompt steps, LR: {current_lr:.2e})")
    
    prompt_policy.eval()
    return losses


def main():
    parser = argparse.ArgumentParser(description="SFT Post-training from RL logs")
    parser.add_argument("--config", type=str, default="hierarchical", help="Config to use")
    parser.add_argument("--rl-log", type=str, required=True, help="Path to RL training log JSON")
    # parser.add_argument("--rl-model-dir", type=str, required=True, help="Directory with RL-trained models")
    parser.add_argument("--epochs", type=int, default=3, help="SFT training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for structure policy")
    parser.add_argument("--prompt-lr", type=float, default=1e-5, help="Learning rate for prompt policy (default: 1e-5, lower due to more training steps)")
    parser.add_argument("--min-reward", type=float, default=4.0, help="Minimum reward for filtering")
    parser.add_argument("--output-dir", type=str, default="models/sft_posttrained", help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--algorithm", type=str, default="grpo", choices=["ppo", "grpo"], help="Model architecture to use (matches your saved checkpoint)")
    
    parser.add_argument("--prompt_model_path", type=str, required=True, help="Path to the prompt policy checkpoint (e.g., models/ppo_models/prompt_policy_...pt)")
    parser.add_argument("--structure_model_path", type=str, required=True, help="Path to the structure policy checkpoint (e.g., models/ppo_models/structure_policy_...pt)")
    parser.add_argument("--dataset", type=str, default=None, choices=["gsm8k", "hotpotqa", "gaia"], help="Dataset name (overrides config)")
    
    args = parser.parse_args()
    
    # Device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Load config
    cfg = load_config(args.config)
    
    if args.dataset:
        cfg.DATASET_NAME = args.dataset
        print(f"Overriding config dataset with: {cfg.DATASET_NAME}")
    
    # Load correct episodes from log
    all_episodes = load_correct_episodes_from_log(args.rl_log, min_reward=args.min_reward)
    
    if len(all_episodes) == 0:
        print("Error: No high-quality episodes found. Try lowering --min-reward threshold.")
        return
    
    # Find RL-trained models (look for _final.pt files)
    # struct_pattern = os.path.join(args.rl_model_dir, "*structure*_final.pt")
    # prompt_pattern = os.path.join(args.rl_model_dir, "*prompt*_final.pt")
    
    # struct_files = glob(struct_pattern)
    # prompt_files = glob(prompt_pattern)
    
    # if not struct_files:
    #     # Try without _final suffix
    #     struct_pattern = os.path.join(args.rl_model_dir, "*structure*.pt")
    #     struct_files = sorted(glob(struct_pattern), key=os.path.getmtime, reverse=True)
    #     if struct_files:
    #         struct_files = [struct_files[0]]  # Use most recent
    
    # if not prompt_files:
    #     prompt_pattern = os.path.join(args.rl_model_dir, "*prompt*.pt")
    #     prompt_files = sorted(glob(prompt_pattern), key=os.path.getmtime, reverse=True)
    #     if prompt_files:
    #         prompt_files = [prompt_files[0]]
    
    # if not struct_files or not prompt_files:
    #     raise FileNotFoundError(
    #         f"RL models not found in {args.rl_model_dir}\n"
    #         f"  Looking for: {struct_pattern} and {prompt_pattern}"
    #     )
    
    # struct_path = struct_files[0]
    # prompt_path = prompt_files[0]
    
    struct_path = args.structure_model_path
    prompt_path = args.prompt_model_path
    
    print(f"\nLoading RL-trained models...")
    print(f"  Structure: {struct_path}")
    print(f"  Prompt: {prompt_path}")
    
    # Load checkpoints
    struct_checkpoint = torch.load(struct_path, map_location=device, weights_only=False)
    prompt_checkpoint = torch.load(prompt_path, map_location=device, weights_only=False)
    
    algorithm = args.algorithm.lower()
    
    if algorithm == "ppo":
        # Initialize policies with same architecture
        structure_policy = MultiDiscretePolicyPPO(
            obs_dim=struct_checkpoint["obs_dim"],
            action_dims=struct_checkpoint["action_dims"]
        ).to(device)
        
        prompt_policy = PolicyNetworkPPO(
            obs_dim=prompt_checkpoint["obs_dim"],
            action_dim=prompt_checkpoint["action_dim"]
        ).to(device)
    else:
        # Initialize policies with same architecture
        structure_policy = MultiDiscretePolicyGRPO(
            obs_dim=struct_checkpoint["obs_dim"],
            action_dims=struct_checkpoint["action_dims"]
        ).to(device)
        
        prompt_policy = PolicyNetworkGRPO(
            obs_dim=prompt_checkpoint["obs_dim"],
            action_dim=prompt_checkpoint["action_dim"]
        ).to(device)
    
    # Load RL weights
    structure_policy.load_state_dict(struct_checkpoint["model_state_dict"])
    prompt_policy.load_state_dict(prompt_checkpoint["model_state_dict"])
    print("✓ Models loaded")
    
    # Create environments for observation generation
    structure_env = StructureEnv(cfg)
    prompt_env = PromptEnv(cfg)
    
    # Train both policies
    print("\n" + "="*60)
    print("Starting SFT Post-training")
    print("="*60)
    
    struct_losses = train_structure_policy_sft(
        structure_policy, all_episodes, structure_env, 
        epochs=args.epochs, lr=args.lr, device=device
    )
    prompt_losses = train_prompt_policy_sft(
        prompt_policy, all_episodes, prompt_env,
        epochs=args.epochs, lr=args.prompt_lr, device=device  # Use separate LR for prompt policy
    )
    
    # Save models (in same format as RL models for compatibility)
    timestamp = int(time.time())
    os.makedirs(args.output_dir, exist_ok=True)
    struct_save_path = os.path.join(args.output_dir, f"structure_policy_sft_{timestamp}.pt")
    prompt_save_path = os.path.join(args.output_dir, f"prompt_policy_sft_{timestamp}.pt")
    
    torch.save({
        "model_state_dict": structure_policy.state_dict(),
        "action_dims": struct_checkpoint["action_dims"],
        "obs_dim": struct_checkpoint["obs_dim"],
        "algorithm": f"{algorithm.upper()}_SFT",
    }, struct_save_path)
    
    torch.save({
        "model_state_dict": prompt_policy.state_dict(),
        "action_dim": prompt_checkpoint["action_dim"],
        "obs_dim": prompt_checkpoint["obs_dim"],
        "algorithm": f"{algorithm.upper()}_SFT",
    }, prompt_save_path)
    
    print(f"\n{'='*60}")
    print("✓ SFT post-training complete!")
    print(f"{'='*60}")
    print(f"\nModels saved to:")
    print(f"  Structure: {struct_save_path}")
    print(f"  Prompt: {prompt_save_path}")
    print(f"\nEvaluate with:")
    print(f"  python eval.py --structure-model {struct_save_path} --prompt-model {prompt_save_path}")


if __name__ == "__main__":
    main()