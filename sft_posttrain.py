"""
Supervised Fine-Tuning (SFT) Post-Training Script

Refines RL-trained models by training on high-quality correct episodes from RL logs.
This helps the model internalize correct behaviors and can improve accuracy by 2-5%.

Usage:
    # PPO Workflow
    python sft_posttrain.py \
        --rl-log logs/training_log_ppo_gaia_1767177868.json \
        --structure_model_path models/ppo_models/structure_policy_gaia_1767177868_final.pt \
        --prompt_model_path models/ppo_models/prompt_policy_gaia_1767177868_final.pt \
        --algorithm ppo --epochs 3

    # GRPO Workflow
    python sft_posttrain.py \
        --rl-log logs/training_log_grpo_gaia_1767177421.json \
        --structure_model_path models/grpo_models/structure_policy_gaia_1767177421_final.pt \
        --prompt_model_path models/grpo_models/prompt_policy_gaia_1767177421_final.pt \
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
from agents_system.worker import LLMWorker
from prompts import library


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
    mapping = {
        "Direct": 0,
        "Reason+Ans": 1,
        "Reason+Verify+Ans": 2,
        "Routing": 3,
        "Parallel-Sectioning": 4,
        "Parallel-Voting": 5,
        "Orchestrator-Workers": 6,
        "Evaluator-Optimizer": 7,
        "Autonomous-Agent": 8
    }
    return mapping.get(workflow_str, 1)  # Default to Reason+Ans if unknown


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


def train_structure_policy_sft(structure_policy, episodes, structure_env, epochs=3, lr=1e-4, device="cuda", entropy_coef=0.01):
    """
    Train structure policy on correct RL episodes with entropy regularization to maintain diversity.
    
    Args:
        entropy_coef: Entropy regularization coefficient to prevent overfitting to most common workflows
    """
    print(f"\n=== Training Structure Policy (Post-RL SFT, entropy_coef={entropy_coef}) ===")
    
    # Analyze workflow distribution in filtered episodes
    workflow_counts = {}
    for ep in episodes:
        wf = ep.get("workflow", "Reason+Ans")
        workflow_counts[wf] = workflow_counts.get(wf, 0) + 1
    print(f"Workflow distribution in filtered episodes:")
    for wf, count in sorted(workflow_counts.items(), key=lambda x: -x[1]):
        print(f"  {wf}: {count} ({count/len(episodes)*100:.1f}%)")
    
    optimizer = optim.Adam(structure_policy.parameters(), lr=lr)
    structure_policy.train()
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_entropy_loss = 0.0
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
            # Log uses agent1_tools/agent2_tools, but also check for reasoner_tools/verifier_tools for compatibility
            reasoner_tools = encode_tools(episode.get("agent1_tools", episode.get("reasoner_tools", [])))
            reasoner_budget = decode_budget(episode.get("reasoner_budget", "Mid"))
            verifier_tools = encode_tools(episode.get("agent2_tools", episode.get("verifier_tools", [])))
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
            total_entropy = 0.0
            for i, logits in enumerate(action_logits_list):
                target = torch.LongTensor([target_action[i]]).to(device)
                total_loss += criterion(logits, target)
                
                # Compute entropy for regularization (encourages diversity)
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
                total_entropy += entropy.mean()
            
            # Average loss across action dimensions
            loss = total_loss / len(action_logits_list)
            entropy_loss = -total_entropy / len(action_logits_list)  # Negative because we want to maximize entropy
            
            # Combined loss: classification loss - entropy regularization
            combined_loss = loss - entropy_coef * entropy_loss
            
            # Backward pass
            optimizer.zero_grad()
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(structure_policy.parameters(), 0.5)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_entropy_loss += entropy_loss.item()
            num_updates += 1
        
        avg_loss = epoch_loss / num_updates if num_updates > 0 else 0.0
        avg_entropy = epoch_entropy_loss / num_updates if num_updates > 0 else 0.0
        losses.append(avg_loss)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}, entropy: {avg_entropy:.4f}")
    
    structure_policy.eval()
    return losses


def train_prompt_policy_sft(prompt_policy, episodes, prompt_env, epochs=3, lr=5e-6, device="cuda"):
    """
    Train prompt policy on correct RL episodes (one step at a time).
    
    Uses lower default learning rate (5e-6) because:
    - Prompt policy processes ~76K steps per epoch (vs 11K for structure)
    - Fine-tuning requires gentler updates to avoid destabilizing pretrained weights
    - Sequential nature means errors compound across steps
    - Lowered from 1e-5 to reduce loss spikes
    """
    print(f"\n=== Training Prompt Policy (Post-RL SFT, LR={lr:.2e}) ===")
    
    optimizer = optim.Adam(prompt_policy.parameters(), lr=lr)
    # Add learning rate scheduler to reduce LR if loss plateaus/increases
    # Reduced patience from 2 to 1 to react faster when loss increases
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, min_lr=1e-6
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
            # Log uses agent1_tools/agent2_tools, but also check for reasoner_tools/verifier_tools for compatibility
            reasoner_tools = encode_tools(episode.get("agent1_tools", episode.get("reasoner_tools", [])))
            reasoner_budget = decode_budget(episode.get("reasoner_budget", "Mid"))
            verifier_tools = encode_tools(episode.get("agent2_tools", episode.get("verifier_tools", [])))
            verifier_budget = decode_budget(episode.get("verifier_budget", "Mid"))
            answerer_budget = decode_budget(episode.get("answerer_budget", "Mid"))
            
            # Set up prompt environment
            prompt_env.set_structure(
                question=question,
                answer="",
                embedding=prompt_env.worker.get_embedding(question),
                structure={
                    "workflow_depth": workflow,
                    "agent1_tools_idx": reasoner_tools,
                    "agent1_budget_idx": reasoner_budget,
                    "agent2_tools_idx": verifier_tools,
                    "agent2_budget_idx": verifier_budget,
                    "answerer_budget_idx": answerer_budget,
                }
            )
            
            # IMPORTANT: Reset prompt environment state for this episode
            # Direct (0) and Parallel-Voting (5) don't need reasoner prompts
            if workflow == 0 or workflow == 5:
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
            # Workflows that use reasoner: 1, 2, 3, 4, 6, 7, 8 (NOT 0 or 5)
            if workflow >= 1 and workflow != 5 and reasoner_prompts:
                prompt_env.prompt_stage = prompt_env.PROMPT_STAGE_REASONER
                prompt_env.prompt_step = 0
                for prompt_idx in reasoner_prompts:
                    train_prompt_step(prompt_idx, "reasoner", max_valid_idx=6)
                    # Update environment state for next observation
                    prompt_env.selected_prompts["reasoner"].append(prompt_idx)
                    prompt_env.prompt_step += 1
            
            # Collect and train on verifier prompt steps
            # Workflows that use verifier: 2, 7 (Reason+Verify+Ans and Evaluator-Optimizer)
            if workflow in [2, 7] and verifier_prompts:
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
    parser.add_argument("--dataset", type=str, required=True, default=None, choices=["gsm8k", "hotpotqa", "gaia", "medqa", "aime25"], help="Dataset name (overrides config)")

    # parser.add_argument("--rl-model-dir", type=str, required=True, help="Directory with RL-trained models")
    parser.add_argument("--epochs", type=int, default=3, help="SFT training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for structure policy")
    parser.add_argument("--prompt-lr", type=float, default=5e-6, help="Learning rate for prompt policy (default: 5e-6, lowered to reduce loss spikes)")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy regularization coefficient for structure policy (prevents overfitting to most common workflow, default: 0.01)")
    parser.add_argument("--min-reward", type=float, default=4.0, help="Minimum reward for filtering")
    parser.add_argument("--output-dir", type=str, default="models/sft_posttrained", help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--algorithm", type=str, default="grpo", choices=["ppo", "grpo"], help="Model architecture to use. Auto-detected from checkpoint if available, otherwise uses this value (default: grpo)")
    
    parser.add_argument("--prompt_model_path", "--prompt-model", type=str, required=True, help="Path to the prompt policy checkpoint (e.g., models/ppo_models/prompt_policy_...pt)")
    parser.add_argument("--structure_model_path", "--structure-model", type=str, required=True, help="Path to the structure policy checkpoint (e.g., models/ppo_models/structure_policy_...pt)")
    
    # API configuration (must match training configuration)
    parser.add_argument("--api", action="store_true", default=False,
                       help="Use OpenRouter API instead of local HuggingFace models (must match training mode)")
    parser.add_argument("--api-model", type=str, default=None,
                       help="OpenRouter model ID (e.g., 'openai/gpt-4o', 'anthropic/claude-3.5-sonnet'). Defaults to OPENROUTER_MODEL env var")
    parser.add_argument("--hf-model", type=str, default=None,
                       help="HuggingFace model name (e.g., 'Qwen/Qwen2.5-7B-Instruct'). Defaults to LLM_MODEL_NAME from config")
    
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
    
    # Update Prompt Atoms based on dataset (same as train.py)
    print(f"Checking prompt atoms for dataset: {cfg.DATASET_NAME}...")
    atoms_path = library._get_atoms_path(cfg.DATASET_NAME)
    
    # 1. Check if atoms exist (Fast path)
    if os.path.exists(atoms_path):
        print(f"  Found existing atoms at {atoms_path}. Loading...")
        library.load_or_create_atoms(cfg.DATASET_NAME, worker=None)
    
    else:
        print(f"  Atoms not found. Initializing temporary worker to generate them...")
        
        # Create temp worker (use API if specified, otherwise HuggingFace)
        import gc
        
        if args.api:
            from agents_system.worker import OpenRouterWorker
            temp_worker = OpenRouterWorker(model_name=args.api_model)
        else:
            # Loading bigger model for better atom generation
            temp_worker = LLMWorker(model_name=args.hf_model or "Qwen/Qwen2.5-7B-Instruct")
        
        library.load_or_create_atoms(cfg.DATASET_NAME, worker=temp_worker)
        
        # CRITICAL: Free memory immediately
        print("  Generation complete. Freeing memory for training...")
        del temp_worker
        gc.collect()
        torch.cuda.empty_cache()
    
    print(f"  Active Atoms: {library.NUM_ATOMS}")
    
    # Load correct episodes from log
    all_episodes = load_correct_episodes_from_log(args.rl_log, min_reward=args.min_reward)
    
    if len(all_episodes) == 0:
        print("Error: No high-quality episodes found. Try lowering --min-reward threshold.")
        return
    
    # Check if log indicates API mode was used during training
    with open(args.rl_log, 'r') as f:
        log_data = json.load(f)
    log_model_type = log_data.get("model_type", None)
    log_model_name = log_data.get("model_name", None)
    
    if log_model_type:
        print(f"\n⚠️  Training log indicates model was trained with: {log_model_type} ({log_model_name})")
        if log_model_type == "API" and not args.api:
            print(f"   WARNING: You're running SFT with HuggingFace mode (--api not set), but training used API mode!")
            print(f"   Consider adding --api flag to match training configuration.")
        elif log_model_type == "HuggingFace" and args.api:
            print(f"   WARNING: You're running SFT with API mode (--api set), but training used HuggingFace mode!")
            print(f"   Consider removing --api flag to match training configuration.")
    
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
    
    # Auto-detect algorithm from checkpoint
    user_specified = args.algorithm.lower()
    detected_algorithm = None
    
    # Try to detect from checkpoint metadata
    if "algorithm" in struct_checkpoint:
        detected_algorithm = struct_checkpoint["algorithm"].lower()
    elif "algorithm" in prompt_checkpoint:
        detected_algorithm = prompt_checkpoint["algorithm"].lower()
    else:
        # Fallback: check for value_head in state_dict (PPO has it, GRPO doesn't)
        struct_state = struct_checkpoint.get("model_state_dict", struct_checkpoint)
        has_value_head = any("value_head" in key for key in struct_state.keys())
        detected_algorithm = "ppo" if has_value_head else "grpo"
    
    # Determine which algorithm to use
    if user_specified and detected_algorithm:
        if user_specified != detected_algorithm:
            print(f"⚠️  WARNING: You specified --algorithm {user_specified.upper()}, but checkpoint contains {detected_algorithm.upper()} model!")
            print(f"   Using detected algorithm: {detected_algorithm.upper()}")
            algorithm = detected_algorithm
        else:
            algorithm = user_specified
            print(f"Using algorithm: {algorithm.upper()} (matches checkpoint)")
    elif detected_algorithm:
        algorithm = detected_algorithm
        print(f"Auto-detected algorithm from checkpoint: {algorithm.upper()}")
    else:
        # Fallback to default if detection fails
        algorithm = user_specified or "grpo"
        print(f"Using algorithm: {algorithm.upper()} (default/fallback)")
    
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
    
    # Create environments for observation generation (with API support if needed)
    print(f"\n{'='*60}")
    print("Initializing environments...")
    if args.api:
        model_name = args.api_model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
        print(f"  Mode: API (OpenRouter)")
        print(f"  Model: {model_name}")
    else:
        from configs.base import LLM_MODEL_NAME
        model_name = args.hf_model or getattr(cfg, "LLM_MODEL_NAME", LLM_MODEL_NAME)
        print(f"  Mode: HuggingFace")
        print(f"  Model: {model_name}")
    print(f"{'='*60}")
    
    structure_env = StructureEnv(cfg, use_api=args.api, api_model=args.api_model, hf_model=args.hf_model)
    prompt_env = PromptEnv(cfg, use_api=args.api, api_model=args.api_model, hf_model=args.hf_model)
    
    # Train both policies
    print("\n" + "="*60)
    print("Starting SFT Post-training")
    print("="*60)
    
    struct_losses = train_structure_policy_sft(
        structure_policy, all_episodes, structure_env, 
        epochs=args.epochs, lr=args.lr, device=device, entropy_coef=args.entropy_coef
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
    print(f"  python scripts/eval_hrl.py --structure-model {struct_save_path} --prompt-model {prompt_save_path}")


if __name__ == "__main__":
    main()