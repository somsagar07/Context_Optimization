"""
Direct Preference Optimization (DPO) Post-Training Script

Trains models using preference pairs from RL logs (correct vs incorrect episodes).
This helps the model learn to distinguish between good and bad decisions,
preventing overfitting and improving generalization.

DPO Loss: Maximizes log-prob of preferred (correct) actions while minimizing
log-prob of dispreferred (incorrect) actions, with KL penalty to reference model.

Usage:
    # After RL training
    python scripts/dpo_posttrain.py --rl-log logs/training_log_grpo_gsm8k_1766872133.json --rl-model-dir models/grpo_models --epochs 3
    python scripts/eval_hrl.py --structure-model models/dpo_posttrained/structure_policy_dpo.pt --prompt-model models/dpo_posttrained/prompt_policy_dpo.pt
"""
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from glob import glob
from collections import defaultdict

from configs import load_config
from algorithms.base import MultiDiscretePolicyGRPO, PolicyNetworkGRPO
from env.structure_env import StructureEnv
from env.prompt_env import PromptEnv


def encode_tools(tools_list):
    """Encode tool list to bitmask index."""
    tool_mapping = {
        "calculator": 1,
        "web_search": 2,
        "python": 4
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


def load_preference_pairs_from_log(log_path, min_reward_correct=4.0, max_reward_incorrect=2.0):
    """
    Load preference pairs (correct vs incorrect) from RL training log.
    Pairs episodes with same or similar questions.
    
    Args:
        log_path: Path to training log JSON
        min_reward_correct: Minimum reward for correct episodes (preferred)
        max_reward_incorrect: Maximum reward for incorrect episodes (dispreferred)
    
    Returns:
        List of preference pair dictionaries: {
            "question": str,
            "preferred": episode_dict (correct),
            "dispreferred": episode_dict (incorrect)
        }
    """
    print(f"Loading log from: {log_path}")
    with open(log_path, 'r') as f:
        log_data = json.load(f)
    
    # Group episodes by question (exact match)
    episodes_by_question = defaultdict(list)
    for episode in log_data.get("episodes", []):
        question = episode.get("question", "")
        if question:
            episodes_by_question[question].append(episode)
    
    # Create preference pairs
    preference_pairs = []
    
    for question, episodes in episodes_by_question.items():
        # Find correct (preferred) episodes
        correct_episodes = [
            ep for ep in episodes 
            if ep.get("correct", False) and ep.get("reward", 0) >= min_reward_correct
        ]
        
        # Find incorrect (dispreferred) episodes
        incorrect_episodes = [
            ep for ep in episodes 
            if not ep.get("correct", False) and ep.get("reward", 0) <= max_reward_incorrect
        ]
        
        # Create pairs: one correct vs one incorrect
        if correct_episodes and incorrect_episodes:
            # Randomly sample from correct/incorrect to ensure diversity
            # (instead of always using best/worst which can bias toward common patterns)
            preferred = random.choice(correct_episodes)
            dispreferred = random.choice(incorrect_episodes)
            
            preference_pairs.append({
                "question": question,
                "preferred": preferred,
                "dispreferred": dispreferred
            })
    
    # If not enough exact matches, try fuzzy matching by question similarity
    if len(preference_pairs) < 100:
        print(f"Only {len(preference_pairs)} exact question matches. Trying fuzzy matching...")
        
        all_correct = [
            ep for ep in log_data.get("episodes", [])
            if ep.get("correct", False) and ep.get("reward", 0) >= min_reward_correct
        ]
        all_incorrect = [
            ep for ep in log_data.get("episodes", [])
            if not ep.get("correct", False) and ep.get("reward", 0) <= max_reward_incorrect
        ]
        
        # Random pairing for different questions (still valid preference learning)
        min_pairs = min(len(all_correct), len(all_incorrect))
        for i in range(min_pairs):
            preference_pairs.append({
                "question": all_correct[i].get("question", ""),
                "preferred": all_correct[i],
                "dispreferred": all_incorrect[i % len(all_incorrect)]
            })
    
    print(f"Created {len(preference_pairs)} preference pairs")
    print(f"  Total episodes in log: {len(log_data.get('episodes', []))}")
    print(f"  Accuracy in log: {log_data.get('accuracy', 0):.1%}")
    return preference_pairs


def compute_dpo_loss(log_probs_preferred, log_probs_dispreferred, 
                     ref_log_probs_preferred, ref_log_probs_dispreferred, beta=0.1):
    """
    Compute DPO loss.
    
    Args:
        log_probs_preferred: Log probs of preferred actions under current policy
        log_probs_dispreferred: Log probs of dispreferred actions under current policy
        ref_log_probs_preferred: Log probs of preferred actions under reference policy
        ref_log_probs_dispreferred: Log probs of dispreferred actions under reference policy
        beta: Temperature parameter (higher = stronger preference signal)
    
    Returns:
        DPO loss scalar
    """
    # Compute log ratios
    log_ratio_preferred = log_probs_preferred - ref_log_probs_preferred
    log_ratio_dispreferred = log_probs_dispreferred - ref_log_probs_dispreferred
    
    # DPO loss: -log(σ(β * (log π_θ(y_w) - log π_ref(y_w) - log π_θ(y_l) + log π_ref(y_l))))
    logits = beta * (log_ratio_preferred - log_ratio_dispreferred)
    loss = -torch.nn.functional.logsigmoid(logits)
    
    return loss.mean()


def train_structure_policy_dpo(structure_policy, structure_policy_ref, preference_pairs, 
                                structure_env, epochs=3, lr=1e-4, batch_size=16, 
                                beta=0.1, entropy_coef=0.01, device="cuda"):
    """
    Train structure policy using DPO on preference pairs.
    
    Args:
        entropy_coef: Entropy regularization coefficient to prevent mode collapse
    """
    print(f"\n=== Training Structure Policy (DPO, batch_size={batch_size}, beta={beta}, entropy={entropy_coef}) ===")
    
    optimizer = optim.Adam(structure_policy.parameters(), lr=lr)
    structure_policy.train()
    structure_policy_ref.eval()  # Reference model is frozen
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle pairs for each epoch
        shuffled_pairs = preference_pairs.copy()
        random.shuffle(shuffled_pairs)
        
        # Process in batches
        for batch_start in tqdm(range(0, len(shuffled_pairs), batch_size), 
                                desc=f"Epoch {epoch+1}/{epochs}"):
            batch_pairs = shuffled_pairs[batch_start:batch_start + batch_size]
            
            # Collect batch data for preferred and dispreferred
            batch_obs_preferred = []
            batch_obs_dispreferred = []
            batch_targets_preferred = []
            batch_targets_dispreferred = []
            
            for pair in batch_pairs:
                question = pair["question"]
                preferred = pair["preferred"]
                dispreferred = pair["dispreferred"]
                
                if not question:
                    continue
                
                # Process preferred episode
                structure_env.current_q = question
                structure_env.current_a = ""
                structure_env.question_embedding = structure_env.worker.get_embedding(question)
                obs_pref = structure_env._get_observation()
                batch_obs_preferred.append(obs_pref)
                
                workflow = decode_workflow(preferred.get("workflow", "Reason+Ans"))
                reasoner_tools = encode_tools(preferred.get("reasoner_tools", []))
                reasoner_budget = decode_budget(preferred.get("reasoner_budget", "Mid"))
                verifier_tools = encode_tools(preferred.get("verifier_tools", []))
                verifier_budget = decode_budget(preferred.get("verifier_budget", "Mid"))
                answerer_budget = decode_budget(preferred.get("answerer_budget", "Mid"))
                
                target_pref = np.array([
                    workflow, reasoner_tools, reasoner_budget,
                    verifier_tools, verifier_budget, answerer_budget,
                ])
                batch_targets_preferred.append(target_pref)
                
                # Process dispreferred episode
                obs_dispref = structure_env._get_observation()  # Same question
                batch_obs_dispreferred.append(obs_dispref)
                
                workflow = decode_workflow(dispreferred.get("workflow", "Reason+Ans"))
                reasoner_tools = encode_tools(dispreferred.get("reasoner_tools", []))
                reasoner_budget = decode_budget(dispreferred.get("reasoner_budget", "Mid"))
                verifier_tools = encode_tools(dispreferred.get("verifier_tools", []))
                verifier_budget = decode_budget(dispreferred.get("verifier_budget", "Mid"))
                answerer_budget = decode_budget(dispreferred.get("answerer_budget", "Mid"))
                
                target_dispref = np.array([
                    workflow, reasoner_tools, reasoner_budget,
                    verifier_tools, verifier_budget, answerer_budget,
                ])
                batch_targets_dispreferred.append(target_dispref)
            
            if len(batch_obs_preferred) == 0:
                continue
            
            # Convert to tensors
            obs_pref_batch = torch.FloatTensor(np.array(batch_obs_preferred)).to(device)
            obs_dispref_batch = torch.FloatTensor(np.array(batch_obs_dispreferred)).to(device)
            targets_pref_batch = np.array(batch_targets_preferred)  # [batch_size, 6]
            targets_dispref_batch = np.array(batch_targets_dispreferred)
            
            # Forward pass: current policy
            action_logits_list_pref = structure_policy(obs_pref_batch)
            action_logits_list_dispref = structure_policy(obs_dispref_batch)
            
            # Forward pass: reference policy (frozen)
            with torch.no_grad():
                ref_logits_list_pref = structure_policy_ref(obs_pref_batch)
                ref_logits_list_dispref = structure_policy_ref(obs_dispref_batch)
            
            # Compute log probs for each action dimension and aggregate
            total_loss = 0.0
            for i, (logits_pref, logits_dispref, ref_logits_pref, ref_logits_dispref) in enumerate(
                zip(action_logits_list_pref, action_logits_list_dispref,
                    ref_logits_list_pref, ref_logits_list_dispref)
            ):
                targets_pref = torch.LongTensor(targets_pref_batch[:, i]).to(device)
                targets_dispref = torch.LongTensor(targets_dispref_batch[:, i]).to(device)
                
                # Get log probs for chosen actions
                log_probs_pref = torch.log_softmax(logits_pref, dim=-1)
                log_probs_dispref = torch.log_softmax(logits_dispref, dim=-1)
                
                log_prob_pref = log_probs_pref.gather(1, targets_pref.unsqueeze(1)).squeeze(1)
                log_prob_dispref = log_probs_dispref.gather(1, targets_dispref.unsqueeze(1)).squeeze(1)
                
                # Reference policy log probs
                ref_log_probs_pref = torch.log_softmax(ref_logits_pref, dim=-1)
                ref_log_probs_dispref = torch.log_softmax(ref_logits_dispref, dim=-1)
                
                ref_log_prob_pref = ref_log_probs_pref.gather(1, targets_pref.unsqueeze(1)).squeeze(1)
                ref_log_prob_dispref = ref_log_probs_dispref.gather(1, targets_dispref.unsqueeze(1)).squeeze(1)
                
                # DPO loss for this action dimension
                loss = compute_dpo_loss(
                    log_prob_pref, log_prob_dispref,
                    ref_log_prob_pref, ref_log_prob_dispref,
                    beta=beta
                )
                total_loss += loss
            
            # Average loss across action dimensions
            dpo_loss = total_loss / len(action_logits_list_pref)
            
            # Add entropy regularization to prevent mode collapse
            # Compute entropy on BOTH preferred and dispreferred observations to encourage diversity
            entropy_loss = 0.0
            for logits_pref, logits_disp in zip(action_logits_list_pref, action_logits_list_dispref):
                # Entropy for preferred
                probs_pref = torch.softmax(logits_pref, dim=-1)
                entropy_pref = -(probs_pref * torch.log(probs_pref + 1e-8)).sum(dim=-1).mean()
                # Entropy for dispreferred
                probs_disp = torch.softmax(logits_disp, dim=-1)
                entropy_disp = -(probs_disp * torch.log(probs_disp + 1e-8)).sum(dim=-1).mean()
                # Average entropy across both
                entropy_loss += (entropy_pref + entropy_disp) / 2.0
            entropy_loss = entropy_loss / len(action_logits_list_pref)
            
            # Total loss: DPO loss - entropy (we want to maximize entropy, so subtract)
            loss = dpo_loss - entropy_coef * entropy_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(structure_policy.parameters(), 0.5)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        losses.append(avg_loss)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
    
    structure_policy.eval()
    return losses


def train_prompt_policy_dpo(prompt_policy, prompt_policy_ref, preference_pairs,
                            prompt_env, epochs=3, lr=1e-5, batch_size=16,
                            beta=0.1, entropy_coef=0.01, device="cuda"):
    """
    Train prompt policy using DPO on preference pairs.
    
    Uses lower default learning rate (1e-5) because:
    - Prompt policy processes many sequential steps
    - Fine-tuning requires gentler updates
    - DPO loss can be more sensitive than supervised loss
    
    Args:
        entropy_coef: Entropy regularization coefficient to prevent mode collapse
    """
    print(f"\n=== Training Prompt Policy (DPO, batch_size={batch_size}, beta={beta}, entropy={entropy_coef}, LR={lr:.2e}) ===")
    
    optimizer = optim.Adam(prompt_policy.parameters(), lr=lr)
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
    )
    
    prompt_policy.train()
    prompt_policy_ref.eval()  # Reference model is frozen
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Collect all prompt step pairs from preference pairs
        all_pairs = []
        
        # Add progress bar for pair collection (this is the slow part)
        for pair in tqdm(preference_pairs, desc=f"Epoch {epoch+1}/{epochs} (collecting pairs)", leave=False):
            question = pair["question"]
            preferred = pair["preferred"]
            dispreferred = pair["dispreferred"]
            
            if not question:
                continue
            
            # Decode structure from preferred episode (use preferred structure)
            # NOTE: We use preferred structure for both, assuming they have same structure
            # If structures differ, we can only compare prompt steps that exist in both
            workflow_pref = decode_workflow(preferred.get("workflow", "Reason+Ans"))
            workflow_disp = decode_workflow(dispreferred.get("workflow", "Reason+Ans"))
            
            # Skip if structures are too different (can't meaningfully compare)
            if workflow_pref != workflow_disp:
                continue  # Skip this pair - structures don't match
            
            workflow = workflow_pref
            reasoner_tools = encode_tools(preferred.get("reasoner_tools", []))
            reasoner_budget = decode_budget(preferred.get("reasoner_budget", "Mid"))
            verifier_tools = encode_tools(preferred.get("verifier_tools", []))
            verifier_budget = decode_budget(preferred.get("verifier_budget", "Mid"))
            answerer_budget = decode_budget(preferred.get("answerer_budget", "Mid"))
            
            structure = {
                "workflow_depth": workflow,
                "reasoner_tools_idx": reasoner_tools,
                "reasoner_budget_idx": reasoner_budget,
                "verifier_tools_idx": verifier_tools,
                "verifier_budget_idx": verifier_budget,
                "answerer_budget_idx": answerer_budget,
            }
            
            # Set up prompt environment
            prompt_env.set_structure(
                question=question,
                answer="",
                embedding=prompt_env.worker.get_embedding(question),
                structure=structure
            )
            
            # IMPORTANT: Reset prompt environment state for this episode
            if workflow == 0:
                prompt_env.prompt_stage = prompt_env.PROMPT_STAGE_ANSWERER
            else:
                prompt_env.prompt_stage = prompt_env.PROMPT_STAGE_REASONER
            prompt_env.prompt_step = 0
            prompt_env.selected_prompts = {"reasoner": [], "verifier": [], "answerer": []}
            
            # Get prompts from both episodes
            pref_reasoner = preferred.get("reasoner_prompts", [])
            pref_verifier = preferred.get("verifier_prompts", [])
            pref_answerer = preferred.get("answerer_prompts", [])
            
            dispref_reasoner = dispreferred.get("reasoner_prompts", [])
            dispref_verifier = dispreferred.get("verifier_prompts", [])
            dispref_answerer = dispreferred.get("answerer_prompts", [])
            
            # Collect reasoner prompt pairs
            if workflow >= 1:
                prompt_env.prompt_stage = prompt_env.PROMPT_STAGE_REASONER
                prompt_env.prompt_step = 0
                # Note: selected_prompts already reset above
                
                max_len = min(len(pref_reasoner), len(dispref_reasoner))
                for step in range(max_len):
                    # Validate prompt indices
                    pref_idx = pref_reasoner[step]
                    dispref_idx = dispref_reasoner[step]
                    
                    # Skip if indices are invalid (should be 1-6 for reasoner)
                    if pref_idx < 1 or pref_idx > 6 or dispref_idx < 1 or dispref_idx > 6:
                        continue
                    
                    obs = prompt_env._get_observation()
                    all_pairs.append({
                        "obs": obs,
                        "preferred": pref_idx,
                        "dispreferred": dispref_idx
                    })
                    # Update state using preferred (for observation consistency)
                    prompt_env.selected_prompts["reasoner"].append(pref_idx)
                    prompt_env.prompt_step += 1
            
            # Collect verifier prompt pairs
            if workflow == 2:
                prompt_env.prompt_stage = prompt_env.PROMPT_STAGE_VERIFIER
                prompt_env.prompt_step = 0
                # Note: reasoner prompts are preserved in selected_prompts
                
                max_len = min(len(pref_verifier), len(dispref_verifier))
                for step in range(max_len):
                    pref_idx = pref_verifier[step]
                    dispref_idx = dispref_verifier[step]
                    
                    # Validate (should be 1-5 for verifier)
                    if pref_idx < 1 or pref_idx > 5 or dispref_idx < 1 or dispref_idx > 5:
                        continue
                    
                    obs = prompt_env._get_observation()
                    all_pairs.append({
                        "obs": obs,
                        "preferred": pref_idx,
                        "dispreferred": dispref_idx
                    })
                    prompt_env.selected_prompts["verifier"].append(pref_idx)
                    prompt_env.prompt_step += 1
            
            # Collect answerer prompt pairs
            prompt_env.prompt_stage = prompt_env.PROMPT_STAGE_ANSWERER
            prompt_env.prompt_step = 0
            # Note: reasoner and verifier prompts are preserved
            
            max_len = min(len(pref_answerer), len(dispref_answerer))
            for step in range(max_len):
                pref_idx = pref_answerer[step]
                dispref_idx = dispref_answerer[step]
                
                # Validate (should be 1-4 for answerer)
                if pref_idx < 1 or pref_idx > 4 or dispref_idx < 1 or dispref_idx > 4:
                    continue
                
                obs = prompt_env._get_observation()
                all_pairs.append({
                    "obs": obs,
                    "preferred": pref_idx,
                    "dispreferred": dispref_idx
                })
                prompt_env.selected_prompts["answerer"].append(pref_idx)
                prompt_env.prompt_step += 1
        
        if len(all_pairs) == 0:
            print("Warning: No prompt step pairs found")
            return []
        
        # Shuffle for each epoch
        random.shuffle(all_pairs)
        
        # Process in batches
        for batch_start in tqdm(range(0, len(all_pairs), batch_size), 
                                desc=f"Epoch {epoch+1}/{epochs} (training)"):
            batch_pairs = all_pairs[batch_start:batch_start + batch_size]
            
            batch_obs = [p["obs"] for p in batch_pairs]
            batch_preferred = [p["preferred"] for p in batch_pairs]
            batch_dispreferred = [p["dispreferred"] for p in batch_pairs]
            
            # Convert to tensors
            obs_batch = torch.FloatTensor(np.array(batch_obs)).to(device)
            targets_pref_batch = torch.LongTensor(batch_preferred).to(device)
            targets_dispref_batch = torch.LongTensor(batch_dispreferred).to(device)
            
            # Forward pass: current policy
            action_logits = prompt_policy(obs_batch)
            
            # Forward pass: reference policy (frozen)
            with torch.no_grad():
                ref_logits = prompt_policy_ref(obs_batch)
            
            # Get log probs for chosen actions
            log_probs = torch.log_softmax(action_logits, dim=-1)
            log_prob_pref = log_probs.gather(1, targets_pref_batch.unsqueeze(1)).squeeze(1)
            log_prob_dispref = log_probs.gather(1, targets_dispref_batch.unsqueeze(1)).squeeze(1)
            
            # Reference policy log probs
            ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
            ref_log_prob_pref = ref_log_probs.gather(1, targets_pref_batch.unsqueeze(1)).squeeze(1)
            ref_log_prob_dispref = ref_log_probs.gather(1, targets_dispref_batch.unsqueeze(1)).squeeze(1)
            
            # DPO loss
            dpo_loss = compute_dpo_loss(
                log_prob_pref, log_prob_dispref,
                ref_log_prob_pref, ref_log_prob_dispref,
                beta=beta
            )
            
            # Add entropy regularization to prevent mode collapse
            # Compute entropy on the action distribution to encourage diversity
            probs = torch.softmax(action_logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            
            # Total loss: DPO loss - entropy (we want to maximize entropy, so subtract)
            loss = dpo_loss - entropy_coef * entropy
            
            # Check for NaN/Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                continue  # Skip this batch
            
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
                optimizer.zero_grad()
                continue  # Skip this batch
            
            torch.nn.utils.clip_grad_norm_(prompt_policy.parameters(), 0.5)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        losses.append(avg_loss)
        
        # Update learning rate scheduler
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f} (processed {len(all_pairs)} prompt step pairs, LR: {current_lr:.2e})")
    
    prompt_policy.eval()
    return losses


def main():
    parser = argparse.ArgumentParser(description="DPO Post-training from RL logs")
    parser.add_argument("--config", type=str, default="hierarchical", help="Config to use")
    parser.add_argument("--rl-log", type=str, required=True, help="Path to RL training log JSON")
    parser.add_argument("--rl-model-dir", type=str, required=True, help="Directory with RL-trained models")
    parser.add_argument("--epochs", type=int, default=3, help="DPO training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for structure policy")
    parser.add_argument("--prompt-lr", type=float, default=1e-5, help="Learning rate for prompt policy (default: 1e-5)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--beta", type=float, default=0.05, help="DPO temperature parameter (lower = less aggressive, prevents overfitting)")
    parser.add_argument("--entropy-coef", type=float, default=0.05, help="Entropy regularization coefficient (prevents mode collapse, higher = more diversity)")
    parser.add_argument("--min-reward-correct", type=float, default=4.0, help="Min reward for correct episodes")
    parser.add_argument("--max-reward-incorrect", type=float, default=2.0, help="Max reward for incorrect episodes")
    parser.add_argument("--output-dir", type=str, default="models/dpo_posttrained", help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Load config
    cfg = load_config(args.config)
    
    # Load preference pairs from log
    preference_pairs = load_preference_pairs_from_log(
        args.rl_log, 
        min_reward_correct=args.min_reward_correct,
        max_reward_incorrect=args.max_reward_incorrect
    )
    
    if len(preference_pairs) == 0:
        print("Error: No preference pairs found. Try adjusting reward thresholds.")
        return
    
    # Find RL-trained models
    struct_pattern = os.path.join(args.rl_model_dir, "*structure*_final.pt")
    prompt_pattern = os.path.join(args.rl_model_dir, "*prompt*_final.pt")
    
    struct_files = glob(struct_pattern)
    prompt_files = glob(prompt_pattern)
    
    if not struct_files:
        struct_pattern = os.path.join(args.rl_model_dir, "*structure*.pt")
        struct_files = sorted(glob(struct_pattern), key=os.path.getmtime, reverse=True)
        if struct_files:
            struct_files = [struct_files[0]]
    
    if not prompt_files:
        prompt_pattern = os.path.join(args.rl_model_dir, "*prompt*.pt")
        prompt_files = sorted(glob(prompt_pattern), key=os.path.getmtime, reverse=True)
        if prompt_files:
            prompt_files = [prompt_files[0]]
    
    if not struct_files or not prompt_files:
        raise FileNotFoundError(
            f"RL models not found in {args.rl_model_dir}\n"
            f"  Looking for: {struct_pattern} and {prompt_pattern}"
        )
    
    struct_path = struct_files[0]
    prompt_path = prompt_files[0]
    
    print(f"\nLoading RL-trained models...")
    print(f"  Structure: {struct_path}")
    print(f"  Prompt: {prompt_path}")
    
    # Load checkpoints
    struct_checkpoint = torch.load(struct_path, map_location=device, weights_only=False)
    prompt_checkpoint = torch.load(prompt_path, map_location=device, weights_only=False)
    
    # Initialize policies (current and reference)
    structure_policy = MultiDiscretePolicyGRPO(
        obs_dim=struct_checkpoint["obs_dim"],
        action_dims=struct_checkpoint["action_dims"]
    ).to(device)
    
    structure_policy_ref = MultiDiscretePolicyGRPO(
        obs_dim=struct_checkpoint["obs_dim"],
        action_dims=struct_checkpoint["action_dims"]
    ).to(device)
    
    prompt_policy = PolicyNetworkGRPO(
        obs_dim=prompt_checkpoint["obs_dim"],
        action_dim=prompt_checkpoint["action_dim"]
    ).to(device)
    
    prompt_policy_ref = PolicyNetworkGRPO(
        obs_dim=prompt_checkpoint["obs_dim"],
        action_dim=prompt_checkpoint["action_dim"]
    ).to(device)
    
    # Load RL weights into both current and reference models
    structure_policy.load_state_dict(struct_checkpoint["model_state_dict"])
    structure_policy_ref.load_state_dict(struct_checkpoint["model_state_dict"])
    prompt_policy.load_state_dict(prompt_checkpoint["model_state_dict"])
    prompt_policy_ref.load_state_dict(prompt_checkpoint["model_state_dict"])
    
    # Freeze reference models
    for param in structure_policy_ref.parameters():
        param.requires_grad = False
    for param in prompt_policy_ref.parameters():
        param.requires_grad = False
    
    print("✓ Models loaded (reference models frozen)")
    
    # Create environments for observation generation
    structure_env = StructureEnv(cfg)
    prompt_env = PromptEnv(cfg)
    
    # Train both policies with DPO
    print("\n" + "="*60)
    print("Starting DPO Post-training")
    print("="*60)
    
    struct_losses = train_structure_policy_dpo(
        structure_policy, structure_policy_ref, preference_pairs, structure_env,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, 
        beta=args.beta, entropy_coef=args.entropy_coef, device=device
    )
    prompt_losses = train_prompt_policy_dpo(
        prompt_policy, prompt_policy_ref, preference_pairs, prompt_env,
        epochs=args.epochs, lr=args.prompt_lr, batch_size=args.batch_size,  # Use separate LR
        beta=args.beta, entropy_coef=args.entropy_coef, device=device
    )
    
    # Save models
    os.makedirs(args.output_dir, exist_ok=True)
    struct_save_path = os.path.join(args.output_dir, "structure_policy_dpo.pt")
    prompt_save_path = os.path.join(args.output_dir, "prompt_policy_dpo.pt")
    
    torch.save({
        "model_state_dict": structure_policy.state_dict(),
        "action_dims": struct_checkpoint["action_dims"],
        "obs_dim": struct_checkpoint["obs_dim"],
        "algorithm": "GRPO_DPO",
    }, struct_save_path)
    
    torch.save({
        "model_state_dict": prompt_policy.state_dict(),
        "action_dim": prompt_checkpoint["action_dim"],
        "obs_dim": prompt_checkpoint["obs_dim"],
        "algorithm": "GRPO_DPO",
    }, prompt_save_path)
    
    print(f"\n{'='*60}")
    print("✓ DPO post-training complete!")
    print(f"{'='*60}")
    print(f"\nModels saved to:")
    print(f"  Structure: {struct_save_path}")
    print(f"  Prompt: {prompt_save_path}")
    print(f"\nEvaluate with:")
    print(f"  python scripts/eval_hrl.py --structure-model {struct_save_path} --prompt-model {prompt_save_path}")


if __name__ == "__main__":
    main()

