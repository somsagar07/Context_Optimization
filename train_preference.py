import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import argparse
from tqdm import tqdm
from collections import deque

# Import from your existing codebase
from train_dual import HierarchicalTrainer
from configs import load_config
import os

class HybridPreferencesTrainer(HierarchicalTrainer):
    """
    Hybrid Trainer:
    - Structure Policy (Manager): Trained via GRPO (Group Relative Policy Optimization)
    - Prompt Policy (Worker): Trained via DPO (Direct Preference Optimization)
    """
    def __init__(self, cfg, device="cuda" if torch.cuda.is_available() else "cpu"):
        # Initialize standard components using the base class
        super().__init__(cfg, device)
        
        # --- DPO Requirement: Reference Model ---
        # We need a frozen copy of the Prompt Policy to calculate KL-divergence implicitly
        print("Initializing Reference Model for DPO...")
        self.ref_prompt_policy = copy.deepcopy(self.prompt_policy)
        self.ref_prompt_policy.eval()
        for p in self.ref_prompt_policy.parameters():
            p.requires_grad = False

        # Hyperparameters
        self.dpo_beta = 0.1  # Controls deviation from reference model (standard: 0.1)
        self.grpo_epsilon = 0.2 # Clip ratio for GRPO (standard PPO-like clipping)

    def get_structure_log_prob(self, policy, obs, actions):
        """
        Calculate log_prob for MultiDiscrete structure actions.
        Matches logic needed for GRPO.
        """
        # actions shape: [Batch, 6]
        # output: [Batch] (sum of log_probs across the 6 dimensions)
        action_logits_list, _ = policy(obs)
        total_log_prob = 0
        
        for i, logits in enumerate(action_logits_list):
            # Log Softmax over the dimension
            log_probs = F.log_softmax(logits, dim=-1)
            # Gather log prob of the chosen action index
            # actions[:, i] -> [Batch] -> unsqueeze to [Batch, 1]
            selected = log_probs.gather(1, actions[:, i].unsqueeze(-1)).squeeze(-1)
            total_log_prob += selected
            
        return total_log_prob

    def get_prompt_sequence_log_probs(self, policy, obs_list, act_list):
        """
        Calculate sum of log probs for a sequence of prompt actions.
        Needed for DPO (log_pi(y_w | x) and log_pi(y_l | x)).
        """
        if not obs_list:
            return torch.tensor(0.0, device=self.device)
            
        obs_t = torch.FloatTensor(np.array(obs_list)).to(self.device)
        act_t = torch.LongTensor(np.array(act_list)).to(self.device)
        
        # Forward pass
        logits, _ = policy(obs_t)
        
        # Calculate log_probs
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Select the log_prob of the actions actually taken
        selected_log_probs = log_probs.gather(1, act_t.unsqueeze(-1)).squeeze(-1)
        
        # Sum over the sequence (trajectory)
        return selected_log_probs.sum()

    def get_ground_truth_reward(self, correct, info):
        """
        The 'Implicit' Reward Function.
        Used to rank trajectories for DPO and calculate scores for GRPO.
        """
        # 1. Correctness (Primary Signal)
        r = 5.0 if correct else 0.0
        
        # 2. Costs (Efficiency Penalties)
        r -= info.get("steps_taken", 1) * self.cfg.COST_PER_STEP
        r -= info.get("tools_used", 0) * self.cfg.COST_TOOL_USAGE
        
        # Token Budget Penalty (Normalized)
        # Using a rough max_tokens estimation for normalization
        max_tokens = 1024 + 512 + 256
        tokens = info.get("total_tokens", 0)
        r -= (tokens / max_tokens) * self.cfg.COST_TOKEN_BUDGET
        
        return r

    def run_hybrid_sampling_step(self):
        """
        The 2x2 Sampling Strategy (Data Collection).
        1. Sample 2 Structure Configurations (S1, S2).
        2. For each Structure, Sample 2 Prompt Trajectories (T1, T2).
        """
        # A. Get Context (Question)
        struct_obs, struct_info = self.structure_env.reset()
        
        current_embedding = self.structure_env.question_embedding
        
        # B. Sample K=2 Structures (The Group for GRPO)
        structures_data = []
        for _ in range(2):
            # Sample structure action (stochastic)
            action, log_prob, _ = self.structure_policy.get_action(struct_obs)
            
            # Convert to dict for Env
            struct_dict = {
                "workflow_depth": int(action[0]),
                "reasoner_tools_idx": int(action[1]),
                "reasoner_budget_idx": int(action[2]),
                "verifier_tools_idx": int(action[3]),
                "verifier_budget_idx": int(action[4]),
                "answerer_budget_idx": int(action[5]),
            }
            
            structures_data.append({
                "action": action,      # For training
                "struct_dict": struct_dict,
                "trajectories": []     # Will hold the prompt runs
            })
        
        # C. For each Structure, Run M=2 Prompt Paths (The Pair for DPO)
        for struct in structures_data:
            # Configure PromptEnv with this structure
            self.prompt_env.set_structure(
                question=struct_info["question"],
                answer=struct_info["answer"],
                embedding=current_embedding,
                structure=struct["struct_dict"]
            )
            
            # Run 2 trajectories
            for _ in range(2):
                prompt_obs, _ = self.prompt_env.reset()
                traj_obs, traj_act = [], []
                
                done = False
                while not done:
                    # Sample from Prompt Policy
                    with torch.no_grad():
                        # We don't need gradients during sampling
                        act, _, _ = self.prompt_policy.get_action(prompt_obs)
                    
                    traj_obs.append(prompt_obs.copy())
                    traj_act.append(act)
                    
                    prompt_obs, _, done, _, info = self.prompt_env.step(act)
                
                # Calculate Outcome
                correct = info.get("correct", False)
                reward = self.get_ground_truth_reward(correct, info)
                
                struct["trajectories"].append({
                    "obs": traj_obs,
                    "actions": traj_act,
                    "reward": reward,
                    "correct": correct
                })

        return struct_obs, structures_data

    def update_step(self, struct_obs, structures_data):
        """
        Perform the Hybrid Update:
        1. Prompt Policy -> DPO Update
        2. Structure Policy -> GRPO Update
        """
        
        # ==========================================
        # Part 1: Prompt Policy Update (DPO)
        # ==========================================
        dpo_loss_accum = 0
        valid_pairs = 0
        
        for struct in structures_data:
            # We have 2 trajectories: A and B
            traj_a = struct["trajectories"][0]
            traj_b = struct["trajectories"][1]
            
            # --- DPO Ranking Logic ---
            # 1. Correctness is King
            if traj_a["correct"] and not traj_b["correct"]:
                win, lose = traj_a, traj_b
            elif traj_b["correct"] and not traj_a["correct"]:
                win, lose = traj_b, traj_a
            else:
                # 2. Tie-breaker: Efficiency (Higher Reward = Lower Cost)
                if traj_a["reward"] > traj_b["reward"]:
                    win, lose = traj_a, traj_b
                else:
                    win, lose = traj_b, traj_a
            
            # Only train if there is a meaningful difference? 
            # (Standard DPO trains on all preferences, even small ones)
            
            # Calculate Sequence Log Probs
            # Policy (Parametric)
            pi_win = self.get_prompt_sequence_log_probs(self.prompt_policy, win["obs"], win["actions"])
            pi_lose = self.get_prompt_sequence_log_probs(self.prompt_policy, lose["obs"], lose["actions"])
            
            # Reference (Frozen) - No Grad
            with torch.no_grad():
                ref_win = self.get_prompt_sequence_log_probs(self.ref_prompt_policy, win["obs"], win["actions"])
                ref_lose = self.get_prompt_sequence_log_probs(self.ref_prompt_policy, lose["obs"], lose["actions"])
            
            # --- The DPO Loss Math ---
            # log(pi_w/ref_w) - log(pi_l/ref_l)
            logits = (pi_win - ref_win) - (pi_lose - ref_lose)
            
            # Loss = -log(sigmoid(beta * logits))
            loss = -F.logsigmoid(self.dpo_beta * logits)
            
            dpo_loss_accum += loss
            valid_pairs += 1

        # Optimization Step for Prompt Policy
        if valid_pairs > 0:
            dpo_final_loss = dpo_loss_accum / valid_pairs
            
            self.prompt_optimizer.zero_grad()
            dpo_final_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.prompt_policy.parameters(), 0.5)
            self.prompt_optimizer.step()
        else:
            dpo_final_loss = torch.tensor(0.0)

        # ==========================================
        # Part 2: Structure Policy Update (GRPO)
        # ==========================================
        
        # 1. Calculate Scores for the Group
        # The score of a structure is the average reward of its trajectories
        scores = []
        for struct in structures_data:
            avg_r = np.mean([t["reward"] for t in struct["trajectories"]])
            scores.append(avg_r)
        
        scores_t = torch.FloatTensor(scores).to(self.device)
        
        # 2. Calculate Advantages (Group Relative)
        # A_i = (Score_i - Mean(Group)) / Std(Group)
        mean_score = scores_t.mean()
        std_score = scores_t.std()
        
        # Avoid division by zero if std is very small
        if std_score < 1e-6:
            std_score = 1.0
            
        advantages = (scores_t - mean_score) / std_score
        
        # 3. Calculate Policy Loss
        grpo_loss_accum = 0
        struct_obs_t = torch.FloatTensor(struct_obs).to(self.device).unsqueeze(0) # [1, Dim]
        
        for i, struct in enumerate(structures_data):
            # We must run forward pass again to get gradients for the action
            action_t = torch.LongTensor([struct["action"]]).to(self.device) # [1, 6]
            
            # Get log probability of the taken action
            log_prob = self.get_structure_log_prob(self.structure_policy, struct_obs_t, action_t)
            
            # Standard GRPO Loss: -Advantage * LogProb
            # We treat advantage as a fixed constant (detach)
            grpo_loss_accum += -advantages[i].detach() * log_prob

        # Optimization Step for Structure Policy
        grpo_final_loss = grpo_loss_accum / len(structures_data)
        
        self.struct_optimizer.zero_grad()
        grpo_final_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.structure_policy.parameters(), 0.5)
        self.struct_optimizer.step()
        
        return dpo_final_loss.item(), grpo_final_loss.item(), mean_score.item()

    def train(self, num_episodes=10, log_every=1):
        print(f"\n🚀 Starting Hybrid Training")
        print(f"   Structure Policy -> GRPO (Group Size=2)")
        print(f"   Prompt Policy    -> DPO (Pairs=2)")
        print(f"   ---------------------------------------")
        
        moving_avg_reward = deque(maxlen=50)
        
        # Progress bar
        pbar = tqdm(range(num_episodes))
        
        for ep in pbar:
            # 1. Run Sampling
            struct_obs, structures_data = self.run_hybrid_sampling_step()
            
            # 2. Update Networks
            dpo_loss, grpo_loss, avg_reward = self.update_step(struct_obs, structures_data)
            
            # 3. Logging
            moving_avg_reward.append(avg_reward)
            smooth_reward = sum(moving_avg_reward) / len(moving_avg_reward)
            
            pbar.set_description(f"R:{smooth_reward:.2f} | DPO:{dpo_loss:.3f} | GRPO:{grpo_loss:.3f}")
            
            if (ep + 1) % log_every == 0:
                # Optional: Log to WandB or file here
                pass
                
            # Periodic Reference Model Update
            # (Some DPO papers suggest slowly updating reference, others keep it frozen)
            if (ep + 1) % 500 == 0:
                print("Updating Reference Model...")
                self.ref_prompt_policy.load_state_dict(self.prompt_policy.state_dict())

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train dual-policy hierarchical RL (True HIRO-style)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", type=str, default="hierarchical",
        help="Configuration to use"
    )
    parser.add_argument(
        "--episodes", type=int, default=500,
        help="Number of episodes to train"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for policy updates"
    )
    parser.add_argument(
        "--log-every", type=int, default=25,
        help="Log every N episodes"
    )
    parser.add_argument(
        "--save-every", type=int, default=2000,
        help="Save models every N episodes"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=["gsm8k", "hotpotqa"],
        help="Override dataset"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load Config
    cfg = load_config(args.config)
    
    # Apply dataset override
    if args.dataset:
        cfg.DATASET_NAME = args.dataset
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Initialize Hybrid Trainer
    trainer = HybridPreferencesTrainer(cfg)
    
    # Run Training
    trainer.train(num_episodes=10, log_every=1)

if __name__ == "__main__":
    main()