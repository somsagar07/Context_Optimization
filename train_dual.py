"""
Training Script for Dual-Policy Hierarchical RL (True HIRO-style with PPO)

TRUE HIERARCHICAL EPISODE:
1. Structure policy picks [workflow, tools, budgets]
2. Prompt policy picks prompts sequentially  
3. Execute LLM
4. Get reward
5. BOTH policies update from the same episode using PPO

This implementation uses PPO (Proximal Policy Optimization) for both policies:
- Clipped objective to prevent large policy updates
- Importance sampling ratio
- Multiple epochs per batch for sample efficiency
- Value function baseline

Usage:
    python train_dual.py                    # Train both policies together
    python train_dual.py --episodes 500     # Train for 500 episodes
    python train_dual.py --ppo-epsilon 0.2 --ppo-epochs 4  # Custom PPO params
"""
import argparse
import os
import time
import json
import numpy as np
from collections import deque
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal

from configs import load_config
from env.structure_env import StructureEnv
from env.prompt_env import PromptEnv


class PolicyNetwork(nn.Module):
    """Simple policy network for discrete actions."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        features = self.network(x)
        action_logits = self.action_head(features)
        value = self.value_head(features)
        return action_logits, value
    
    def get_action(self, obs, deterministic=False):
        """Get action from observation."""
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        # Move to same device as model
        device = next(self.parameters()).device
        obs = obs.to(device)
            
        action_logits, value = self.forward(obs)
        probs = torch.softmax(action_logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = Categorical(probs)
            action = dist.sample()
        
        log_prob = torch.log(probs.gather(1, action.unsqueeze(-1)) + 1e-8)
        
        return action.item(), log_prob.item(), value.item()
    
    def get_log_prob_and_value(self, obs, action):
        """Compute log probability and value for given obs/action (for gradients)."""
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if not isinstance(action, torch.Tensor):
            action = torch.LongTensor([action])
        if action.dim() == 0:
            action = action.unsqueeze(0)
        
        device = next(self.parameters()).device
        obs = obs.to(device)
        action = action.to(device)
        
        action_logits, value = self.forward(obs)
        probs = torch.softmax(action_logits, dim=-1)
        log_prob = torch.log(probs.gather(1, action.unsqueeze(-1)) + 1e-8)
        
        return log_prob.squeeze(), value.squeeze()


class MultiDiscretePolicy(nn.Module):
    """Policy network for MultiDiscrete action space [3, 8, 3, 8, 3, 3]."""
    
    def __init__(self, obs_dim: int, action_dims: list, hidden_dim: int = 256):
        super().__init__()
        self.action_dims = action_dims  # [3, 8, 3, 8, 3, 3]
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Separate head for each action dimension
        self.action_heads = nn.ModuleList([
            nn.Linear(hidden_dim, dim) for dim in action_dims
        ])
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        features = self.network(x)
        action_logits = [head(features) for head in self.action_heads]
        value = self.value_head(features)
        return action_logits, value
    
    def get_action(self, obs, deterministic=False):
        """Get MultiDiscrete action from observation."""
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        # Move to same device as model
        device = next(self.parameters()).device
        obs = obs.to(device)
            
        action_logits_list, value = self.forward(obs)
        
        actions = []
        log_probs = []
        
        for logits in action_logits_list:
            probs = torch.softmax(logits, dim=-1)
            
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                dist = Categorical(probs)
                action = dist.sample()
            
            log_prob = torch.log(probs.gather(1, action.unsqueeze(-1)) + 1e-8)
            actions.append(action.item())
            log_probs.append(log_prob.item())
        
        return np.array(actions), sum(log_probs), value.item()
    
    def get_log_prob_and_value(self, obs, action):
        """Compute log probability and value for given obs/action (for gradients)."""
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if not isinstance(action, torch.Tensor):
            action = torch.LongTensor(action)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        
        device = next(self.parameters()).device
        obs = obs.to(device)
        action = action.to(device)
        
        action_logits_list, value = self.forward(obs)
        
        log_probs = []
        for i, logits in enumerate(action_logits_list):
            probs = torch.softmax(logits, dim=-1)
            act = action[:, i]
            log_prob = torch.log(probs.gather(1, act.unsqueeze(-1)) + 1e-8)
            log_probs.append(log_prob)
        
        total_log_prob = sum(log_probs)
        return total_log_prob.squeeze(), value.squeeze()


class HierarchicalTrainer:
    """
    Custom trainer for true hierarchical RL.
    
    Each episode:
    1. Structure policy picks configuration
    2. Prompt policy picks prompts (multiple steps)
    3. Execute workflow
    4. Both policies get the same final reward
    5. Both policies update with policy gradient
    """
    
    def __init__(self, cfg, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.cfg = cfg
        self.device = device
        
        # Create environments
        self.structure_env = StructureEnv(cfg)
        self.prompt_env = PromptEnv(cfg)
        
        # Get observation dimensions
        struct_obs_dim = self.structure_env.observation_space.shape[0]
        prompt_obs_dim = self.prompt_env.observation_space.shape[0]
        
        # Get action dimensions
        struct_action_dims = list(self.structure_env.action_space.nvec)  # [3, 8, 3, 8, 3, 3]
        prompt_action_dim = self.prompt_env.action_space.n  # 7
        
        print(f"Structure obs dim: {struct_obs_dim}, action dims: {struct_action_dims}")
        print(f"Prompt obs dim: {prompt_obs_dim}, action dim: {prompt_action_dim}")
        
        # Create policy networks
        print("\nInitializing Structure Policy...")
        self.structure_policy = MultiDiscretePolicy(
            struct_obs_dim, struct_action_dims, hidden_dim=256
        ).to(device)
        
        print("Initializing Prompt Policy...")
        self.prompt_policy = PolicyNetwork(
            prompt_obs_dim, prompt_action_dim, hidden_dim=256
        ).to(device)
        
        # Optimizers
        self.struct_optimizer = optim.Adam(
            self.structure_policy.parameters(), 
            lr=cfg.STRUCTURE_LEARNING_RATE
        )
        self.prompt_optimizer = optim.Adam(
            self.prompt_policy.parameters(), 
            lr=cfg.PROMPT_LEARNING_RATE
        )
        
        # Metrics
        self.episode_count = 0
        self.correct_count = 0
        self.total_reward = 0.0
        self.rewards_history = deque(maxlen=100)
        
        # Logging
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = None
        self.episode_logs = []  # Store all episode data
        
    def run_episode(self, deterministic=False):
        """
        Run a single hierarchical episode.
        
        Returns:
            dict with episode info and experiences for both policies
        """
        # Reset and get question
        struct_obs, struct_info = self.structure_env.reset()
        question = struct_info["question"]
        answer = struct_info["answer"]
        
        # ========================================
        # STEP 1: Structure Policy Decision
        # ========================================
        # Get action and store old log prob for PPO importance sampling
        struct_action, struct_log_prob_old, struct_value_old = self.structure_policy.get_action(
            struct_obs, deterministic=deterministic
        )
        
        # Parse structure action
        workflow_depth = int(struct_action[0])
        reasoner_tools_idx = int(struct_action[1])
        reasoner_budget_idx = int(struct_action[2])
        verifier_tools_idx = int(struct_action[3])
        verifier_budget_idx = int(struct_action[4])
        answerer_budget_idx = int(struct_action[5])
        
        # ========================================
        # STEP 2: Prompt Policy Decisions (Sequential)
        # ========================================
        # Set up prompt environment with the structure
        self.prompt_env.current_q = question
        self.prompt_env.current_a = answer
        self.prompt_env.question_embedding = self.structure_env.question_embedding.copy()
        
        self.prompt_env.workflow_depth = workflow_depth
        self.prompt_env.reasoner_tools_idx = reasoner_tools_idx
        self.prompt_env.reasoner_budget_idx = reasoner_budget_idx
        self.prompt_env.verifier_tools_idx = verifier_tools_idx
        self.prompt_env.verifier_budget_idx = verifier_budget_idx
        self.prompt_env.answerer_budget_idx = answerer_budget_idx
        
        # Initialize prompt selection state
        if workflow_depth == 0:
            self.prompt_env.prompt_stage = self.prompt_env.PROMPT_STAGE_ANSWERER
        else:
            self.prompt_env.prompt_stage = self.prompt_env.PROMPT_STAGE_REASONER
        
        self.prompt_env.prompt_step = 0
        self.prompt_env.selected_prompts = {"reasoner": [], "verifier": [], "answerer": []}
        self.prompt_env._structure_set = True
        
        # Collect prompt experiences (store obs/actions, old log probs, and old values for PPO)
        prompt_obs_list = []
        prompt_actions = []
        prompt_log_probs_old = []
        prompt_values_old = []
        prompt_obs = self.prompt_env._get_observation()
        
        done = False
        while not done:
            # Prompt policy decision - store old log prob and value for PPO
            prompt_action, prompt_log_prob_old, prompt_value_old = self.prompt_policy.get_action(
                prompt_obs, deterministic=deterministic
            )
            
            # Store obs, action, old log prob, and old value for PPO importance sampling
            prompt_obs_list.append(prompt_obs.copy())
            prompt_actions.append(prompt_action)
            prompt_log_probs_old.append(prompt_log_prob_old)
            prompt_values_old.append(prompt_value_old)
            
            # Take step
            next_obs, step_reward, done, _, info = self.prompt_env.step(prompt_action)
            prompt_obs = next_obs
        
        # ========================================
        # STEP 3: Compute Final Reward
        # ========================================
        correct = info.get("correct", False)
        correctness = 1.0 if correct else 0.0
        
        final_reward = correctness * 5.0
        final_reward -= info.get("steps_taken", 1) * self.cfg.COST_PER_STEP
        final_reward -= info.get("tools_used", 0) * self.cfg.COST_TOOL_USAGE
        max_tokens = 1024 + 512 + 256
        token_penalty = (info.get("total_tokens", 256) / max_tokens) * self.cfg.COST_TOKEN_BUDGET
        final_reward -= token_penalty
        
        # Update metrics
        self.episode_count += 1
        self.correct_count += 1 if correct else 0
        self.total_reward += final_reward
        self.rewards_history.append(final_reward)
        
        # Log episode data
        episode_log = {
            "episode": self.episode_count,
            "correct": correct,
            "reward": final_reward,
            "workflow": ["Direct", "Reason+Ans", "Reason+Verify+Ans"][workflow_depth],
            "num_prompt_steps": len(prompt_actions),
            "steps_taken": info.get("steps_taken", 0),
            "tools_used": info.get("tools_used", 0),
            "total_tokens": info.get("total_tokens", 0),
            "reasoner_tools": info.get("reasoner_tools", []),
            "verifier_tools": info.get("verifier_tools", []),
            "reasoner_prompts": info.get("reasoner_prompts", []),
            "verifier_prompts": info.get("verifier_prompts", []),
            "answerer_prompts": info.get("answerer_prompts", []),
            "reasoner_budget": info.get("reasoner_budget", "N/A"),
            "verifier_budget": info.get("verifier_budget", "N/A"),
            "answerer_budget": info.get("answerer_budget", "N/A"),
            "question": info.get("question", ""),
        }
        self.episode_logs.append(episode_log)
        
        return {
            "struct_obs": struct_obs,
            "struct_action": struct_action,
            "struct_log_prob_old": struct_log_prob_old,
            "struct_value_old": struct_value_old,
            "prompt_obs_list": prompt_obs_list,
            "prompt_actions": prompt_actions,
            "prompt_log_probs_old": prompt_log_probs_old,
            "prompt_values_old": prompt_values_old,
            "reward": final_reward,
            "correct": correct,
            "workflow": ["Direct", "Reason+Ans", "Reason+Verify+Ans"][workflow_depth],
            "num_prompt_steps": len(prompt_actions),
            "info": info,
        }
    
    def update_policies(self, episodes: list, gamma: float = 0.95, 
                       ppo_epsilon: float = 0.2, ppo_epochs: int = 4,
                       struct_entropy_coef: float = 0.05, prompt_entropy_coef: float = 0.05,
                       gae_lambda: float = 0.95):
        """
        Update both policies using PPO (Proximal Policy Optimization).
        
        PPO features:
        - Clipped objective to prevent large policy updates
        - Importance sampling ratio
        - Multiple epochs on same batch
        - Value function baseline
        - GAE (Generalized Advantage Estimation) for better credit assignment
        
        Args:
            gamma: Discount factor for future rewards (0.95 is standard)
            ppo_epsilon: PPO clipping parameter (0.2 is standard)
            ppo_epochs: Number of PPO update epochs per batch (4 is standard)
            struct_entropy_coef: Entropy bonus for structure policy (higher = more exploration)
            prompt_entropy_coef: Entropy bonus for prompt policy (higher = more exploration)
            gae_lambda: GAE lambda parameter (0.95 = slight bias reduction, 1.0 = MC returns)
        """
        # PPO hyperparameters (optimized for stability)
        value_coef = 0.5  # Standard value function coefficient
        max_grad_norm = 0.5  # Gradient clipping for stability
        
        # Prepare structure policy data
        struct_obs_list = []
        struct_actions_list = []
        struct_log_probs_old_list = []
        struct_values_old_list = []
        struct_returns_list = []
        
        for ep in episodes:
            struct_obs_list.append(ep["struct_obs"])
            struct_actions_list.append(ep["struct_action"])
            struct_log_probs_old_list.append(ep["struct_log_prob_old"])
            struct_values_old_list.append(ep["struct_value_old"])
            struct_returns_list.append(ep["reward"])
        
        # Convert to tensors (old log probs and values don't need gradients)
        struct_obs_tensor = torch.FloatTensor(np.array(struct_obs_list)).to(self.device)
        struct_actions_tensor = torch.LongTensor(np.array(struct_actions_list)).to(self.device)
        struct_log_probs_old = torch.FloatTensor(struct_log_probs_old_list).to(self.device).detach()
        struct_values_old = torch.FloatTensor(struct_values_old_list).to(self.device).detach()
        struct_returns = torch.FloatTensor(struct_returns_list).to(self.device).detach()
        
        # Compute advantages (using old values as baseline)
        # Detach to ensure no gradients flow through advantages during PPO updates
        struct_advantages = (struct_returns - struct_values_old).detach()
        if len(struct_advantages) > 1:
            struct_advantages = (struct_advantages - struct_advantages.mean()) / (struct_advantages.std() + 1e-8)
        
        # PPO update for structure policy (multiple epochs)
        for epoch in range(ppo_epochs):
            # Recompute log probs and values with current policy
            struct_log_probs_new = []
            struct_values_new = []
            
            for i in range(len(struct_obs_list)):
                log_prob, value = self.structure_policy.get_log_prob_and_value(
                    struct_obs_list[i], struct_actions_list[i]
                )
                struct_log_probs_new.append(log_prob)
                struct_values_new.append(value)
            
            struct_log_probs_new = torch.stack(struct_log_probs_new)
            struct_values_new = torch.stack(struct_values_new)
            
            # Importance sampling ratio
            ratio = torch.exp(struct_log_probs_new - struct_log_probs_old)
            
            # PPO clipped objective
            surr1 = ratio * struct_advantages
            surr2 = torch.clamp(ratio, 1.0 - ppo_epsilon, 1.0 + ppo_epsilon) * struct_advantages
            struct_policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            struct_value_loss = nn.MSELoss()(struct_values_new, struct_returns)
            
            # Entropy bonus (for exploration)
            # For MultiDiscrete, compute entropy per dimension and sum
            struct_entropy = 0.0
            with torch.no_grad():
                action_logits_list, _ = self.structure_policy.forward(struct_obs_tensor)
                for logits in action_logits_list:
                    probs = torch.softmax(logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                    struct_entropy += entropy
            
            struct_loss = struct_policy_loss + value_coef * struct_value_loss - struct_entropy_coef * struct_entropy
            
            self.struct_optimizer.zero_grad()
            struct_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.structure_policy.parameters(), max_grad_norm)
            self.struct_optimizer.step()
        
        # Prepare prompt policy data
        prompt_obs_all = []
        prompt_actions_all = []
        prompt_log_probs_old_all = []
        prompt_values_old_all = []
        prompt_returns_all = []
        
        for ep in episodes:
            reward = ep["reward"]
            prompt_obs_list = ep["prompt_obs_list"]
            prompt_actions = ep["prompt_actions"]
            prompt_log_probs_old = ep["prompt_log_probs_old"]
            prompt_values_old = ep["prompt_values_old"]
            num_steps = len(prompt_obs_list)
            
            for i in range(num_steps):
                steps_to_end = num_steps - i - 1
                discounted_return = reward * (gamma ** steps_to_end)
                
                prompt_obs_all.append(prompt_obs_list[i])
                prompt_actions_all.append(prompt_actions[i])
                prompt_log_probs_old_all.append(prompt_log_probs_old[i])
                prompt_values_old_all.append(prompt_values_old[i])
                prompt_returns_all.append(discounted_return)
        
        if prompt_obs_all:
            # Convert to tensors (old log probs and values don't need gradients)
            prompt_obs_tensor = torch.FloatTensor(np.array(prompt_obs_all)).to(self.device)
            prompt_actions_tensor = torch.LongTensor(prompt_actions_all).to(self.device)
            prompt_log_probs_old = torch.FloatTensor(prompt_log_probs_old_all).to(self.device).detach()
            prompt_values_old = torch.FloatTensor(prompt_values_old_all).to(self.device).detach()
            prompt_returns = torch.FloatTensor(prompt_returns_all).to(self.device).detach()
            
            # Compute advantages using OLD values (consistent with structure policy)
            # Detach to ensure no gradients flow through advantages during PPO updates
            prompt_advantages = (prompt_returns - prompt_values_old).detach()
            if len(prompt_advantages) > 1:
                prompt_advantages = (prompt_advantages - prompt_advantages.mean()) / (prompt_advantages.std() + 1e-8)
            
            # PPO update for prompt policy (multiple epochs)
            for epoch in range(ppo_epochs):
                # Recompute log probs and values with current policy
                prompt_log_probs_new = []
                prompt_values_new = []
                
                for i in range(len(prompt_obs_all)):
                    log_prob, value = self.prompt_policy.get_log_prob_and_value(
                        prompt_obs_all[i], prompt_actions_all[i]
                    )
                    prompt_log_probs_new.append(log_prob)
                    prompt_values_new.append(value)
                
                prompt_log_probs_new = torch.stack(prompt_log_probs_new)
                prompt_values_new = torch.stack(prompt_values_new)
                
                # Importance sampling ratio
                ratio = torch.exp(prompt_log_probs_new - prompt_log_probs_old)
                
                # PPO clipped objective
                surr1 = ratio * prompt_advantages
                surr2 = torch.clamp(ratio, 1.0 - ppo_epsilon, 1.0 + ppo_epsilon) * prompt_advantages
                prompt_policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                prompt_value_loss = nn.MSELoss()(prompt_values_new, prompt_returns)
                
                # Entropy bonus
                with torch.no_grad():
                    action_logits, _ = self.prompt_policy.forward(prompt_obs_tensor)
                    probs = torch.softmax(action_logits, dim=-1)
                    prompt_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                
                prompt_loss = prompt_policy_loss + value_coef * prompt_value_loss - prompt_entropy_coef * prompt_entropy
                
                self.prompt_optimizer.zero_grad()
                prompt_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.prompt_policy.parameters(), max_grad_norm)
                self.prompt_optimizer.step()
            
            return struct_loss.item(), prompt_loss.item()
        else:
            return struct_loss.item(), 0.0
    
    def train(self, num_episodes: int = 1000, batch_size: int = 16, 
              log_every: int = 50, save_every: int = 200,
              ppo_epsilon: float = 0.2, ppo_epochs: int = 4,
              save_log_every: int = 500,
              struct_entropy_coef: float = 0.05, prompt_entropy_coef: float = 0.05,
              gae_lambda: float = 0.95):
        """
        Train both policies using PPO (Proximal Policy Optimization).
        
        Args:
            num_episodes: Total episodes to train
            batch_size: Batch size before update (32-64 recommended)
            log_every: Print stats every N episodes
            save_every: Save models every N episodes
            ppo_epsilon: PPO clipping parameter (0.2 is standard, 0.1-0.3 range)
            ppo_epochs: Number of PPO update epochs per batch (4 is standard, 3-10 range)
            save_log_every: Save log file every N episodes
            struct_entropy_coef: Entropy bonus for structure policy (higher = more exploration)
            prompt_entropy_coef: Entropy bonus for prompt policy (higher = more exploration)
            gae_lambda: GAE lambda parameter (0.95 = slight bias reduction, 1.0 = MC returns)
        """
        # Initialize log file
        timestamp = int(time.time())
        self.log_path = os.path.join(self.log_dir, f"training_log_dual_{self.cfg.DATASET_NAME}_{timestamp}.json")
        print(f"Training log will be saved to: {self.log_path}")
        
        print("\n" + "=" * 70)
        print("TRUE HIERARCHICAL TRAINING (PPO)")
        print("=" * 70)
        print(f"  Episodes:         {num_episodes}")
        print(f"  Batch Size:       {batch_size}")
        print(f"  PPO Epsilon:      {ppo_epsilon} (clipping range)")
        print(f"  PPO Epochs:       {ppo_epochs} (updates per batch)")
        print(f"  Structure Entropy: {struct_entropy_coef}")
        print(f"  Prompt Entropy:    {prompt_entropy_coef}")
        print(f"  Gamma:            {self.cfg.PROMPT_GAMMA}")
        print(f"  GAE Lambda:       {gae_lambda}")
        print(f"  Value Coef:       0.5")
        print(f"  Grad Clip:        0.5")
        print(f"  Structure LR:     {self.cfg.STRUCTURE_LEARNING_RATE}")
        print(f"  Prompt LR:        {self.cfg.PROMPT_LEARNING_RATE}")
        print(f"  Dataset:          {self.cfg.DATASET_NAME}")
        print(f"  Device:           {self.device}")
        print("=" * 70)
        print("\nEach episode:")
        print("  1. Structure policy picks [workflow, tools, budgets]")
        print("  2. Prompt policy picks prompts sequentially")
        print("  3. Execute LLM")
        print("  4. Both policies update from same reward (PPO)")
        print("=" * 70 + "\n")
        
        start_time = time.time()
        batch = []
        
        # Create progress bar
        pbar = tqdm(total=num_episodes, desc="Training", unit="ep")
        
        for ep in range(num_episodes):
            # Run episode
            result = self.run_episode()
            batch.append(result)
            
            # Update when batch is full (PPO with multiple epochs)
            if len(batch) >= batch_size:
                struct_loss, prompt_loss = self.update_policies(
                    batch, 
                    gamma=self.cfg.PROMPT_GAMMA,
                    ppo_epsilon=ppo_epsilon,
                    ppo_epochs=ppo_epochs,
                    struct_entropy_coef=struct_entropy_coef,
                    prompt_entropy_coef=prompt_entropy_coef,
                    gae_lambda=gae_lambda
                )
                batch = []
            
            # Update progress bar
            acc = self.correct_count / self.episode_count * 100 if self.episode_count > 0 else 0
            avg_reward = self.total_reward / self.episode_count if self.episode_count > 0 else 0
            recent_reward = np.mean(self.rewards_history) if self.rewards_history else 0
            
            # Calculate ETA
            elapsed = time.time() - start_time
            if ep > 0:
                time_per_ep = elapsed / (ep + 1)
                eta_seconds = time_per_ep * (num_episodes - ep - 1)
                eta_str = f"{eta_seconds/3600:.1f}h" if eta_seconds > 3600 else f"{eta_seconds/60:.1f}m"
            else:
                eta_str = "?"
            
            # Update progress bar
            pbar.set_postfix({
                'Acc': f'{acc:.1f}%',
                'AvgRew': f'{avg_reward:.2f}',
                'Recent': f'{recent_reward:.2f}',
                'ETA': eta_str
            })
            pbar.update(1)
            
            # Detailed log every log_every episodes
            if (ep + 1) % log_every == 0:
                print(f"\n[Episode {ep+1:5d}/{num_episodes}] "
                      f"Accuracy: {acc:5.1f}% | "
                      f"Avg Reward: {avg_reward:6.3f} | "
                      f"Recent (100): {recent_reward:6.3f} | "
                      f"Elapsed: {elapsed/3600:.2f}h | "
                      f"ETA: {eta_str}")
            
            # Save checkpoints
            if (ep + 1) % save_every == 0:
                self.save_models(f"_ep{ep+1}")
            
            # Save log periodically
            if (ep + 1) % save_log_every == 0:
                self._save_log()
        
        # Update with remaining batch
        if batch:
            self.update_policies(
                batch, 
                gamma=self.cfg.PROMPT_GAMMA,
                ppo_epsilon=ppo_epsilon,
                ppo_epochs=ppo_epochs,
                struct_entropy_coef=struct_entropy_coef,
                prompt_entropy_coef=prompt_entropy_coef,
                gae_lambda=gae_lambda
            )
        
        # Close progress bar
        pbar.close()
        
        # Final stats
        elapsed = time.time() - start_time
        final_acc = self.correct_count / self.episode_count * 100
        avg_reward = self.total_reward / self.episode_count
        
        # Save final log
        self._save_log()
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"  Episodes:     {self.episode_count}")
        print(f"  Final Acc:    {final_acc:.1f}%")
        print(f"  Avg Reward:   {avg_reward:.3f}")
        print(f"  Time:         {elapsed:.0f}s ({elapsed/num_episodes:.2f}s/ep)")
        print(f"  Log saved to: {self.log_path}")
        print("=" * 70)
        
        return final_acc, avg_reward
    
    def _save_log(self):
        """Save training log to JSON file."""
        if self.log_path and self.episode_logs:
            rewards = [e["reward"] for e in self.episode_logs]
            correct_episodes = [e for e in self.episode_logs if e["correct"]]
            
            summary = {
                "total_episodes": len(self.episode_logs),
                "total_correct": len(correct_episodes),
                "accuracy": len(correct_episodes) / len(self.episode_logs) if self.episode_logs else 0.0,
                "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
                "min_reward": float(np.min(rewards)) if rewards else 0.0,
                "max_reward": float(np.max(rewards)) if rewards else 0.0,
                "std_reward": float(np.std(rewards)) if rewards else 0.0,
                "avg_steps_taken": float(np.mean([e["steps_taken"] for e in self.episode_logs])) if self.episode_logs else 0.0,
                "avg_tools_used": float(np.mean([e["tools_used"] for e in self.episode_logs])) if self.episode_logs else 0.0,
                "avg_tokens": float(np.mean([e["total_tokens"] for e in self.episode_logs])) if self.episode_logs else 0.0,
                "avg_prompt_steps": float(np.mean([e["num_prompt_steps"] for e in self.episode_logs])) if self.episode_logs else 0.0,
                "workflow_distribution": {
                    workflow: sum(1 for e in self.episode_logs if e["workflow"] == workflow)
                    for workflow in ["Direct", "Reason+Ans", "Reason+Verify+Ans"]
                },
                "episodes": self.episode_logs
            }
            
            with open(self.log_path, "w") as f:
                json.dump(summary, f, indent=2)
    
    def save_models(self, suffix: str = ""):
        """Save both policy models."""
        timestamp = int(time.time())
        dataset = self.cfg.DATASET_NAME
        
        struct_path = f"models/structure_policy_{dataset}_{timestamp}{suffix}.pt"
        prompt_path = f"models/prompt_policy_{dataset}_{timestamp}{suffix}.pt"
        
        torch.save({
            "model_state_dict": self.structure_policy.state_dict(),
            "optimizer_state_dict": self.struct_optimizer.state_dict(),
            "action_dims": self.structure_env.action_space.nvec.tolist(),
            "obs_dim": self.structure_env.observation_space.shape[0],
        }, struct_path)
        
        torch.save({
            "model_state_dict": self.prompt_policy.state_dict(),
            "optimizer_state_dict": self.prompt_optimizer.state_dict(),
            "action_dim": self.prompt_env.action_space.n,
            "obs_dim": self.prompt_env.observation_space.shape[0],
        }, prompt_path)
        
        print(f"\nModels saved:")
        print(f"  Structure: {struct_path}")
        print(f"  Prompt:    {prompt_path}")
        
        return struct_path, prompt_path


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
    parser.add_argument(
        "--ppo-epsilon", type=float, default=0.2,
        help="PPO clipping parameter (default: 0.2)"
    )
    parser.add_argument(
        "--ppo-epochs", type=int, default=4,
        help="Number of PPO update epochs per batch (default: 4)"
    )
    parser.add_argument(
        "--entropy-coef", type=float, default=0.05,
        help="Entropy coefficient for exploration (default: 0.05, higher=more exploration)"
    )
    parser.add_argument(
        "--struct-entropy-coef", type=float, default=None,
        help="Separate entropy coef for structure policy (default: same as --entropy-coef)"
    )
    parser.add_argument(
        "--prompt-entropy-coef", type=float, default=None,
        help="Separate entropy coef for prompt policy (default: same as --entropy-coef)"
    )
    parser.add_argument(
        "--gae-lambda", type=float, default=0.95,
        help="GAE lambda parameter (0.95=standard, 1.0=MC returns, 0.0=TD(0))"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    cfg = load_config(args.config)
    print(f"Loaded config: {args.config}")
    
    # Apply dataset override
    if args.dataset:
        cfg.DATASET_NAME = args.dataset
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Create trainer
    trainer = HierarchicalTrainer(cfg)
    
    # Determine entropy coefficients
    struct_ent = args.struct_entropy_coef if args.struct_entropy_coef is not None else args.entropy_coef
    prompt_ent = args.prompt_entropy_coef if args.prompt_entropy_coef is not None else args.entropy_coef
    
    print(f"  Entropy Coefficients:")
    print(f"    Structure: {struct_ent}")
    print(f"    Prompt:    {prompt_ent}")
    
    # Train
    trainer.train(
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        log_every=args.log_every,
        save_every=args.save_every,
        ppo_epsilon=args.ppo_epsilon,
        ppo_epochs=args.ppo_epochs,
        struct_entropy_coef=struct_ent,
        prompt_entropy_coef=prompt_ent,
        gae_lambda=args.gae_lambda
    )
    
    # Save final models
    struct_path, prompt_path = trainer.save_models("_final")
    
    print("\nTo evaluate, run:")
    print(f"  python eval_dual.py --structure-model {struct_path} --prompt-model {prompt_path}")


if __name__ == "__main__":
    main()
