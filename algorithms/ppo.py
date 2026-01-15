"""
PPO (Proximal Policy Optimization) for Hierarchical RL.

Features:
- Value function baseline for advantage estimation
- Clipped surrogate objective
- Multiple epochs per batch
"""
import numpy as np
import torch
import torch.nn as nn

from algorithms.base import (
    Algorithm, BaseTrainer,
    MultiDiscretePolicyPPO, PolicyNetworkPPO
)


class PPOTrainer(BaseTrainer):
    """PPO trainer with value function."""
    
    algorithm = Algorithm.PPO
    
    def __init__(self, cfg, device="cuda" if torch.cuda.is_available() else "cpu", use_action_masking=False, use_api=False, api_model=None, hf_model=None):
        super().__init__(cfg, device, use_action_masking=use_action_masking, use_api=use_api, api_model=api_model, hf_model=hf_model)
        
        print(f"Initializing PPO (with value heads)...")
        print(f"  Structure: obs={self.struct_obs_dim}, actions={self.struct_action_dims}")
        print(f"  Prompt: obs={self.prompt_obs_dim}, actions={self.prompt_action_dim}")
        if self.use_action_masking:
            print(f"  ✓ Action masking ENABLED (two-stage selection: workflow first, then mask)")
        else:
            print(f"  Action masking DISABLED (standard selection)")
        
        self.structure_policy = MultiDiscretePolicyPPO(
            self.struct_obs_dim, self.struct_action_dims
        ).to(device)
        
        self.prompt_policy = PolicyNetworkPPO(
            self.prompt_obs_dim, self.prompt_action_dim
        ).to(device)
        
        self._init_optimizers()
    
    def update_policies(self, episodes: list, **kwargs):
        """PPO policy update with value function baseline."""
        gamma = kwargs.get("gamma", self.cfg.PROMPT_GAMMA)
        clip_epsilon = kwargs.get("clip_epsilon", 0.2)
        epochs = kwargs.get("epochs", 4)
        struct_ent_coef = kwargs.get("struct_entropy_coef", 0.05)
        prompt_ent_coef = kwargs.get("prompt_entropy_coef", 0.05)
        value_coef = 0.5
        max_grad_norm = 0.5
        
        # ========== STRUCTURE POLICY ==========
        struct_obs = [ep["struct_obs"] for ep in episodes]
        struct_actions = [ep["struct_action"] for ep in episodes]
        struct_log_probs_old = torch.FloatTensor([ep["struct_log_prob"] for ep in episodes]).to(self.device).detach()
        struct_values_old = torch.FloatTensor([ep["struct_value"] for ep in episodes]).to(self.device).detach()
        struct_returns = torch.FloatTensor([ep["reward"] for ep in episodes]).to(self.device).detach()
        
        struct_obs_tensor = torch.FloatTensor(np.array(struct_obs)).to(self.device)
        
        # Compute advantages
        struct_advantages = (struct_returns - struct_values_old).detach()
        if len(struct_advantages) > 1:
            struct_advantages = (struct_advantages - struct_advantages.mean()) / (struct_advantages.std() + 1e-8)
        
        # Get action masks from episodes
        struct_action_masks = [ep.get("struct_action_mask", None) for ep in episodes]
        
        for _ in range(epochs):
            log_probs_new, values_new = [], []
            for i in range(len(struct_obs)):
                lp, v = self.structure_policy.get_log_prob_and_value(
                    struct_obs[i], struct_actions[i], 
                    action_mask=struct_action_masks[i],
                    use_two_stage_masking=self.use_action_masking,
                    structure_env=self.structure_env if self.use_action_masking else None
                )
                log_probs_new.append(lp)
                values_new.append(v)
            
            log_probs_new = torch.stack(log_probs_new)
            values_new = torch.stack(values_new)
            
            ratio = torch.exp(log_probs_new - struct_log_probs_old)
            surr1 = ratio * struct_advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * struct_advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values_new, struct_returns)
            # Compute entropy with per-sample masks to match actual action constraints
            if self.use_action_masking:
                entropy_list = []
                for i in range(len(struct_obs)):
                    workflow_idx = int(struct_actions[i][0])
                    mask = self.structure_env._get_action_mask(workflow_depth=workflow_idx)
                    entropy_list.append(
                        self.structure_policy.get_entropy(struct_obs[i], action_mask=mask).mean()
                    )
                entropy = torch.stack(entropy_list).mean()
            else:
                struct_mask = struct_action_masks[0] if struct_action_masks[0] is not None else None
                entropy = self.structure_policy.get_entropy(struct_obs_tensor, action_mask=struct_mask).mean()
            
            loss = policy_loss + value_coef * value_loss - struct_ent_coef * entropy
            
            self.struct_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.structure_policy.parameters(), max_grad_norm)
            self.struct_optimizer.step()
        
        # ========== PROMPT POLICY ==========
        prompt_obs_all, prompt_actions_all = [], []
        prompt_log_probs_old_all, prompt_values_old_all, prompt_returns_all = [], [], []
        
        for ep in episodes:
            num_steps = len(ep["prompt_obs_list"])
            for i in range(num_steps):
                prompt_obs_all.append(ep["prompt_obs_list"][i])
                prompt_actions_all.append(ep["prompt_actions"][i])
                prompt_log_probs_old_all.append(ep["prompt_log_probs"][i])
                prompt_values_old_all.append(ep["prompt_values"][i])
                prompt_returns_all.append(ep["reward"] * (gamma ** (num_steps - i - 1)))
        
        if not prompt_obs_all:
            return
        
        prompt_obs_tensor = torch.FloatTensor(np.array(prompt_obs_all)).to(self.device)
        prompt_log_probs_old = torch.FloatTensor(prompt_log_probs_old_all).to(self.device).detach()
        prompt_values_old = torch.FloatTensor(prompt_values_old_all).to(self.device).detach()
        prompt_returns = torch.FloatTensor(prompt_returns_all).to(self.device).detach()
        
        prompt_advantages = (prompt_returns - prompt_values_old).detach()
        if len(prompt_advantages) > 1:
            prompt_advantages = (prompt_advantages - prompt_advantages.mean()) / (prompt_advantages.std() + 1e-8)
        
        for _ in range(epochs):
            log_probs_new, values_new = [], []
            for i in range(len(prompt_obs_all)):
                lp, v = self.prompt_policy.get_log_prob_and_value(prompt_obs_all[i], prompt_actions_all[i])
                log_probs_new.append(lp)
                values_new.append(v)
            
            log_probs_new = torch.stack(log_probs_new)
            values_new = torch.stack(values_new)
            
            ratio = torch.exp(log_probs_new - prompt_log_probs_old)
            surr1 = ratio * prompt_advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * prompt_advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values_new, prompt_returns)
            entropy = self.prompt_policy.get_entropy(prompt_obs_tensor).mean()
            
            loss = policy_loss + value_coef * value_loss - prompt_ent_coef * entropy
            
            self.prompt_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.prompt_policy.parameters(), max_grad_norm)
            self.prompt_optimizer.step()
    
    def _print_config(self, num_episodes, batch_size, kwargs):
        super()._print_config(num_episodes, batch_size, kwargs)
        print(f"  Clip Epsilon: {kwargs.get('clip_epsilon', 0.2)}")
        print(f"  PPO Epochs:   {kwargs.get('epochs', 4)}")
        print(f"  Entropy:      {kwargs.get('struct_entropy_coef', 0.05)}")
        print("  Features:     Value function baseline")

