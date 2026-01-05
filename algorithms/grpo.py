"""
GRPO (Group Relative Policy Optimization) for Hierarchical RL.

From DeepSeek-Math paper - designed for sparse reward environments.

Key Features:
- NO value function (critic-free)
- Group-relative advantages: A_i = (R_i - mean(R)) / std(R)
- Better for sparse rewards (correctness signal)
- Optional KL regularization to reference policy
"""
import copy
import numpy as np
import torch

from algorithms.base import (
    Algorithm, BaseTrainer,
    MultiDiscretePolicyGRPO, PolicyNetworkGRPO
)


class GRPOTrainer(BaseTrainer):
    """GRPO trainer - critic-free with group-relative advantages."""
    
    algorithm = Algorithm.GRPO
    
    def __init__(self, cfg, device="cuda" if torch.cuda.is_available() else "cpu", use_action_masking=False, use_api=False, api_model=None):
        super().__init__(cfg, device, use_action_masking=use_action_masking, use_api=use_api, api_model=api_model)
        
        print(f"Initializing GRPO (critic-free)...")
        print(f"  Structure: obs={self.struct_obs_dim}, actions={self.struct_action_dims}")
        print(f"  Prompt: obs={self.prompt_obs_dim}, actions={self.prompt_action_dim}")
        if self.use_action_masking:
            print(f"  ✓ Action masking ENABLED (two-stage selection: workflow first, then mask)")
        else:
            print(f"  Action masking DISABLED (standard selection)")
        
        self.structure_policy = MultiDiscretePolicyGRPO(
            self.struct_obs_dim, self.struct_action_dims
        ).to(device)
        
        self.prompt_policy = PolicyNetworkGRPO(
            self.prompt_obs_dim, self.prompt_action_dim
        ).to(device)
        
        self._init_optimizers()
        
        # Reference policies for KL regularization
        self.structure_ref = None
        self.prompt_ref = None
        self.episodes_since_ref_update = 0
    
    def _update_reference_policies(self):
        """Create frozen copies for KL regularization."""
        self.structure_ref = copy.deepcopy(self.structure_policy)
        self.prompt_ref = copy.deepcopy(self.prompt_policy)
        for p in self.structure_ref.parameters():
            p.requires_grad = False
        for p in self.prompt_ref.parameters():
            p.requires_grad = False
    
    def _compute_group_advantages(self, rewards: list) -> torch.Tensor:
        """
        GRPO core: Group-relative advantage computation.
        
        A_i = (R_i - mean(R)) / (std(R) + eps)
        """
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        std = rewards_t.std()
        
        if std < 1e-6:
            return torch.zeros_like(rewards_t)
        
        return ((rewards_t - rewards_t.mean()) / (std + 1e-8)).detach()
    
    def update_policies(self, episodes: list, **kwargs):
        """GRPO policy update with group-relative advantages."""
        gamma = kwargs.get("gamma", self.cfg.PROMPT_GAMMA)
        clip_epsilon = kwargs.get("clip_epsilon", 0.2)
        epochs = kwargs.get("epochs", 4)
        struct_ent_coef = kwargs.get("struct_entropy_coef", 0.05)
        prompt_ent_coef = kwargs.get("prompt_entropy_coef", 0.05)
        kl_coef = kwargs.get("kl_coef", 0.0)
        ref_update_every = kwargs.get("ref_update_every", 1000)
        max_grad_norm = 0.5
        
        # Update reference policies periodically
        self.episodes_since_ref_update += len(episodes)
        if kl_coef > 0:
            if self.structure_ref is None or self.episodes_since_ref_update >= ref_update_every:
                self._update_reference_policies()
                self.episodes_since_ref_update = 0
        
        # ========== STRUCTURE POLICY ==========
        struct_obs = [ep["struct_obs"] for ep in episodes]
        struct_actions = [ep["struct_action"] for ep in episodes]
        struct_log_probs_old = torch.FloatTensor([ep["struct_log_prob"] for ep in episodes]).to(self.device).detach()
        struct_rewards = [ep["reward"] for ep in episodes]
        
        struct_obs_tensor = torch.FloatTensor(np.array(struct_obs)).to(self.device)
        
        # GRPO: Group-relative advantages (NO value function)
        struct_advantages = self._compute_group_advantages(struct_rewards)
        
        # Get action masks from episodes
        struct_action_masks = [ep.get("struct_action_mask", None) for ep in episodes]
        
        for _ in range(epochs):
            log_probs_new = torch.stack([
                self.structure_policy.get_log_prob(
                    struct_obs[i], struct_actions[i],
                    action_mask=struct_action_masks[i],
                    use_two_stage_masking=self.use_action_masking,
                    structure_env=self.structure_env if self.use_action_masking else None
                )
                for i in range(len(struct_obs))
            ])
            
            ratio = torch.exp(log_probs_new - struct_log_probs_old)
            surr1 = ratio * struct_advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * struct_advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            # Compute entropy with masks
            struct_mask = struct_action_masks[0] if struct_action_masks[0] is not None else None
            entropy = self.structure_policy.get_entropy(struct_obs_tensor, action_mask=struct_mask).mean()
            
            # Optional KL regularization
            kl_loss = 0.0
            if kl_coef > 0 and self.structure_ref is not None:
                with torch.no_grad():
                    ref_log_probs = torch.stack([
                        self.structure_ref.get_log_prob(
                            struct_obs[i], struct_actions[i],
                            action_mask=struct_action_masks[i],
                            use_two_stage_masking=self.use_action_masking,
                            structure_env=self.structure_env if self.use_action_masking else None
                        )
                        for i in range(len(struct_obs))
                    ])
                kl_loss = (log_probs_new - ref_log_probs).mean()
            
            loss = policy_loss - struct_ent_coef * entropy + kl_coef * kl_loss
            
            self.struct_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.structure_policy.parameters(), max_grad_norm)
            self.struct_optimizer.step()
        
        # ========== PROMPT POLICY ==========
        prompt_obs_all, prompt_actions_all = [], []
        prompt_log_probs_old_all, prompt_rewards_all = [], []
        
        for ep in episodes:
            num_steps = len(ep["prompt_obs_list"])
            for i in range(num_steps):
                prompt_obs_all.append(ep["prompt_obs_list"][i])
                prompt_actions_all.append(ep["prompt_actions"][i])
                prompt_log_probs_old_all.append(ep["prompt_log_probs"][i])
                prompt_rewards_all.append(ep["reward"] * (gamma ** (num_steps - i - 1)))
        
        if not prompt_obs_all:
            return
        
        prompt_obs_tensor = torch.FloatTensor(np.array(prompt_obs_all)).to(self.device)
        prompt_log_probs_old = torch.FloatTensor(prompt_log_probs_old_all).to(self.device).detach()
        
        # GRPO: Group-relative advantages
        prompt_advantages = self._compute_group_advantages(prompt_rewards_all)
        
        for _ in range(epochs):
            log_probs_new = torch.stack([
                self.prompt_policy.get_log_prob(prompt_obs_all[i], prompt_actions_all[i])
                for i in range(len(prompt_obs_all))
            ])
            
            ratio = torch.exp(log_probs_new - prompt_log_probs_old)
            surr1 = ratio * prompt_advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * prompt_advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy = self.prompt_policy.get_entropy(prompt_obs_tensor).mean()
            
            # Optional KL regularization
            kl_loss = 0.0
            if kl_coef > 0 and self.prompt_ref is not None:
                with torch.no_grad():
                    ref_log_probs = torch.stack([
                        self.prompt_ref.get_log_prob(prompt_obs_all[i], prompt_actions_all[i])
                        for i in range(len(prompt_obs_all))
                    ])
                kl_loss = (log_probs_new - ref_log_probs).mean()
            
            loss = policy_loss - prompt_ent_coef * entropy + kl_coef * kl_loss
            
            self.prompt_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.prompt_policy.parameters(), max_grad_norm)
            self.prompt_optimizer.step()
    
    def _print_config(self, num_episodes, batch_size, kwargs):
        super()._print_config(num_episodes, batch_size, kwargs)
        print(f"  Clip Epsilon: {kwargs.get('clip_epsilon', 0.2)}")
        print(f"  GRPO Epochs:  {kwargs.get('epochs', 4)}")
        print(f"  Entropy:      {kwargs.get('struct_entropy_coef', 0.05)}")
        print(f"  KL Coef:      {kwargs.get('kl_coef', 0.0)}")
        print("  Features:     Critic-free, group-relative advantages")

