"""
Base classes for hierarchical RL algorithms.

Contains:
- Algorithm enum
- Policy network classes (PPO and GRPO variants)
- BaseTrainer with shared logic
"""
import os
import re
import time
import json
import numpy as np
import threading
from abc import ABC, abstractmethod
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from env.structure_env import StructureEnv
from env.prompt_env import PromptEnv


def flatten_per_turn_structure(episodes, gamma=0.99):
    """Flatten per-turn structure decisions into individual entries.

    Per-turn episodes store lists for struct_obs/action/log_prob/value (one
    per dialog turn).  Single-config episodes store scalars.  This normalises
    both into flat lists so PPO/GRPO update code can treat every entry
    identically.  Rewards are gamma-discounted from the terminal turn.
    """
    flat_obs, flat_actions, flat_log_probs = [], [], []
    flat_values, flat_masks, flat_returns = [], [], []

    for ep in episodes:
        if ep.get("per_turn_config") and isinstance(ep.get("struct_obs"), list):
            n = len(ep["struct_obs"])
            for t in range(n):
                flat_obs.append(ep["struct_obs"][t])
                flat_actions.append(ep["struct_action"][t])
                flat_log_probs.append(float(ep["struct_log_prob"][t]))
                flat_values.append(float(ep["struct_value"][t]) if "struct_value" in ep else 0.0)
                mask_list = ep.get("struct_action_mask")
                flat_masks.append(mask_list[t] if mask_list else None)
                flat_returns.append(ep["reward"] * (gamma ** (n - t - 1)))
        else:
            flat_obs.append(ep["struct_obs"])
            flat_actions.append(ep["struct_action"])
            flat_log_probs.append(float(ep["struct_log_prob"]))
            flat_values.append(float(ep.get("struct_value", 0.0)))
            flat_masks.append(ep.get("struct_action_mask"))
            flat_returns.append(ep["reward"])

    return flat_obs, flat_actions, flat_log_probs, flat_values, flat_masks, flat_returns


class Algorithm(Enum):
    """Supported RL algorithms."""
    PPO = "ppo"
    GRPO = "grpo"


# =============================================================================
# POLICY NETWORKS
# =============================================================================

class PolicyNetworkPPO(nn.Module):
    """Discrete policy network with value head (for PPO)."""
    
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
        return self.action_head(features), self.value_head(features)
    
    def get_action(self, obs, deterministic=False):
        obs = self._to_tensor(obs)
        action_logits, value = self.forward(obs)
        probs = torch.softmax(action_logits, dim=-1)
        
        action = torch.argmax(probs, dim=-1) if deterministic else Categorical(probs).sample()
        log_prob = torch.log(probs.gather(1, action.unsqueeze(-1)) + 1e-8)
        
        return action.item(), log_prob.item(), value.item()
    
    def get_log_prob_and_value(self, obs, action):
        obs = self._to_tensor(obs)
        action = self._action_to_tensor(action)
        
        action_logits, value = self.forward(obs)
        probs = torch.softmax(action_logits, dim=-1)
        log_prob = torch.log(probs.gather(1, action.unsqueeze(-1)) + 1e-8)
        
        return log_prob.squeeze(), value.squeeze()
    
    def get_entropy(self, obs):
        obs = self._to_tensor(obs)
        action_logits, _ = self.forward(obs)
        probs = torch.softmax(action_logits, dim=-1)
        return -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
    
    def _to_tensor(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return obs.to(next(self.parameters()).device)
    
    def _action_to_tensor(self, action):
        if not isinstance(action, torch.Tensor):
            action = torch.LongTensor([action])
        if action.dim() == 0:
            action = action.unsqueeze(0)
        return action.to(next(self.parameters()).device)


class PolicyNetworkGRPO(nn.Module):
    """Discrete policy network WITHOUT value head (for GRPO - critic-free)."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        return self.action_head(self.network(x))
    
    def get_action(self, obs, deterministic=False):
        obs = self._to_tensor(obs)
        action_logits = self.forward(obs)
        probs = torch.softmax(action_logits, dim=-1)
        
        action = torch.argmax(probs, dim=-1) if deterministic else Categorical(probs).sample()
        log_prob = torch.log(probs.gather(1, action.unsqueeze(-1)) + 1e-8)
        
        return action.item(), log_prob.item(), 0.0  # No value
    
    def get_log_prob(self, obs, action):
        obs = self._to_tensor(obs)
        action = self._action_to_tensor(action)
        
        action_logits = self.forward(obs)
        probs = torch.softmax(action_logits, dim=-1)
        return torch.log(probs.gather(1, action.unsqueeze(-1)) + 1e-8).squeeze()
    
    def get_entropy(self, obs):
        obs = self._to_tensor(obs)
        action_logits = self.forward(obs)
        probs = torch.softmax(action_logits, dim=-1)
        return -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
    
    def _to_tensor(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return obs.to(next(self.parameters()).device)
    
    def _action_to_tensor(self, action):
        if not isinstance(action, torch.Tensor):
            action = torch.LongTensor([action])
        if action.dim() == 0:
            action = action.unsqueeze(0)
        return action.to(next(self.parameters()).device)


class MultiDiscretePolicyPPO(nn.Module):
    """MultiDiscrete policy with value head (for PPO structure policy)."""
    
    def __init__(self, obs_dim: int, action_dims: list, hidden_dim: int = 256):
        super().__init__()
        self.action_dims = action_dims
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_heads = nn.ModuleList([nn.Linear(hidden_dim, dim) for dim in action_dims])
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        features = self.network(x)
        return [head(features) for head in self.action_heads], self.value_head(features)
    
    def get_action(self, obs, deterministic=False, action_mask=None, use_two_stage_masking=False, structure_env=None):
        """
        Get action from policy.
        
        Args:
            obs: Observation
            deterministic: If True, use argmax instead of sampling
            action_mask: Pre-computed action mask (list of boolean arrays)
            use_two_stage_masking: If True, do two-stage selection (workflow first, then mask others)
            structure_env: StructureEnv instance (needed for two-stage masking)
        """
        obs = self._to_tensor(obs)
        action_logits_list, value = self.forward(obs)
        
        actions, log_probs = [], []
        
        if use_two_stage_masking and structure_env is not None:
            # Two-stage selection: workflow first, then mask other dimensions
            # Stage 1: Select workflow (dimension 0)
            workflow_logits = action_logits_list[0]
            workflow_probs = torch.softmax(workflow_logits, dim=-1)
            workflow_action = torch.argmax(workflow_probs, dim=-1) if deterministic else Categorical(workflow_probs).sample()
            workflow_idx = workflow_action.item()
            actions.append(workflow_idx)
            log_probs.append(torch.log(workflow_probs.gather(1, workflow_action.unsqueeze(-1)) + 1e-8).item())
            
            # Stage 2: Get mask based on selected workflow and select other dimensions
            workflow_mask = structure_env._get_action_mask(workflow_depth=workflow_idx)
            
            # Select remaining dimensions with masking
            for i in range(1, len(action_logits_list)):
                logits = action_logits_list[i]
                # Apply mask for this dimension
                if i < len(workflow_mask):
                    mask = torch.tensor(workflow_mask[i], dtype=torch.bool, device=logits.device)
                    expected_action_dim = logits.shape[-1]
                    if mask.shape[0] != expected_action_dim:
                        raise ValueError(
                            f"Mask size mismatch for action dimension {i}: "
                            f"mask has size {mask.shape[0]}, but logits action dimension is {expected_action_dim}. "
                            f"This suggests a mismatch between the action space definition and the mask."
                        )
                    logits = logits.masked_fill(~mask, float('-inf'))
                
                probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(probs, dim=-1) if deterministic else Categorical(probs).sample()
                actions.append(action.item())
                log_probs.append(torch.log(probs.gather(1, action.unsqueeze(-1)) + 1e-8).item())
        else:
            # Standard single-stage selection with optional pre-computed mask
            for i, logits in enumerate(action_logits_list):
                # Apply action mask if provided
                if action_mask is not None and i < len(action_mask):
                    mask = torch.tensor(action_mask[i], dtype=torch.bool, device=logits.device)
                    # Set masked logits to -inf before softmax
                    logits = logits.masked_fill(~mask, float('-inf'))
                
                probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(probs, dim=-1) if deterministic else Categorical(probs).sample()
                actions.append(action.item())
                log_probs.append(torch.log(probs.gather(1, action.unsqueeze(-1)) + 1e-8).item())
        
        return np.array(actions), sum(log_probs), value.item()
    
    def get_log_prob_and_value(self, obs, action, action_mask=None, use_two_stage_masking=False, structure_env=None):
        obs = self._to_tensor(obs)
        action = self._action_to_tensor(action)
        
        action_logits_list, value = self.forward(obs)
        
        # If using two-stage masking, reconstruct mask from workflow in action
        if use_two_stage_masking and structure_env is not None and action.shape[0] > 0:
            workflow_idx = int(action[0, 0].item())  # action[0, 0] is workflow index (first dim of first action)
            action_mask = structure_env._get_action_mask(workflow_depth=workflow_idx)
        
        log_probs = []
        for i, logits in enumerate(action_logits_list):
            # Apply action mask if provided
            if action_mask is not None and i < len(action_mask):
                mask = torch.tensor(action_mask[i], dtype=torch.bool, device=logits.device)
                logits = logits.masked_fill(~mask, float('-inf'))
            
            probs = torch.softmax(logits, dim=-1)
            log_probs.append(torch.log(probs.gather(1, action[:, i].unsqueeze(-1)) + 1e-8))
        
        return sum(log_probs).squeeze(), value.squeeze()
    
    def get_entropy(self, obs, action_mask=None):
        obs = self._to_tensor(obs)
        action_logits_list, _ = self.forward(obs)
        
        total_entropy = 0.0
        for i, logits in enumerate(action_logits_list):
            # Apply action mask if provided
            if action_mask is not None and i < len(action_mask):
                mask = torch.tensor(action_mask[i], dtype=torch.bool, device=logits.device)
                logits = logits.masked_fill(~mask, float('-inf'))
            
            probs = torch.softmax(logits, dim=-1)
            total_entropy = total_entropy + (-(probs * torch.log(probs + 1e-8)).sum(dim=-1))
        return total_entropy
    
    def _to_tensor(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return obs.to(next(self.parameters()).device)
    
    def _action_to_tensor(self, action):
        if not isinstance(action, torch.Tensor):
            action = torch.LongTensor(action)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        return action.to(next(self.parameters()).device)


class MultiDiscretePolicyGRPO(nn.Module):
    """MultiDiscrete policy WITHOUT value head (for GRPO structure policy)."""
    
    def __init__(self, obs_dim: int, action_dims: list, hidden_dim: int = 256):
        super().__init__()
        self.action_dims = action_dims
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_heads = nn.ModuleList([nn.Linear(hidden_dim, dim) for dim in action_dims])
        
    def forward(self, x):
        features = self.network(x)
        return [head(features) for head in self.action_heads]
    
    def get_action(self, obs, deterministic=False, action_mask=None, use_two_stage_masking=False, structure_env=None):
        """
        Get action from policy.
        
        Args:
            obs: Observation
            deterministic: If True, use argmax instead of sampling
            action_mask: Pre-computed action mask (list of boolean arrays)
            use_two_stage_masking: If True, do two-stage selection (workflow first, then mask others)
            structure_env: StructureEnv instance (needed for two-stage masking)
        """
        obs = self._to_tensor(obs)
        action_logits_list = self.forward(obs)
        
        actions, log_probs = [], []
        
        if use_two_stage_masking and structure_env is not None:
            # Two-stage selection: workflow first, then mask other dimensions
            # Stage 1: Select workflow (dimension 0)
            workflow_logits = action_logits_list[0]
            workflow_probs = torch.softmax(workflow_logits, dim=-1)
            workflow_action = torch.argmax(workflow_probs, dim=-1) if deterministic else Categorical(workflow_probs).sample()
            workflow_idx = workflow_action.item()
            actions.append(workflow_idx)
            log_probs.append(torch.log(workflow_probs.gather(1, workflow_action.unsqueeze(-1)) + 1e-8).item())
            
            # Stage 2: Get mask based on selected workflow and select other dimensions
            workflow_mask = structure_env._get_action_mask(workflow_depth=workflow_idx)
            
            # Select remaining dimensions with masking
            for i in range(1, len(action_logits_list)):
                logits = action_logits_list[i]
                # Apply mask for this dimension
                if i < len(workflow_mask):
                    mask = torch.tensor(workflow_mask[i], dtype=torch.bool, device=logits.device)
                    logits = logits.masked_fill(~mask, float('-inf'))
                
                probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(probs, dim=-1) if deterministic else Categorical(probs).sample()
                actions.append(action.item())
                log_probs.append(torch.log(probs.gather(1, action.unsqueeze(-1)) + 1e-8).item())
        else:
            # Standard single-stage selection with optional pre-computed mask
            for i, logits in enumerate(action_logits_list):
                # Apply action mask if provided
                if action_mask is not None and i < len(action_mask):
                    mask = torch.tensor(action_mask[i], dtype=torch.bool, device=logits.device)
                    logits = logits.masked_fill(~mask, float('-inf'))
                
                probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(probs, dim=-1) if deterministic else Categorical(probs).sample()
                actions.append(action.item())
                log_probs.append(torch.log(probs.gather(1, action.unsqueeze(-1)) + 1e-8).item())
        
        return np.array(actions), sum(log_probs), 0.0  # No value
    
    def get_log_prob(self, obs, action, action_mask=None, use_two_stage_masking=False, structure_env=None):
        obs = self._to_tensor(obs)
        action = self._action_to_tensor(action)
        
        # If using two-stage masking, reconstruct mask from workflow in action
        if use_two_stage_masking and structure_env is not None and action.shape[0] > 0:
            workflow_idx = int(action[0, 0].item())  # action[0, 0] is workflow index (first dim of first action)
            action_mask = structure_env._get_action_mask(workflow_depth=workflow_idx)
        
        action_logits_list = self.forward(obs)
        
        log_probs = []
        for i, logits in enumerate(action_logits_list):
            # Apply action mask if provided
            if action_mask is not None and i < len(action_mask):
                mask = torch.tensor(action_mask[i], dtype=torch.bool, device=logits.device)
                logits = logits.masked_fill(~mask, float('-inf'))
            
            probs = torch.softmax(logits, dim=-1)
            log_probs.append(torch.log(probs.gather(1, action[:, i].unsqueeze(-1)) + 1e-8))
        
        return sum(log_probs).squeeze()
    
    def get_entropy(self, obs, action_mask=None):
        obs = self._to_tensor(obs)
        action_logits_list = self.forward(obs)
        
        total_entropy = 0.0
        for i, logits in enumerate(action_logits_list):
            # Apply action mask if provided
            if action_mask is not None and i < len(action_mask):
                mask = torch.tensor(action_mask[i], dtype=torch.bool, device=logits.device)
                logits = logits.masked_fill(~mask, float('-inf'))
            
            probs = torch.softmax(logits, dim=-1)
            total_entropy = total_entropy + (-(probs * torch.log(probs + 1e-8)).sum(dim=-1))
        return total_entropy
    
    def _to_tensor(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return obs.to(next(self.parameters()).device)
    
    def _action_to_tensor(self, action):
        if not isinstance(action, torch.Tensor):
            action = torch.LongTensor(action)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        return action.to(next(self.parameters()).device)


# =============================================================================
# BASE TRAINER
# =============================================================================

class BaseTrainer(ABC):
    """
    Base trainer for hierarchical RL.
    
    Subclasses implement algorithm-specific update logic.
    """
    
    algorithm: Algorithm = None
    
    def __init__(self, cfg, device="cuda" if torch.cuda.is_available() else "cpu", use_action_masking=False, use_api=False, api_model=None, hf_model=None):
        self.cfg = cfg
        self.device = device
        self.use_action_masking = use_action_masking
        self.use_api = use_api
        self.api_model = api_model
        self.hf_model = hf_model
        
        self.model_name = api_model if use_api else hf_model
        if self.model_name is None:
            try:
                from configs.base import LLM_MODEL_NAME
                self.model_name = LLM_MODEL_NAME
            except:
                self.model_name = "default"
                print(f"Warning: LLM_MODEL_NAME not found in configs.base.py, using default model name: {self.model_name}")
        
        # Create environments
        self.structure_env = StructureEnv(cfg, use_api=use_api, api_model=api_model, hf_model=hf_model)
        self.prompt_env = PromptEnv(cfg, use_api=use_api, api_model=api_model, hf_model=hf_model)
        
        # Dimensions
        self.struct_obs_dim = self.structure_env.observation_space.shape[0]
        self.prompt_obs_dim = self.prompt_env.observation_space.shape[0]
        self.struct_action_dims = list(self.structure_env.action_space.nvec)
        self.prompt_action_dim = self.prompt_env.action_space.n
        
        # Policies (created by subclass)
        self.structure_policy = None
        self.prompt_policy = None
        
        # Optimizers (created after policies)
        self.struct_optimizer = None
        self.prompt_optimizer = None
        
        # Metrics
        self.episode_count = 0
        self.correct_count = 0
        self.total_reward = 0.0
        self.rewards_history = deque(maxlen=100)
        self.tool_usage_history = deque(maxlen=100)
        
        # Logging
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = None
        self.episode_logs = []
        
        # Reward config
        self.reward_scale = 1.0
        self.tool_bonus = 0.0
        # Tau2-specific reward weights (only used when info contains tau2_* keys).
        # Stage A (graded final score): weighted sum of per-component pass fractions.
        # Each weight is the MAX contribution that component can add to the reward
        # when fully passed. Sum of weights = max possible Stage A reward.
        self.tau2_w_action = 2.0
        self.tau2_w_communicate = 1.0
        self.tau2_w_env = 2.0
        self.tau2_w_nl = 1.0
        # Stage A2: big bonus when all required components pass (perfect run).
        self.tau2_completion_bonus = 5.0
        # Stage B/C: per-pass bonuses and per-miss penalties (additional gradient
        # on top of Stage A's smooth fraction).
        self.tau2_action_bonus = 0.5
        self.tau2_communicate_bonus = 0.3
        self.tau2_env_bonus = 0.5
        self.tau2_nl_bonus = 0.3
        self.tau2_action_penalty = 1.0
        self.tau2_communicate_penalty = 0.5
        self.tau2_env_penalty = 0.5
        # Stage E: hygiene bonus for clean termination
        self.tau2_clean_termination_bonus = 0.2

        # Stage F: anti-stall penalty. We observed (telecom run, ep 2500-2999)
        # that the structure policy can find a "do-nothing" local optimum: the
        # agent emits one generic stalling line, the user simulator gives up,
        # and the episode terminates with all per-component totals = 0. With
        # totals=0 the stage_a/bonuses/penalties all evaluate to 0, leaving
        # the episode at ~0 reward — which dominates "try-and-fail" runs that
        # earn ~-0.4. Once the policy learns reward(stall) > reward(try), it
        # collapses. The gate below detects these stall episodes deterministically
        # and subtracts a fixed penalty so trying is always preferred. Tunable.
        self.tau2_stall_penalty = 1.0
        self.tau2_stall_max_turns = 2          # stall iff dialog ended in ≤ 2 turns
        self.tau2_stall_max_tokens = 500       # AND used <500 agent tokens
        # (we also require tools_count==0 and all component totals==0 below)

    def _compute_episode_reward(
        self,
        info: dict,
        accumulated_reward: float,
        agent1_tools_list: list,
        agent2_tools_list: list,
    ) -> tuple:
        """Compute final episode reward and a per-episode log-payload of all signals.

        Returns (final_reward, reward_log) where reward_log is a dict of components
        suitable to merge into the per-episode log entry. `final_reward` is what gets
        used for advantage / GAE computation by the caller.

        Two paths:
          - tau2_* datasets: shaped reward over tau2's per-component breakdown.
          - everything else: original binary-correctness formula (unchanged).
        """
        is_tau2 = bool(info.get("tau2_reward_basis") is not None and "tau2_action_total" in info)

        # ---- Standard (non-tau2) path: original formula, unchanged ----
        if not is_tau2:
            correct = bool(info.get("correct", False))
            tools_used = int(info.get("tools_used", 0))
            num_tools_selected = len(agent1_tools_list) + len(agent2_tools_list)
            tools_selected = num_tools_selected > 0

            final_reward = (1.0 if correct else 0.0) * 5.0 * self.reward_scale
            final_reward += accumulated_reward
            final_reward -= info.get("steps_taken", 1) * self.cfg.COST_PER_STEP

            if tools_used > 0:
                final_reward += tools_used * self.tool_bonus
                if correct:
                    final_reward += tools_used * 0.2
                if tools_selected:
                    tool_usage_ratio = tools_used / num_tools_selected
                    if tool_usage_ratio >= 0.5:
                        final_reward += 0.3 * tool_usage_ratio
            elif tools_selected:
                unused_tools_penalty = 0.5 * num_tools_selected
                final_reward -= unused_tools_penalty
                if not correct:
                    final_reward -= 0.3 * num_tools_selected

            max_tokens = 2048 + 1024 + 512
            final_reward -= (info.get("total_tokens", 256) / max_tokens) * self.cfg.COST_TOKEN_BUDGET

            return final_reward, {
                "reward_path": "standard",
                "binary_correct": correct,
            }

        # ---- Tau2 shaped path ----
        # Components only count if their type is in the task's reward_basis.
        # tau2 reward_basis values look like 'RewardType.ACTION', 'RewardType.COMMUNICATE',
        # 'RewardType.DB', 'RewardType.ENV_ASSERTION', 'RewardType.NL_ASSERTION'.
        rb = set(str(x) for x in (info.get("tau2_reward_basis") or []))
        def _in_rb(*needles):
            return any(any(n in r for n in needles) for r in rb)

        action_pass = int(info.get("tau2_action_pass", 0))
        action_total = int(info.get("tau2_action_total", 0))
        action_missed = int(info.get("tau2_action_missed", 0))
        comm_pass = int(info.get("tau2_communicate_pass", 0))
        comm_total = int(info.get("tau2_communicate_total", 0))
        comm_missed = int(info.get("tau2_communicate_missed", 0))
        env_pass = int(info.get("tau2_env_pass", 0))
        env_total = int(info.get("tau2_env_total", 0))
        env_missed = int(info.get("tau2_env_missed", 0))
        nl_pass = int(info.get("tau2_nl_pass", 0))
        nl_total = int(info.get("tau2_nl_total", 0))
        nl_missed = int(info.get("tau2_nl_missed", 0))
        original = float(info.get("tau2_original_reward", 0.0))
        correctness = float(info.get("correctness", original))
        term_reason = str(info.get("tau2_termination_reason", "") or "")

        # Stage A: weighted sum of per-component pass fractions.
        # Replaces the multiplicative tau2 reward (which goes to 0 if ANY
        # component fully fails) with an additive sum so partial progress
        # is rewarded smoothly. Only counts components that are in this
        # task's reward_basis (so e.g. NL_ASSERTION-less tasks aren't
        # punished for missing NL passes).
        components = []
        if _in_rb("ACTION") and action_total > 0:
            components.append((self.tau2_w_action, action_pass / action_total))
        if _in_rb("COMMUNICATE") and comm_total > 0:
            components.append((self.tau2_w_communicate, comm_pass / comm_total))
        if _in_rb("DB", "ENV_ASSERTION") and env_total > 0:
            components.append((self.tau2_w_env, env_pass / env_total))
        if _in_rb("NL_ASSERTION") and nl_total > 0:
            components.append((self.tau2_w_nl, nl_pass / nl_total))

        if components:
            stage_a_score = sum(w * frac for w, frac in components)
        else:
            # Fallback: no per-component data. Use the multiplicative original.
            stage_a_score = correctness * (self.tau2_w_action + self.tau2_w_env)

        final_reward = stage_a_score * self.reward_scale

        # Stage A2: big completion bonus when ALL required components fully pass.
        # This is the "you nailed the whole task" signal that's much larger than
        # any single per-component contribution, so it dominates the gradient
        # for fully-correct trajectories.
        all_passed = bool(components) and all(frac >= 0.999 for _, frac in components)
        if all_passed:
            final_reward += self.tau2_completion_bonus * self.reward_scale

        # Stage B: per-component shaping (only weighted if in reward_basis)
        action_bonus_total = 0.0
        comm_bonus_total = 0.0
        env_bonus_total = 0.0
        nl_bonus_total = 0.0
        if _in_rb("ACTION") and action_pass > 0:
            action_bonus_total = action_pass * self.tau2_action_bonus
        if _in_rb("COMMUNICATE") and comm_pass > 0:
            comm_bonus_total = comm_pass * self.tau2_communicate_bonus
        if _in_rb("DB", "ENV_ASSERTION") and env_pass > 0:
            env_bonus_total = env_pass * self.tau2_env_bonus
        if _in_rb("NL_ASSERTION") and nl_pass > 0:
            nl_bonus_total = nl_pass * self.tau2_nl_bonus
        final_reward += action_bonus_total + comm_bonus_total + env_bonus_total + nl_bonus_total

        # Stage C: per-component penalties for missed required components
        action_penalty_total = 0.0
        comm_penalty_total = 0.0
        env_penalty_total = 0.0
        if _in_rb("ACTION") and action_missed > 0:
            action_penalty_total = action_missed * self.tau2_action_penalty
        if _in_rb("COMMUNICATE") and comm_missed > 0:
            comm_penalty_total = comm_missed * self.tau2_communicate_penalty
        if _in_rb("DB", "ENV_ASSERTION") and env_missed > 0:
            env_penalty_total = env_missed * self.tau2_env_penalty
        final_reward -= (action_penalty_total + comm_penalty_total + env_penalty_total)

        # Stage D: efficiency penalties (same shape as standard path)
        final_reward -= info.get("steps_taken", 1) * self.cfg.COST_PER_STEP
        max_tokens = 2048 + 1024 + 512
        final_reward -= (info.get("total_tokens", 256) / max_tokens) * self.cfg.COST_TOKEN_BUDGET

        # Stage E: hygiene bonus
        clean_term = ("AGENT_STOP" in term_reason or "USER_STOP" in term_reason) if term_reason else False
        if clean_term:
            final_reward += self.tau2_clean_termination_bonus

        # Stage F: anti-stall penalty. Detect "did-nothing" episodes — short
        # dialog, no tool calls, and tau2 returned no scoring breakdown (all
        # per-component totals = 0, which happens when the user simulator gave
        # up before any gold check fired). Without this gate, such episodes
        # net ~0 reward and dominate "try-and-fail" runs that net ~-0.4,
        # causing the structure policy to collapse to "Direct workflow / 0
        # tools / 1 turn". AGENT_STOP is excluded because that's a legitimate
        # explicit termination by the agent (e.g., transferring to a human
        # after gathering needed info), not a stall.
        no_components = (action_total + comm_total + env_total + nl_total) == 0
        # Exclusions:
        #   AGENT_STOP — agent intentionally ended (e.g., transferred to human)
        #   too_many_errors — exogenous API failures (not the agent's fault);
        #     these are rare (~1% of episodes) but penalizing them would
        #     misattribute infrastructure issues to the policy.
        is_exogenous_failure = "too_many_errors" in term_reason.lower()
        stall_detected = (
            int(info.get("tau2_turns", 0)) <= self.tau2_stall_max_turns
            and int(info.get("tools_count", 0)) == 0
            and int(info.get("total_tokens", 0)) < self.tau2_stall_max_tokens
            and no_components
            and "AGENT_STOP" not in term_reason
            and not is_exogenous_failure
        )
        stall_penalty_applied = self.tau2_stall_penalty if stall_detected else 0.0
        if stall_detected:
            final_reward -= stall_penalty_applied

        # Per-step / atom-selection rewards from PromptEnv (kept for parity with standard path)
        final_reward += accumulated_reward

        reward_log = {
            "reward_path": "tau2",
            "binary_correct": all_passed,
            "tau2_original_reward": original,
            "tau2_correctness": correctness,
            "tau2_stage_a_score": stage_a_score,
            "tau2_completion_bonus_applied": float(self.tau2_completion_bonus * self.reward_scale) if all_passed else 0.0,
            "tau2_reward_basis": sorted(rb),
            "tau2_action_pass": action_pass, "tau2_action_total": action_total, "tau2_action_missed": action_missed,
            "tau2_communicate_pass": comm_pass, "tau2_communicate_total": comm_total, "tau2_communicate_missed": comm_missed,
            "tau2_env_pass": env_pass, "tau2_env_total": env_total, "tau2_env_missed": env_missed,
            "tau2_nl_pass": nl_pass, "tau2_nl_total": nl_total, "tau2_nl_missed": nl_missed,
            "tau2_termination_reason": term_reason,
            "tau2_bonus_action": action_bonus_total,
            "tau2_bonus_communicate": comm_bonus_total,
            "tau2_bonus_env": env_bonus_total,
            "tau2_bonus_nl": nl_bonus_total,
            "tau2_penalty_action": action_penalty_total,
            "tau2_penalty_communicate": comm_penalty_total,
            "tau2_penalty_env": env_penalty_total,
            "tau2_clean_termination": clean_term,
            "tau2_stall_detected": stall_detected,
            "tau2_stall_penalty_applied": stall_penalty_applied,
            "scaled_reward": final_reward,
        }
        return final_reward, reward_log
    
    def _sanitize_folder_name(self, model_name: str) -> str:
        """Sanitize model name for logging."""
        # Split from / and get last part
        model_name = model_name.split('/')[-1]
        
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', model_name)
        sanitized = sanitized.replace(' ', '_')
        sanitized = sanitized.replace('.', '_')
        sanitized = sanitized.strip('. ')
        sanitized = re.sub(r'_+', '_', sanitized)
        
        if not sanitized:
            return "default"
        return sanitized
    
    def _init_optimizers(self, struct_lr=None, prompt_lr=None):
        """
        Initialize optimizers after policies are created.
        
        Args:
            struct_lr: Override structure learning rate (default: use config)
            prompt_lr: Override prompt learning rate (default: use config)
        """
        struct_lr = struct_lr if struct_lr is not None else self.cfg.STRUCTURE_LEARNING_RATE
        prompt_lr = prompt_lr if prompt_lr is not None else self.cfg.PROMPT_LEARNING_RATE
        
        self.struct_optimizer = optim.Adam(
            self.structure_policy.parameters(), lr=struct_lr
        )
        self.prompt_optimizer = optim.Adam(
            self.prompt_policy.parameters(), lr=prompt_lr
        )
    
    def load_pretrained(self, structure_path=None, prompt_path=None, reset_optimizers=True):
        """
        Load pretrained model weights (e.g., from SFT post-training).
        
        Args:
            structure_path: Path to pretrained structure policy checkpoint
            prompt_path: Path to pretrained prompt policy checkpoint
            reset_optimizers: If True, reset optimizer state after loading (recommended)
        """
        if structure_path:
            if not os.path.exists(structure_path):
                raise FileNotFoundError(f"Pretrained structure model not found: {structure_path}")
            print(f"\nLoading pretrained structure policy from: {structure_path}")
            checkpoint = torch.load(structure_path, map_location=self.device, weights_only=False)
            
            # Check if dimensions match
            if checkpoint.get("obs_dim") != self.struct_obs_dim:
                print(f"  Warning: obs_dim mismatch ({checkpoint.get('obs_dim')} vs {self.struct_obs_dim})")
            if checkpoint.get("action_dims") != self.struct_action_dims:
                print(f"  Warning: action_dims mismatch ({checkpoint.get('action_dims')} vs {self.struct_action_dims})")
            
            self.structure_policy.load_state_dict(checkpoint["model_state_dict"])
            self.structure_policy.train()  # Ensure train mode
            print("  ✓ Structure policy loaded")
        
        if prompt_path:
            if not os.path.exists(prompt_path):
                raise FileNotFoundError(f"Pretrained prompt model not found: {prompt_path}")
            print(f"\nLoading pretrained prompt policy from: {prompt_path}")
            checkpoint = torch.load(prompt_path, map_location=self.device, weights_only=False)
            
            # Check if dimensions match
            if checkpoint.get("obs_dim") != self.prompt_obs_dim:
                print(f"  Warning: obs_dim mismatch ({checkpoint.get('obs_dim')} vs {self.prompt_obs_dim})")
            if checkpoint.get("action_dim") != self.prompt_action_dim:
                print(f"  Warning: action_dim mismatch ({checkpoint.get('action_dim')} vs {self.prompt_action_dim})")
            
            self.prompt_policy.load_state_dict(checkpoint["model_state_dict"])
            self.prompt_policy.train()  # Ensure train mode
            print("  ✓ Prompt policy loaded")
        
        if (structure_path or prompt_path) and reset_optimizers:
            # Reset optimizer state to match new weights
            # This is important because optimizers maintain momentum/state per parameter
            print("  Resetting optimizer state to match pretrained weights...")
            # Use current learning rates (can be overridden via train args)
            self._init_optimizers()
            print("  ✓ Optimizers reset")
        
        if structure_path or prompt_path:
            print("  Ready to continue RL training from pretrained models\n")
    
    def _decode_tools(self, idx: int) -> list:
        """Decode tool bitmask to tool names.

        Delegates to the structure env so tau2 datasets (variable-size domain
        toolkits, 2^N action space) and default datasets (fixed 4-tool toolkit)
        share a single decoder.
        """
        return self.structure_env._decode_tools(idx)

    def load_training_log(self, log_path: str) -> None:
        """Resume cumulative training stats from a previous run's training-log JSON.

        Restores: episode_count, correct_count, total_reward, rewards_history (last
        100), tool_usage_history (last 100), episode_logs, and self.log_path so new
        episodes APPEND to the same file. Optimizer state and rollout buffers are
        intentionally NOT restored — PPO/GRPO are on-policy and reset_optimizers in
        load_pretrained() drops the stale momentum.
        """
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"Training log not found: {log_path}")
        with open(log_path, "r") as f:
            payload = json.load(f)

        # _save_log writes a wrapper dict with metadata + 'episodes' list. Tolerate
        # both shapes: a bare list of episode dicts, or {"episodes": [...]}.
        if isinstance(payload, dict) and "episodes" in payload:
            episodes = payload["episodes"]
        elif isinstance(payload, list):
            episodes = payload
        else:
            raise ValueError(
                f"Unrecognized training-log shape in {log_path}: expected list "
                f"or dict with 'episodes' key, got {type(payload).__name__}"
            )

        # Reconstruct state.
        self.episode_logs = list(episodes)
        self.episode_count = len(self.episode_logs)
        self.correct_count = sum(1 for e in self.episode_logs if e.get("correct"))
        self.total_reward = float(sum(float(e.get("reward", 0.0)) for e in self.episode_logs))

        # rewards_history / tool_usage_history are deque(maxlen=100) — only last 100
        # matter for the rolling mean displayed in the bar postfix.
        self.rewards_history.clear()
        for e in self.episode_logs[-100:]:
            self.rewards_history.append(float(e.get("reward", 0.0)))
        self.tool_usage_history.clear()
        for e in self.episode_logs[-100:]:
            self.tool_usage_history.append(int(e.get("tools_used", 0)))

        # Reuse the same log file so subsequent _save_log() calls APPEND (rewrite
        # the full self.episode_logs list, which now includes both old + new).
        self.log_path = log_path

        print(
            f"  ✓ Resumed from training log: {log_path}\n"
            f"    episodes={self.episode_count} correct={self.correct_count} "
            f"total_reward={self.total_reward:.2f} "
            f"recent_avg={float(np.mean(self.rewards_history)) if self.rewards_history else 0.0:.3f}"
        )
    
    def run_episode(self, deterministic=False):
        """Run a single hierarchical episode."""
        # Per-turn-config mode (tau2 only, opt-in via cfg.PER_TURN_CONFIG):
        # delegate to multiturn_rollout which re-runs structure+prompt policies
        # at every dialog turn. Returns per-turn LISTS for struct_obs/action/etc.
        # PPO/GRPO update_policies must flatten these via flatten_per_turn_structure.
        ds_name = getattr(self.prompt_env.dataset, "name", "") or ""
        per_turn = bool(getattr(self.cfg, "PER_TURN_CONFIG", False)) and ds_name.startswith("tau2_")
        if per_turn:
            return self._run_episode_per_turn(deterministic=deterministic)

        # Reset and get question
        struct_obs, struct_info = self.structure_env.reset()
        question, answer = struct_info["question"], struct_info["answer"]
        action_mask = struct_info.get("action_mask", None)
        
        # Structure policy decision
        # Use two-stage masking if enabled: select workflow first, then mask other dimensions
        if self.use_action_masking:
            struct_action, struct_log_prob, struct_value = self.structure_policy.get_action(
                struct_obs, deterministic=deterministic,
                use_two_stage_masking=True, structure_env=self.structure_env
            )
        else:
            struct_action, struct_log_prob, struct_value = self.structure_policy.get_action(
                struct_obs, deterministic=deterministic, action_mask=action_mask
            )
        
        # Parse structure
        workflow_depth = int(struct_action[0])
        agent1_tools_idx = int(struct_action[1])
        agent1_budget_idx = int(struct_action[2])
        agent2_tools_idx = int(struct_action[3])
        agent2_budget_idx = int(struct_action[4])
        answerer_budget_idx = int(struct_action[5])
        
        # Set up prompt env
        self.prompt_env.current_q = question
        self.prompt_env.current_a = answer
        self.prompt_env.question_embedding = self.structure_env.question_embedding.copy()
        self.prompt_env.workflow_depth = workflow_depth
        self.prompt_env.agent1_tools_idx = agent1_tools_idx
        self.prompt_env.agent1_budget_idx = agent1_budget_idx
        self.prompt_env.agent2_tools_idx = agent2_tools_idx
        self.prompt_env.agent2_budget_idx = agent2_budget_idx
        self.prompt_env.answerer_budget_idx = answerer_budget_idx
        
        # Direct (0) and Parallel-Voting (5) don't need reasoner prompts
        if workflow_depth == 0 or workflow_depth == 5:
            self.prompt_env.prompt_stage = self.prompt_env.PROMPT_STAGE_ANSWERER
        else:
            # All other workflows start with reasoner
            self.prompt_env.prompt_stage = self.prompt_env.PROMPT_STAGE_REASONER
        
        self.prompt_env.prompt_step = 0
        self.prompt_env.selected_prompts = {"reasoner": [], "verifier": [], "answerer": []}
        self.prompt_env._structure_set = True
        
        # Prompt policy rollout
        prompt_obs_list, prompt_actions, prompt_log_probs, prompt_values = [], [], [], []
        prompt_obs = self.prompt_env._get_observation()
        accumulated_reward = 0.0
        
        done = False
        while not done:
            action, log_prob, value = self.prompt_policy.get_action(prompt_obs, deterministic)
            prompt_obs_list.append(prompt_obs.copy())
            prompt_actions.append(action)
            prompt_log_probs.append(log_prob)
            prompt_values.append(value)
            
            next_obs, step_reward, done, _, info = self.prompt_env.step(action)
            accumulated_reward += step_reward
            prompt_obs = next_obs
        
        # Compute final reward (tau2 datasets get a shaped per-component reward;
        # everything else gets the original binary-correctness formula).
        agent1_tools_list = self._decode_tools(agent1_tools_idx)
        agent2_tools_list = self._decode_tools(agent2_tools_idx)
        final_reward, reward_log = self._compute_episode_reward(
            info, accumulated_reward, agent1_tools_list, agent2_tools_list
        )
        correct = bool(reward_log.get("binary_correct", info.get("correct", False)))
        tools_used = info.get("tools_used", 0)
        num_tools_selected = len(agent1_tools_list) + len(agent2_tools_list)
        
        # Update metrics
        self.episode_count += 1
        self.correct_count += 1 if correct else 0
        self.total_reward += final_reward
        self.rewards_history.append(final_reward)
        self.tool_usage_history.append(tools_used)
        
        # Log (the per-tau2-component breakdown is merged in via reward_log when applicable)
        log_entry = {
            "episode": self.episode_count,
            "correct": correct,
            "reward": final_reward,                        # scaled reward used for training
            "workflow": [
                "Direct", "Reason+Ans", "Reason+Verify+Ans",
                "Routing", "Parallel-Sectioning", "Parallel-Voting",
                "Orchestrator-Workers", "Evaluator-Optimizer", "Autonomous-Agent"
            ][workflow_depth],
            "num_prompt_steps": len(prompt_actions),
            "steps_taken": info.get("steps_taken", 0),
            "tools_used": tools_used,
            "total_tokens": info.get("total_tokens", 0),
            "agent1_tools": self._decode_tools(agent1_tools_idx),
            "agent2_tools": self._decode_tools(agent2_tools_idx),
            "reasoner_prompts": info.get("reasoner_prompts", []),
            "verifier_prompts": info.get("verifier_prompts", []),
            "answerer_prompts": info.get("answerer_prompts", []),
            "reasoner_budget": ["Low", "Mid", "High"][agent1_budget_idx] if workflow_depth in [1, 2, 3, 4, 6, 7, 8] else "N/A",
            "verifier_budget": ["Low", "Mid", "High"][agent2_budget_idx] if workflow_depth in [2, 7] else "N/A",
            "answerer_budget": ["Low", "Mid", "High"][answerer_budget_idx],
            "question": info.get("question", ""),
            "final_answer": info.get("final_answer", ""),
            "ground_truth": info.get("ground_truth", "")
        }
        log_entry.update(reward_log)
        # Forward any tau2_* diagnostics in info that reward_log didn't already
        # include (e.g. per-turn shaping, shaping_mode, turn count). Mirrors
        # the parallel path's forwarding so logs are consistent regardless of
        # --num-workers.
        for k, v in (info or {}).items():
            if k.startswith("tau2_") and k not in log_entry:
                log_entry[k] = v
        self.episode_logs.append(log_entry)

        return {
            "struct_obs": struct_obs,
            "struct_action": struct_action,
            "struct_log_prob": struct_log_prob,
            "struct_value": struct_value,
            "struct_action_mask": action_mask,  # Store action mask for policy updates
            "prompt_obs_list": prompt_obs_list,
            "prompt_actions": prompt_actions,
            "prompt_log_probs": prompt_log_probs,
            "prompt_values": prompt_values,
            "reward": final_reward,
            "correct": correct,
            "info": info,
        }

    def _run_episode_per_turn(self, deterministic=False):
        """Per-turn-config rollout: structure + prompt policies re-decided each
        dialog turn. The trainer treats every turn as additional (s,a,r) tuples
        for the buffer, so credit propagates back via γ-discount as usual.

        Returns the SAME shape as run_episode() so the caller can stay agnostic.
        Only difference: struct_obs / struct_action / etc. are now LISTS (one
        entry per turn) and the result dict has 'per_turn_rewards' set."""
        from env.multiturn_rollout import multiturn_rollout

        # Reset structure_env to load a fresh task (provides question/answer/embedding)
        struct_obs, struct_info = self.structure_env.reset()
        question = struct_info.get("question", "")
        answer = struct_info.get("answer", "")
        domain = self.prompt_env.dataset.domain
        task_id = answer  # tau2 datasets store task_id in the 'answer' slot

        solo_mode = bool(getattr(self.cfg, "TAU2_SOLO_MODE", False))
        max_turns = int(getattr(self.cfg, "TAU2_MAX_TURNS", 8))
        use_shaped = bool(getattr(self.cfg, "USE_SHAPED_REWARDS", True))
        shaping_mode = str(getattr(self.cfg, "SHAPING_MODE", "training"))

        traj = multiturn_rollout(
            structure_policy=self.structure_policy,
            prompt_policy=self.prompt_policy,
            structure_env=self.structure_env,
            prompt_env=self.prompt_env,
            domain=domain,
            task_id=task_id,
            deterministic=deterministic,
            max_turns=max_turns,
            solo_mode=solo_mode,
            use_shaped_rewards=use_shaped,
            shaping_mode=shaping_mode,
            use_action_masking=self.use_action_masking,
        )

        exec_info = traj["exec_info"]
        # Aggregate to a single "final_reward" the existing logger expects.
        # We sum per-turn shaping (already includes terminal at last turn) plus
        # any additional component-based reward from _compute_episode_reward.
        info_for_reward = dict(exec_info)
        info_for_reward.setdefault("correct", traj["terminal_reward"] >= 0.999)
        info_for_reward.setdefault("correctness", float(min(max(traj["terminal_reward"], 0.0), 1.0)))
        info_for_reward.setdefault("steps_taken", exec_info.get("tau2_turns", 1))
        info_for_reward.setdefault("tools_used", exec_info.get("tools_count", 0))
        info_for_reward.setdefault("total_tokens", 0)

        agent1_tools_list = []  # per-turn — log first turn for backward-compat
        agent2_tools_list = []
        if traj["structure_actions"]:
            first_struct = traj["structure_actions"][0]
            agent1_tools_list = self._decode_tools(int(first_struct[1]))
            agent2_tools_list = self._decode_tools(int(first_struct[3]))

        # Use the trainer's shaped-reward computation on the aggregated info.
        # Per-turn shaping is already summed inside per_turn_rewards; we just
        # use the trainer's final calculation as the "total" reward signal
        # (matches what _compute_episode_reward does for single-config mode).
        final_reward, reward_log = self._compute_episode_reward(
            info_for_reward, sum(traj["per_turn_rewards"]),
            agent1_tools_list, agent2_tools_list,
        )
        correct = bool(reward_log.get("binary_correct", info_for_reward.get("correct", False)))
        tools_used = info_for_reward.get("tools_used", 0)

        # Update aggregate metrics
        self.episode_count += 1
        self.correct_count += 1 if correct else 0
        self.total_reward += final_reward
        self.rewards_history.append(final_reward)
        self.tool_usage_history.append(tools_used)

        log_entry = {
            "episode": self.episode_count,
            "correct": correct,
            "reward": final_reward,
            "workflow": "PerTurnReconfig",
            "num_prompt_steps": len(traj["prompt_actions"]),
            "num_dialog_turns": len(traj["per_turn_rewards"]),
            "steps_taken": info_for_reward.get("steps_taken", 0),
            "tools_used": tools_used,
            "total_tokens": info_for_reward.get("total_tokens", 0),
            "agent1_tools": agent1_tools_list,
            "agent2_tools": agent2_tools_list,
            "per_turn_rewards": list(map(float, traj["per_turn_rewards"])),
            "per_turn_workflow": [int(a[0]) for a in traj["structure_actions"]],
        }
        log_entry.update(reward_log)
        for k, v in (info_for_reward or {}).items():
            if k.startswith("tau2_") and k not in log_entry:
                log_entry[k] = v
        self.episode_logs.append(log_entry)

        return {
            "struct_obs": traj["structure_obs_list"],          # list, one per turn
            "struct_action": traj["structure_actions"],        # list
            "struct_log_prob": traj["structure_log_probs"],    # list
            "struct_value": traj["structure_values"],          # list
            "struct_action_mask": traj["structure_action_masks"],  # list
            "prompt_obs_list": traj["prompt_obs_list"],
            "prompt_actions": traj["prompt_actions"],
            "prompt_log_probs": traj["prompt_log_probs"],
            "prompt_values": traj["prompt_values"],
            "reward": final_reward,
            "correct": correct,
            "info": info_for_reward,
            "per_turn_rewards": traj["per_turn_rewards"],
            "per_turn_config": True,
        }


    def _run_episode_with_envs(self, ep_num, structure_env, prompt_env, deterministic=False):
        """
        Run a single episode using provided environment instances.
        Used for parallel training where each worker has its own environments.
        """
        # Reset and get question
        struct_obs, struct_info = structure_env.reset()
        question, answer = struct_info["question"], struct_info["answer"]
        action_mask = struct_info.get("action_mask", None)
        
        # Structure policy decision
        if self.use_action_masking:
            struct_action, struct_log_prob, struct_value = self.structure_policy.get_action(
                struct_obs, deterministic=deterministic,
                use_two_stage_masking=True, structure_env=structure_env
            )
        else:
            struct_action, struct_log_prob, struct_value = self.structure_policy.get_action(
                struct_obs, deterministic=deterministic, action_mask=action_mask
            )
        
        # Parse structure
        workflow_depth = int(struct_action[0])
        agent1_tools_idx = int(struct_action[1])
        agent1_budget_idx = int(struct_action[2])
        agent2_tools_idx = int(struct_action[3])
        agent2_budget_idx = int(struct_action[4])
        answerer_budget_idx = int(struct_action[5])
        
        # Set up prompt env
        prompt_env.current_q = question
        prompt_env.current_a = answer
        prompt_env.question_embedding = structure_env.question_embedding.copy()
        prompt_env.workflow_depth = workflow_depth
        prompt_env.agent1_tools_idx = agent1_tools_idx
        prompt_env.agent1_budget_idx = agent1_budget_idx
        prompt_env.agent2_tools_idx = agent2_tools_idx
        prompt_env.agent2_budget_idx = agent2_budget_idx
        prompt_env.answerer_budget_idx = answerer_budget_idx
        
        # Direct (0) and Parallel-Voting (5) don't need reasoner prompts
        if workflow_depth == 0 or workflow_depth == 5:
            prompt_env.prompt_stage = prompt_env.PROMPT_STAGE_ANSWERER
        else:
            prompt_env.prompt_stage = prompt_env.PROMPT_STAGE_REASONER
        
        prompt_env.prompt_step = 0
        prompt_env.selected_prompts = {"reasoner": [], "verifier": [], "answerer": []}
        prompt_env._structure_set = True
        
        # Prompt policy rollout
        prompt_obs_list, prompt_actions, prompt_log_probs, prompt_values = [], [], [], []
        prompt_obs = prompt_env._get_observation()
        accumulated_reward = 0.0
        
        done = False
        while not done:
            action, log_prob, value = self.prompt_policy.get_action(prompt_obs, deterministic)
            prompt_obs_list.append(prompt_obs.copy())
            prompt_actions.append(action)
            prompt_log_probs.append(log_prob)
            prompt_values.append(value)
            
            next_obs, step_reward, done, _, info = prompt_env.step(action)
            accumulated_reward += step_reward
            prompt_obs = next_obs
        
        # Compute final reward (tau2 datasets: shaped reward; others: original formula)
        agent1_tools_list = self._decode_tools(agent1_tools_idx)
        agent2_tools_list = self._decode_tools(agent2_tools_idx)
        final_reward, reward_log = self._compute_episode_reward(
            info, accumulated_reward, agent1_tools_list, agent2_tools_list
        )
        correct = bool(reward_log.get("binary_correct", info.get("correct", False)))
        tools_used = info.get("tools_used", 0)

        # Create episode log entry
        episode_log = {
            "episode": ep_num,
            "correct": correct,
            "reward": final_reward,                        # scaled reward used for training
            "workflow": [
                "Direct", "Reason+Ans", "Reason+Verify+Ans",
                "Routing", "Parallel-Sectioning", "Parallel-Voting",
                "Orchestrator-Workers", "Evaluator-Optimizer", "Autonomous-Agent"
            ][workflow_depth],
            "num_prompt_steps": len(prompt_actions),
            "steps_taken": info.get("steps_taken", 0),
            "tools_used": tools_used,
            "total_tokens": info.get("total_tokens", 0),
            "agent1_tools": agent1_tools_list,
            "agent2_tools": agent2_tools_list,
            "reasoner_prompts": info.get("reasoner_prompts", []),
            "verifier_prompts": info.get("verifier_prompts", []),
            "answerer_prompts": info.get("answerer_prompts", []),
            "reasoner_budget": ["Low", "Mid", "High"][agent1_budget_idx] if workflow_depth in [1, 2, 3, 4, 6, 7, 8] else "N/A",
            "verifier_budget": ["Low", "Mid", "High"][agent2_budget_idx] if workflow_depth in [2, 7] else "N/A",
            "answerer_budget": ["Low", "Mid", "High"][answerer_budget_idx],
            "question": info.get("question", ""),
            "final_answer": info.get("final_answer", ""),
            "ground_truth": info.get("ground_truth", "")
        }
        episode_log.update(reward_log)
        # Also forward any tau2_* diagnostics in info that reward_log didn't
        # populate (e.g. per-turn shaping breakdown, shaping_mode, turn count).
        for k, v in (info or {}).items():
            if k.startswith("tau2_") and k not in episode_log:
                episode_log[k] = v

        return {
            "episode_num": ep_num,
            "struct_obs": struct_obs,
            "struct_action": struct_action,
            "struct_log_prob": struct_log_prob,
            "struct_value": struct_value,
            "struct_action_mask": action_mask,
            "prompt_obs_list": prompt_obs_list,
            "prompt_actions": prompt_actions,
            "prompt_log_probs": prompt_log_probs,
            "prompt_values": prompt_values,
            "reward": final_reward,
            "correct": correct,
            "info": info,
            "episode_log": episode_log,
        }
    
    @abstractmethod
    def update_policies(self, episodes: list, **kwargs):
        """Algorithm-specific policy update."""
        pass
    
    def _get_model_name_for_filename(self):
        """Get model name for use in filename (sanitized)."""
        if self.use_api:
            model_name = self.api_model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
        else:
            if self.hf_model:
                model_name = self.hf_model
            else:
                try:
                    from configs.base import LLM_MODEL_NAME
                    model_name = LLM_MODEL_NAME
                except:
                    model_name = "unknown"
        
        # Sanitize model name for filename: replace / with -, replace spaces with _
        model_name_safe = model_name.replace("/", "-").replace(" ", "_")
        return model_name_safe
    
    def train(self, num_episodes: int = 1000, batch_size: int = 32,
              log_every: int = 50, save_every: int = 2000,
              save_log_every: int = 500, num_workers: int = 1, **kwargs):
        """
        Main training loop.
        
        Args:
            num_episodes: Number of training episodes
            batch_size: Batch size for policy updates
            log_every: Logging frequency
            save_every: Model checkpoint frequency
            save_log_every: Log file save frequency
            num_workers: Number of parallel workers (only used with --api mode)
        """
        self.reward_scale = kwargs.get("reward_scale", 1.0)
        self.tool_bonus = kwargs.get("tool_bonus", 0.0)
        # Tau2 reward weight overrides (only applied when set; otherwise __init__ defaults stand)
        for k in (
            "tau2_w_action", "tau2_w_communicate", "tau2_w_env", "tau2_w_nl",
            "tau2_completion_bonus",
        ):
            if k in kwargs and kwargs[k] is not None:
                setattr(self, k, float(kwargs[k]))
        
        # Check if parallel mode should be used (only for API mode)
        use_parallel = self.use_api and num_workers > 1
        
        if use_parallel:
            return self._train_parallel(num_episodes, batch_size, log_every, 
                                       save_every, save_log_every, num_workers, **kwargs)
        else:
            return self._train_sequential(num_episodes, batch_size, log_every,
                                         save_every, save_log_every, **kwargs)
    
    def _train_sequential(self, num_episodes: int = 1000, batch_size: int = 32,
                          log_every: int = 50, save_every: int = 2000,
                          save_log_every: int = 500, **kwargs):
        """Sequential training loop (original implementation)."""
        timestamp = int(time.time())
        algo = self.algorithm.value
        model_name = self._get_model_name_for_filename()
        # If the log_path is already set (e.g., load_training_log was called to
        # resume a prior run), keep appending to it. Otherwise create a fresh file.
        if not self.log_path:
            self.log_path = os.path.join(self.log_dir, f"training_log_{model_name}_{algo}_{self.cfg.DATASET_NAME}_{timestamp}.json")
        
        print("\n" + "=" * 70)
        print(f"HIERARCHICAL TRAINING ({algo.upper()})")
        print(f"Log will be saved to: {self.log_path} (every {save_log_every} episodes)")
        print("=" * 70)
        self._print_config(num_episodes, batch_size, kwargs)
        print("=" * 70 + "\n")
        
        start_time = time.time()
        batch = []
        pbar = tqdm(total=num_episodes, desc=f"{algo.upper()} Training", unit="ep")
        
        for ep in range(num_episodes):
            result = self.run_episode()
            batch.append(result)
            
            if len(batch) >= batch_size:
                self.update_policies(batch, **kwargs)
                batch = []
            
            # Progress
            acc = self.correct_count / self.episode_count * 100 if self.episode_count > 0 else 0
            avg_rew = self.total_reward / self.episode_count if self.episode_count > 0 else 0
            recent = np.mean(self.rewards_history) if self.rewards_history else 0
            tools = np.mean(self.tool_usage_history) if self.tool_usage_history else 0
            
            elapsed = time.time() - start_time
            eta = f"{(elapsed/(ep+1))*(num_episodes-ep-1)/60:.1f}m" if ep > 0 else "?"
            
            pbar.set_postfix({'Acc': f'{acc:.1f}%', 'Rew': f'{avg_rew:.2f}', 'Tools': f'{tools:.1f}', 'ETA': eta})
            pbar.update(1)
            
            if (ep + 1) % log_every == 0:
                # Show tool selection stats
                recent_episodes = self.episode_logs[-log_every:] if len(self.episode_logs) >= log_every else self.episode_logs
                tools_selected = sum(1 for e in recent_episodes if e.get("agent1_tools") or e.get("agent2_tools"))
                correct_count = sum(1 for e in recent_episodes if e.get("correct", False))
                print(f"\n[{ep+1}/{num_episodes}] Acc: {acc:.1f}% ({self.correct_count}/{self.episode_count}) | Reward: {avg_rew:.3f} | Tools Used: {tools:.2f} | Tools Selected: {tools_selected}/{len(recent_episodes)} | Recent Correct: {correct_count}/{len(recent_episodes)}")
            
            if (ep + 1) % save_every == 0:
                self.save_models(f"_ep{ep+1}")
            
            if (ep + 1) % save_log_every == 0:
                self._save_log()
        
        if batch:
            self.update_policies(batch, **kwargs)
        
        pbar.close()
        self._save_log()
        
        final_acc = self.correct_count / self.episode_count * 100
        print(f"\n{'='*70}\nTRAINING COMPLETE: {final_acc:.1f}% accuracy\n{'='*70}")
        
        return final_acc
    
    def _train_parallel(self, num_episodes: int = 1000, batch_size: int = 32,
                       log_every: int = 50, save_every: int = 2000,
                       save_log_every: int = 500, num_workers: int = 4, **kwargs):
        """Parallel training loop using ThreadPoolExecutor (API mode only)."""
        from utils.get_dataset import get_dataset_loader
        
        timestamp = int(time.time())
        algo = self.algorithm.value
        model_name = self._get_model_name_for_filename()
        # Resumed runs already have self.log_path set by load_training_log; keep it.
        if not self.log_path:
            self.log_path = os.path.join(self.log_dir, f"training_log_{model_name}_{algo}_{self.cfg.DATASET_NAME}_{timestamp}.json")

        print("\n" + "=" * 70)
        print(f"HIERARCHICAL TRAINING ({algo.upper()}) - PARALLEL MODE")
        print(f"Log will be saved to: {self.log_path} (every {save_log_every} episodes)")
        print("=" * 70)
        self._print_config(num_episodes, batch_size, kwargs)
        print(f"  Workers:     {num_workers} (parallel)")
        print("=" * 70 + "\n")
        
        # Load dataset once and share across workers
        print(f"Loading dataset once for {num_workers} workers...")
        shared_dataset = get_dataset_loader(self.cfg.DATASET_NAME, is_eval=False)
        print(f"  Dataset loaded: {len(shared_dataset.tasks) if hasattr(shared_dataset, 'tasks') else len(shared_dataset.data)} samples")
        
        # Pre-create environment pairs for each worker
        print(f"\nPre-creating {num_workers} environment pairs...")
        env_pools = []
        for i in range(num_workers):
            structure_env = StructureEnv(self.cfg, is_eval=False, use_api=self.use_api,
                                        api_model=self.api_model, hf_model=self.hf_model,
                                        dataset=shared_dataset)
            prompt_env = PromptEnv(self.cfg, is_eval=False, use_api=self.use_api,
                                  api_model=self.api_model, hf_model=self.hf_model,
                                  dataset=shared_dataset)
            env_pools.append((structure_env, prompt_env))
        print(f"  ✓ Created {num_workers} environment pairs")
        
        # Pre-initialize embedders sequentially to avoid race conditions
        print(f"\nPre-initializing embedders...")
        initialized_count = 0
        for i, (structure_env, prompt_env) in enumerate(env_pools):
            for env_name, env in [("structure", structure_env), ("prompt", prompt_env)]:
                try:
                    if hasattr(env, 'worker') and hasattr(env.worker, 'embedder'):
                        embedder = env.worker.embedder
                        if hasattr(embedder, '_init_embedder') and not embedder._initialized:
                            embedder._init_embedder()
                            initialized_count += 1
                except Exception as e:
                    print(f"    ⚠ Warning: Could not initialize {env_name} embedder {i}: {e}")
        print(f"  ✓ Pre-initialized {initialized_count} embedders\n")
        
        # Locks for thread-safe access
        env_locks = [threading.Lock() for _ in range(num_workers)]
        metrics_lock = threading.Lock()
        
        # Thread-safe metrics
        episode_results = {}  # ep_num -> result dict
        completed_count = [0]
        
        def worker_fn(ep_num):
            """Worker function to run a single episode."""
            env_idx = ep_num % num_workers
            structure_env, prompt_env = env_pools[env_idx]
            
            with env_locks[env_idx]:
                result = self._run_episode_with_envs(ep_num, structure_env, prompt_env, deterministic=False)
            
            # Thread-safe result storage
            with metrics_lock:
                episode_results[ep_num] = result
                completed_count[0] += 1
            
            return result
        
        start_time = time.time()
        batch = []
        episode_buffer = {}  # Store episodes until we can process them in order
        
        print(f"Starting parallel training with {num_workers} workers...")
        # Sentinel used to mark a failed episode so the in-order consumer can advance
        # past it instead of deadlocking. Without this, a single worker exception
        # (e.g. an OpenRouter 429 on episode 0) leaves episode_buffer[0] missing
        # forever and the bar stays at 0/N.
        _FAILED_EPISODE = object()
        failed_episodes = 0

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all episodes
            futures = {executor.submit(worker_fn, ep): ep for ep in range(num_episodes)}

            # Process results as they complete, maintaining batch order
            with tqdm(total=num_episodes, desc=f"{algo.upper()} Training ({num_workers} workers)", unit="ep") as pbar:
                next_episode = 0  # Track which episode we're waiting for

                for future in as_completed(futures):
                    ep_num = futures[future]
                    try:
                        result = future.result()
                        episode_buffer[ep_num] = result
                    except Exception as e:
                        failed_episodes += 1
                        print(f"\n⚠ Error in episode {ep_num}: {e}")
                        import traceback
                        traceback.print_exc()
                        # Sentinel so the in-order consumer can skip this slot.
                        episode_buffer[ep_num] = _FAILED_EPISODE

                    # Tick the bar on EVERY worker completion (out-of-order is fine
                    # for visual progress) so the user gets immediate feedback even
                    # when episode 0 is the slowest. Postfix shows live metrics
                    # across whatever has been processed in-order so far; until the
                    # first in-order episode lands these read 0.0% / 0.0.
                    pbar.update(1)
                    with metrics_lock:
                        acc_live = self.correct_count / self.episode_count * 100 if self.episode_count > 0 else 0.0
                        avg_rew_live = self.total_reward / self.episode_count if self.episode_count > 0 else 0.0
                        recent_live = float(np.mean(self.rewards_history)) if self.rewards_history else 0.0
                        tools_live = np.mean(self.tool_usage_history) if self.tool_usage_history else 0.0
                    pbar.set_postfix({
                        'Acc': f'{acc_live:.1f}%',
                        'Rew': f'{avg_rew_live:.2f}',
                        'Recent': f'{recent_live:.2f}',
                        'Tools': f'{tools_live:.1f}',
                        'in_order': f'{next_episode}',
                        'failed': failed_episodes,
                    })

                    try:
                        # Process episodes in order as they become available.
                        # Note: this no longer ticks pbar (already done above) but
                        # still drives policy updates / logging / checkpoints.
                        while next_episode in episode_buffer:
                            result = episode_buffer.pop(next_episode)
                            if result is _FAILED_EPISODE:
                                next_episode += 1
                                continue
                            
                            # Remove episode_num from result dict before adding to batch
                            batch_result = {k: v for k, v in result.items() if k != "episode_num" and k != "episode_log"}
                            batch.append(batch_result)
                            
                            # Update metrics thread-safely
                            with metrics_lock:
                                self.episode_count += 1
                                self.correct_count += 1 if result["correct"] else 0
                                self.total_reward += result["reward"]
                                self.rewards_history.append(result["reward"])
                                self.tool_usage_history.append(result["info"].get("tools_used", 0))
                                self.episode_logs.append(result["episode_log"])
                            
                            # Update policies when batch is full
                            if len(batch) >= batch_size:
                                self.update_policies(batch, **kwargs)
                                batch = []
                            
                            # Progress tracking (postfix only — bar already ticked above)
                            with metrics_lock:
                                acc = self.correct_count / self.episode_count * 100 if self.episode_count > 0 else 0
                                avg_rew = self.total_reward / self.episode_count if self.episode_count > 0 else 0
                                recent = np.mean(self.rewards_history) if self.rewards_history else 0
                                tools = np.mean(self.tool_usage_history) if self.tool_usage_history else 0

                            elapsed = time.time() - start_time
                            eta = f"{(elapsed/(next_episode+1))*(num_episodes-next_episode-1)/60:.1f}m" if next_episode > 0 else "?"

                            pbar.set_postfix({
                                'Acc': f'{acc:.1f}%',
                                'Rew': f'{avg_rew:.2f}',
                                'Recent': f'{recent:.2f}',
                                'Tools': f'{tools:.1f}',
                                'ETA': eta,
                                'in_order': f'{next_episode+1}',
                                'failed': failed_episodes,
                            })

                            # Logging
                            if (next_episode + 1) % log_every == 0:
                                with metrics_lock:
                                    recent_episodes = self.episode_logs[-log_every:] if len(self.episode_logs) >= log_every else self.episode_logs
                                    tools_selected = sum(1 for e in recent_episodes if e.get("agent1_tools") or e.get("agent2_tools"))
                                    correct_count = sum(1 for e in recent_episodes if e.get("correct", False))
                                    print(f"\n[{next_episode+1}/{num_episodes}] Acc: {acc:.1f}% ({self.correct_count}/{self.episode_count}) | Reward: {avg_rew:.3f} | Tools Used: {tools:.2f} | Tools Selected: {tools_selected}/{len(recent_episodes)} | Recent Correct: {correct_count}/{len(recent_episodes)}")

                                    # Early-warning: detect user-sim / API failures
                                    one_turn_count = sum(1 for e in recent_episodes if int(e.get("tau2_turns", 99)) <= 1)
                                    if len(recent_episodes) >= 10 and one_turn_count / len(recent_episodes) > 0.6:
                                        print(f"\n{'='*70}")
                                        print(f"  WARNING: {one_turn_count}/{len(recent_episodes)} recent episodes ended in <=1 turn!")
                                        print(f"  This usually means the user simulator API is failing (quota/auth).")
                                        print(f"  Check OPENROUTER_USER_MODEL in .env and API key validity.")
                                        print(f"{'='*70}")

                            # Checkpointing
                            if (next_episode + 1) % save_every == 0:
                                self.save_models(f"_ep{next_episode+1}")

                            if (next_episode + 1) % save_log_every == 0:
                                self._save_log()

                            next_episode += 1

                    except Exception as e:
                        # An error here means the consumer side blew up (e.g., an
                        # update_policies failure). The bar has already been ticked.
                        print(f"\n⚠ Error processing episode {ep_num}: {e}")
                        import traceback
                        traceback.print_exc()
        
        # Process any remaining episodes in buffer
        while next_episode < num_episodes:
            if next_episode in episode_buffer:
                result = episode_buffer.pop(next_episode)
                if result is _FAILED_EPISODE:
                    next_episode += 1
                    continue
                batch_result = {k: v for k, v in result.items() if k != "episode_num" and k != "episode_log"}
                batch.append(batch_result)

                with metrics_lock:
                    self.episode_count += 1
                    self.correct_count += 1 if result["correct"] else 0
                    self.total_reward += result["reward"]
                    self.rewards_history.append(result["reward"])
                    self.tool_usage_history.append(result["info"].get("tools_used", 0))
                    self.episode_logs.append(result["episode_log"])

                if len(batch) >= batch_size:
                    self.update_policies(batch, **kwargs)
                    batch = []

                next_episode += 1
            else:
                # Should never happen — every submitted future writes to the buffer
                # (success or sentinel). If we hit this, mark the slot failed and move on.
                next_episode += 1
        
        # Final batch update
        if batch:
            self.update_policies(batch, **kwargs)
        
        self._save_log()
        
        final_acc = self.correct_count / self.episode_count * 100
        print(f"\n{'='*70}\nTRAINING COMPLETE: {final_acc:.1f}% accuracy\n{'='*70}")
        
        return final_acc
    
    def _print_config(self, num_episodes, batch_size, kwargs):
        print(f"  Episodes:     {num_episodes}")
        print(f"  Batch Size:   {batch_size}")
        print(f"  Dataset:      {self.cfg.DATASET_NAME}")
        print(f"  Device:       {self.device}")
        print(f"  Tool Bonus:   {kwargs.get('tool_bonus', 0.0)}")
    
    def _save_log(self):
        if self.log_path and self.episode_logs:
            print(f"\n[LOG] Saving log to {self.log_path} ({len(self.episode_logs)} episodes)...")
            rewards = [e["reward"] for e in self.episode_logs]
            
            # Determine model information
            if self.use_api:
                model_type = "API"
                model_name = self.api_model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
            else:
                model_type = "HuggingFace"
                if self.hf_model:
                    model_name = self.hf_model
                else:
                    # Fallback to config default
                    try:
                        from configs.base import LLM_MODEL_NAME
                        model_name = LLM_MODEL_NAME
                    except:
                        model_name = "unknown"
            
            summary = {
                "algorithm": self.algorithm.value.upper(),
                "model_type": model_type,
                "model_name": model_name,
                "total_episodes": len(self.episode_logs),
                "accuracy": sum(1 for e in self.episode_logs if e["correct"]) / len(self.episode_logs),
                "avg_reward": float(np.mean(rewards)),
                "avg_tools_used": float(np.mean([e["tools_used"] for e in self.episode_logs])),
                "workflow_distribution": {
                    w: sum(1 for e in self.episode_logs if e["workflow"] == w)
                    for w in [
                        "Direct", "Reason+Ans", "Reason+Verify+Ans",
                        "Routing", "Parallel-Sectioning", "Parallel-Voting",
                        "Orchestrator-Workers", "Evaluator-Optimizer", "Autonomous-Agent"
                    ]
                },
                "episodes": self.episode_logs
            }
            with open(self.log_path, "w") as f:
                json.dump(summary, f, indent=2)
    
    def save_models(self, suffix: str = ""):
        print(f"Saving models for {self.model_name}")
        model_folder = self._sanitize_folder_name(self.model_name)
        model_dir = os.path.join("models", self.algorithm.value, self.cfg.DATASET_NAME, model_folder)
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = int(time.time())
        struct_path = f"{model_dir}/structure_policy_{self.cfg.DATASET_NAME}_{timestamp}{suffix}.pt"
        prompt_path = f"{model_dir}/prompt_policy_{self.cfg.DATASET_NAME}_{timestamp}{suffix}.pt"
        
        torch.save({
            "model_state_dict": self.structure_policy.state_dict(),
            "action_dims": self.struct_action_dims,
            "obs_dim": self.struct_obs_dim,
            "algorithm": self.algorithm.value.upper(),
        }, struct_path)
        
        torch.save({
            "model_state_dict": self.prompt_policy.state_dict(),
            "action_dim": self.prompt_action_dim,
            "obs_dim": self.prompt_obs_dim,
            "algorithm": self.algorithm.value.upper(),
        }, prompt_path)
        
        print(f"\nSaved: {struct_path}, {prompt_path}")
        return struct_path, prompt_path