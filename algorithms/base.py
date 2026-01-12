"""
Base classes for hierarchical RL algorithms.

Contains:
- Algorithm enum
- Policy network classes (PPO and GRPO variants)
- BaseTrainer with shared logic
"""
import os
import time
import json
import numpy as np
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from env.structure_env import StructureEnv
from env.prompt_env import PromptEnv


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
        """Decode tool bitmask to tool names."""
        tools = []
        if idx & 1: tools.append("calculator")
        if idx & 2: tools.append("web_search")
        if idx & 4: tools.append("python")
        if idx & 8: tools.append("ocr_reader")
        return tools
    
    def run_episode(self, deterministic=False):
        """Run a single hierarchical episode."""
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
        
        # Compute final reward
        correct = info.get("correct", False)
        tools_used = info.get("tools_used", 0)
        
        # Base correctness reward
        final_reward = (1.0 if correct else 0.0) * 5.0 * self.reward_scale
        final_reward += accumulated_reward
        final_reward -= info.get("steps_taken", 1) * self.cfg.COST_PER_STEP
        
        # Check if tools were selected (from structure action)
        agent1_tools_list = self._decode_tools(agent1_tools_idx)
        agent2_tools_list = self._decode_tools(agent2_tools_idx)
        num_tools_selected = len(agent1_tools_list) + len(agent2_tools_list)
        tools_selected = num_tools_selected > 0
        
        # Tool usage reward: encourage tool usage, especially when it leads to correct answers
        if tools_used > 0:
            # Base bonus for using tools
            final_reward += tools_used * self.tool_bonus
            # Extra bonus if tools were used AND answer is correct (tools helped!)
            if correct:
                final_reward += tools_used * 0.2  # Additional 0.2 per tool when correct
            
            # Bonus for using a high percentage of selected tools
            if tools_selected:
                tool_usage_ratio = tools_used / num_tools_selected
                if tool_usage_ratio >= 0.5:  # Used at least half of selected tools
                    final_reward += 0.3 * tool_usage_ratio  # Up to 0.3 bonus for full usage
        elif tools_selected:
            # STRONG penalty for selecting tools but not using them
            # This is the key issue: tools are selected but LLM doesn't use them
            # Penalty is proportional to number of tools selected (more tools = bigger waste)
            unused_tools_penalty = 0.5 * num_tools_selected  # 0.5 per unused tool
            final_reward -= unused_tools_penalty
            
            # Additional penalty if answer is wrong (tools would have helped!)
            if not correct:
                final_reward -= 0.3 * num_tools_selected  # Extra 0.3 per tool when wrong
        
        max_tokens = 1024 + 512 + 256
        final_reward -= (info.get("total_tokens", 256) / max_tokens) * self.cfg.COST_TOKEN_BUDGET
        
        # Update metrics
        self.episode_count += 1
        self.correct_count += 1 if correct else 0
        self.total_reward += final_reward
        self.rewards_history.append(final_reward)
        self.tool_usage_history.append(tools_used)
        
        # Log
        self.episode_logs.append({
            "episode": self.episode_count,
            "correct": correct,
            "reward": final_reward,
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
        })
        
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
    
    @abstractmethod
    def update_policies(self, episodes: list, **kwargs):
        """Algorithm-specific policy update."""
        pass
    
    def train(self, num_episodes: int = 1000, batch_size: int = 32,
              log_every: int = 50, save_every: int = 2000,
              save_log_every: int = 500, **kwargs):
        """Main training loop."""
        self.reward_scale = kwargs.get("reward_scale", 1.0)
        self.tool_bonus = kwargs.get("tool_bonus", 0.0)
        
        timestamp = int(time.time())
        algo = self.algorithm.value
        self.log_path = os.path.join(self.log_dir, f"training_log_{algo}_{self.cfg.DATASET_NAME}_{timestamp}.json")
        
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
        model_dir = f"models/{self.algorithm.value}_models"
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

