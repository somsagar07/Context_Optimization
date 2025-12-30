"""
Structure Environment - High-Level Policy (HIRO-style Manager)

Single-step environment that selects the overall agent structure:
- Workflow type (Direct / Reason+Ans / Reason+Verify+Ans)
- Tools for reasoner and verifier
- Token budgets for all agents

NOTE: In TRUE hierarchical setup, this env does NOT execute the LLM.
It only picks the structure. The PromptEnv handles execution.
Both policies learn from the final reward.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
sys.path.append('..')

from agents_system import LLMWorker
from utils import get_dataset_loader


class StructureEnv(gym.Env):
    """
    High-level policy environment for structure decisions.
    
    Action Space: MultiDiscrete([3, 8, 3, 8, 3, 3])
        - workflow: 3 options (Direct=0, Reason+Ans=1, Reason+Verify+Ans=2)
        - reasoner_tools: 8 options (binary encoding of 3 tools)
            - NOTE: Updated to 16 options for 4 tools
        - reasoner_budget: 3 options (Low=0, Mid=1, High=2)
        - verifier_tools: 8 options (binary encoding of 3 tools)
            - NOTE: Updated to 16 options for 4 tools
        - verifier_budget: 3 options
        - answerer_budget: 3 options
    
    This env does NOT execute the workflow - it only selects the structure.
    The PromptEnv handles the actual LLM execution after prompt selection.
    """
    
    def __init__(self, cfg=None):
        """
        Args:
            cfg: Configuration module
        """
        super().__init__()
        
        # Store config
        if cfg is None:
            from configs import load_config
            cfg = load_config("hierarchical")
        self.cfg = cfg
        
        # Initialize components
        self.worker = LLMWorker()
        self.dataset = get_dataset_loader(cfg.DATASET_NAME)
        
        # Structure dimensions: [workflow, r_tools, r_budget, v_tools, v_budget, a_budget]
        # NOTE: Updated tools to 16 options for 4 tools
        self.structure_dims = np.array([3, 16, 3, 16, 3, 3])
        
        # Action space: MultiDiscrete for interpretable structure decisions
        self.action_space = spaces.MultiDiscrete(self.structure_dims)
        
        # Observation space: question embedding + question statistics
        hidden_size = self.worker.model.config.hidden_size
        num_stats = 8
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(hidden_size + num_stats,),
            dtype=np.float32
        )
        
        # Episode state
        self.current_q = None
        self.current_a = None
        self.question_embedding = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Get new question
        self.current_q, self.current_a = self.dataset.get_sample()
        self.question_embedding = self.worker.get_embedding(self.current_q)
        
        info = {
            "question": self.current_q,
            "answer": self.current_a,
        }
        
        return self._get_observation(), info
    
    def _get_observation(self):
        """Build observation: question embedding + statistics."""
        q_len = len(self.current_q)
        word_count = len(self.current_q.split())
        has_numbers = float(any(c.isdigit() for c in self.current_q))
        num_count = sum(1 for c in self.current_q if c.isdigit())
        
        has_multi_step = float(any(kw in self.current_q.lower() for kw in 
                                   ['then', 'after', 'before', 'first', 'total']))
        has_comparison = float(any(kw in self.current_q.lower() for kw in 
                                   ['more', 'less', 'difference', 'compare']))
        
        stats = np.array([
            q_len / 500.0,
            word_count / 100.0,
            has_numbers,
            num_count / 20.0,
            has_multi_step,
            has_comparison,
            0.0,
            0.0,
        ], dtype=np.float32)
        
        return np.concatenate([self.question_embedding, stats]).astype(np.float32)
    
    def step(self, action):
        """
        Process structure decision. Does NOT execute LLM.
        
        Returns the structure info so PromptEnv can use it.
        Reward is 0 here - actual reward comes from PromptEnv after execution.
        """
        # Parse MultiDiscrete action
        workflow_depth = int(action[0])
        reasoner_tools_idx = int(action[1])
        reasoner_budget_idx = int(action[2])
        verifier_tools_idx = int(action[3])
        verifier_budget_idx = int(action[4])
        answerer_budget_idx = int(action[5])
        
        # Store structure info for handoff to PromptEnv
        info = {
            "question": self.current_q,
            "answer": self.current_a,
            "embedding": self.question_embedding,
            "structure": {
                "workflow_depth": workflow_depth,
                "reasoner_tools_idx": reasoner_tools_idx,
                "reasoner_budget_idx": reasoner_budget_idx,
                "verifier_tools_idx": verifier_tools_idx,
                "verifier_budget_idx": verifier_budget_idx,
                "answerer_budget_idx": answerer_budget_idx,
            },
            "workflow": ["Direct", "Reason+Ans", "Reason+Verify+Ans"][workflow_depth],
        }
        
        # No reward here - will get real reward after PromptEnv executes
        # This terminates the structure selection (one-step env)
        return self._get_observation(), 0.0, True, False, info
