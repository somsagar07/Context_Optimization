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

from agents_system import LLMWorker, OpenRouterWorker
from utils import get_dataset_loader


class StructureEnv(gym.Env):
    """
    High-level policy environment for structure decisions.
    
    Action Space: MultiDiscrete([9, 16, 3, 16, 3, 3])
        - workflow: 9 options
            0=Direct, 1=Reason+Ans, 2=Reason+Verify+Ans,
            3=Routing, 4=Parallel-Sectioning, 5=Parallel-Voting,
            6=Orchestrator-Workers, 7=Evaluator-Optimizer, 8=Autonomous-Agent
        - agent1_tools: 16 options (binary encoding of 4 tools)
        - agent1_budget: 3 options (Low=0, Mid=1, High=2)
        - agent2_tools: 16 options (binary encoding of 4 tools)
        - agent2_budget: 3 options
        - answerer_budget: 3 options
    
    This env does NOT execute the workflow - it only selects the structure.
    The PromptEnv handles the actual LLM execution after prompt selection.
    """
    
    def __init__(self, cfg=None, is_eval=False, use_api=False, api_model=None, hf_model=None):
        """
        Args:
            cfg: Configuration module
            use_api: If True, use OpenRouterWorker instead of LLMWorker
            api_model: OpenRouter model ID (e.g., 'openai/gpt-4o')
            hf_model: HuggingFace model name (e.g., 'Qwen/Qwen2.5-7B-Instruct'). Defaults to LLM_MODEL_NAME from config
        """
        super().__init__()
        
        # Store config
        if cfg is None:
            from configs import load_config
            cfg = load_config("hierarchical")
        self.cfg = cfg
        
        # Initialize components
        if use_api:
            self.worker = OpenRouterWorker(model_name=api_model)
        else:
            self.worker = LLMWorker(model_name=hf_model)
        self.dataset = get_dataset_loader(cfg.DATASET_NAME, is_eval=is_eval)
        
        # Structure dimensions: [workflow, agent1_tools, agent1_budget, agent2_tools, agent2_budget, answerer_budget]
        # NOTE: Updated to 9 workflows (3 original + 6 from Anthropic patterns)
        self.structure_dims = np.array([9, 16, 3, 16, 3, 3])
        
        # Action space: MultiDiscrete for interpretable structure decisions
        self.action_space = spaces.MultiDiscrete(self.structure_dims)
        
        # Observation space: question embedding + question statistics
        # Question embedding is 1024D from MetaCLIP-H14
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
            "action_mask": self._get_action_mask(),
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
    
    def _get_action_mask(self, workflow_depth=None):
        """
        Compute action mask for MultiDiscrete action space.
        Returns a list of boolean arrays, one per action dimension.
        True = valid action, False = invalid (masked).
        
        Args:
            workflow_depth: If provided, masks based on workflow requirements.
                           If None, returns all-valid mask (for initial selection).
        """
        # All workflows are valid
        workflow_mask = np.ones(9, dtype=bool)
        
        # All tool combinations are valid (16 options)
        agent1_tools_mask = np.ones(16, dtype=bool)
        agent2_tools_mask = np.ones(16, dtype=bool)
        
        # All budgets are valid (3 options)
        agent1_budget_mask = np.ones(3, dtype=bool)
        agent2_budget_mask = np.ones(3, dtype=bool)
        answerer_budget_mask = np.ones(3, dtype=bool)
        
        # Workflow-dependent masking: Mask agent2 for workflows that don't use it
        # Workflows 0, 1, 5 don't use agent2 (verifier/workers/aspect2)
        if workflow_depth is not None and workflow_depth in [0, 1, 5]:
            # These workflows don't use agent2, so mask agent2_tools and agent2_budget
            # Keep at least one valid action (index 0) to avoid all-masked dimension
            agent2_tools_mask = np.zeros(16, dtype=bool)
            agent2_tools_mask[0] = True  # Keep "no tools" as valid (will be ignored anyway)
            agent2_budget_mask = np.zeros(3, dtype=bool)
            agent2_budget_mask[0] = True  # Keep "Low" as valid (will be ignored anyway)
        
        return [
            workflow_mask,
            agent1_tools_mask,
            agent1_budget_mask,
            agent2_tools_mask,
            agent2_budget_mask,
            answerer_budget_mask
        ]
    
    def step(self, action):
        """
        Process structure decision. Does NOT execute LLM.
        
        Returns the structure info so PromptEnv can use it.
        Reward is 0 here - actual reward comes from PromptEnv after execution.
        """
        # Parse MultiDiscrete action
        workflow_depth = int(action[0])
        agent1_tools_idx = int(action[1])
        agent1_budget_idx = int(action[2])
        agent2_tools_idx = int(action[3])
        agent2_budget_idx = int(action[4])
        answerer_budget_idx = int(action[5])
        
        # Store structure info for handoff to PromptEnv
        info = {
            "question": self.current_q,
            "answer": self.current_a,
            "embedding": self.question_embedding,
            "structure": {
                "workflow_depth": workflow_depth,
                "agent1_tools_idx": agent1_tools_idx,
                "agent1_budget_idx": agent1_budget_idx,
                "agent2_tools_idx": agent2_tools_idx,
                "agent2_budget_idx": agent2_budget_idx,
                "answerer_budget_idx": answerer_budget_idx,
            },
            "workflow": [
                "Direct", "Reason+Ans", "Reason+Verify+Ans",
                "Routing", "Parallel-Sectioning", "Parallel-Voting",
                "Orchestrator-Workers", "Evaluator-Optimizer", "Autonomous-Agent"
            ][workflow_depth],
        }
        
        # No reward here - will get real reward after PromptEnv executes
        # This terminates the structure selection (one-step env)
        info["action_mask"] = self._get_action_mask()
        return self._get_observation(), 0.0, True, False, info
