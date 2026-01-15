import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
sys.path.append('..')

from agents_system import LLMWorker
from agents_system.workflows import get_workflow
from tools import ToolRegistry
from utils import get_dataset_loader


class MultiStepAgentEnv(gym.Env):
    """
    Multi-Step RL Environment for LLM Agent Configuration.
    
    Instead of choosing all 6 action dimensions at once (5,184 combinations),
    the agent makes SEQUENTIAL decisions, enabling proper credit assignment.
    
    Episode Structure (4 steps max):
        Step 0: Choose workflow depth [0, 1, 2, ..., 8] → action_space = 9
        Step 1: Choose reasoner config (tools + budget) → action_space = 24 (8 tools × 3 budgets)
            NOTE: Edited to 16 budget for tools, so action_space = 48
        Step 2: Choose verifier config (if depth=2) → action_space = 24
            NOTE: Edited to 16 budget for tools, so action_space = 48
        Step 3: Choose answerer budget → action_space = 3
        → Execute workflow → Final reward → Done
    
    Benefits:
        - Smaller action space per step (3-24 vs 5,184)
        - Temporal credit assignment via multi-step returns
        - Intermediate rewards guide learning
        - Agent can learn dependencies between choices
    """
    
    # Decision stages
    STAGE_WORKFLOW = 0
    STAGE_REASONER = 1  
    STAGE_VERIFIER = 2
    STAGE_ANSWERER = 3
    STAGE_EXECUTE = 4  # Terminal pseudo-stage
    
    def __init__(self, cfg=None):
        """
        Initialize the environment.
        
        Args:
            cfg: Configuration module. If None, imports default config.
        """
        super(MultiStepAgentEnv, self).__init__()
        
        # Store config (import default if not provided)
        if cfg is None:
            from configs import load_config
            cfg = load_config("multi_step")  # Default to multi_step for backwards compatibility
        self.cfg = cfg
        
        self.worker = LLMWorker()
        self.tools = ToolRegistry()
        self.dataset = get_dataset_loader(cfg.DATASET_NAME)
        # Initialize workflow instances (lazy loading)
        self._workflow_instances = {}
        
        # Action space: max across all stages (reasoner/verifier have 24 = 8×3)
        # NOTE: Edited to 16 tools, so max is 48
        # We'll mask invalid actions per stage
        self.action_space = spaces.Discrete(48)
        
        # Token budgets
        self.TOKEN_BUDGETS = {
            "reasoner": {0: 256, 1: 512, 2: 1024},
            "verifier": {0: 128, 1: 256, 2: 512},
            "answerer": {0: 64, 1: 128, 2: 256}
        }
        
        # Observation: question embedding + stage encoding + partial decisions
        # - question_embedding: hidden_size (1024 for MetaCLIP-H14)
        # - stage_onehot: 4 dims (which decision stage we're in)
        # - workflow_chosen: 9 dims (one-hot of depth)
        # - reasoner_chosen: 24 dims (one-hot of tools×budget)
        # - verifier_chosen: 24 dims (one-hot of tools×budget)
        hidden_size = self.worker.model.config.hidden_size
        obs_size = hidden_size + 4 + 9 + 48 + 48  # Total observation (updated workflow to 9)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # Episode state
        self.current_q = None
        self.current_a = None
        self.question_embedding = None
        self.stage = 0
        
        # Accumulated decisions
        self.workflow_depth = None
        self.agent1_tools = None
        self.agent1_budget = None
        self.agent2_tools = None
        self.agent2_budget = None
        self.answerer_budget = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Get new question
        self.current_q, self.current_a = self.dataset.get_sample()
        self.question_embedding = self.worker.get_embedding(self.current_q)
        
        # Reset stage and decisions
        self.stage = self.STAGE_WORKFLOW
        self.workflow_depth = None
        self.agent1_tools = None
        self.agent1_budget = None
        self.agent2_tools = None
        self.agent2_budget = None
        self.answerer_budget = None
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Build observation: question embedding + stage + partial decisions."""
        # Stage one-hot (4 stages)
        stage_onehot = np.zeros(4, dtype=np.float32)
        stage_onehot[min(self.stage, 3)] = 1.0
        
        # Workflow choice one-hot (3 options)
        workflow_onehot = np.zeros(9, dtype=np.float32)
        if self.workflow_depth is not None:
            workflow_onehot[self.workflow_depth] = 1.0
        
        # Reasoner choice one-hot (24 = 8 tools × 3 budgets)
        # NOTE: Edited to 16 tools, so action space is 48
        reasoner_onehot = np.zeros(48, dtype=np.float32)
        if self.agent1_tools is not None and self.agent1_budget is not None:
            idx = self._encode_config(self.agent1_tools, self.agent1_budget)
            reasoner_onehot[idx] = 1.0
        
        # Verifier choice one-hot (24)
        # NOTE: Edited to 16 tools, so action space is 48
        verifier_onehot = np.zeros(48, dtype=np.float32)
        if self.agent2_tools is not None and self.agent2_budget is not None:
            idx = self._encode_config(self.agent2_tools, self.agent2_budget)
            verifier_onehot[idx] = 1.0
        
        # Concatenate all
        obs = np.concatenate([
            self.question_embedding,
            stage_onehot,
            workflow_onehot,
            reasoner_onehot,
            verifier_onehot
        ]).astype(np.float32)
        
        return obs
    
    def _encode_config(self, tools_idx: int, budget_idx: int) -> int:
        """Encode tools (0-15) and budget (0-2) into single index (0-47)."""
        return tools_idx * 3 + budget_idx
    
    def _decode_config(self, action: int) -> tuple:
        """Decode action into tools index and budget index."""
        tools_idx = action // 3
        budget_idx = action % 3
        return tools_idx, budget_idx
    
    def _get_workflow(self, workflow_depth: int):
        """Get or create workflow instance for the given depth."""
        if workflow_depth not in self._workflow_instances:
            self._workflow_instances[workflow_depth] = get_workflow(
                workflow_depth, self.worker, self.tools
            )
        return self._workflow_instances[workflow_depth]
    
    def _decode_tools(self, idx: int) -> list:
        """Decode tool index to list of tool names."""
        tools = []
        if idx & 1: tools.append("calculator")
        if idx & 2: tools.append("web_search")
        if idx & 4: tools.append("python")
        if idx & 8: tools.append("ocr_reader") # Added this tool
        return tools
    
    def _get_valid_actions(self) -> np.ndarray:
        """Return mask of valid actions for current stage."""
        mask = np.zeros(48, dtype=np.float32)
        
        if self.stage == self.STAGE_WORKFLOW:
            # 9 valid workflows: [0, 1, 2, 3, 4, 5, 6, 7, 8]
            mask[:9] = 1.0
        elif self.stage == self.STAGE_REASONER:
            # 48 valid: all combinations of tools (16) × budget (3)
            # NOTE: Edited to 16 tools, so 48 combinations
            mask[:48] = 1.0
        elif self.stage == self.STAGE_VERIFIER:
            # 48 valid (only reached if depth == 2)
            mask[:48] = 1.0
        elif self.stage == self.STAGE_ANSWERER:
            # Only 3 valid: [0, 1, 2] for answerer budget
            mask[:3] = 1.0
            
        return mask
    
    def step(self, action):
        """Execute one decision step."""
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        # Clamp action to valid range for current stage
        action = int(action)
        
        if self.stage == self.STAGE_WORKFLOW:
            # Choose workflow depth
            self.workflow_depth = min(action, 8)
            
            # Small shaping reward: prefer simpler workflows for efficiency
            # (will be outweighed by correctness if complex is needed)
            # Normalize: Direct=0 gets max bonus, most complex gets 0
            efficiency_bonus = max(0, (8 - self.workflow_depth) * 0.01)
            reward = efficiency_bonus
            
            # Next stage depends on depth
            if self.workflow_depth == 0:
                # Direct answer: skip to answerer
                self.stage = self.STAGE_ANSWERER
            else:
                # Need reasoning
                self.stage = self.STAGE_REASONER
                
        elif self.stage == self.STAGE_REASONER:
            # Choose reasoner tools + budget
            tools_idx, budget_idx = self._decode_config(min(action, 47))
            self.agent1_tools = tools_idx
            self.agent1_budget = budget_idx
            
            # Small shaping: prefer lower budgets (efficiency)
            efficiency_bonus = (2 - budget_idx) * 0.01
            reward = efficiency_bonus
            
            # Next stage: workflows 2 and 7 need verifier
            if self.workflow_depth in [2, 7]:
                self.stage = self.STAGE_VERIFIER
            else:
                self.stage = self.STAGE_ANSWERER
                
        elif self.stage == self.STAGE_VERIFIER:
            # Choose verifier tools + budget
            tools_idx, budget_idx = self._decode_config(min(action, 47))
            self.agent2_tools = tools_idx
            self.agent2_budget = budget_idx
            
            efficiency_bonus = (2 - budget_idx) * 0.01
            reward = efficiency_bonus
            
            self.stage = self.STAGE_ANSWERER
            
        elif self.stage == self.STAGE_ANSWERER:
            # Choose answerer budget and EXECUTE
            self.answerer_budget = min(action, 2)
            
            # Execute the full workflow
            final_text, execution_info = self._execute_workflow()
            
            # Calculate final reward
            correctness = self.dataset.evaluate_correctness(final_text, self.current_a)
            
            # Main reward: correctness
            reward = correctness * 5.0
            
            # Cost penalties (use config values)
            reward -= execution_info["steps"] * self.cfg.COST_PER_STEP
            reward -= execution_info["tools_count"] * self.cfg.COST_TOOL_USAGE
            
            # Token penalty (normalized)
            max_tokens = 2048 + 1024 + 512  # reasoner_high + verifier_high + answerer_high
            token_penalty = (execution_info["total_tokens"] / max_tokens) * self.cfg.COST_TOKEN_BUDGET
            reward -= token_penalty
            
            terminated = True
            info = {
                "query": self.current_q,
                "correct": correctness == 1.0,
                "workflow": [
                    "Direct", "Reason+Ans", "Reason+Verify+Ans",
                    "Routing", "Parallel-Sectioning", "Parallel-Voting",
                    "Orchestrator-Workers", "Evaluator-Optimizer", "Autonomous-Agent"
                ][self.workflow_depth] if self.workflow_depth is not None and self.workflow_depth < 9 else "Unknown",
                "steps_taken": execution_info["steps"],
                "tools_used": execution_info["tools_count"],
                "agent1_tools": execution_info.get("agent1_tools", []),
                "agent2_tools": execution_info.get("agent2_tools", []),
                "agent1_budget": ["Low", "Mid", "High"][self.agent1_budget] if self.agent1_budget is not None else "N/A",
                "agent2_budget": ["Low", "Mid", "High"][self.agent2_budget] if self.agent2_budget is not None else "N/A",
                "answerer_budget": ["Low", "Mid", "High"][self.answerer_budget],
                "total_tokens": execution_info["total_tokens"],
                "episode_length": self._get_episode_length(),
                "final_answer": final_text,
                "ground_truth": self.current_a,
            }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_episode_length(self) -> int:
        """Get number of decision steps for this episode."""
        if self.workflow_depth == 0 or self.workflow_depth == 5:
            return 2  # workflow + answerer (Direct or Parallel-Voting)
        elif self.workflow_depth in [2, 7]:
            return 4  # workflow + reasoner + verifier + answerer
        else:
            return 3  # workflow + reasoner + answerer (all other workflows)
    
    def _execute_workflow(self) -> tuple:
        """Execute the configured workflow and return (final_text, info)."""
        # Get workflow instance and execute
        workflow = self._get_workflow(self.workflow_depth)
        
        # Get token counts
        agent1_tokens = self.TOKEN_BUDGETS["reasoner"][self.agent1_budget] if self.agent1_budget is not None else 256
        agent2_tokens = self.TOKEN_BUDGETS["verifier"][self.agent2_budget] if self.agent2_budget is not None else 128
        answerer_tokens = self.TOKEN_BUDGETS["answerer"][self.answerer_budget]
        
        # Decode tools
        agent1_tools = self._decode_tools(self.agent1_tools) if self.agent1_tools is not None else []
        agent2_tools = self._decode_tools(self.agent2_tools) if self.agent2_tools is not None else []
        
        # Execute workflow
        final_text, exec_info = workflow.execute(
            self.current_q,
            agent1_tools,
            self.agent1_budget if self.agent1_budget is not None else 1,
            agent2_tools,
            self.agent2_budget if self.agent2_budget is not None else 1,
            self.answerer_budget,
            agent1_tokens,
            agent2_tokens,
            answerer_tokens
        )
        
        # Add tool lists to execution info
        exec_info["agent1_tools"] = agent1_tools
        exec_info["agent2_tools"] = agent2_tools
        
        return final_text, exec_info
    
    def _execute_agent_step(self, method, question, context=None, tools=None, tokens=256):
        """Execute a single agent step with optional tool execution."""
        tools = tools or []
        
        if context:
            response = method(question, context, tools=tools, tokens=tokens)
        else:
            response = method(question, tools=tools, tokens=tokens)
        
        # Parse and execute tools if any
        if tools:
            tool_result = self.tools.parse_and_execute(response, tools)
            if tool_result:
                response += f"\nTool Output: {tool_result}"
        
        return response
