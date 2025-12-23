import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
sys.path.append('..')

from agents_system import LLMWorker
from tools import ToolRegistry
from utils import get_dataset_loader


class MultiStepAgentEnv(gym.Env):
    """
    Multi-Step RL Environment for LLM Agent Configuration.
    
    Instead of choosing all 6 action dimensions at once (5,184 combinations),
    the agent makes SEQUENTIAL decisions, enabling proper credit assignment.
    
    Episode Structure (4 steps max):
        Step 0: Choose workflow depth [0, 1, 2] → action_space = 3
        Step 1: Choose reasoner config (tools + budget) → action_space = 24 (8 tools × 3 budgets)
        Step 2: Choose verifier config (if depth=2) → action_space = 24
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
            import config as cfg
        self.cfg = cfg
        
        self.worker = LLMWorker()
        self.tools = ToolRegistry()
        self.dataset = get_dataset_loader(cfg.DATASET_NAME)
        
        # Action space: max across all stages (reasoner/verifier have 24 = 8×3)
        # We'll mask invalid actions per stage
        self.action_space = spaces.Discrete(24)
        
        # Token budgets
        self.TOKEN_BUDGETS = {
            "reasoner": {0: 256, 1: 512, 2: 1024},
            "verifier": {0: 128, 1: 256, 2: 512},
            "answerer": {0: 64, 1: 128, 2: 256}
        }
        
        # Observation: question embedding + stage encoding + partial decisions
        # - question_embedding: hidden_size (e.g., 1536 for Qwen2.5-1.5B)
        # - stage_onehot: 4 dims (which decision stage we're in)
        # - workflow_chosen: 3 dims (one-hot of depth)
        # - reasoner_chosen: 24 dims (one-hot of tools×budget)
        # - verifier_chosen: 24 dims (one-hot of tools×budget)
        hidden_size = self.worker.model.config.hidden_size
        obs_size = hidden_size + 4 + 3 + 24 + 24  # Total observation
        
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
        self.reasoner_tools = None
        self.reasoner_budget = None
        self.verifier_tools = None
        self.verifier_budget = None
        self.answerer_budget = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Get new question
        self.current_q, self.current_a = self.dataset.get_sample()
        self.question_embedding = self.worker.get_embedding(self.current_q)
        
        # Reset stage and decisions
        self.stage = self.STAGE_WORKFLOW
        self.workflow_depth = None
        self.reasoner_tools = None
        self.reasoner_budget = None
        self.verifier_tools = None
        self.verifier_budget = None
        self.answerer_budget = None
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Build observation: question embedding + stage + partial decisions."""
        # Stage one-hot (4 stages)
        stage_onehot = np.zeros(4, dtype=np.float32)
        stage_onehot[min(self.stage, 3)] = 1.0
        
        # Workflow choice one-hot (3 options)
        workflow_onehot = np.zeros(3, dtype=np.float32)
        if self.workflow_depth is not None:
            workflow_onehot[self.workflow_depth] = 1.0
        
        # Reasoner choice one-hot (24 = 8 tools × 3 budgets)
        reasoner_onehot = np.zeros(24, dtype=np.float32)
        if self.reasoner_tools is not None and self.reasoner_budget is not None:
            idx = self._encode_config(self.reasoner_tools, self.reasoner_budget)
            reasoner_onehot[idx] = 1.0
        
        # Verifier choice one-hot (24)
        verifier_onehot = np.zeros(24, dtype=np.float32)
        if self.verifier_tools is not None and self.verifier_budget is not None:
            idx = self._encode_config(self.verifier_tools, self.verifier_budget)
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
        """Encode tools (0-7) and budget (0-2) into single index (0-23)."""
        return tools_idx * 3 + budget_idx
    
    def _decode_config(self, action: int) -> tuple:
        """Decode action into tools index and budget index."""
        tools_idx = action // 3
        budget_idx = action % 3
        return tools_idx, budget_idx
    
    def _decode_tools(self, idx: int) -> list:
        """Decode tool index to list of tool names."""
        tools = []
        if idx & 1: tools.append("calculator")
        if idx & 2: tools.append("web_search")
        if idx & 4: tools.append("python")
        return tools
    
    def _get_valid_actions(self) -> np.ndarray:
        """Return mask of valid actions for current stage."""
        mask = np.zeros(24, dtype=np.float32)
        
        if self.stage == self.STAGE_WORKFLOW:
            # Only 3 valid: [0, 1, 2] for workflow depth
            mask[:3] = 1.0
        elif self.stage == self.STAGE_REASONER:
            # 24 valid: all combinations of tools (8) × budget (3)
            mask[:24] = 1.0
        elif self.stage == self.STAGE_VERIFIER:
            # 24 valid (only reached if depth == 2)
            mask[:24] = 1.0
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
            self.workflow_depth = min(action, 2)
            
            # Small shaping reward: prefer simpler workflows for efficiency
            # (will be outweighed by correctness if complex is needed)
            efficiency_bonus = (2 - self.workflow_depth) * 0.02
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
            tools_idx, budget_idx = self._decode_config(min(action, 23))
            self.reasoner_tools = tools_idx
            self.reasoner_budget = budget_idx
            
            # Small shaping: prefer lower budgets (efficiency)
            efficiency_bonus = (2 - budget_idx) * 0.01
            reward = efficiency_bonus
            
            # Next stage
            if self.workflow_depth == 2:
                self.stage = self.STAGE_VERIFIER
            else:
                self.stage = self.STAGE_ANSWERER
                
        elif self.stage == self.STAGE_VERIFIER:
            # Choose verifier tools + budget
            tools_idx, budget_idx = self._decode_config(min(action, 23))
            self.verifier_tools = tools_idx
            self.verifier_budget = budget_idx
            
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
            max_tokens = 1024 + 512 + 256
            token_penalty = (execution_info["total_tokens"] / max_tokens) * self.cfg.COST_TOKEN_BUDGET
            reward -= token_penalty
            
            terminated = True
            info = {
                "query": self.current_q,
                "correct": correctness == 1.0,
                "workflow": ["Direct", "Reason+Ans", "Reason+Verify+Ans"][self.workflow_depth],
                "steps_taken": execution_info["steps"],
                "tools_used": execution_info["tools_count"],
                "reasoner_tools": execution_info.get("reasoner_tools", []),
                "verifier_tools": execution_info.get("verifier_tools", []),
                "reasoner_budget": ["Low", "Mid", "High"][self.reasoner_budget] if self.reasoner_budget is not None else "N/A",
                "verifier_budget": ["Low", "Mid", "High"][self.verifier_budget] if self.verifier_budget is not None else "N/A",
                "answerer_budget": ["Low", "Mid", "High"][self.answerer_budget],
                "total_tokens": execution_info["total_tokens"],
                "episode_length": self._get_episode_length(),
            }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_episode_length(self) -> int:
        """Get number of decision steps for this episode."""
        if self.workflow_depth == 0:
            return 2  # workflow + answerer
        elif self.workflow_depth == 1:
            return 3  # workflow + reasoner + answerer
        else:
            return 4  # workflow + reasoner + verifier + answerer
    
    def _execute_workflow(self) -> tuple:
        """Execute the configured workflow and return (final_text, info)."""
        # Get token counts
        answerer_tokens = self.TOKEN_BUDGETS["answerer"][self.answerer_budget]
        
        execution_info = {
            "steps": 0,
            "tools_count": 0,
            "total_tokens": 0,
            "reasoner_tools": [],
            "verifier_tools": [],
        }
        
        if self.workflow_depth == 0:
            # Direct Answer
            final_text = self._execute_agent_step(
                self.worker.answer_direct,
                self.current_q,
                tokens=answerer_tokens
            )
            execution_info["steps"] = 1
            execution_info["total_tokens"] = answerer_tokens
            
        elif self.workflow_depth == 1:
            # Reason -> Answer
            reasoner_tokens = self.TOKEN_BUDGETS["reasoner"][self.reasoner_budget]
            reasoner_tools = self._decode_tools(self.reasoner_tools)
            execution_info["reasoner_tools"] = reasoner_tools
            
            reasoning = self._execute_agent_step(
                self.worker.reason,
                self.current_q,
                tools=reasoner_tools,
                tokens=reasoner_tokens
            )
            execution_info["tools_count"] += len(reasoner_tools)
            
            final_text = self._execute_agent_step(
                self.worker.answer_with_context,
                self.current_q,
                context=reasoning,
                tokens=answerer_tokens
            )
            execution_info["steps"] = 2
            execution_info["total_tokens"] = reasoner_tokens + answerer_tokens
            
        else:  # workflow_depth == 2
            # Reason -> Verify -> Answer
            reasoner_tokens = self.TOKEN_BUDGETS["reasoner"][self.reasoner_budget]
            verifier_tokens = self.TOKEN_BUDGETS["verifier"][self.verifier_budget]
            reasoner_tools = self._decode_tools(self.reasoner_tools)
            verifier_tools = self._decode_tools(self.verifier_tools)
            execution_info["reasoner_tools"] = reasoner_tools
            execution_info["verifier_tools"] = verifier_tools
            
            reasoning = self._execute_agent_step(
                self.worker.reason,
                self.current_q,
                tools=reasoner_tools,
                tokens=reasoner_tokens
            )
            execution_info["tools_count"] += len(reasoner_tools)
            
            critique = self._execute_agent_step(
                self.worker.verify,
                self.current_q,
                context=reasoning,
                tools=verifier_tools,
                tokens=verifier_tokens
            )
            execution_info["tools_count"] += len(verifier_tools)
            
            context = f"Reasoning: {reasoning}\nReview: {critique}"
            final_text = self._execute_agent_step(
                self.worker.answer_with_context,
                self.current_q,
                context=context,
                tokens=answerer_tokens
            )
            execution_info["steps"] = 3
            execution_info["total_tokens"] = reasoner_tokens + verifier_tokens + answerer_tokens
        
        return final_text, execution_info
    
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
