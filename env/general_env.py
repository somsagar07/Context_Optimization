import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
sys.path.append('..')

from agents_system import LLMWorker
from agents_system.worker import OpenRouterWorker
from agents_system.workflows import get_workflow
from tools import ToolRegistry
from utils import get_dataset_loader


class GeneralAgentEnv(gym.Env):
    """
    Single-Step RL Environment for optimizing LLM agent configurations.
    
    Action Space: [9, 16, 3, 16, 3, 3] = 62,208 combinations
        0: Workflow Type [0=Direct, 1=Reason+Ans, 2=Reason+Verify+Ans,
                          3=Routing, 4=Parallel-Sectioning, 5=Parallel-Voting,
                          6=Orchestrator-Workers, 7=Evaluator-Optimizer, 8=Autonomous-Agent]
        1: Agent1 Tools [0-15: 4 tools binary encoded]
        2: Agent1 Budget [0=Low, 1=Mid, 2=High]
        3: Agent2 Tools [0-15: 4 tools binary encoded]
        4: Agent2 Budget [0=Low, 1=Mid, 2=High]
        5: Answerer Budget [0=Low, 1=Mid, 2=High]
    
    Observation: Question embedding from MetaCLIP-H14 (1024D)
    
    NOTE: This is a single-step environment (contextual bandit).
    All actions are selected at once, no temporal credit assignment.
    """
    
    def __init__(self, cfg=None, is_eval=False, use_api=False, api_model=None, hf_model=None):
        """
        Initialize the environment.
        
        Args:
            cfg: Configuration module. If None, imports default config.
            is_eval: If True, loads evaluation dataset.
            use_api: If True, use OpenRouter API instead of local HuggingFace model.
            api_model: OpenRouter model ID (e.g., "openai/gpt-4o"). Required if use_api=True.
            hf_model: HuggingFace model name. If None, uses config default.
        """
        super(GeneralAgentEnv, self).__init__()
        
        # Store config (import default if not provided)
        if cfg is None:
            from configs import load_config
            cfg = load_config("single_step")  # Default to single_step for GeneralAgentEnv
        self.cfg = cfg
        
        # Initialize worker based on API or HuggingFace mode
        if use_api:
            if not api_model:
                raise ValueError("api_model is required when use_api=True")
            self.worker = OpenRouterWorker(model_name=api_model)
        else:
            self.worker = LLMWorker(model_name=hf_model)
        
        self.tools = ToolRegistry()
        self.dataset = get_dataset_loader(cfg.DATASET_NAME, is_eval=is_eval)
        
        # Initialize workflow instances (lazy loading)
        self._workflow_instances = {}
        
        # NOTE: Updated to 9 workflows (3 original + 6 from Anthropic patterns)
        # Action space: [9 workflows, 16 agent1_tools, 3 agent1_budget, 
        #                16 agent2_tools, 3 agent2_budget, 3 answerer_budget]
        self.action_space = spaces.MultiDiscrete([9, 16, 3, 16, 3, 3])
        
        # Token budgets for each level (Low/Mid/High)
        self.TOKEN_BUDGETS = {
            "reasoner": {0: 256, 1: 512, 2: 1024},
            "verifier": {0: 128, 1: 256, 2: 512},
            "answerer": {0: 64, 1: 128, 2: 256}
        }
        
        # Observation Space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.worker.model.config.hidden_size,), 
            dtype=np.float32
        )

        self.current_q = None
        self.current_a = None

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_q, self.current_a = self.dataset.get_sample()
        
        return self.worker.get_embedding(self.current_q), {"action_mask": self._get_action_mask()}
    
    def _get_workflow(self, workflow_depth: int):
        """Get or create workflow instance for the given depth."""
        if workflow_depth not in self._workflow_instances:
            self._workflow_instances[workflow_depth] = get_workflow(
                workflow_depth, self.worker, self.tools
            )
        return self._workflow_instances[workflow_depth]
    
    def _decode_tools(self, idx: int) -> list:
        """
        Decode tool index to list of tools (binary encoding).
        3 tools = 2^3 = 8 combinations
        Bit 0: calculator, Bit 1: web_search, Bit 2: python, Bit 3: ocr_reader
        """
        tools = []
        if idx & 1: tools.append("calculator")
        if idx & 2: tools.append("web_search")
        if idx & 4: tools.append("python")
        if idx & 8: tools.append("ocr_reader") # Added this tool
        return tools

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

    def step(self, action):
        # Decode action
        workflow_depth = action[0]
        agent1_tools = self._decode_tools(action[1])
        agent1_budget = action[2]
        agent2_tools = self._decode_tools(action[3])
        agent2_budget = action[4]
        answerer_budget = action[5]
        
        # Get token counts
        agent1_tokens = self.TOKEN_BUDGETS["reasoner"][agent1_budget]
        agent2_tokens = self.TOKEN_BUDGETS["verifier"][agent2_budget]
        answerer_tokens = self.TOKEN_BUDGETS["answerer"][answerer_budget]
        
        # Get workflow instance and execute
        workflow = self._get_workflow(workflow_depth)
        
        # Execute workflow (workflow 2 already has use_verifier set in get_workflow)
        final_text, exec_info = workflow.execute(
                self.current_q, 
            agent1_tools,
            agent1_budget,
            agent2_tools,
            agent2_budget,
            answerer_budget,
            agent1_tokens,
            agent2_tokens,
            answerer_tokens
            )
        
        cost_steps = exec_info["steps"]
        tools_used_count = exec_info["tools_count"]
        total_tokens_used = exec_info["total_tokens"]
        # Calculate Reward
        correctness = self.dataset.evaluate_correctness(final_text, self.current_a)
        reward = correctness * 5.0 
        
        # Penalties (use config values)
        reward -= cost_steps * self.cfg.COST_PER_STEP
        reward -= tools_used_count * self.cfg.COST_TOOL_USAGE
        
        # Token budget penalty (normalized)
        max_tokens = 2048 + 1024 + 512  # reasoner_high + verifier_high + answerer_high
        token_penalty = (total_tokens_used / max_tokens) * self.cfg.COST_TOKEN_BUDGET
        reward -= token_penalty
        
        terminated = True
        truncated = False
        
        info = {
            "query": self.current_q,
            "correct": correctness == 1.0,
            "steps_taken": cost_steps,
            "workflow": [
                "Direct", "Reason+Ans", "Reason+Verify+Ans",
                "Routing", "Parallel-Sectioning", "Parallel-Voting",
                "Orchestrator-Workers", "Evaluator-Optimizer", "Autonomous-Agent"
            ][workflow_depth],
            "tools_used": tools_used_count,
            "agent1_tools": agent1_tools,
            "agent2_tools": agent2_tools,
            "agent1_budget": ["Low", "Mid", "High"][agent1_budget],
            "agent2_budget": ["Low", "Mid", "High"][agent2_budget],
            "answerer_budget": ["Low", "Mid", "High"][answerer_budget],
            "total_tokens": total_tokens_used,
            "final_answer": final_text,
            "ground_truth": self.current_a,
            "action_mask": self._get_action_mask()
        }
        
        return self.worker.get_embedding(final_text), reward, terminated, truncated, info
