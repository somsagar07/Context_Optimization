import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
sys.path.append('..')

from agents_system import LLMWorker
from tools import ToolRegistry
from utils import get_dataset_loader
import config

class GeneralAgentEnv(gym.Env):
    """
    RL Environment for optimizing LLM agent configurations.
    
    Action Space: [3, 8, 3, 8, 3, 3] = 5,184 combinations
        0: Workflow Depth [0=Direct, 1=Reason+Ans, 2=Reason+Verify+Ans]
        1: Reasoner Tools [0-7: 3 tools binary encoded]
        2: Reasoner Budget [0=Low, 1=Mid, 2=High]
        3: Verifier Tools [0-7: 3 tools binary encoded]
        4: Verifier Budget [0=Low, 1=Mid, 2=High]
        5: Answerer Budget [0=Low, 1=Mid, 2=High]
    
    Observation: Question embedding from LLM hidden states
    """
    
    def __init__(self):
        super(GeneralAgentEnv, self).__init__()
        
        self.worker = LLMWorker()
        self.tools = ToolRegistry()
        self.dataset = get_dataset_loader(config.DATASET_NAME)
        
        # Action Space
        self.action_space = spaces.MultiDiscrete([3, 8, 3, 8, 3, 3])
        
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_q, self.current_a = self.dataset.get_sample()
        return self.worker.get_embedding(self.current_q), {}
    
    def _decode_tools(self, idx: int) -> list:
        """
        Decode tool index to list of tools (binary encoding).
        3 tools = 2^3 = 8 combinations
        Bit 0: calculator, Bit 1: web_search, Bit 2: python
        """
        tools = []
        if idx & 1: tools.append("calculator")
        if idx & 2: tools.append("web_search")
        if idx & 4: tools.append("python")
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
        reasoner_tools = self._decode_tools(action[1])
        reasoner_budget = action[2]
        verifier_tools = self._decode_tools(action[3])
        verifier_budget = action[4]
        answerer_budget = action[5]
        
        # Get token counts
        reasoner_tokens = self.TOKEN_BUDGETS["reasoner"][reasoner_budget]
        verifier_tokens = self.TOKEN_BUDGETS["verifier"][verifier_budget]
        answerer_tokens = self.TOKEN_BUDGETS["answerer"][answerer_budget]
        
        final_text = ""
        cost_steps = 0
        tools_used_count = 0
        total_tokens_used = 0
        
        # Execute workflow based on depth
        if workflow_depth == 0:
            # Direct Answer
            final_text = self._execute_agent_step(
                self.worker.answer_direct, 
                self.current_q, 
                tokens=answerer_tokens
            )
            cost_steps = 1
            total_tokens_used = answerer_tokens
            
        elif workflow_depth == 1:
            # Reason -> Answer
            reasoning = self._execute_agent_step(
                self.worker.reason,
                self.current_q,
                tools=reasoner_tools,
                tokens=reasoner_tokens
            )
            if reasoner_tools: 
                tools_used_count += len(reasoner_tools)
            
            final_text = self._execute_agent_step(
                self.worker.answer_with_context,
                self.current_q,
                context=reasoning,
                tokens=answerer_tokens
            )
            cost_steps = 2
            total_tokens_used = reasoner_tokens + answerer_tokens
            
        elif workflow_depth == 2:
            # Reason -> Verify -> Answer
            reasoning = self._execute_agent_step(
                self.worker.reason,
                self.current_q,
                tools=reasoner_tools,
                tokens=reasoner_tokens
            )
            if reasoner_tools: 
                tools_used_count += len(reasoner_tools)

            critique = self._execute_agent_step(
                self.worker.verify,
                self.current_q,
                context=reasoning,
                tools=verifier_tools,
                tokens=verifier_tokens
            )
            if verifier_tools: 
                tools_used_count += len(verifier_tools)
            
            context = f"Reasoning: {reasoning}\nReview: {critique}"
            final_text = self._execute_agent_step(
                self.worker.answer_with_context,
                self.current_q,
                context=context,
                tokens=answerer_tokens
            )
            cost_steps = 3
            total_tokens_used = reasoner_tokens + verifier_tokens + answerer_tokens
        
        # Calculate Reward
        correctness = self.dataset.evaluate_correctness(final_text, self.current_a)
        reward = correctness * 2.0 
        
        # Penalties
        reward -= cost_steps * config.COST_PER_STEP
        reward -= tools_used_count * config.COST_TOOL_USAGE
        
        # Token budget penalty (normalized)
        max_tokens = 1024 + 512 + 256
        token_penalty = (total_tokens_used / max_tokens) * config.COST_TOKEN_BUDGET
        reward -= token_penalty
        
        terminated = True
        truncated = False
        
        info = {
            "query": self.current_q,
            "correct": correctness == 1.0,
            "steps_taken": cost_steps,
            "workflow": ["Direct", "Reason+Ans", "Reason+Verify+Ans"][workflow_depth],
            "tools_used": tools_used_count,
            "reasoner_tools": reasoner_tools,
            "verifier_tools": verifier_tools,
            "reasoner_budget": ["Low", "Mid", "High"][reasoner_budget],
            "verifier_budget": ["Low", "Mid", "High"][verifier_budget],
            "answerer_budget": ["Low", "Mid", "High"][answerer_budget],
            "total_tokens": total_tokens_used
        }
        
        return self.worker.get_embedding(final_text), reward, terminated, truncated, info

