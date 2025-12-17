# env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from worker import LLMWorker
from tools import ToolRegistry
from data_loader import get_dataset_loader
import config

class GeneralAgentEnv(gym.Env):
    def __init__(self):
        super(GeneralAgentEnv, self).__init__()
        
        self.worker = LLMWorker()
        self.tools = ToolRegistry()
        self.dataset = get_dataset_loader(config.DATASET_NAME)
        
        # Action Space:
        # 0: Workflow Depth [0, 1, 2]
        # 1: Reasoner Tools [0=None, 1=Calc, 2=Search, 3=Both]
        # 2: Verifier Tools [0=None, 1=Calc, 2=Search, 3=Both]
        # 3: Context Budget [0=Low, 1=High]
        self.action_space = spaces.MultiDiscrete([3, 4, 4, 2])
        
        # Observation: Embedding of the question
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
    
    def _decode_tools(self, idx):
        if idx == 1: return ["calculator"]
        if idx == 2: return ["web_search"]
        if idx == 3: return ["calculator", "web_search"]
        return []

    def _execute_agent_step(self, method, question, context=None, tools=[], tokens=256):
        """Runs generation + tool execution loop"""
        if context:
            response = method(question, context, tools=tools, tokens=tokens)
        else:
            response = method(question, tools=tools, tokens=tokens)
            
        # Parse and execute tools if any
        if tools:
            tool_result = self.tools.parse_and_execute(response, tools)
            if tool_result:
                # If tool used, simple approach: append result
                response += f"\nTool Output: {tool_result}"
        return response

    def step(self, action):
        workflow_depth = action[0]
        reasoner_tools = self._decode_tools(action[1])
        verifier_tools = self._decode_tools(action[2])
        high_budget = action[3] == 1
        
        # Tokens scaler
        t_mult = 2 if high_budget else 1
        
        final_text = ""
        cost_steps = 0
        tools_used_count = 0
        
        # --- EXECUTE WORKFLOW ---
        if workflow_depth == 0:
            # Direct Answer (No Tools allowed for speed)
            final_text = self._execute_agent_step(
                self.worker.answer_direct, 
                self.current_q, 
                tokens=128 * t_mult
            )
            cost_steps = 1
            
        elif workflow_depth == 1:
            # Reason -> Answer
            reasoning = self._execute_agent_step(
                self.worker.reason,
                self.current_q,
                tools=reasoner_tools,
                tokens=512 * t_mult
            )
            if reasoner_tools: tools_used_count += 1
            
            final_text = self._execute_agent_step(
                self.worker.answer_with_context,
                self.current_q,
                context=reasoning,
                tokens=128 * t_mult
            )
            cost_steps = 2 
            
        elif workflow_depth == 2:
            # Reason -> Verify -> Answer
            reasoning = self._execute_agent_step(
                self.worker.reason,
                self.current_q,
                tools=reasoner_tools,
                tokens=512 * t_mult
            )
            if reasoner_tools: tools_used_count += 1

            critique = self._execute_agent_step(
                self.worker.verify,
                self.current_q,
                context=reasoning, # Verify takes reasoning as context
                tools=verifier_tools,
                tokens=256 * t_mult
            )
            if verifier_tools: tools_used_count += 1
            
            context = f"Reasoning: {reasoning}\nReview: {critique}"
            final_text = self._execute_agent_step(
                self.worker.answer_with_context,
                self.current_q,
                context=context,
                tokens=128 * t_mult
            )
            cost_steps = 3
        
        # 3. Calculate Reward
        # A. Performance
        correctness = self.dataset.evaluate_correctness(final_text, self.current_a)
        reward = correctness * 2.0 
        
        # B. Penalties 
        reward -= cost_steps * config.COST_PER_STEP
        reward -= tools_used_count * config.COST_TOOL_USAGE
        if high_budget: reward -= 0.05
        
        terminated = True
        truncated = False
        
        info = {
            "query": self.current_q,
            "correct": correctness == 1.0,
            "steps_taken": cost_steps,
            "workflow": ["Direct", "Reason+Ans", "Reason+Verify+Ans"][workflow_depth],
            "tools_loaded": len(reasoner_tools) + len(verifier_tools),
            "budget": "High" if high_budget else "Low"
        }
        
        return self.worker.get_embedding(final_text), reward, terminated, truncated, info