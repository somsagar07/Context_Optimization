"""
Prompt Environment - Low-Level Policy (HIRO-style Worker)

Multi-step environment for sequential prompt selection.
Receives structure decisions from the high-level policy and:
1. Sequentially selects prompts for each agent (Reasoner → Verifier → Answerer)
2. Executes the full workflow
3. Returns final reward based on correctness and efficiency

This separation allows:
- Structure policy to learn WHAT configuration to use
- Prompt policy to learn HOW to prompt each agent
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
sys.path.append('..')

from agents_system import LLMWorker
from tools import ToolRegistry
from utils import get_dataset_loader
from prompts.library import (
    PROMPT_ATOMS, NUM_ATOMS, build_prompt_suffix,
    NUM_REASONER_ATOMS, NUM_VERIFIER_ATOMS, NUM_ANSWERER_ATOMS
)
import re

class PromptEnv(gym.Env):
    """
    Low-level policy environment for prompt selection.
    
    Given a structure decision (workflow, tools, budgets), this environment:
    1. Sequentially selects prompts for each relevant agent
    2. Executes the configured workflow with selected prompts
    3. Returns reward based on correctness and efficiency
    
    Action Space: Discrete(max_atoms)
        - 0 = DONE (stop selecting prompts for current agent)
        - 1-N = Select that prompt atom
    
    Observation Space:
        - Question embedding
        - Structure decisions (from high-level policy)
        - Current prompt stage (reasoner/verifier/answerer)
        - Current prompt step (which prompt we're selecting)
        - Already selected prompts (per agent)
    """
    
    # Prompt selection stages
    PROMPT_STAGE_REASONER = 0
    PROMPT_STAGE_VERIFIER = 1
    PROMPT_STAGE_ANSWERER = 2
    
    # Token budget mappings
    TOKEN_BUDGETS = {
        "reasoner": {0: 256, 1: 512, 2: 1024},
        "verifier": {0: 128, 1: 256, 2: 512},
        "answerer": {0: 64, 1: 128, 2: 256}
    }
    
    def __init__(self, cfg=None, is_eval=False):
        super().__init__()
        
        # Store config
        if cfg is None:
            from configs import load_config
            cfg = load_config("hierarchical")
        self.cfg = cfg
        
        # Prompt configuration
        self.MAX_PROMPTS_PER_AGENT = getattr(cfg, 'MAX_PROMPTS_PER_AGENT', 3)
        
        # Number of prompt atoms per agent
        self.num_reasoner_atoms = NUM_REASONER_ATOMS
        self.num_verifier_atoms = NUM_VERIFIER_ATOMS
        self.num_answerer_atoms = NUM_ANSWERER_ATOMS
        self.max_prompt_atoms = max(NUM_REASONER_ATOMS, NUM_VERIFIER_ATOMS, NUM_ANSWERER_ATOMS)
        
        # Action space: Discrete for prompt selection
        self.action_space = spaces.Discrete(self.max_prompt_atoms)
        
        # Initialize components
        self.worker = LLMWorker()
        self.tools = ToolRegistry()
        self.dataset = get_dataset_loader(cfg.DATASET_NAME, is_eval=is_eval)
        
        # Observation space components
        hidden_size = self.worker.model.config.hidden_size
        obs_size = (
            hidden_size +                       # Question embedding
            6 +                                 # Structure decisions (normalized)
            3 +                                 # Prompt stage one-hot
            self.MAX_PROMPTS_PER_AGENT +        # Prompt step one-hot
            NUM_REASONER_ATOMS +                # Reasoner prompts mask
            NUM_VERIFIER_ATOMS +                # Verifier prompts mask
            NUM_ANSWERER_ATOMS                  # Answerer prompts mask
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # Episode state - set externally via set_structure()
        self.current_q = None
        self.current_a = None
        self.question_embedding = None
        
        # Structure decisions (from high-level policy)
        self.workflow_depth = 0
        self.reasoner_tools_idx = 0
        self.reasoner_budget_idx = 0
        self.verifier_tools_idx = 0
        self.verifier_budget_idx = 0
        self.answerer_budget_idx = 0
        
        # Prompt selection state
        self.prompt_stage = self.PROMPT_STAGE_REASONER
        self.prompt_step = 0
        self.selected_prompts = {
            "reasoner": [],
            "verifier": [],
            "answerer": [],
        }
        
        # Flag to track if structure has been set
        self._structure_set = False
        
    def set_structure(self, question: str, answer: str, embedding: np.ndarray, structure: dict):
        """
        Set the structure decision from the high-level policy.
        Must be called before reset() when using externally.
        
        Args:
            question: The current question
            answer: The ground truth answer
            embedding: Pre-computed question embedding
            structure: Dict with workflow_depth, tools, budgets
        """
        self.current_q = question
        self.current_a = answer
        self.question_embedding = embedding
        
        self.workflow_depth = structure["workflow_depth"]
        self.reasoner_tools_idx = structure["reasoner_tools_idx"]
        self.reasoner_budget_idx = structure["reasoner_budget_idx"]
        self.verifier_tools_idx = structure["verifier_tools_idx"]
        self.verifier_budget_idx = structure["verifier_budget_idx"]
        self.answerer_budget_idx = structure["answerer_budget_idx"]
        
        self._structure_set = True
        
    def reset(self, seed=None, options=None):
        """
        Reset prompt selection state.
        
        If structure not set externally, samples a new question
        and uses default structure (for standalone testing).
        """
        super().reset(seed=seed)
        
        # If structure not set, sample new question (standalone mode)
        if not self._structure_set:
            self.current_q, self.current_a = self.dataset.get_sample()
            self.question_embedding = self.worker.get_embedding(self.current_q)
            # Default structure
            self.workflow_depth = 0
            self.reasoner_tools_idx = 0
            self.reasoner_budget_idx = 1
            self.verifier_tools_idx = 0
            self.verifier_budget_idx = 1
            self.answerer_budget_idx = 1
        
        # Reset prompt selection
        if self.workflow_depth == 0:
            # Direct: only answerer prompts
            self.prompt_stage = self.PROMPT_STAGE_ANSWERER
        else:
            # Start with reasoner
            self.prompt_stage = self.PROMPT_STAGE_REASONER
            
        self.prompt_step = 0
        self.selected_prompts = {"reasoner": [], "verifier": [], "answerer": []}
        
        # Reset structure flag for next episode
        self._structure_set = False
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Build observation vector."""
        # Structure decisions (normalized)
        structure_vec = np.array([
            self.workflow_depth / 2.0,
            self.reasoner_tools_idx / 7.0,
            self.reasoner_budget_idx / 2.0,
            self.verifier_tools_idx / 7.0,
            self.verifier_budget_idx / 2.0,
            self.answerer_budget_idx / 2.0,
        ], dtype=np.float32)
        
        # Prompt stage one-hot
        stage_onehot = np.zeros(3, dtype=np.float32)
        stage_onehot[self.prompt_stage] = 1.0
        
        # Prompt step one-hot
        step_onehot = np.zeros(self.MAX_PROMPTS_PER_AGENT, dtype=np.float32)
        if self.prompt_step < self.MAX_PROMPTS_PER_AGENT:
            step_onehot[self.prompt_step] = 1.0
        
        # Selected prompts masks
        reasoner_mask = np.zeros(NUM_REASONER_ATOMS, dtype=np.float32)
        for idx in self.selected_prompts["reasoner"]:
            if idx < NUM_REASONER_ATOMS:
                reasoner_mask[idx] = 1.0
        
        verifier_mask = np.zeros(NUM_VERIFIER_ATOMS, dtype=np.float32)
        for idx in self.selected_prompts["verifier"]:
            if idx < NUM_VERIFIER_ATOMS:
                verifier_mask[idx] = 1.0
        
        answerer_mask = np.zeros(NUM_ANSWERER_ATOMS, dtype=np.float32)
        for idx in self.selected_prompts["answerer"]:
            if idx < NUM_ANSWERER_ATOMS:
                answerer_mask[idx] = 1.0
        
        # Concatenate all
        obs = np.concatenate([
            self.question_embedding,
            structure_vec,
            stage_onehot,
            step_onehot,
            reasoner_mask,
            verifier_mask,
            answerer_mask,
        ]).astype(np.float32)
        
        return obs
    
    def step(self, action):
        """Execute prompt selection step."""
        action = int(action)
        reward = 0.0
        terminated = False
        info = {}
        
        # Determine current agent and num atoms
        if self.prompt_stage == self.PROMPT_STAGE_REASONER:
            agent = "reasoner"
            num_atoms = NUM_REASONER_ATOMS
        elif self.prompt_stage == self.PROMPT_STAGE_VERIFIER:
            agent = "verifier"
            num_atoms = NUM_VERIFIER_ATOMS
        else:
            agent = "answerer"
            num_atoms = NUM_ANSWERER_ATOMS
        
        # Clamp action to valid range
        action = min(action, num_atoms - 1)
        
        # Action 0 = DONE with this agent's prompts
        if action == 0 or self.prompt_step >= self.MAX_PROMPTS_PER_AGENT:
            # No efficiency reward - let final correctness determine value
            # This prevents the policy from being biased toward skipping prompts
            reward = 0.0
            self._advance_prompt_stage()
        else:
            # Select this prompt (if not already selected)
            if action not in self.selected_prompts[agent]:
                self.selected_prompts[agent].append(action)
                reward = 0.02  # Small positive reward for selecting a prompt
            else:
                reward = -0.01  # Small penalty for selecting duplicate
            self.prompt_step += 1
            
            if self.prompt_step >= self.MAX_PROMPTS_PER_AGENT:
                self._advance_prompt_stage()
        
        # Check if we should execute
        if self._all_prompts_done():
            final_text, exec_info = self._execute_workflow()
            
            # Calculate final reward
            # Base correctness reward
            correctness = self.dataset.evaluate_correctness(final_text, self.current_a)
            reward += correctness * 5.0
            
            # Check the dataset and apply reward/penalties
            if self.dataset.name in ["gaia"]:
                # Reward for valid code execution (encourages syntax correctness and tool use)
                if exec_info["valid_code_count"] > 0:
                    reward += 0.2 * exec_info["valid_code_count"]
                
                # Reward for accessing the specific file (encourages file usage)
                if exec_info["file_access_count"] > 0:
                    reward += 0.5
                
                # Reward for providing a final answer (encourages format compliance)
                if "Final Answer:" in final_text:
                    reward += 0.1
            
            # Cost penalties
            reward -= exec_info["steps"] * self.cfg.COST_PER_STEP
            reward -= exec_info["tools_count"] * self.cfg.COST_TOOL_USAGE
            
            # Token penalty
            max_tokens = 1024 + 512 + 256
            token_penalty = (exec_info["total_tokens"] / max_tokens) * self.cfg.COST_TOKEN_BUDGET
            reward -= token_penalty
            
            terminated = True
            info = {
                "question": self.current_q,
                "correct": correctness == 1.0,
                "workflow": ["Direct", "Reason+Ans", "Reason+Verify+Ans"][self.workflow_depth],
                "steps_taken": exec_info["steps"],
                "tools_used": exec_info["tools_count"],
                "reasoner_prompts": self.selected_prompts["reasoner"],
                "verifier_prompts": self.selected_prompts["verifier"],
                "answerer_prompts": self.selected_prompts["answerer"],
                "total_tokens": exec_info["total_tokens"],
                # Budget info
                "reasoner_budget": ["Low", "Mid", "High"][self.reasoner_budget_idx] if self.workflow_depth >= 1 else "N/A",
                "verifier_budget": ["Low", "Mid", "High"][self.verifier_budget_idx] if self.workflow_depth == 2 else "N/A",
                "answerer_budget": ["Low", "Mid", "High"][self.answerer_budget_idx],
                "final_answer": final_text,
                "ground_truth": self.current_a,
            }
        
        return self._get_observation(), reward, terminated, False, info
    
    def _advance_prompt_stage(self):
        """Move to next prompt stage."""
        self.prompt_step = 0
        
        if self.prompt_stage == self.PROMPT_STAGE_REASONER:
            if self.workflow_depth == 2:
                self.prompt_stage = self.PROMPT_STAGE_VERIFIER
            else:
                self.prompt_stage = self.PROMPT_STAGE_ANSWERER
        elif self.prompt_stage == self.PROMPT_STAGE_VERIFIER:
            self.prompt_stage = self.PROMPT_STAGE_ANSWERER
        elif self.prompt_stage == self.PROMPT_STAGE_ANSWERER:
            # Mark as done
            self.prompt_stage = -1  # Invalid stage = done
    
    def _all_prompts_done(self):
        """Check if all prompt stages are complete."""
        return self.prompt_stage == -1
    
    def _decode_tools(self, idx: int) -> list:
        """Decode tool index to list of tool names."""
        tools = []
        if idx & 1: tools.append("calculator")
        if idx & 2: tools.append("web_search")
        if idx & 4: tools.append("python")
        if idx & 8: tools.append("ocr_reader")
        return tools
    
    def _process_tool_calls(self, text_response: str, allowed_tools: list) -> tuple:
        """
        Manually parse and execute tools to track dense reward metrics.
        Returns: (updated_text, stats_dict)
        """
        stats = {
            "tool_calls": 0,
            "valid_code": False,
            "file_access": False
        }
        
        # 1. Identify the target file from the prompt (if any)
        target_file = None
        if self.current_q:
            match = re.search(r"File Attachment:\s*(.+)", self.current_q)
            if match:
                target_file = match.group(1).strip()

        # 2. Parse TOOL: ... || QUERY: ... pattern
        # This matches the format expected by your ToolRegistry
        tool_matches = list(re.finditer(r"TOOL:\s*(\w+)\s*\|\|\s*QUERY:\s*(.*)", text_response))
        
        updated_text = text_response
        
        for match in tool_matches:
            t_name, t_query = match.groups()
            t_name = t_name.strip().lower()
            t_query = t_query.strip()
            
            if t_name in allowed_tools:
                stats["tool_calls"] += 1
                
                # METRIC: File Access
                # Check if the generated code/query references the file path
                if target_file and target_file in t_query:
                    stats["file_access"] = True
                
                # Execute the tool
                # We assume self.tools.execute(name, query) exists
                try:
                    tool_result = self.tools.execute(t_name, t_query)
                except Exception as e:
                    tool_result = f"Error executing tool: {e}"

                # METRIC: Valid Code
                # If python tool runs without "Error" or "Syntax Error", it's valid
                if t_name == "python":
                    if "Error:" not in tool_result and "Syntax Error:" not in tool_result:
                        stats["valid_code"] = True
                
                updated_text += f"\nTool Output: {tool_result}"
                
        return updated_text, stats
    
    def _execute_workflow(self) -> tuple:
        """Execute the configured workflow and return (final_text, info)."""
        # Build prompt suffixes
        reasoner_suffix = build_prompt_suffix("reasoner", self.selected_prompts["reasoner"])
        verifier_suffix = build_prompt_suffix("verifier", self.selected_prompts["verifier"])
        answerer_suffix = build_prompt_suffix("answerer", self.selected_prompts["answerer"])
        
        # Get token counts
        reasoner_tokens = self.TOKEN_BUDGETS["reasoner"][self.reasoner_budget_idx]
        verifier_tokens = self.TOKEN_BUDGETS["verifier"][self.verifier_budget_idx]
        answerer_tokens = self.TOKEN_BUDGETS["answerer"][self.answerer_budget_idx]
        
        # Get tools
        reasoner_tools = self._decode_tools(self.reasoner_tools_idx)
        verifier_tools = self._decode_tools(self.verifier_tools_idx)
        
        exec_info = {
            "steps": 0,
            "tools_count": 0,
            "total_tokens": 0,
            "valid_code_count": 0,
            "file_access_count": 0,
        }
        
        # Helper to merge stats
        def merge_stats(stats):
            exec_info["tools_count"] += stats["tool_calls"]
            if stats["valid_code"]: exec_info["valid_code_count"] += 1
            if stats["file_access"]: exec_info["file_access_count"] += 1
        
        if self.workflow_depth == 0:
            # Direct Answer
            final_text = self.worker.answer_direct(
                self.current_q,
                tools=[],
                tokens=answerer_tokens,
                prompt_suffix=answerer_suffix
            )
            exec_info["steps"] = 1
            exec_info["total_tokens"] = answerer_tokens
            
        elif self.workflow_depth == 1:
            # Reason -> Answer
            reasoning = self.worker.reason(
                self.current_q,
                tools=reasoner_tools,
                tokens=reasoner_tokens,
                prompt_suffix=reasoner_suffix
            )
            exec_info["tools_count"] += len(reasoner_tools)
            
            # Execute tools if any
            if reasoner_tools:
                # tool_result = self.tools.parse_and_execute(reasoning, reasoner_tools)
                # if tool_result:
                #     reasoning += f"\nTool Output: {tool_result}"
                reasoning, stats = self._process_tool_calls(reasoning, reasoner_tools)
                merge_stats(stats)
            
            final_text = self.worker.answer_with_context(
                self.current_q,
                context=reasoning,
                tools=[],
                tokens=answerer_tokens,
                prompt_suffix=answerer_suffix
            )
            exec_info["steps"] = 2
            exec_info["total_tokens"] = reasoner_tokens + answerer_tokens
            
        else:  # workflow_depth == 2
            # Reason -> Verify -> Answer
            reasoning = self.worker.reason(
                self.current_q,
                tools=reasoner_tools,
                tokens=reasoner_tokens,
                prompt_suffix=reasoner_suffix
            )
            exec_info["tools_count"] += len(reasoner_tools)
            
            if reasoner_tools:
                # tool_result = self.tools.parse_and_execute(reasoning, reasoner_tools)
                # if tool_result:
                #     reasoning += f"\nTool Output: {tool_result}"
                reasoning, stats = self._process_tool_calls(reasoning, reasoner_tools)
                merge_stats(stats)
            
            critique = self.worker.verify(
                self.current_q,
                reasoning=reasoning,
                tools=verifier_tools,
                tokens=verifier_tokens,
                prompt_suffix=verifier_suffix
            )
            exec_info["tools_count"] += len(verifier_tools)
            
            if verifier_tools:
                # tool_result = self.tools.parse_and_execute(critique, verifier_tools)
                # if tool_result:
                #     critique += f"\nTool Output: {tool_result}"
                critique, stats = self._process_tool_calls(critique, verifier_tools)
                merge_stats(stats)
            
            context = f"Reasoning: {reasoning}\nReview: {critique}"
            final_text = self.worker.answer_with_context(
                self.current_q,
                context=context,
                tokens=answerer_tokens,
                prompt_suffix=answerer_suffix
            )
            exec_info["steps"] = 3
            exec_info["total_tokens"] = reasoner_tokens + verifier_tokens + answerer_tokens
        
        return final_text, exec_info

