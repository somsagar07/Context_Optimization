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
from agents_system.workflows import get_workflow
from tools import ToolRegistry
from utils import get_dataset_loader
from prompts.library import (
    PROMPT_ATOMS, NUM_ATOMS, build_prompt_suffix,
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
        self.num_reasoner_atoms = NUM_ATOMS["reasoner"]
        self.num_verifier_atoms = NUM_ATOMS["verifier"]
        self.num_answerer_atoms = NUM_ATOMS["answerer"]
        self.max_prompt_atoms = max(self.num_reasoner_atoms, self.num_verifier_atoms, self.num_answerer_atoms)
        
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
            self.num_reasoner_atoms +                # Reasoner prompts mask
            self.num_verifier_atoms +                # Verifier prompts mask
            self.num_answerer_atoms                  # Answerer prompts mask
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
        self.agent1_tools_idx = 0
        self.agent1_budget_idx = 0
        self.agent2_tools_idx = 0
        self.agent2_budget_idx = 0
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
        self.agent1_tools_idx = structure["agent1_tools_idx"]
        self.agent1_budget_idx = structure["agent1_budget_idx"]
        self.agent2_tools_idx = structure["agent2_tools_idx"]
        self.agent2_budget_idx = structure["agent2_budget_idx"]
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
            self.agent1_tools_idx = 0
            self.agent1_budget_idx = 1
            self.agent2_tools_idx = 0
            self.agent2_budget_idx = 1
            self.answerer_budget_idx = 1
        
        # Reset prompt selection
        # Direct (0) and Parallel-Voting (5) don't need reasoner prompts
        if self.workflow_depth == 0 or self.workflow_depth == 5:
            # Direct or Parallel-Voting: only answerer prompts
            self.prompt_stage = self.PROMPT_STAGE_ANSWERER
        else:
            # All other workflows start with reasoner
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
            self.workflow_depth / 8.0,  # Normalize for 9 workflows (0-8)
            self.agent1_tools_idx / 15.0,  # Normalize for 16 tool options (0-15)
            self.agent1_budget_idx / 2.0,
            self.agent2_tools_idx / 15.0,  # Normalize for 16 tool options (0-15)
            self.agent2_budget_idx / 2.0,
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
        reasoner_mask = np.zeros(self.num_reasoner_atoms, dtype=np.float32)
        for idx in self.selected_prompts["reasoner"]:
            if idx < self.num_reasoner_atoms:
                reasoner_mask[idx] = 1.0
        
        verifier_mask = np.zeros(self.num_verifier_atoms, dtype=np.float32)
        for idx in self.selected_prompts["verifier"]:
            if idx < self.num_verifier_atoms:
                verifier_mask[idx] = 1.0
        
        answerer_mask = np.zeros(self.num_answerer_atoms, dtype=np.float32)
        for idx in self.selected_prompts["answerer"]:
            if idx < self.num_answerer_atoms:
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
            num_atoms = self.num_reasoner_atoms
        elif self.prompt_stage == self.PROMPT_STAGE_VERIFIER:
            agent = "verifier"
            num_atoms = self.num_verifier_atoms
        else:
            agent = "answerer"
            num_atoms = self.num_answerer_atoms
        
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
            
            # Calculate correctness (reward will be added in base.py to avoid double-counting)
            correctness = self.dataset.evaluate_correctness(final_text, self.current_a)
            
            # Dataset-specific bonuses (keep these as they're unique to prompt selection)
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
            
            # NOTE: Correctness reward and penalties (steps, tools, tokens) are applied
            # in base.py to avoid double-counting. Only intermediate step rewards and
            # dataset-specific bonuses are added here.
            
            terminated = True
            info = {
                "question": self.current_q,
                "correct": correctness == 1.0,
                "workflow": [
                    "Direct", "Reason+Ans", "Reason+Verify+Ans",
                    "Routing", "Parallel-Sectioning", "Parallel-Voting",
                    "Orchestrator-Workers", "Evaluator-Optimizer", "Autonomous-Agent"
                ][self.workflow_depth],
                "steps_taken": exec_info["steps"],
                "tools_used": exec_info["tools_count"],
                "reasoner_prompts": self.selected_prompts["reasoner"],
                "verifier_prompts": self.selected_prompts["verifier"],
                "answerer_prompts": self.selected_prompts["answerer"],
                "total_tokens": exec_info["total_tokens"],
                # Budget info
                "reasoner_budget": ["Low", "Mid", "High"][self.agent1_budget_idx] if self.workflow_depth in [1, 2, 3, 4, 6, 7, 8] else "N/A",
                "verifier_budget": ["Low", "Mid", "High"][self.agent2_budget_idx] if self.workflow_depth in [2, 7] else "N/A",
                "answerer_budget": ["Low", "Mid", "High"][self.answerer_budget_idx],
                "final_answer": final_text,
                "ground_truth": self.current_a,
            }
        
        return self._get_observation(), reward, terminated, False, info
    
    def _advance_prompt_stage(self):
        """Move to next prompt stage."""
        self.prompt_step = 0
        
        if self.prompt_stage == self.PROMPT_STAGE_REASONER:
            # Workflows 2 and 7 need verifier stage
            if self.workflow_depth in [2, 7]:
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
        agent1_tokens = self.TOKEN_BUDGETS["reasoner"][self.agent1_budget_idx]
        agent2_tokens = self.TOKEN_BUDGETS["verifier"][self.agent2_budget_idx]
        answerer_tokens = self.TOKEN_BUDGETS["answerer"][self.answerer_budget_idx]
        
        # Get tools
        agent1_tools = self._decode_tools(self.agent1_tools_idx)
        agent2_tools = self._decode_tools(self.agent2_tools_idx)
        
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
        
        # Direct (0) workflow
        if self.workflow_depth == 0:
            # Direct Answer (can use agent1_tools)
            answerer_tools = agent1_tools if agent1_tools else []
            final_text = self.worker.answer_direct(
                self.current_q,
                tools=answerer_tools,
                tokens=answerer_tokens,
                prompt_suffix=answerer_suffix
            )
            # Process tool calls if tools were used
            if answerer_tools:
                final_text, stats = self._process_tool_calls(final_text, answerer_tools)
                exec_info["tools_count"] += stats.get("tool_calls", 0)
                merge_stats(stats)
            exec_info["steps"] = 1
            exec_info["total_tokens"] = answerer_tokens
            
        elif self.workflow_depth == 1:
            # Reason -> Answer
            reasoning = self.worker.reason(
                self.current_q,
                tools=agent1_tools,
                tokens=agent1_tokens,
                prompt_suffix=reasoner_suffix
            )
            # Execute tools if any
            if agent1_tools:
                reasoning, stats = self._process_tool_calls(reasoning, agent1_tools)
                exec_info["tools_count"] += stats.get("tool_calls", 0)
                merge_stats(stats)
            
            # Answerer just synthesizes - no tools needed (reasoner already did computation)
            final_text = self.worker.answer_with_context(
                self.current_q,
                context=reasoning,
                tools=[],
                tokens=answerer_tokens,
                prompt_suffix=answerer_suffix
            )
            exec_info["steps"] = 2
            exec_info["total_tokens"] = agent1_tokens + answerer_tokens
            
        elif self.workflow_depth in [2, 7]:
            # Reason -> Verify -> Answer (workflow 2) or Evaluator-Optimizer (workflow 7)
            reasoning = self.worker.reason(
                self.current_q,
                tools=agent1_tools,
                tokens=agent1_tokens,
                prompt_suffix=reasoner_suffix
            )
            
            if agent1_tools:
                reasoning, stats = self._process_tool_calls(reasoning, agent1_tools)
                exec_info["tools_count"] += stats.get("tool_calls", 0)
                merge_stats(stats)
            
            critique = self.worker.verify(
                self.current_q,
                reasoning=reasoning,
                tools=agent2_tools,
                tokens=agent2_tokens,
                prompt_suffix=verifier_suffix
            )
            
            if agent2_tools:
                critique, stats = self._process_tool_calls(critique, agent2_tools)
                exec_info["tools_count"] += stats.get("tool_calls", 0)
                merge_stats(stats)
            
            context = f"Reasoning: {reasoning}\nReview: {critique}"
            # Answerer just synthesizes - no tools needed (reasoner/verifier already did computation)
            final_text = self.worker.answer_with_context(
                self.current_q,
                context=context,
                tools=[],
                tokens=answerer_tokens,
                prompt_suffix=answerer_suffix
            )
            exec_info["steps"] = 3
            exec_info["total_tokens"] = agent1_tokens + agent2_tokens + answerer_tokens
            
        elif self.workflow_depth == 3:
            # Routing: Classify → Route to one of two reasoners → Answer
            classification_prompt = (
                f"Classify this question into one category: [simple, complex, multi-step]. "
                f"Respond with ONLY the category name. Question: {self.current_q}"
            )
            classification = self.worker.reason(
                classification_prompt,
                tools=agent1_tools,
                tokens=agent1_tokens // 3,
                prompt_suffix=reasoner_suffix
            )
            if agent1_tools:
                classification, stats = self._process_tool_calls(classification, agent1_tools)
                merge_stats(stats)
            exec_info["steps"] += 1
            exec_info["total_tokens"] += agent1_tokens // 3
            
            # Route to one of two reasoners based on classification
            classification_lower = classification.lower()
            if "multi-step" in classification_lower or "multi step" in classification_lower:
                # Route to Reasoner2 (agent2)
                reasoning_tools = agent2_tools if agent2_tools else agent1_tools
                reasoning_tokens = agent2_tokens if agent2_tools else agent1_tokens // 2
                reasoner_name = "Reasoner2 (agent2)"
            else:
                # Route to Reasoner1 (agent1)
                reasoning_tools = agent1_tools
                reasoning_tokens = agent1_tokens // 2
                reasoner_name = "Reasoner1 (agent1)"
            
            reasoning = self.worker.reason(
                self.current_q,
                tools=reasoning_tools,
                tokens=reasoning_tokens,
                prompt_suffix=reasoner_suffix
            )
            if reasoning_tools:
                reasoning, stats = self._process_tool_calls(reasoning, reasoning_tools)
                exec_info["tools_count"] += stats.get("tool_calls", 0)
                merge_stats(stats)
            exec_info["steps"] += 1
            exec_info["total_tokens"] += reasoning_tokens
            
            # Answerer just synthesizes - no tools needed
            final_text = self.worker.answer_with_context(
                self.current_q,
                context=f"Classification: {classification}\nRouted to: {reasoner_name}\nReasoning: {reasoning}",
                tools=[],
                tokens=answerer_tokens,
                prompt_suffix=answerer_suffix
            )
            exec_info["steps"] += 1
            exec_info["total_tokens"] += answerer_tokens
            
        elif self.workflow_depth == 4:
            # Parallel-Sectioning: Aspect1 breaks down → Worker1 (agent2) + Worker2 (agent1) → Combine
            # Step 1: Aspect1 breaks down into subtasks
            breakdown_prompt = (
                f"Break down this question into 2 independent subtasks that can be solved in parallel. "
                f"Question: {self.current_q}"
            )
            breakdown = self.worker.reason(
                breakdown_prompt,
                tools=agent1_tools,
                tokens=agent1_tokens // 3,
                prompt_suffix=reasoner_suffix
            )
            if agent1_tools:
                breakdown, stats = self._process_tool_calls(breakdown, agent1_tools)
                merge_stats(stats)
            exec_info["steps"] += 1
            exec_info["total_tokens"] += agent1_tokens // 3
            
            # Step 2: Process subtasks in parallel (Worker1 uses agent2, Worker2 reuses agent1)
            tokens_per_worker = agent1_tokens // 3
            
            # Worker1: Uses agent2_tools
            worker1_tools = agent2_tools if agent2_tools else agent1_tools
            worker1_prompt = f"Subtask 1: {breakdown}\nOriginal question: {self.current_q}\nSolve this subtask."
            result1 = self.worker.reason(
                worker1_prompt,
                tools=worker1_tools,
                tokens=tokens_per_worker,
                prompt_suffix=reasoner_suffix
            )
            if worker1_tools:
                result1, stats = self._process_tool_calls(result1, worker1_tools)
                exec_info["tools_count"] += stats.get("tool_calls", 0)
                merge_stats(stats)
            exec_info["steps"] += 1
            exec_info["total_tokens"] += tokens_per_worker
            
            # Worker2: Reuses agent1_tools (same agent, different subtask)
            worker2_prompt = f"Subtask 2: {breakdown}\nOriginal question: {self.current_q}\nSolve this subtask."
            result2 = self.worker.reason(
                worker2_prompt,
                tools=agent1_tools,
                tokens=tokens_per_worker,
                prompt_suffix=reasoner_suffix
            )
            if agent1_tools:
                result2, stats = self._process_tool_calls(result2, agent1_tools)
                exec_info["tools_count"] += stats.get("tool_calls", 0)
                merge_stats(stats)
            exec_info["steps"] += 1
            exec_info["total_tokens"] += tokens_per_worker
            
            # Step 3: Answerer synthesizes
            combined_context = f"Task breakdown: {breakdown}\nWorker 1 result: {result1}\nWorker 2 result: {result2}"
            final_text = self.worker.answer_with_context(
                self.current_q,
                context=combined_context,
                tools=[],
                tokens=answerer_tokens,
                prompt_suffix=answerer_suffix
            )
            exec_info["steps"] += 1
            exec_info["total_tokens"] += answerer_tokens
            
        elif self.workflow_depth == 5:
            # Parallel-Voting: Run same task multiple times and aggregate
            num_votes = 3
            votes = []
            # Use agent1_tools if available (for calculations/verification)
            vote_tools = agent1_tools if agent1_tools else []
            for i in range(num_votes):
                vote = self.worker.answer_direct(
                    self.current_q,
                    tools=vote_tools,
                    tokens=answerer_tokens // num_votes,
                    prompt_suffix=answerer_suffix
                )
                # Process tool calls if tools were used
                if vote_tools:
                    vote, stats = self._process_tool_calls(vote, vote_tools)
                    exec_info["tools_count"] += stats.get("tool_calls", 0)
                    merge_stats(stats)
                votes.append(vote)
            
            # Answerer just synthesizes - no tools needed (votes already did computation)
            votes_text = "\n".join([f"Vote {i+1}: {v}" for i, v in enumerate(votes)])
            final_text = self.worker.answer_with_context(
                self.current_q,
                context=f"Multiple attempts:\n{votes_text}\nProvide the most consistent answer.",
                tools=[],
                tokens=answerer_tokens,
                prompt_suffix=answerer_suffix
            )
            exec_info["steps"] = num_votes + 2
            exec_info["total_tokens"] = answerer_tokens * 2
            
        elif self.workflow_depth == 6:
            # Orchestrator-Workers: Central LLM breaks down and delegates
            breakdown = self.worker.reason(
                f"Break down this task into subtasks. Task: {self.current_q}",
                tools=agent1_tools,
                tokens=agent1_tokens // 3,
                prompt_suffix=reasoner_suffix
            )
            if agent1_tools:
                breakdown, stats = self._process_tool_calls(breakdown, agent1_tools)
                merge_stats(stats)
            
            # Use agent2_tools for workers
            worker_tools = agent2_tools if agent2_tools else agent1_tools
            worker1_result = self.worker.reason(
                f"Subtask 1: {breakdown}\nOriginal question: {self.current_q}",
                tools=worker_tools,
                tokens=agent1_tokens // 3,
                prompt_suffix=reasoner_suffix
            )
            if worker_tools:
                worker1_result, stats = self._process_tool_calls(worker1_result, worker_tools)
                exec_info["tools_count"] += stats.get("tool_calls", 0)
                merge_stats(stats)
            
            worker2_result = self.worker.reason(
                f"Subtask 2: {breakdown}\nOriginal question: {self.current_q}",
                tools=worker_tools,
                tokens=agent1_tokens // 3,
                prompt_suffix=reasoner_suffix
            )
            if worker_tools:
                worker2_result, stats = self._process_tool_calls(worker2_result, worker_tools)
                exec_info["tools_count"] += stats.get("tool_calls", 0)
                merge_stats(stats)
            
            # Answerer just synthesizes - no tools needed (workers already did computation)
            synthesis_context = f"Task breakdown: {breakdown}\nWorker 1: {worker1_result}\nWorker 2: {worker2_result}"
            final_text = self.worker.answer_with_context(
                self.current_q,
                context=synthesis_context,
                tools=[],
                tokens=answerer_tokens,
                prompt_suffix=answerer_suffix
            )
            exec_info["steps"] = 4
            exec_info["total_tokens"] = agent1_tokens + answerer_tokens
            
        elif self.workflow_depth == 7:
            # Evaluator-Optimizer: Generate → Evaluate → Refine loop
            max_iterations = 3
            current_answer = None
            
            for iteration in range(max_iterations):
                if iteration == 0:
                    current_answer = self.worker.reason(
                        self.current_q,
                        tools=agent1_tools,
                        tokens=agent1_tokens // max_iterations,
                        prompt_suffix=reasoner_suffix
                    )
                    if agent1_tools:
                        current_answer, stats = self._process_tool_calls(current_answer, agent1_tools)
                        merge_stats(stats)
                else:
                    current_answer = self.worker.reason(
                        f"Question: {self.current_q}\nPrevious attempt: {current_answer}\nImprove this answer.",
                        tools=agent1_tools,
                        tokens=agent1_tokens // max_iterations,
                        prompt_suffix=reasoner_suffix
                    )
                    if agent1_tools:
                        current_answer, stats = self._process_tool_calls(current_answer, agent1_tools)
                        merge_stats(stats)
                
                evaluation = self.worker.verify(
                    self.current_q,
                    reasoning=current_answer,
                    tools=agent2_tools,
                    tokens=agent2_tokens // max_iterations,
                    prompt_suffix=verifier_suffix
                )
                if agent2_tools:
                    evaluation, stats = self._process_tool_calls(evaluation, agent2_tools)
                    merge_stats(stats)
                
                if "correct" in evaluation.lower() and iteration > 0:
                    break
            
            # Answerer just synthesizes - no tools needed (generator/evaluator already did computation)
            final_text = self.worker.answer_with_context(
                self.current_q,
                context=f"Refined reasoning: {current_answer}",
                tools=[],
                tokens=answerer_tokens,
                prompt_suffix=answerer_suffix
            )
            exec_info["steps"] = max_iterations * 2 + 1
            exec_info["total_tokens"] = agent1_tokens + agent2_tokens + answerer_tokens
            
        elif self.workflow_depth == 8:
            # Autonomous Agent: LLM uses tools autonomously in a loop
            max_iterations = 5
            context_history = []
            
            for iteration in range(max_iterations):
                # Use agent1_tools for first iteration, agent2_tools for later iterations
                iteration_tools = agent1_tools if iteration == 0 else (agent2_tools if agent2_tools else agent1_tools)
                
                if iteration == 0:
                    reasoning = self.worker.reason(
                        self.current_q,
                        tools=iteration_tools,
                        tokens=agent1_tokens // max_iterations,
                        prompt_suffix=reasoner_suffix
                    )
                else:
                    reasoning = self.worker.reason(
                        f"Question: {self.current_q}\nPrevious context: {context_history[-1]}\nContinue reasoning.",
                        tools=iteration_tools,
                        tokens=agent1_tokens // max_iterations,
                        prompt_suffix=reasoner_suffix
                    )
                
                if iteration_tools:
                    reasoning, stats = self._process_tool_calls(reasoning, iteration_tools)
                    exec_info["tools_count"] += stats.get("tool_calls", 0)
                    merge_stats(stats)
                
                context_history.append(reasoning)
                
                if iteration >= 2:
                    break
            
            # Answerer just synthesizes - no tools needed (previous iterations already did computation)
            combined_context = "\n".join([f"Step {i+1}: {ctx}" for i, ctx in enumerate(context_history)])
            final_text = self.worker.answer_with_context(
                self.current_q,
                context=combined_context,
                tools=[],
                tokens=answerer_tokens,
                prompt_suffix=answerer_suffix
            )
            exec_info["steps"] = len(context_history) + 1
            exec_info["total_tokens"] = agent1_tokens + answerer_tokens
        
        return final_text, exec_info

