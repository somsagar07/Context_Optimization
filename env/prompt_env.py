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

from agents_system import LLMWorker, OpenRouterWorker
from agents_system.workflows import get_workflow, get_openrouter_workflow
from tools import ToolRegistry
from utils import get_dataset_loader
from prompts.library import (
    PROMPT_ATOMS, NUM_ATOMS, build_prompt_suffix,
)
import re

# Handle tau2 dataset
from tools.tau2_tool_registry import Tau2ToolRegistry
try:
    from tau2_executions_wrapper import Tau2ExecutionWrapper
except ImportError:
    # Try importing from parent directory
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from tau2_executions_wrapper import Tau2ExecutionWrapper

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
    
    # Token budget mappings (2x increased from original to prevent truncation)
    TOKEN_BUDGETS = {
        "reasoner": {0: 512, 1: 1024, 2: 2048},
        "verifier": {0: 256, 1: 512, 2: 1024},
        "answerer": {0: 128, 1: 256, 2: 512}
    }
    
    def __init__(self, cfg=None, is_eval=False, use_api=False, api_model=None, hf_model=None):
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
        if use_api:
            self.worker = OpenRouterWorker(model_name=api_model)
            self.get_workflow_func = get_openrouter_workflow
        else:
            self.worker = LLMWorker(model_name=hf_model)
            self.get_workflow_func = get_workflow
        # self.tools = ToolRegistry()
        self.dataset = get_dataset_loader(cfg.DATASET_NAME, is_eval=is_eval)
        
        # Check if this is a tau2 dataset
        self.is_tau2 = hasattr(self.dataset, 'domain') and self.dataset.name.startswith('tau2_')
        
        # Initialize tools based on dataset type
        if self.is_tau2:
            self.tools = Tau2ToolRegistry(self.dataset.domain)
            
            if isinstance(self.tools, Tau2ToolRegistry):
                tau2_descriptions = self.tools.get_tool_prompt_descriptions()
                if tau2_descriptions:
                    self.worker.additional_tool_descriptions = tau2_descriptions
                    print(f"✓ Loaded {len(tau2_descriptions)} tau2 tool descriptions for worker")
                else:
                    print(f"⚠ Warning: No tool descriptions returned from tau2 registry")
            
                # self.worker.additional_tool_descriptions = tau2_descriptions
            
            self.tau2_executor = Tau2ExecutionWrapper(
                self.dataset.domain,
                self.worker,
                self.tools,
                use_api=use_api
            )
        else:
            # Use the already imported ToolRegistry from line 22
            self.tools = ToolRegistry()
            self.tau2_executor = None
        
        # Observation space components
        # Question embedding is 1024D from MetaCLIP-H14
        hidden_size = self.worker.model.config.hidden_size
        obs_size = (
            hidden_size +                       # Question embedding (1024D)
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
        
    def set_structure(self, question: str, answer: str, embedding: np.ndarray, structure: dict, task=None):
        """
        Set the structure decision from the high-level policy.
        Must be called before reset() when using externally.
        
        Args:
            question: The current question
            answer: The ground truth answer
            embedding: Pre-computed question embedding
            structure: Dict with workflow_depth, tools, budgets
            task: Task object for tau2 datasets
        """
        self.current_q = question
        self.current_a = answer
        self.question_embedding = embedding
        
        # # For tau2, store task object if available
        # if self.is_tau2 and isinstance(answer, dict):
        #     self.current_task = answer
        # For tau2, store task object if provided
        if self.is_tau2:
            if task is not None:
                self.current_task = task
            elif isinstance(answer, dict):
                self.current_task = answer
        
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
        if self.is_tau2 and isinstance(self.tools, Tau2ToolRegistry):
            max_tool_idx = self.tools.get_max_tool_index()
        else:
            max_tool_idx = 15
        # Structure decisions (normalized)
        structure_vec = np.array([
            self.workflow_depth / 8.0,  # Normalize for 9 workflows (0-8)
            self.agent1_tools_idx / max(max_tool_idx, 1.0),  # Normalize for 16 tool options (0-15)
            self.agent1_budget_idx / 2.0,
            self.agent2_tools_idx / max(max_tool_idx, 1.0),  # Normalize for 16 tool options (0-15)
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
        
        # Clamp action to valid range; out-of-range becomes DONE
        if action >= num_atoms:
            action = 0
        
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
        if self.is_tau2 and isinstance(self.tools, Tau2ToolRegistry):
            return self.tools.decode_tool_index(idx)
        else:
            # Original tool decoding
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
        # If tau2, use execution wrapper
        if self.is_tau2 and self.tau2_executor:
            # Get task_id from dataset
            task_obj = getattr(self, 'current_task', None)
            if task_obj is None:
                # Fallback: try to get from dataset
                _, task_obj = self.dataset.get_sample()
            
            task_id = task_obj.get('task_id', 'Unknown') if isinstance(task_obj, dict) else 'Unknown'
            
            # Execute tau2 conversation
            pass_k_reward, exec_info = self.tau2_executor.execute_conversation(
                task_id=task_id,
                workflow_depth=self.workflow_depth,
                agent1_tools_idx=self.agent1_tools_idx,
                agent1_budget_idx=self.agent1_budget_idx,
                agent2_tools_idx=self.agent2_tools_idx,
                agent2_budget_idx=self.agent2_budget_idx,
                answerer_budget_idx=self.answerer_budget_idx,
                selected_prompts=self.selected_prompts
            )
            
            # Format final text from conversation history
            conversation_history = exec_info.get("conversation_history", [])
            final_text = "\n".join([f"{role}: {msg}" for role, msg in conversation_history[-3:]])  # Last 3 turns
            
            # pass_k_reward is the actual reward from Tau2 environment (not binary)
            exec_info["pass_k"] = pass_k_reward
            exec_info["tau2_reward"] = pass_k_reward  # Store Tau2's shaped reward
            exec_info["final_reward"] = exec_info.get("final_reward", pass_k_reward)  # Keep final_reward from wrapper
            exec_info["correct"] = pass_k_reward > 0  # Binary correctness for logging/metrics
            
            return final_text, exec_info
        
        # Original workflow execution for non-tau2 datasets
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
        
        # Get workflow instance using the appropriate function (HuggingFace or OpenRouter)
        workflow = self.get_workflow_func(
            self.workflow_depth, self.worker, self.tools
        )
        
        # Special handling for workflow 2 (Reason+Verify+Ans)
        if self.workflow_depth == 2:
            if hasattr(workflow, 'use_verifier'):
                workflow.use_verifier = True
        
        # Execute workflow with prompt suffixes
        prompt_suffixes = {
            "reasoner": reasoner_suffix,
            "verifier": verifier_suffix,
            "answerer": answerer_suffix
        }
        
        final_text, exec_info = workflow.execute(
            self.current_q,
            agent1_tools,
            self.agent1_budget_idx,
            agent2_tools,
            self.agent2_budget_idx,
            self.answerer_budget_idx,
            agent1_tokens,
            agent2_tokens,
            answerer_tokens,
            prompt_suffixes=prompt_suffixes
        )
        
        return final_text, exec_info

