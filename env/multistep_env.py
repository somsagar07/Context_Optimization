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
from prompts.library import NUM_ATOMS, PROMPT_ATOMS, build_prompt_suffix


class MultiStepAgentEnv(gym.Env):
    """
    Multi-Step RL Environment for LLM Agent Configuration.
    
    Instead of choosing all 6 action dimensions at once (5,184 combinations),
    the agent makes SEQUENTIAL decisions, enabling proper credit assignment.
    
    Episode Structure (without prompt learning, 4 steps max):
        Step 0: Choose workflow depth [0, 1, 2, ..., 8] → action_space = 9
        Step 1: Choose reasoner config (tools + budget) → action_space = 48
        Step 2: Choose verifier config (if depth in [2,7]) → action_space = 48
        Step 3: Choose answerer budget → action_space = 3
        → Execute workflow → Final reward → Done
    
    Episode Structure (WITH prompt learning, up to 7 steps):
        Step 0: Choose workflow depth [0-8]
        Step 1: Choose reasoner config (tools + budget)
        Step 2: Choose verifier config (if needed)
        Step 3: Choose answerer budget
        Step 4: Choose reasoner prompt [0=none, 1-6=prompt atoms]
        Step 5: Choose verifier prompt (if needed) [0=none, 1-5=prompt atoms]
        Step 6: Choose answerer prompt [0=none, 1-4=prompt atoms]
        → Execute workflow with prompts → Final reward → Done
    
    Benefits:
        - Smaller action space per step
        - Temporal credit assignment via multi-step returns
        - Intermediate rewards guide learning
        - Agent can learn dependencies between choices
        - With learn_prompts=True, also learns which prompts to use
    """
    
    # Decision stages (structure)
    STAGE_WORKFLOW = 0
    STAGE_REASONER = 1  
    STAGE_VERIFIER = 2
    STAGE_ANSWERER = 3
    # Decision stages (prompts) - only used if learn_prompts=True
    STAGE_REASONER_PROMPT = 4
    STAGE_VERIFIER_PROMPT = 5
    STAGE_ANSWERER_PROMPT = 6
    STAGE_EXECUTE = 7  # Terminal pseudo-stage
    
    def __init__(self, cfg=None, is_eval=False, use_api=False, api_model=None, hf_model=None, 
                 learn_prompts=False):
        """
        Initialize the environment.
        
        Args:
            cfg: Configuration module. If None, imports default config.
            is_eval: If True, loads evaluation dataset (for future use).
            use_api: If True, use OpenRouter API instead of local HuggingFace model.
            api_model: OpenRouter model ID (e.g., "openai/gpt-4o"). Required if use_api=True.
            hf_model: HuggingFace model name. If None, uses config default.
            learn_prompts: If True, add prompt selection steps to the episode.
        """
        super(MultiStepAgentEnv, self).__init__()
        
        # Store config (import default if not provided)
        if cfg is None:
            from configs import load_config
            cfg = load_config("multi_step")  # Default to multi_step for backwards compatibility
        self.cfg = cfg
        self.learn_prompts = learn_prompts
        
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
        
        # Prompt atom counts (loaded from library)
        self.num_reasoner_atoms = NUM_ATOMS.get("reasoner", 7)  # 0=none + 6 prompts
        self.num_verifier_atoms = NUM_ATOMS.get("verifier", 6)  # 0=none + 5 prompts
        self.num_answerer_atoms = NUM_ATOMS.get("answerer", 5)  # 0=none + 4 prompts
        self.max_prompt_atoms = max(self.num_reasoner_atoms, self.num_verifier_atoms, self.num_answerer_atoms)
        
        # Action space: max across all stages
        # Structure stages: max 48 (16 tools × 3 budgets)
        # Prompt stages: max 7 (reasoner atoms)
        # Use max of both = 48
        self.action_space = spaces.Discrete(48)
        
        # Token budgets
        self.TOKEN_BUDGETS = {
            "reasoner": {0: 256, 1: 512, 2: 1024},
            "verifier": {0: 128, 1: 256, 2: 512},
            "answerer": {0: 64, 1: 128, 2: 256}
        }
        
        # Observation: question embedding + stage encoding + partial decisions + prompt choices
        # - question_embedding: hidden_size (1024 for MetaCLIP-H14)
        # - stage_onehot: 7 dims (which decision stage we're in, expanded for prompt stages)
        # - workflow_chosen: 9 dims (one-hot of depth)
        # - reasoner_config: 48 dims (one-hot of tools×budget)
        # - verifier_config: 48 dims (one-hot of tools×budget)
        # - reasoner_prompt: 7 dims (one-hot of prompt atom, if learn_prompts)
        # - verifier_prompt: 6 dims (one-hot of prompt atom, if learn_prompts)
        # - answerer_prompt: 5 dims (one-hot of prompt atom, if learn_prompts)
        hidden_size = self.worker.model.config.hidden_size
        
        if learn_prompts:
            num_stages = 7  # Stages 0-6
            obs_size = hidden_size + num_stages + 9 + 48 + 48 + self.num_reasoner_atoms + self.num_verifier_atoms + self.num_answerer_atoms
        else:
            num_stages = 4  # Stages 0-3
            obs_size = hidden_size + num_stages + 9 + 48 + 48
        
        self.num_stages_obs = num_stages
        
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
        
        # Accumulated decisions (structure)
        self.workflow_depth = None
        self.agent1_tools = None
        self.agent1_budget = None
        self.agent2_tools = None
        self.agent2_budget = None
        self.answerer_budget = None
        
        # Accumulated decisions (prompts) - only used if learn_prompts=True
        self.reasoner_prompt = None  # Index into REASONER_ATOMS (0=none)
        self.verifier_prompt = None  # Index into VERIFIER_ATOMS (0=none)
        self.answerer_prompt = None  # Index into ANSWERER_ATOMS (0=none)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Get new question
        self.current_q, self.current_a = self.dataset.get_sample()
        self.question_embedding = self.worker.get_embedding(self.current_q)
        
        # Reset stage and structure decisions
        self.stage = self.STAGE_WORKFLOW
        self.workflow_depth = None
        self.agent1_tools = None
        self.agent1_budget = None
        self.agent2_tools = None
        self.agent2_budget = None
        self.answerer_budget = None
        
        # Reset prompt decisions (only used if learn_prompts=True)
        self.reasoner_prompt = None
        self.verifier_prompt = None
        self.answerer_prompt = None
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Build observation: question embedding + stage + partial decisions (+ prompts if learn_prompts)."""
        # Stage one-hot (varies by learn_prompts)
        stage_onehot = np.zeros(self.num_stages_obs, dtype=np.float32)
        stage_onehot[min(self.stage, self.num_stages_obs - 1)] = 1.0
        
        # Workflow choice one-hot (9 options)
        workflow_onehot = np.zeros(9, dtype=np.float32)
        if self.workflow_depth is not None:
            workflow_onehot[self.workflow_depth] = 1.0
        
        # Reasoner config one-hot (16 tools × 3 budgets = 48)
        reasoner_config_onehot = np.zeros(48, dtype=np.float32)
        if self.agent1_tools is not None and self.agent1_budget is not None:
            idx = self._encode_config(self.agent1_tools, self.agent1_budget)
            reasoner_config_onehot[idx] = 1.0
        
        # Verifier config one-hot (16 tools × 3 budgets = 48)
        verifier_config_onehot = np.zeros(48, dtype=np.float32)
        if self.agent2_tools is not None and self.agent2_budget is not None:
            idx = self._encode_config(self.agent2_tools, self.agent2_budget)
            verifier_config_onehot[idx] = 1.0
        
        # Base observation (structure only)
        obs_parts = [
            self.question_embedding,
            stage_onehot,
            workflow_onehot,
            reasoner_config_onehot,
            verifier_config_onehot
        ]
        
        # Add prompt observations if learning prompts
        if self.learn_prompts:
            # Reasoner prompt one-hot
            reasoner_prompt_onehot = np.zeros(self.num_reasoner_atoms, dtype=np.float32)
            if self.reasoner_prompt is not None:
                reasoner_prompt_onehot[self.reasoner_prompt] = 1.0
            
            # Verifier prompt one-hot
            verifier_prompt_onehot = np.zeros(self.num_verifier_atoms, dtype=np.float32)
            if self.verifier_prompt is not None:
                verifier_prompt_onehot[self.verifier_prompt] = 1.0
            
            # Answerer prompt one-hot
            answerer_prompt_onehot = np.zeros(self.num_answerer_atoms, dtype=np.float32)
            if self.answerer_prompt is not None:
                answerer_prompt_onehot[self.answerer_prompt] = 1.0
            
            obs_parts.extend([reasoner_prompt_onehot, verifier_prompt_onehot, answerer_prompt_onehot])
        
        # Concatenate all
        obs = np.concatenate(obs_parts).astype(np.float32)
        
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
            mask[:48] = 1.0
        elif self.stage == self.STAGE_VERIFIER:
            # 48 valid (only reached if depth in [2, 7])
            mask[:48] = 1.0
        elif self.stage == self.STAGE_ANSWERER:
            # Only 3 valid: [0, 1, 2] for answerer budget
            mask[:3] = 1.0
        # Prompt stages (only if learn_prompts=True)
        elif self.stage == self.STAGE_REASONER_PROMPT:
            # 0 = no prompt, 1-6 = prompt atoms
            mask[:self.num_reasoner_atoms] = 1.0
        elif self.stage == self.STAGE_VERIFIER_PROMPT:
            # 0 = no prompt, 1-5 = prompt atoms
            mask[:self.num_verifier_atoms] = 1.0
        elif self.stage == self.STAGE_ANSWERER_PROMPT:
            # 0 = no prompt, 1-4 = prompt atoms
            mask[:self.num_answerer_atoms] = 1.0
            
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
            efficiency_bonus = max(0, (8 - self.workflow_depth) * 0.01)
            reward = efficiency_bonus
            
            # Next stage depends on depth
            # Workflow 0 (Direct) and 5 (Parallel-Voting) skip reasoner config
            if self.workflow_depth in [0, 5]:
                # Skip to answerer
                self.stage = self.STAGE_ANSWERER
            else:
                # Need reasoning config
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
            # Choose answerer budget
            self.answerer_budget = min(action, 2)
            
            if self.learn_prompts:
                # Continue to prompt selection stages
                # Workflow 0 (Direct) and 5 (Parallel-Voting) skip reasoner prompt
                if self.workflow_depth in [0, 5]:
                    # Only answerer prompt needed
                    self.stage = self.STAGE_ANSWERER_PROMPT
                else:
                    # Need reasoner prompt first
                    self.stage = self.STAGE_REASONER_PROMPT
            else:
                # No prompt learning: execute immediately
                terminated, reward, info = self._execute_and_compute_reward()
        
        # ===== PROMPT SELECTION STAGES (only if learn_prompts=True) =====
        
        elif self.stage == self.STAGE_REASONER_PROMPT:
            # Choose reasoner prompt: 0=none, 1-6=prompt atoms
            self.reasoner_prompt = min(action, self.num_reasoner_atoms - 1)
            
            # Next: verifier prompt (if verifier used) or answerer prompt
            if self.workflow_depth in [2, 7]:
                self.stage = self.STAGE_VERIFIER_PROMPT
            else:
                self.stage = self.STAGE_ANSWERER_PROMPT
                
        elif self.stage == self.STAGE_VERIFIER_PROMPT:
            # Choose verifier prompt: 0=none, 1-5=prompt atoms
            self.verifier_prompt = min(action, self.num_verifier_atoms - 1)
            
            # Next: answerer prompt
            self.stage = self.STAGE_ANSWERER_PROMPT
            
        elif self.stage == self.STAGE_ANSWERER_PROMPT:
            # Choose answerer prompt: 0=none, 1-4=prompt atoms
            self.answerer_prompt = min(action, self.num_answerer_atoms - 1)
            
            # Now execute the full workflow with prompts
            terminated, reward, info = self._execute_and_compute_reward()
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _execute_and_compute_reward(self) -> tuple:
        """Execute the workflow and compute the final reward. Returns (terminated, reward, info)."""
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
        
        # Add prompt info if learning prompts
        if self.learn_prompts:
            info["reasoner_prompt"] = self.reasoner_prompt
            info["verifier_prompt"] = self.verifier_prompt
            info["answerer_prompt"] = self.answerer_prompt
        
        return True, reward, info
    
    def _get_episode_length(self) -> int:
        """Get number of decision steps for this episode."""
        # Base episode length (structure decisions)
        if self.workflow_depth == 0:
            # Direct answer: workflow + answerer
            base_length = 2
        elif self.workflow_depth == 5:
            # Parallel-Voting: workflow + answerer (no reasoner config needed)
            base_length = 2
        elif self.workflow_depth in [2, 7]:
            # Workflows with verifier: workflow + reasoner + verifier + answerer
            base_length = 4
        else:
            # Standard workflows: workflow + reasoner + answerer
            base_length = 3
        
        # Add prompt steps if learning prompts
        if self.learn_prompts:
            if self.workflow_depth in [0, 5]:
                # Direct / Parallel-Voting: only answerer prompt
                base_length += 1
            elif self.workflow_depth in [2, 7]:
                # Verifier workflows: reasoner + verifier + answerer prompts
                base_length += 3
            else:
                # Standard: reasoner + answerer prompts
                base_length += 2
        
        return base_length
    
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
        
        # Build prompt suffixes dict (if learning prompts)
        prompt_suffixes = None
        
        if self.learn_prompts:
            prompt_suffixes = {}
            
            # Build prompt suffix for reasoner (if selected, 0=none)
            if self.reasoner_prompt is not None and self.reasoner_prompt > 0:
                # Index is offset by 1 (0=none, 1=atom[0], 2=atom[1], etc.)
                prompt_suffixes["reasoner"] = build_prompt_suffix("reasoner", [self.reasoner_prompt])
            
            # Build prompt suffix for verifier (if selected)
            if self.verifier_prompt is not None and self.verifier_prompt > 0:
                prompt_suffixes["verifier"] = build_prompt_suffix("verifier", [self.verifier_prompt])
            
            # Build prompt suffix for answerer (if selected)
            if self.answerer_prompt is not None and self.answerer_prompt > 0:
                prompt_suffixes["answerer"] = build_prompt_suffix("answerer", [self.answerer_prompt])
        
        # Execute workflow with optional prompt suffixes
        final_text, exec_info = workflow.execute(
            self.current_q,
            agent1_tools,
            self.agent1_budget if self.agent1_budget is not None else 1,
            agent2_tools,
            self.agent2_budget if self.agent2_budget is not None else 1,
            self.answerer_budget,
            agent1_tokens,
            agent2_tokens,
            answerer_tokens,
            prompt_suffixes=prompt_suffixes
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
