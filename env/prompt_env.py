"""
Prompt Environment - Low-Level Policy

Multi-step environment for sequential prompt selection.
Receives structure decisions from the high-level policy and:
1. Sequentially selects prompts for each agent (Reasoner → Verifier → Answerer)
2. Executes the full workflow
3. Returns final reward based on correctness and efficiency

This separation allows:
- Structure policy to learn WHAT configuration to use
- Prompt policy to learn HOW to prompt each agent
"""
import os
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
    PROMPT_ATOMS, build_prompt_suffix,
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
    
    # Token budget mappings (2x increased from original to prevent truncation)
    TOKEN_BUDGETS = {
        "reasoner": {0: 512, 1: 1024, 2: 2048},
        "verifier": {0: 256, 1: 512, 2: 1024},
        "answerer": {0: 128, 1: 256, 2: 512}
    }
    
    def __init__(self, cfg=None, is_eval=False, use_api=False, api_model=None, hf_model=None, dataset=None):
        """
        Args:
            cfg: Configuration module
            is_eval: If True, use evaluation split of dataset
            use_api: If True, use OpenRouterWorker instead of LLMWorker
            api_model: OpenRouter model ID (e.g., 'openai/gpt-4o')
            hf_model: HuggingFace model name (e.g., 'Qwen/Qwen2.5-7B-Instruct')
            dataset: Pre-loaded dataset (optional, to avoid reloading in parallel evaluation)
        """
        super().__init__()
        
        # Store config
        if cfg is None:
            from configs import load_config
            cfg = load_config("hierarchical")
        self.cfg = cfg
        
        # Prompt configuration
        self.MAX_PROMPTS_PER_AGENT = getattr(cfg, 'MAX_PROMPTS_PER_AGENT', 3)
        
        # Number of prompt atoms per agent. Read from PROMPT_ATOMS directly (mutated
        # in place by load_or_create_atoms) rather than from library.NUM_ATOMS, since
        # NUM_ATOMS is rebound by refresh_counts() — the imported binding here would
        # otherwise point to the pre-load snapshot and the action space would be sized
        # to base atoms only, making v2 atoms unreachable by the policy.
        self.num_reasoner_atoms = len(PROMPT_ATOMS["reasoner"])
        self.num_verifier_atoms = len(PROMPT_ATOMS["verifier"])
        self.num_answerer_atoms = len(PROMPT_ATOMS["answerer"])
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
        
        # Use provided dataset or load new one
        self.dataset = dataset if dataset is not None else get_dataset_loader(cfg.DATASET_NAME, is_eval=is_eval)
        
        # Dataset-aware tool registry: tau2_* datasets get a domain-scoped registry
        # exposing the domain's task-relevant tools; everything else gets the default 4-tool registry.
        from tools import get_tool_registry
        self.tools = get_tool_registry(getattr(self.dataset, "name", None))

        # Tool action decoder: tau2 datasets bitmask over semantic groups via the
        # registry; default datasets use the 4-tool bitmask in _decode_tools.
        ds_name = getattr(self.dataset, "name", "") or getattr(cfg, "DATASET_NAME", "") or ""
        if ds_name.startswith("tau2_"):
            self._tau2_registry = self.tools  # Tau2ToolRegistry
            self._tau2_num_groups = self._tau2_registry.num_groups()
        else:
            self._tau2_registry = None
            self._tau2_num_groups = 0
        
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

        # When True, the multi-step atom-selection loop terminates without
        # invoking _execute_workflow. Used by per-turn-config rollout: that
        # rollout runs the workflow itself once per dialog turn and only
        # needs PromptEnv's atom-selection state machine, not its execution.
        self.skip_execute = False
        
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

        # Tool index ranges over 2^G for tau2 datasets (group bitmask) or 2^4=16
        # for default datasets. Normalize so the value stays in [0,1].
        if self._tau2_registry is not None:
            max_tool_idx = (2 ** self._tau2_num_groups) - 1
        else:
            max_tool_idx = 15
        # Structure decisions (normalized)
        structure_vec = np.array([
            self.workflow_depth / 8.0,  # Normalize for 9 workflows (0-8)
            self.agent1_tools_idx / max(max_tool_idx, 1.0),
            self.agent1_budget_idx / 2.0,
            self.agent2_tools_idx / max(max_tool_idx, 1.0),
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
        # Train-time prompt-dimension ablation: force DONE on every prompt slot
        # so no learned atoms are ever appended (ARC_FREEZE_PROMPT=1). No-op otherwise.
        if os.environ.get("ARC_FREEZE_PROMPT"):
            action = 0
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
            if self.skip_execute:
                # Per-turn-config mode: terminate here. Caller drives the
                # actual workflow execution and env.step itself.
                return self._get_observation(), reward, True, False, {
                    "selected_prompts": dict(self.selected_prompts),
                    "skipped_execute": True,
                }
            final_text, exec_info = self._execute_workflow()
            
            # Calculate correctness
            correctness = self.dataset.evaluate_correctness(final_text, self.current_a)
            correct = correctness == 1.0
            
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
                "correct": correct,  # Use the correctly computed correctness
                "correctness": float(correctness),  # raw fractional [0,1] (= tau2 product reward for tau2 datasets)
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
            # Forward tau2 per-component breakdown (only present when running a tau2 dataset)
            for k, v in (exec_info or {}).items():
                if k.startswith("tau2_"):
                    info[k] = v
        
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
        """Decode tool action index to a list of tool names.

        Tau2 datasets: bit i selects semantic group i; all tools in selected
        groups are returned (deduped, in registry order).
        Default datasets: 4-bit mask over [calculator, web_search, python, ocr_reader].
        """
        if self._tau2_registry is not None:
            return self._tau2_registry.decode_group_mask(idx)

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
        # Original workflow execution
        # Build prompt suffixes
        reasoner_suffix = build_prompt_suffix("reasoner", self.selected_prompts["reasoner"])
        verifier_suffix = build_prompt_suffix("verifier", self.selected_prompts["verifier"])
        answerer_suffix = build_prompt_suffix("answerer", self.selected_prompts["answerer"])

        # Get token counts
        agent1_tokens = self.TOKEN_BUDGETS["reasoner"][self.agent1_budget_idx]
        agent2_tokens = self.TOKEN_BUDGETS["verifier"][self.agent2_budget_idx]
        answerer_tokens = self.TOKEN_BUDGETS["answerer"][self.answerer_budget_idx]

        # Tau2 datasets: workflow runs as a black-box agent inside a multi-turn gym env
        # rather than as a single-shot Q&A. The structure policy still picks a
        # per-agent tool subset (2^N action space over the domain's N tools), and that
        # selection is honored inside _execute_tau2_workflow.
        ds_name = getattr(self.dataset, "name", "") or ""
        if ds_name.startswith("tau2_"):
            return self._execute_tau2_workflow(
                reasoner_suffix, verifier_suffix, answerer_suffix,
                agent1_tokens, agent2_tokens, answerer_tokens,
            )

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
        
        # Update question embedding to use the output embedding (like standard datasets)
        # This ensures the next state uses the embedding of the output, not the original question
        # This matches the behavior in general_env.py where next_obs = embedding(final_text)
        try:
            self.question_embedding = self.worker.get_embedding(final_text)
        except Exception as e:
            # Fallback: if embedding fails, keep original question embedding
            print(f"  ⚠ Warning: Could not update embedding from final_text: {e}")

        return final_text, exec_info

    def _execute_tau2_workflow(
        self,
        reasoner_suffix, verifier_suffix, answerer_suffix,
        agent1_tokens, agent2_tokens, answerer_tokens,
    ) -> tuple:
        """Tau2 dialog rollout: configured workflow acts as the agent in a multi-turn
        conversation with tau2's user simulator. Reward comes from evaluate_simulation."""
        from tau2_code.src import ConfiguredAgent, tau2_dialog_rollout

        # task_id occupies the 'answer' slot for tau2 datasets (see Tau2Dataset.get_sample).
        task_id = self.current_a
        domain = self.dataset.domain

        workflow = self.get_workflow_func(self.workflow_depth, self.worker, self.tools)
        if self.workflow_depth == 2 and hasattr(workflow, "use_verifier"):
            workflow.use_verifier = True

        # The structure policy chooses a tool subset per agent (2^N action space
        # over the domain's N tools). We respect that selection here so the policy
        # can actually learn which tools matter for which task.
        agent1_tools = self._decode_tools(self.agent1_tools_idx)
        agent2_tools = self._decode_tools(self.agent2_tools_idx)

        agent = ConfiguredAgent(
            workflow=workflow,
            worker=self.worker,
            tool_registry=self.tools,
            prompt_suffixes={
                "reasoner": reasoner_suffix,
                "verifier": verifier_suffix,
                "answerer": answerer_suffix,
            },
            agent1_tools=agent1_tools,
            agent2_tools=agent2_tools,
            agent1_tokens=agent1_tokens,
            agent2_tokens=agent2_tokens,
            answerer_tokens=answerer_tokens,
            agent1_budget=self.agent1_budget_idx,
            agent2_budget=self.agent2_budget_idx,
            answerer_budget=self.answerer_budget_idx,
        )

        max_turns = int(getattr(self.cfg, "TAU2_MAX_TURNS", 12))
        # Shaping toggles read off cfg (set by train.py / eval scripts).
        # Defaults match: training=ON, mode=training. Eval scripts override.
        use_shaped = bool(getattr(self.cfg, "USE_SHAPED_REWARDS", True))
        shaping_mode = str(getattr(self.cfg, "SHAPING_MODE", "training"))
        shaping_cfg = getattr(self.cfg, "SHAPING_CFG", None)
        transcript, exec_info, reward = tau2_dialog_rollout(
            agent, domain=domain, task_id=task_id, max_turns=max_turns,
            use_shaped_rewards=use_shaped,
            shaping_mode=shaping_mode,
            shaping_cfg=shaping_cfg,
        )
        # Cache the reward on the dataset so evaluate_correctness can return it.
        if hasattr(self.dataset, "cache_reward"):
            self.dataset.cache_reward(task_id, reward)

        # NOTE: we deliberately DO NOT re-embed the transcript here.
        # Transcripts are unique per episode -> always a precomputed-cache miss ->
        # forces MetaCLIP-H14 to load on GPU inside every API worker process and
        # run a forward pass per episode (huge wall-clock cost, and the trigger
        # for prior CUDA OOM with --num-workers > 1). The task-description
        # embedding (already in self.question_embedding from set_structure) is a
        # good enough state representation for the next obs.

        return transcript, exec_info

