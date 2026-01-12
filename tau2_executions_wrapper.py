"""
Tau2 Execution Wrapper - Executes full tau2 conversations with HRL configuration.
"""
import os
import gymnasium as gym
from typing import Dict, List, Tuple, Optional
from agents_system.workflows import get_workflow, get_openrouter_workflow
from prompts.library import build_prompt_suffix

class Tau2ExecutionWrapper:
    """
    Wrapper to execute tau2 conversations using HRL-selected configuration.
    
    Flow:
    1. Receives HRL configuration (workflow, tools, prompts, budgets)
    2. Runs full tau2 conversation (agent ↔ user simulator)
    3. Returns Pass^k reward and execution info
    """
    
    def __init__(self, domain: str, worker, tools_registry, use_api: bool = False):
        """
        Initialize tau2 execution wrapper.
        
        Args:
            domain: tau2 domain (airline, retail, telecom)
            worker: LLMWorker or OpenRouterWorker instance
            tools_registry: Tau2ToolRegistry instance
            use_api: Whether using OpenRouter API
        """
        self.domain = domain
        self.worker = worker
        self.tools_registry = tools_registry
        self.use_api = use_api
        self.env = None
        self._env_initialized = False
        # Don't initialize env here - do it lazily when needed
        # self._init_env()
    
    def _init_env(self):
        """Initialize tau2 gym environment (lazy initialization)."""
        if self._env_initialized:
            return
        
        if 'TAU2_DATA_DIR' not in os.environ:
            possible_paths = [
                './data_cache/tau2_data_root',
                '../data_cache/tau2_data_root',
                os.path.join(os.path.dirname(__file__), '../../data_cache/tau2_data_root')
            ]
            for path in possible_paths:
                abs_path = os.path.abspath(path)
                if os.path.exists(abs_path):
                    os.environ['TAU2_DATA_DIR'] = abs_path
                    break
        
        try:
            import tau2
            from tau2.gym import register_gym_agent, TAU_BENCH_ENV_ID
            
            # Register gym environments first
            register_gym_agent()
            
            # Note: We can't create the environment here because we need a task_id
            # which we'll get in execute_conversation. Just mark that registration is done.
            self._env_initialized = True
            self._tau2_gym_ready = True
            print(f"  ✓ Registered tau2 gym environments")
            
        except Exception as e:
            print(f"  ⚠ Warning: Could not initialize tau2 gym environments: {e}")
            print(f"     Will attempt to execute conversations without gym environment")
            self._env_initialized = True
            self._tau2_gym_ready = False
            self.env = None
    
    def execute_conversation(
        self,
        task_id: str,
        workflow_depth: int,
        agent1_tools_idx: int,
        agent1_budget_idx: int,
        agent2_tools_idx: int,
        agent2_budget_idx: int,
        answerer_budget_idx: int,
        selected_prompts: Dict[str, List[int]],
        max_turns: int = 50
    ) -> Tuple[float, Dict]:
        """
        Execute full tau2 conversation with HRL configuration.
        """
        # Initialize environment lazily if not already done
        if not self._env_initialized:
            self._init_env()
        
        # If gym registration failed, return placeholder
        if not getattr(self, '_tau2_gym_ready', False):
            print(f"  ⚠ Warning: Cannot execute tau2 conversation - gym environments not registered")
            execution_info = {
                "steps": 0,
                "tools_count": 0,
                "total_tokens": 0,
                "conversation_history": [],
                "pass_k": 0.0,
                "final_reward": 0.0,
                "task_completed": False,
                "error": "tau2 gym environment not available"
            }
            return 0.0, execution_info
        
        # Create environment for this specific task
        try:
            from tau2.gym import TAU_BENCH_ENV_ID
            import gymnasium as gym
            
            # Create environment with domain and task_id
            self.env = gym.make(TAU_BENCH_ENV_ID, domain=self.domain, task_id=task_id)
            print(f"  ✓ Created tau2 environment for task: {task_id}")
        except Exception as e:
            print(f"  ⚠ Warning: Cannot create tau2 environment for task {task_id}: {e}")
            execution_info = {
                "steps": 0,
                "tools_count": 0,
                "total_tokens": 0,
                "conversation_history": [],
                "pass_k": 0.0,
                "final_reward": 0.0,
                "task_completed": False,
                "error": f"Could not create environment: {e}"
            }
            return 0.0, execution_info
        
        # Build prompt suffixes
        reasoner_suffix = build_prompt_suffix("reasoner", selected_prompts.get("reasoner", []))
        verifier_suffix = build_prompt_suffix("verifier", selected_prompts.get("verifier", []))
        answerer_suffix = build_prompt_suffix("answerer", selected_prompts.get("answerer", []))
        
        # Get token budgets
        TOKEN_BUDGETS = {
            "reasoner": {0: 256, 1: 512, 2: 1024},
            "verifier": {0: 128, 1: 256, 2: 512},
            "answerer": {0: 64, 1: 128, 2: 256}
        }
        agent1_tokens = TOKEN_BUDGETS["reasoner"][agent1_budget_idx]
        agent2_tokens = TOKEN_BUDGETS["verifier"][agent2_budget_idx]
        answerer_tokens = TOKEN_BUDGETS["answerer"][answerer_budget_idx]
        
        # Decode tools
        agent1_tools = self.tools_registry.decode_tool_index(agent1_tools_idx)
        agent2_tools = self.tools_registry.decode_tool_index(agent2_tools_idx)
        
        # Get workflow
        if self.use_api:
            workflow = get_openrouter_workflow(workflow_depth, self.worker, self.tools_registry)
        else:
            workflow = get_workflow(workflow_depth, self.worker, self.tools_registry)
        
        # Special handling for workflow 2
        if workflow_depth == 2 and hasattr(workflow, 'use_verifier'):
            workflow.use_verifier = True
        
        # Reset environment with task
        obs, info = self.env.reset()
        initial_message = info.get("message", "") if isinstance(info, dict) else ""
        
        # Run conversation loop
        conversation_history = []
        turn_count = 0
        total_tokens = 0
        tools_used_count = 0
        
        current_message = initial_message
        done = False
        
        while not done and turn_count < max_turns:
            # Agent responds using workflow
            prompt_suffixes = {
                "reasoner": reasoner_suffix,
                "verifier": verifier_suffix,
                "answerer": answerer_suffix
            }
            
            # Execute workflow to get agent response
            agent_response, exec_info = workflow.execute(
                current_message,
                agent1_tools,
                agent1_budget_idx,
                agent2_tools,
                agent2_budget_idx,
                answerer_budget_idx,
                agent1_tokens,
                agent2_tokens,
                answerer_tokens,
                prompt_suffixes=prompt_suffixes
            )
            
            conversation_history.append(("agent", agent_response))
            total_tokens += exec_info.get("total_tokens", 0)
            tools_used_count += exec_info.get("tools_count", 0)
            turn_count += 1
            
            # Step environment with agent response
            obs, reward, terminated, truncated, info = self.env.step(agent_response)
            
            done = terminated or truncated
            
            if not done:
                # Get user response from info
                if isinstance(info, dict):
                    user_message = info.get("message", "")
                else:
                    user_message = str(obs) if obs else ""
                if user_message:
                    current_message = user_message
                    conversation_history.append(("user", user_message))
                else:
                    done = True
        
        # Calculate Pass^k reward (k=1 for now, can be extended)
        pass_k_reward = 1.0 if reward > 0 else 0.0
        
        # Get additional metrics from info
        execution_info = {
            "steps": turn_count,
            "tools_count": tools_used_count,
            "total_tokens": total_tokens,
            "conversation_history": conversation_history,
            "pass_k": pass_k_reward,
            "final_reward": reward,
            "task_completed": done and reward > 0
        }
        
        return pass_k_reward, execution_info