"""
Tau2 Execution Wrapper - Executes full tau2 conversations with HRL configuration.
"""
import os

# Suppress verbose tau2 logging (tau2 uses loguru, not standard logging)
os.environ.setdefault("LOGURU_LEVEL", "WARNING")
# Try to configure loguru before tau2 imports it
try:
    from loguru import logger
    logger.remove()  # Remove default handler
    logger.add(lambda msg: None, level="WARNING")  # Only show WARNING and above
except ImportError:
    pass  # loguru not available yet

# Load .env file FIRST (before tau2 imports that might need API keys)
try:
    from dotenv import load_dotenv
    # Try to load from project root
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        # Try current directory
        load_dotenv()
except ImportError:
    # python-dotenv not installed, skip
    pass
except Exception:
    # Failed to load, continue anyway
    pass

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
    
    def _extract_user_message(self, obs) -> str:
        """
        Extract the last user message from observation.
        
        The observation from tau2 gym environment is a list of message objects.
        We need to find the last message with role='user' and extract its content.
        
        Args:
            obs: Observation from tau2 gym environment (list of message objects or dict)
            
        Returns:
            str: Content of the last user message, or empty string if not found
        """
        if not obs:
            return ""
        
        # If obs is a list of message objects
        if isinstance(obs, list):
            # Iterate backwards to find the last user message
            for msg in reversed(obs):
                # Check if message has role attribute
                if hasattr(msg, 'role'):
                    if msg.role == 'user' and hasattr(msg, 'content'):
                        return msg.content or ""
                # Also check if it's a dict-like object
                elif isinstance(msg, dict):
                    if msg.get('role') == 'user':
                        return msg.get('content', '')
        
        # If obs is a dict, try to get message directly
        if isinstance(obs, dict):
            return obs.get('content', '') if obs.get('role') == 'user' else ''
        
        # Fallback: try to convert to string
        return str(obs) if obs else ""
    
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
            
            # Share environment with tool registry so tools can be executed
            if hasattr(self.tools_registry, 'set_env'):
                self.tools_registry.set_env(self.env)
            elif hasattr(self.tools_registry, 'env'):
                self.tools_registry.env = self.env
            
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
        
        # Extract initial message from observation (tau2 returns messages in obs)
        initial_message = self._extract_user_message(obs)
        # Fallback to info dict if observation extraction failed
        if not initial_message and isinstance(info, dict):
            initial_message = info.get("message", "")
        
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
                # Extract user message from observation (tau2 returns messages in obs)
                user_message = self._extract_user_message(obs)
                # Fallback to info dict if observation extraction failed
                if not user_message and isinstance(info, dict):
                    user_message = info.get("message", "")
                
                if user_message:
                    current_message = user_message
                    conversation_history.append(("user", user_message))
                else:
                    # No user message found, conversation might be done
                    done = True
        
        # Calculate Pass^k reward (k=1 for now, can be extended)
        # pass_k_reward = 1.0 if reward > 0 else 0.0
        pass_k_reward = reward
        
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