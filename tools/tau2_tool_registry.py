"""
Tau2 Tool Registry - Wraps tau2 domain-specific tools.
Dynamically loads tools from tau2 domains and maps them to indices.
"""
import os
import json
os.environ.setdefault("LOGURU_LEVEL", "WARNING")
# Configure loguru to suppress DEBUG and INFO logs
try:
    from loguru import logger
    # Remove all existing handlers
    logger.remove()
    # Add handler that only shows WARNING and above
    logger.add(lambda msg: None, level="WARNING", filter=lambda record: record["level"].name in ["WARNING", "ERROR", "CRITICAL"])
except (ImportError, Exception):
    # loguru not available yet or already configured, continue
    pass

import gymnasium as gym

class Tau2ToolRegistry:
    """
    Tool registry for tau2 domains.
    Dynamically loads tools from tau2 gym environments.
    """
    
    def __init__(self, domain: str):
        """
        Initialize tau2 tool registry for a domain.
        
        Args:
            domain: tau2 domain name (airline, retail, telecom)
        """
        self.domain = domain
        self.env = None  # Will be set by Tau2ExecutionWrapper when env is created
        self.available_tools = {}
        self.tool_names = []
        self._initialized = False
        self._tool_kit = None  # Store reference to tool kit for execution
        
    def _init_tools(self):
        """Initialize tools from tau2 environment."""
        if self._initialized:
            return
            
        # Set tau2 data directory if not already set
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
            from tau2.registry import registry
            
            # Get tools directly from domain environment using registry
            try:
                # Get the domain environment constructor and create it
                env_constructor = registry.get_env_constructor(self.domain)
                domain_env = env_constructor()
                
                # Get tools from the domain environment
                # domain_env.tools is a ToolKitBase instance (e.g., AirlineTools)
                # We need to access its .tools property (returns Dict[str, Callable])
                # or use .get_tools() method (returns Dict[str, Tool])
                if hasattr(domain_env, 'tools'):
                    tool_kit = domain_env.tools  # This is a ToolKitBase instance
                    
                    # Access the tools property which returns Dict[str, Callable]
                    if hasattr(tool_kit, 'tools'):
                        tools_dict = tool_kit.tools  # This is the property that returns the dict
                        self.tool_names = list(tools_dict.keys())
                        self._tool_kit = tool_kit  # Store for later execution
                    elif hasattr(tool_kit, 'get_tools'):
                        # Alternative: use get_tools() which returns Dict[str, Tool]
                        tools_dict = tool_kit.get_tools()
                        self.tool_names = list(tools_dict.keys())
                        self._tool_kit = tool_kit  # Store for later execution
                    else:
                        raise AttributeError("ToolKit does not have tools property or get_tools method")
                    
                    if self.tool_names:
                        print(f"  ✓ Successfully loaded {len(self.tool_names)} tools from tau2 registry")
                else:
                    raise AttributeError("Domain environment does not have tools attribute")
                    
            except Exception as e:
                print(f"  ⚠ Could not get tools from tau2 registry: {e}")
                # Fallback: use known tools for each domain
                domain_tools = {
                    "airline": ["search_flights", "book_flight", "cancel_booking", "get_booking_info"],
                    "retail": ["search_products", "add_to_cart", "checkout", "get_order_status"],
                    "telecom": ["check_account", "update_plan", "get_billing_info", "troubleshoot"]
                }
                self.tool_names = domain_tools.get(self.domain, [])
                print(f"  ⚠ Using fallback tool list for {self.domain}: {self.tool_names}")
            
            # Create tool mapping
            for tool_name in self.tool_names:
                self.available_tools[tool_name] = self._create_tool_wrapper(tool_name)
            
            self._initialized = True
            print(f"Tau2 Tool Registry initialized for {self.domain}: {len(self.tool_names)} tools")
            if self.tool_names:
                print(f"  Tools: {', '.join(self.tool_names)}")
            
        except Exception as e:
            print(f"Warning: Could not initialize tau2 tools: {e}")
            print("Falling back to empty tool registry. Tools will need to be configured manually.")
            # Use fallback tools
            domain_tools = {
                "airline": ["search_flights", "book_flight", "cancel_booking", "get_booking_info"],
                "retail": ["search_products", "add_to_cart", "checkout", "get_order_status"],
                "telecom": ["check_account", "update_plan", "get_billing_info", "troubleshoot"]
            }
            self.tool_names = domain_tools.get(self.domain, [])
            for tool_name in self.tool_names:
                self.available_tools[tool_name] = self._create_tool_wrapper(tool_name)
            self._initialized = True
            print(f"  Using fallback tools: {self.tool_names}")
    
    def set_env(self, env):
        """Set the Tau2 gym environment for tool execution."""
        self.env = env
        
        # Try to get tool kit from environment if available
        if env:
            if hasattr(env, 'tools'):
                self._tool_kit = env.tools
            elif hasattr(env, 'domain_env') and hasattr(env.domain_env, 'tools'):
                self._tool_kit = env.domain_env.tools
            elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'tools'):
                self._tool_kit = env.unwrapped.tools
            
            # Update tool wrappers to use the new environment
            # Recreate wrappers so they have access to the env
            if self._initialized and self.tool_names:
                for tool_name in self.tool_names:
                    self.available_tools[tool_name] = self._create_tool_wrapper(tool_name)
    
    def _create_tool_wrapper(self, tool_name: str):
        """Create a wrapper function for a tau2 tool."""
        def tool_wrapper(query: str) -> str:
            """Execute tau2 tool with query."""
            try:
                # Parse JSON query if it's a JSON string
                # Tau2 tools expect keyword arguments, not JSON strings
                try:
                    query_dict = json.loads(query)
                    # If parsing succeeded, we have a dict - pass as keyword args
                    use_kwargs = True
                except (json.JSONDecodeError, TypeError):
                    # Not JSON, use as-is (for backward compatibility)
                    query_dict = {}
                    use_kwargs = False
                
                # Try multiple methods to execute the tool
                # Method 1: Direct tool kit execution (most common)
                if self._tool_kit:
                    if hasattr(self._tool_kit, 'tools') and tool_name in self._tool_kit.tools:
                        tool_func = self._tool_kit.tools[tool_name]
                        if use_kwargs:
                            # Pass as keyword arguments (Tau2 tools expect this)
                            result = tool_func(**query_dict)
                        else:
                            # Pass as string (fallback for non-JSON queries)
                            result = tool_func(query)
                        return str(result)
                    elif hasattr(self._tool_kit, 'execute'):
                        if use_kwargs:
                            result = self._tool_kit.execute(tool_name, **query_dict)
                        else:
                            result = self._tool_kit.execute(tool_name, query)
                        return str(result)
                
                # Method 2: Environment-level execution
                if self.env:
                    if hasattr(self.env, 'execute_tool'):
                        if use_kwargs:
                            result = self.env.execute_tool(tool_name, **query_dict)
                        else:
                            result = self.env.execute_tool(tool_name, query)
                        return str(result)
                    elif hasattr(self.env, 'tools') and hasattr(self.env.tools, 'execute'):
                        if use_kwargs:
                            result = self.env.tools.execute(tool_name, **query_dict)
                        else:
                            result = self.env.tools.execute(tool_name, query)
                        return str(result)
                
                # Fallback: return error message
                return f"Tool {tool_name} not available (tau2 environment not set or tool not found)"
            except Exception as e:
                return f"Error executing {tool_name}: {str(e)}"
        
        return tool_wrapper
    
    def get_tool_count(self) -> int:
        """Get number of available tools."""
        if not self._initialized:
            self._init_tools()
        return len(self.tool_names)
    
    def get_tool_names(self) -> list:
        """Get list of available tool names."""
        if not self._initialized:
            self._init_tools()
        return self.tool_names.copy()
    
    def decode_tool_index(self, idx: int) -> list:
        """
        Decode tool index to list of tool names (binary encoding).
        
        Args:
            idx: Tool index (binary encoding, e.g., 1=first tool, 2=second, 4=third, etc.)
            
        Returns:
            List of tool names
        """
        if not self._initialized:
            self._init_tools()
        
        tools = []
        for i, tool_name in enumerate(self.tool_names):
            if idx & (1 << i):
                tools.append(tool_name)
        return tools
    
    def get_max_tool_index(self) -> int:
        """Get maximum valid tool index (2^num_tools - 1)."""
        num_tools = self.get_tool_count()
        return (1 << num_tools) - 1 if num_tools > 0 else 0
    
    def execute(self, tool_name: str, query: str) -> str:
        """Execute a tool by name."""
        if not self._initialized:
            self._init_tools()
        
        if tool_name in self.available_tools:
            return self.available_tools[tool_name](query)
        return f"Error: Tool {tool_name} not found"
    
    def get_tool_prompt_descriptions(self) -> dict:
        """
        Get tool descriptions in the format expected by LLMWorker prompts.
        
        Returns:
            Dictionary mapping tool names to their prompt descriptions
        """
        if not self._initialized:
            self._init_tools()
        
        descriptions = {}
        
        # Try to get descriptions from tau2 library if available
        try:
            import tau2
            from tau2.registry import registry
            
            env_constructor = registry.get_env_constructor(self.domain)
            domain_env = env_constructor()
            
            if hasattr(domain_env, 'tools'):
                tool_kit = domain_env.tools
                
                # Try to get tools with descriptions
                if hasattr(tool_kit, 'get_tools'):
                    tools_dict = tool_kit.get_tools()
                    # If get_tools returns Tool objects with descriptions
                    for tool_name in self.tool_names:
                        if tool_name in tools_dict:
                            tool_obj = tools_dict[tool_name]
                            # Check if tool has description attribute
                            if hasattr(tool_obj, 'description'):
                                desc = tool_obj.description
                            elif hasattr(tool_obj, '__doc__') and tool_obj.__doc__:
                                desc = tool_obj.__doc__.strip()
                            else:
                                desc = None
                            
                            if desc:
                                # Format description similar to generic tools
                                descriptions[tool_name] = (
                                    f"{tool_name} - {desc}\n"
                                    f"  Example: TOOL: {tool_name} || QUERY: <your_query>"
                                )
        except Exception as e:
            print(f"Warning: Could not get tool descriptions from tau2 registry: {e}")
            pass  # Fall back to generic descriptions
        
        # Generate generic descriptions for tools without descriptions
        for tool_name in self.tool_names:
            if tool_name not in descriptions:
                # Create a readable description from tool name
                readable_name = tool_name.replace('_', ' ').title()
                descriptions[tool_name] = (
                    f"{tool_name} - {readable_name} for {self.domain} domain.\n"
                    f"  Example: TOOL: {tool_name} || QUERY: <your_query>"
                )
        
        return descriptions