"""
Tau2 Tool Registry - Wraps tau2 domain-specific tools.
Dynamically loads tools from tau2 domains and maps them to indices.
"""
import os
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
        self.env = None
        self.available_tools = {}
        self.tool_names = []
        self._initialized = False
        
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
                    elif hasattr(tool_kit, 'get_tools'):
                        # Alternative: use get_tools() which returns Dict[str, Tool]
                        tools_dict = tool_kit.get_tools()
                        self.tool_names = list(tools_dict.keys())
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
    
    def _create_tool_wrapper(self, tool_name: str):
        """Create a wrapper function for a tau2 tool."""
        def tool_wrapper(query: str) -> str:
            """Execute tau2 tool with query."""
            try:
                # Tau2 tools are typically accessed through the environment
                # This is a placeholder - actual implementation depends on tau2 API
                if self.env and hasattr(self.env, 'execute_tool'):
                    result = self.env.execute_tool(tool_name, query)
                    return str(result)
                else:
                    # Fallback: return error message
                    return f"Tool {tool_name} not available (tau2 integration incomplete)"
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