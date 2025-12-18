import re
from .calculator import calculator
from .web_search import web_search
from .python_executor import python_executor

class ToolRegistry:
    """Central registry for all available tools."""
    
    def __init__(self):
        self.available_tools = {
            "calculator": calculator,
            "web_search": web_search,
            "python": python_executor
        }
    
    def execute(self, tool_name: str, query: str) -> str:
        """Execute a tool by name with the given query."""
        if tool_name in self.available_tools:
            return self.available_tools[tool_name](query)
        return "Error: Tool not found"
    
    def parse_and_execute(self, response_text: str, allowed_tools: list) -> str:
        """
        Scans model output for tool calls like: TOOL: <name> || QUERY: <query>
        """
        if not allowed_tools:
            return ""

        pattern = r"TOOL:\s*(\w+)\s*\|\|\s*QUERY:\s*(.*)"
        match = re.search(pattern, response_text)
        
        if match:
            t_name, t_query = match.groups()
            t_name = t_name.strip().lower()
            
            if t_name in allowed_tools:
                result = self.execute(t_name, t_query)
                return f"\n[System] Tool Output: {result}\n"
        
        return ""
    
    def list_tools(self) -> list:
        """Return list of available tool names."""
        return list(self.available_tools.keys())

