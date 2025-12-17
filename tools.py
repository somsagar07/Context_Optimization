# tools.py
import re
from sympy import sympify, N
from duckduckgo_search import DDGS

class ToolRegistry:
    def __init__(self):
        # Map tool names to their functions
        self.available_tools = {
            "calculator": self.calculator,
            "web_search": self.web_search
        }

    def calculator(self, query: str) -> str:
        """Safe math evaluation using SymPy."""
        try:
            # Clean non-math chars
            clean_expr = re.sub(r'[^\d+\-*/().^]', '', query)
            if not clean_expr: return "Error: Invalid Math"
            val = N(sympify(clean_expr))
            
            # Formatting
            if abs(val - round(val)) < 1e-9:
                return str(int(round(val)))
            return str(round(float(val), 4))
        except:
            return "Error: Calc Failed"

    def web_search(self, query: str) -> str:
        """DuckDuckGo Search Wrapper."""
        try:
            with DDGS() as ddgs:
                # Get 1 result to save context window space
                results = list(ddgs.text(query, max_results=1))
                if results:
                    return f"Search Result: {results[0]['body']}"
                return "No results found."
        except Exception as e:
            return f"Search Error: {str(e)}"

    def execute(self, tool_name: str, query: str) -> str:
        if tool_name in self.available_tools:
            return self.available_tools[tool_name](query)
        return "Error: Tool not found"

    def parse_and_execute(self, response_text: str, allowed_tools: list) -> str:
        """
        Scans model output for tool calls like: TOOL: <name> || QUERY: <query>
        """
        if not allowed_tools:
            return ""

        # Regex for generic tool usage: "TOOL: calculator || QUERY: 25*50"
        # You can prompt the LLM to output this specific format.
        pattern = r"TOOL:\s*(\w+)\s*\|\|\s*QUERY:\s*(.*)"
        match = re.search(pattern, response_text)
        
        if match:
            t_name, t_query = match.groups()
            t_name = t_name.strip().lower()
            
            if t_name in allowed_tools:
                result = self.execute(t_name, t_query)
                return f"\n[System] Tool Output: {result}\n"
        
        return ""