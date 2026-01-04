"""
Base workflow class that all workflows inherit from.
Provides common functionality for tool execution and result tracking.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional


class BaseWorkflow(ABC):
    """Base class for all workflow implementations."""
    
    def __init__(self, worker, tools_registry):
        """
        Initialize workflow with worker and tools.
        
        Args:
            worker: LLMWorker instance
            tools_registry: ToolRegistry instance
        """
        self.worker = worker
        self.tools = tools_registry
    
    @abstractmethod
    def execute(
        self,
        question: str,
        agent1_tools: List[str],
        agent1_budget: int,
        agent2_tools: List[str],
        agent2_budget: int,
        answerer_budget: int,
        agent1_tokens: int,
        agent2_tokens: int,
        answerer_tokens: int,
        prompt_suffixes: Optional[Dict[str, str]] = None
    ) -> Tuple[str, Dict]:
        """
        Execute the workflow.
        
        Args:
            question: The question to answer
            agent1_tools: List of tool names for agent 1 (e.g., reasoner, orchestrator, aspect1)
            agent1_budget: Budget index (0=Low, 1=Mid, 2=High)
            agent2_tools: List of tool names for agent 2 (e.g., verifier, workers, aspect2)
            agent2_budget: Budget index (0=Low, 1=Mid, 2=High)
            answerer_budget: Budget index (0=Low, 1=Mid, 2=High)
            agent1_tokens: Token limit for agent 1
            agent2_tokens: Token limit for agent 2
            answerer_tokens: Token limit for answerer
            prompt_suffixes: Optional dict with 'reasoner', 'verifier', 'answerer' prompt suffixes
            
        Returns:
            Tuple of (final_answer, execution_info)
            execution_info contains: steps, tools_count, total_tokens, etc.
        """
        pass
    
    def _execute_agent_step(
        self,
        method,
        question: str,
        context: Optional[str] = None,
        tools: Optional[List[str]] = None,
        tokens: int = 256,
        prompt_suffix: Optional[str] = None
    ) -> str:
        """
        Execute a single agent step with optional tool execution.
        
        Args:
            method: Worker method to call (reason, verify, answer_direct, answer_with_context)
            question: The question
            context: Optional context/reasoning
            tools: Optional list of tool names
            tokens: Token budget
            prompt_suffix: Optional prompt suffix
            
        Returns:
            Response text
        """
        tools = tools or []
        
        if context:
            response = method(question, context, tools=tools, tokens=tokens, prompt_suffix=prompt_suffix)
        else:
            response = method(question, tools=tools, tokens=tokens, prompt_suffix=prompt_suffix)
        
        # Parse and execute tools if any
        if tools:
            tool_result = self.tools.parse_and_execute(response, tools)
            if tool_result:
                response += f"\nTool Output: {tool_result}"
        
        return response
    
    def _process_tool_calls(
        self,
        text_response: str,
        allowed_tools: List[str]
    ) -> Tuple[str, Dict]:
        """
        Process tool calls in response text.
        
        Args:
            text_response: Response text that may contain tool calls
            allowed_tools: List of allowed tool names
            
        Returns:
            Tuple of (updated_response, stats_dict)
        """
        if not allowed_tools:
            return text_response, {"tool_calls": 0, "valid_code": False, "file_access": False}
        
        # Simple tool call detection and execution
        import re
        tool_calls = 0
        pattern = r"TOOL:\s*(\w+)\s*\|\|\s*QUERY:\s*(.*)"
        matches = re.finditer(pattern, text_response)
        
        for match in matches:
            t_name, t_query = match.groups()
            t_name = t_name.strip().lower()
            
            if t_name in allowed_tools:
                try:
                    result = self.tools.execute(t_name, t_query)
                    text_response = text_response.replace(
                        match.group(0),
                        f"\n[System] Tool Output: {result}\n"
                    )
                    tool_calls += 1
                except Exception as e:
                    text_response = text_response.replace(
                        match.group(0),
                        f"\n[System] Tool Error: {str(e)}\n"
                    )
        
        stats = {
            "tool_calls": tool_calls,
            "valid_code": False,  # Could be enhanced
            "file_access": False  # Could be enhanced
        }
        
        return text_response, stats

