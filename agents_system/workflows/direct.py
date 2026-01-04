"""Direct workflow: Single LLM call to answer directly."""
from typing import Dict, List, Tuple, Optional
from .base import BaseWorkflow


class DirectWorkflow(BaseWorkflow):
    """Workflow 0: Direct answer without reasoning or verification."""
    
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
        """Execute direct answer workflow."""
        prompt_suffixes = prompt_suffixes or {}
        answerer_suffix = prompt_suffixes.get("answerer", None)
        
        # Use agent1_tools for the direct answerer (can use any combination of tools)
        answerer_tools = agent1_tools if agent1_tools else []
        final_text = self.worker.answer_direct(
            question,
            tools=answerer_tools,
            tokens=answerer_tokens,
            prompt_suffix=answerer_suffix
        )
        
        # Process tool calls if tools were used
        exec_info = {
            "steps": 1,
            "tools_count": 0,
            "total_tokens": answerer_tokens,
            "valid_code_count": 0,
            "file_access_count": 0,
        }
        
        if answerer_tools:
            final_text, stats = self._process_tool_calls(final_text, answerer_tools)
            exec_info["tools_count"] += stats.get("tool_calls", 0)
            exec_info["valid_code_count"] += stats.get("valid_code", 0)
            exec_info["file_access_count"] += stats.get("file_access", 0)
        
        return final_text, exec_info

