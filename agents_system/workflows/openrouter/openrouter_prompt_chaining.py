"""OpenRouter Prompt chaining workflows: Reason+Ans and Reason+Verify+Ans using API."""
from typing import Dict, List, Tuple, Optional
from ..base import BaseWorkflow


class OpenRouterPromptChainingWorkflow(BaseWorkflow):
    """OpenRouter Workflows 1 & 2: Prompt chaining patterns using API."""
    
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
        prompt_suffixes: Optional[Dict[str, str]] = None,
        use_verifier: bool = False
    ) -> Tuple[str, Dict]:
        """
        Execute prompt chaining workflow using OpenRouter API.
        
        Args:
            use_verifier: If True, uses Reason+Verify+Ans (workflow 2), 
                         else Reason+Ans (workflow 1)
        """
        prompt_suffixes = prompt_suffixes or {}
        reasoner_suffix = prompt_suffixes.get("reasoner", None)
        verifier_suffix = prompt_suffixes.get("verifier", None)
        answerer_suffix = prompt_suffixes.get("answerer", None)
        
        exec_info = {
            "steps": 0,
            "tools_count": 0,
            "total_tokens": 0,
            "valid_code_count": 0,
            "file_access_count": 0,
        }
        
        # Step 1: Reason (agent1)
        reasoning = self.worker.reason(
            question,
            tools=agent1_tools,
            tokens=agent1_tokens,
            prompt_suffix=reasoner_suffix
        )
        if agent1_tools:
            reasoning, stats = self._process_tool_calls(reasoning, agent1_tools)
            exec_info["tools_count"] += stats.get("tool_calls", 0)
            exec_info["valid_code_count"] += stats.get("valid_code", 0)
            exec_info["file_access_count"] += stats.get("file_access", 0)
        
        exec_info["steps"] += 1
        exec_info["total_tokens"] += agent1_tokens
        
        # Step 2: Verify (if needed, agent2)
        if use_verifier:
            critique = self.worker.verify(
                question,
                reasoning=reasoning,
                tools=agent2_tools,
                tokens=agent2_tokens,
                prompt_suffix=verifier_suffix
            )
            if agent2_tools:
                critique, stats = self._process_tool_calls(critique, agent2_tools)
                exec_info["tools_count"] += stats.get("tool_calls", 0)
                exec_info["valid_code_count"] += stats.get("valid_code", 0)
                exec_info["file_access_count"] += stats.get("file_access", 0)
            
            context = f"Reasoning: {reasoning}\nReview: {critique}"
            exec_info["steps"] += 1
            exec_info["total_tokens"] += agent2_tokens
        else:
            context = reasoning
        
        # Step 3: Answer (synthesizes context - no tools needed, just formatting)
        final_text = self.worker.answer_with_context(
            question,
            context=context,
            tools=[],  # Answerer just synthesizes - previous agents already did computation
            tokens=answerer_tokens,
            prompt_suffix=answerer_suffix
        )
        
        exec_info["steps"] += 1
        exec_info["total_tokens"] += answerer_tokens
        
        return final_text, exec_info

