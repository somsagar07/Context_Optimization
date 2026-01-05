"""Autonomous agent workflow: LLM uses tools autonomously in a loop."""
from typing import Dict, List, Tuple, Optional
from ..base import BaseWorkflow


class AutonomousAgentWorkflow(BaseWorkflow):
    """Workflow 8: Autonomous Agent - LLM uses tools autonomously in a loop."""
    
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
        max_iterations: int = 5
    ) -> Tuple[str, Dict]:
        """Execute autonomous agent workflow."""
        prompt_suffixes = prompt_suffixes or {}
        reasoner_suffix = prompt_suffixes.get("reasoner", None)
        answerer_suffix = prompt_suffixes.get("answerer", None)
        
        exec_info = {
            "steps": 0,
            "tools_count": 0,
            "total_tokens": 0,
            "valid_code_count": 0,
            "file_access_count": 0,
        }
        
        context_history = []
        tokens_per_iteration = agent1_tokens // max_iterations
        
        for iteration in range(max_iterations):
            # Agent reasons and decides on next action
            # Use agent1_tools for first iteration, agent2_tools for later iterations
            iteration_tools = agent1_tools if iteration == 0 else (agent2_tools if agent2_tools else agent1_tools)
            
            if iteration == 0:
                reasoning = self.worker.reason(
                    question,
                    tools=iteration_tools,
                    tokens=tokens_per_iteration,
                    prompt_suffix=reasoner_suffix
                )
            else:
                reasoning_prompt = (
                    f"Question: {question}\n"
                    f"Previous context: {context_history[-1]}\n"
                    f"Continue reasoning."
                )
                reasoning = self.worker.reason(
                    reasoning_prompt,
                    tools=iteration_tools,
                    tokens=tokens_per_iteration,
                    prompt_suffix=reasoner_suffix
                )
            
            # Process tool calls (for all iterations)
            if iteration_tools:
                reasoning, stats = self._process_tool_calls(reasoning, iteration_tools)
                exec_info["tools_count"] += stats.get("tool_calls", 0)
                exec_info["valid_code_count"] += stats.get("valid_code", 0)
                exec_info["file_access_count"] += stats.get("file_access", 0)
            
            context_history.append(reasoning)
            exec_info["steps"] += 1
            exec_info["total_tokens"] += tokens_per_iteration
            
            # Simple stopping condition: stop after 3 iterations or if we have enough info
            if iteration >= 2:
                break
        
        # Final answer (answerer just synthesizes - no tools needed)
        combined_context = "\n".join([f"Step {i+1}: {ctx}" for i, ctx in enumerate(context_history)])
        final_text = self.worker.answer_with_context(
            question,
            context=combined_context,
            tools=[],  # Answerer just synthesizes - previous iterations already did computation
            tokens=answerer_tokens,
            prompt_suffix=answerer_suffix
        )
        exec_info["steps"] += 1
        exec_info["total_tokens"] += answerer_tokens
        
        return final_text, exec_info

