"""Evaluator-optimizer workflow: Generate → Evaluate → Refine loop."""
from typing import Dict, List, Tuple, Optional
from ..base import BaseWorkflow


class EvaluatorOptimizerWorkflow(BaseWorkflow):
    """Workflow 7: Evaluator-Optimizer - generate → evaluate → refine loop."""
    
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
        max_iterations: int = 3
    ) -> Tuple[str, Dict]:
        """Execute evaluator-optimizer workflow."""
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
        
        current_answer = None
        
        for iteration in range(max_iterations):
            # Generate/Refine
            if iteration == 0:
                current_answer = self.worker.reason(
                    question,
                    tools=agent1_tools,
                    tokens=agent1_tokens // max_iterations,
                    prompt_suffix=reasoner_suffix
                )
                if agent1_tools:
                    current_answer, stats = self._process_tool_calls(current_answer, agent1_tools)
                    exec_info["tools_count"] += stats.get("tool_calls", 0)
                    exec_info["valid_code_count"] += stats.get("valid_code", 0)
                    exec_info["file_access_count"] += stats.get("file_access", 0)
            else:
                refine_prompt = (
                    f"Question: {question}\n"
                    f"Previous attempt: {current_answer}\n"
                    f"Improve this answer."
                )
                current_answer = self.worker.reason(
                    refine_prompt,
                    tools=agent1_tools,
                    tokens=agent1_tokens // max_iterations,
                    prompt_suffix=reasoner_suffix
                )
                if agent1_tools:
                    current_answer, stats = self._process_tool_calls(current_answer, agent1_tools)
                    exec_info["tools_count"] += stats.get("tool_calls", 0)
                    exec_info["valid_code_count"] += stats.get("valid_code", 0)
                    exec_info["file_access_count"] += stats.get("file_access", 0)
            
            exec_info["steps"] += 1
            exec_info["total_tokens"] += agent1_tokens // max_iterations
            
            # Evaluate
            evaluation = self.worker.verify(
                question,
                reasoning=current_answer,
                tools=agent2_tools,
                tokens=agent2_tokens // max_iterations,
                prompt_suffix=verifier_suffix
            )
            if agent2_tools:
                evaluation, stats = self._process_tool_calls(evaluation, agent2_tools)
                exec_info["tools_count"] += stats.get("tool_calls", 0)
                exec_info["valid_code_count"] += stats.get("valid_code", 0)
                exec_info["file_access_count"] += stats.get("file_access", 0)
            exec_info["steps"] += 1
            exec_info["total_tokens"] += agent2_tokens // max_iterations
            
            # Check if evaluation says it's good enough
            if "correct" in evaluation.lower() and iteration > 0:
                break
        
        # Final answer (answerer just synthesizes - no tools needed)
        final_text = self.worker.answer_with_context(
            question,
            context=f"Refined reasoning: {current_answer}",
            tools=[],  # Answerer just synthesizes - generator/evaluator already did computation
            tokens=answerer_tokens,
            prompt_suffix=answerer_suffix
        )
        exec_info["steps"] += 1
        exec_info["total_tokens"] += answerer_tokens
        
        return final_text, exec_info

