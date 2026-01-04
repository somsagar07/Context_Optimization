"""Orchestrator-workers workflow: Central LLM breaks down and delegates."""
from typing import Dict, List, Tuple, Optional
from .base import BaseWorkflow


class OrchestratorWorkersWorkflow(BaseWorkflow):
    """Workflow 6: Orchestrator-Workers - central LLM breaks down and delegates."""
    
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
        num_workers: int = 2
    ) -> Tuple[str, Dict]:
        """Execute orchestrator-workers workflow."""
        prompt_suffixes = prompt_suffixes or {}
        orchestrator_suffix = prompt_suffixes.get("orchestrator", None)
        reasoner_suffix = prompt_suffixes.get("reasoner", None)
        answerer_suffix = prompt_suffixes.get("answerer", None)
        
        exec_info = {
            "steps": 0,
            "tools_count": 0,
            "total_tokens": 0,
            "valid_code_count": 0,
            "file_access_count": 0,
        }
        
        # Step 1: Orchestrator breaks down the task (uses agent1_tools)
        breakdown_prompt = f"Break down this task into subtasks. Task: {question}"
        breakdown = self.worker.reason(
            breakdown_prompt,
            tools=agent1_tools,
            tokens=agent1_tokens // (num_workers + 1),
            prompt_suffix=orchestrator_suffix
        )
        if agent1_tools:
            breakdown, stats = self._process_tool_calls(breakdown, agent1_tools)
            exec_info["tools_count"] += stats.get("tool_calls", 0)
            exec_info["valid_code_count"] += stats.get("valid_code", 0)
            exec_info["file_access_count"] += stats.get("file_access", 0)
        exec_info["steps"] += 1
        exec_info["total_tokens"] += agent1_tokens // (num_workers + 1)
        
        # Step 2: Workers handle subtasks (use agent2_tools for different tool set)
        worker_results = []
        tokens_per_worker = agent1_tokens // (num_workers + 1)
        # Use agent2_tools for workers to allow different tool configuration
        worker_tools = agent2_tools if agent2_tools else agent1_tools
        
        for i in range(num_workers):
            worker_prompt = f"Subtask {i+1}: {breakdown}\nOriginal question: {question}"
            worker_result = self.worker.reason(
                worker_prompt,
                tools=worker_tools,
                tokens=tokens_per_worker,
                prompt_suffix=reasoner_suffix
            )
            if worker_tools:
                worker_result, stats = self._process_tool_calls(worker_result, worker_tools)
                exec_info["tools_count"] += stats.get("tool_calls", 0)
                exec_info["valid_code_count"] += stats.get("valid_code", 0)
                exec_info["file_access_count"] += stats.get("file_access", 0)
            
            worker_results.append(worker_result)
            exec_info["steps"] += 1
            exec_info["total_tokens"] += tokens_per_worker
        
        # Step 3: Orchestrator synthesizes results (answerer just synthesizes - no tools needed)
        synthesis_context = f"Task breakdown: {breakdown}\n"
        for i, result in enumerate(worker_results):
            synthesis_context += f"Worker {i+1}: {result}\n"
        
        final_text = self.worker.answer_with_context(
            question,
            context=synthesis_context,
            tools=[],  # Answerer just synthesizes - workers already did computation
            tokens=answerer_tokens,
            prompt_suffix=answerer_suffix
        )
        exec_info["steps"] += 1
        exec_info["total_tokens"] += answerer_tokens
        
        return final_text, exec_info

