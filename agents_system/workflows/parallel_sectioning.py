"""Parallel sectioning workflow: Break task into independent parallel subtasks."""
from typing import Dict, List, Tuple, Optional
from .base import BaseWorkflow


class ParallelSectioningWorkflow(BaseWorkflow):
    """Workflow 4: Parallel-Sectioning - break into independent parallel subtasks."""
    
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
        """Execute parallel sectioning workflow."""
        prompt_suffixes = prompt_suffixes or {}
        # orchestrator_suffix = prompt_suffixes.get("orchestrator", None)
        reasoner_suffix = prompt_suffixes.get("reasoner", None)
        answerer_suffix = prompt_suffixes.get("answerer", None)
        # aggregator_suffix = prompt_suffixes.get("aggregator", None)
        
        exec_info = {
            "steps": 0,
            "tools_count": 0,
            "total_tokens": 0,
            "valid_code_count": 0,
            "file_access_count": 0,
        }
        
        # Step 1: Aspect1 agent breaks down the task into subtasks (uses agent1_tools)
        breakdown_prompt = (
            f"Break down this question into 2 independent subtasks that can be solved in parallel. "
            f"Question: {question}"
        )
        breakdown = self.worker.reason(
            breakdown_prompt,
            tools=agent1_tools,
            tokens=agent1_tokens // 3,
            prompt_suffix=reasoner_suffix #orchestrator_suffix
        )
        if agent1_tools:
            breakdown, stats = self._process_tool_calls(breakdown, agent1_tools)
            exec_info["tools_count"] += stats.get("tool_calls", 0)
            exec_info["valid_code_count"] += stats.get("valid_code", 0)
            exec_info["file_access_count"] += stats.get("file_access", 0)
        exec_info["steps"] += 1
        exec_info["total_tokens"] += agent1_tokens // 3
        
        # Step 2: Process subtasks in parallel (simulated sequentially)
        # Worker1 uses agent2_tools, Worker2 reuses agent1_tools with different context
        tokens_per_worker = agent1_tokens // 3
        
        # Worker1: Uses agent2_tools (or agent1_tools if agent2 not available)
        worker1_tools = agent2_tools if agent2_tools else agent1_tools
        worker1_prompt = f"Subtask 1: {breakdown}\nOriginal question: {question}\nSolve this subtask."
        result1 = self.worker.reason(
            worker1_prompt,
            tools=worker1_tools,
            tokens=tokens_per_worker,
            prompt_suffix=reasoner_suffix
        )
        if worker1_tools:
            result1, stats = self._process_tool_calls(result1, worker1_tools)
            exec_info["tools_count"] += stats.get("tool_calls", 0)
            exec_info["valid_code_count"] += stats.get("valid_code", 0)
            exec_info["file_access_count"] += stats.get("file_access", 0)
        exec_info["steps"] += 1
        exec_info["total_tokens"] += tokens_per_worker
        
        # Worker2: Reuses agent1_tools (same agent, different subtask)
        worker2_prompt = f"Subtask 2: {breakdown}\nOriginal question: {question}\nSolve this subtask."
        result2 = self.worker.reason(
            worker2_prompt,
            tools=agent1_tools,
            tokens=tokens_per_worker,
            prompt_suffix=reasoner_suffix
        )
        if agent1_tools:
            result2, stats = self._process_tool_calls(result2, agent1_tools)
            exec_info["tools_count"] += stats.get("tool_calls", 0)
            exec_info["valid_code_count"] += stats.get("valid_code", 0)
            exec_info["file_access_count"] += stats.get("file_access", 0)
        exec_info["steps"] += 1
        exec_info["total_tokens"] += tokens_per_worker
        
        # Step 3: Synthesize results (answerer just synthesizes - no tools needed)
        combined_context = f"Task breakdown: {breakdown}\nWorker 1 result: {result1}\nWorker 2 result: {result2}"
        final_text = self.worker.answer_with_context(
            question,
            context=combined_context,
            tools=[],  # Answerer just synthesizes - previous agents already did computation
            tokens=answerer_tokens,
            prompt_suffix= answerer_suffix # aggregator_suffix
        )
        exec_info["steps"] += 1
        exec_info["total_tokens"] += answerer_tokens
        
        return final_text, exec_info

