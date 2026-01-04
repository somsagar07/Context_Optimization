"""Routing workflow: Classify input and route to specialized handlers."""
from typing import Dict, List, Tuple, Optional
from .base import BaseWorkflow


class RoutingWorkflow(BaseWorkflow):
    """Workflow 3: Routing - classify input and route to specialized handler."""
    
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
        """Execute routing workflow."""
        prompt_suffixes = prompt_suffixes or {}
        reasoner_suffix = prompt_suffixes.get("reasoner", None)
        router_suffix = prompt_suffixes.get("router", None)
        answerer_suffix = prompt_suffixes.get("answerer", None)
        
        exec_info = {
            "steps": 0,
            "tools_count": 0,
            "total_tokens": 0,
            "valid_code_count": 0,
            "file_access_count": 0,
        }
        
        # Step 1: Classify (uses agent1_tools)
        classification_prompt = (
            f"Classify this question into one category: [simple, complex, multi-step]. "
            f"Respond with ONLY the category name. Question: {question}"
        )
        classification = self.worker.reason(
            classification_prompt,
            tools=agent1_tools,
            tokens=agent1_tokens // 3,
            prompt_suffix=router_suffix
        )
        if agent1_tools:
            classification, stats = self._process_tool_calls(classification, agent1_tools)
            exec_info["tools_count"] += stats.get("tool_calls", 0)
            exec_info["valid_code_count"] += stats.get("valid_code", 0)
            exec_info["file_access_count"] += stats.get("file_access", 0)
        exec_info["steps"] += 1
        exec_info["total_tokens"] += agent1_tokens // 3
        
        # Step 2: Route to one of two reasoners based on classification
        # Simple/complex → use agent1 (Reasoner1), Multi-step → use agent2 (Reasoner2)
        classification_lower = classification.lower()
        if "multi-step" in classification_lower or "multi step" in classification_lower:
            # Route to Reasoner2 (agent2)
            reasoning_tools = agent2_tools if agent2_tools else agent1_tools
            reasoning_tokens = agent2_tokens if agent2_tools else agent1_tokens // 2
            reasoner_name = "Reasoner2 (agent2)"
        else:
            # Route to Reasoner1 (agent1)
            reasoning_tools = agent1_tools
            reasoning_tokens = agent1_tokens // 2
            reasoner_name = "Reasoner1 (agent1)"
        
        reasoning = self.worker.reason(
            question,
            tools=reasoning_tools,
            tokens=reasoning_tokens,
            prompt_suffix=reasoner_suffix
        )
        if reasoning_tools:
            reasoning, stats = self._process_tool_calls(reasoning, reasoning_tools)
            exec_info["tools_count"] += stats.get("tool_calls", 0)
            exec_info["valid_code_count"] += stats.get("valid_code", 0)
            exec_info["file_access_count"] += stats.get("file_access", 0)
        
        exec_info["steps"] += 1
        exec_info["total_tokens"] += reasoning_tokens
        
        # Step 3: Answer with classification and routing context (synthesizes - no tools needed)
        final_text = self.worker.answer_with_context(
            question,
            context=f"Classification: {classification}\nRouted to: {reasoner_name}\nReasoning: {reasoning}",
            tools=[],  # Answerer just synthesizes - previous agents already did computation
            tokens=answerer_tokens,
            prompt_suffix=answerer_suffix
        )
        exec_info["steps"] += 1
        exec_info["total_tokens"] += answerer_tokens
        
        return final_text, exec_info

