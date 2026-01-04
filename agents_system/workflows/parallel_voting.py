"""Parallel voting workflow: Run same task multiple times and aggregate."""
from typing import Dict, List, Tuple, Optional
from .base import BaseWorkflow


class ParallelVotingWorkflow(BaseWorkflow):
    """Workflow 5: Parallel-Voting - run same task multiple times and aggregate."""
    
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
        num_votes: int = 3
    ) -> Tuple[str, Dict]:
        """Execute parallel voting workflow."""
        prompt_suffixes = prompt_suffixes or {}
        answerer_suffix = prompt_suffixes.get("answerer", None)
        
        exec_info = {
            "steps": 0,
            "tools_count": 0,
            "total_tokens": 0,
            "valid_code_count": 0,
            "file_access_count": 0,
        }
        
        # Generate multiple votes (can use tools if provided)
        # Note: Parallel-Voting typically doesn't use reasoner, but if tools are provided,
        # we can use them in the answerer step for calculations/verification
        votes = []
        for i in range(num_votes):
            # Use agent1_tools if available (for calculations/verification)
            vote_tools = agent1_tools if agent1_tools else []
            vote = self.worker.answer_direct(
                question,
                tools=vote_tools,
                tokens=answerer_tokens // num_votes,
                prompt_suffix=answerer_suffix
            )
            # Process tool calls if tools were used
            if vote_tools:
                vote, stats = self._process_tool_calls(vote, vote_tools)
                exec_info["tools_count"] += stats.get("tool_calls", 0)
                exec_info["valid_code_count"] += stats.get("valid_code", 0)
                exec_info["file_access_count"] += stats.get("file_access", 0)
            
            votes.append(vote)
            exec_info["steps"] += 1
            exec_info["total_tokens"] += answerer_tokens // num_votes
        
        # Aggregate votes (answerer just synthesizes - no tools needed)
        votes_text = "\n".join([f"Vote {i+1}: {v}" for i, v in enumerate(votes)])
        final_text = self.worker.answer_with_context(
            question,
            context=f"Multiple attempts:\n{votes_text}\nProvide the most consistent answer.",
            tools=[],  # Answerer just synthesizes - votes already did computation
            tokens=answerer_tokens,
            prompt_suffix=answerer_suffix
        )
        exec_info["steps"] += 1
        exec_info["total_tokens"] += answerer_tokens
        
        return final_text, exec_info

