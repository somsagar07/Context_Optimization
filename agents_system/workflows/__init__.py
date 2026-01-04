"""
Workflow implementations for different agent patterns.
Each workflow implements a specific pattern from Anthropic's agent design guide.
"""

from .base import BaseWorkflow
from .direct import DirectWorkflow
from .prompt_chaining import PromptChainingWorkflow
from .routing import RoutingWorkflow
from .parallel_sectioning import ParallelSectioningWorkflow
from .parallel_voting import ParallelVotingWorkflow
from .orchestrator_workers import OrchestratorWorkersWorkflow
from .evaluator_optimizer import EvaluatorOptimizerWorkflow
from .autonomous_agent import AutonomousAgentWorkflow

__all__ = [
    'BaseWorkflow',
    'DirectWorkflow',
    'PromptChainingWorkflow',
    'RoutingWorkflow',
    'ParallelSectioningWorkflow',
    'ParallelVotingWorkflow',
    'OrchestratorWorkersWorkflow',
    'EvaluatorOptimizerWorkflow',
    'AutonomousAgentWorkflow',
]

# Workflow registry for easy lookup
WORKFLOW_REGISTRY = {
    0: DirectWorkflow,
    1: PromptChainingWorkflow,  # Reason+Ans
    2: PromptChainingWorkflow,  # Reason+Verify+Ans (uses use_verifier=True)
    3: RoutingWorkflow,
    4: ParallelSectioningWorkflow,
    5: ParallelVotingWorkflow,
    6: OrchestratorWorkersWorkflow,
    7: EvaluatorOptimizerWorkflow,
    8: AutonomousAgentWorkflow,
}

def get_workflow(workflow_depth: int, worker, tools_registry):
    """
    Get workflow instance for the given depth.
    
    Args:
        workflow_depth: Workflow index (0-8)
        worker: LLMWorker instance
        tools_registry: ToolRegistry instance
        
    Returns:
        Workflow instance
    """
    workflow_class = WORKFLOW_REGISTRY.get(workflow_depth)
    if workflow_class is None:
        raise ValueError(f"Unknown workflow depth: {workflow_depth}")
    
    workflow = workflow_class(worker, tools_registry)
    
    # Special handling for workflow 2 (Reason+Verify+Ans)
    if workflow_depth == 2 and isinstance(workflow, PromptChainingWorkflow):
        workflow.use_verifier = True
    
    return workflow

