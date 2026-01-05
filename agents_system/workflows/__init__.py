"""
Workflow implementations for different agent patterns.
Each workflow implements a specific pattern from Anthropic's agent design guide.
"""

from .base import BaseWorkflow

# HuggingFace workflow imports
from .hugging_face import (
    DirectWorkflow,
    PromptChainingWorkflow,
    RoutingWorkflow,
    ParallelSectioningWorkflow,
    ParallelVotingWorkflow,
    OrchestratorWorkersWorkflow,
    EvaluatorOptimizerWorkflow,
    AutonomousAgentWorkflow,
)

# OpenRouter workflow imports
from .openrouter import (
    OpenRouterDirectWorkflow,
    OpenRouterPromptChainingWorkflow,
    OpenRouterRoutingWorkflow,
    OpenRouterParallelSectioningWorkflow,
    OpenRouterParallelVotingWorkflow,
    OpenRouterOrchestratorWorkersWorkflow,
    OpenRouterEvaluatorOptimizerWorkflow,
    OpenRouterAutonomousAgentWorkflow,
)

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
    # OpenRouter workflows
    'OpenRouterDirectWorkflow',
    'OpenRouterPromptChainingWorkflow',
    'OpenRouterRoutingWorkflow',
    'OpenRouterParallelSectioningWorkflow',
    'OpenRouterParallelVotingWorkflow',
    'OpenRouterOrchestratorWorkersWorkflow',
    'OpenRouterEvaluatorOptimizerWorkflow',
    'OpenRouterAutonomousAgentWorkflow',
    # Workflow getters
    'get_workflow',
    'get_openrouter_workflow',
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

# OpenRouter workflow registry
OPENROUTER_WORKFLOW_REGISTRY = {
    0: OpenRouterDirectWorkflow,
    1: OpenRouterPromptChainingWorkflow,  # Reason+Ans
    2: OpenRouterPromptChainingWorkflow,  # Reason+Verify+Ans (uses use_verifier=True)
    3: OpenRouterRoutingWorkflow,
    4: OpenRouterParallelSectioningWorkflow,
    5: OpenRouterParallelVotingWorkflow,
    6: OpenRouterOrchestratorWorkersWorkflow,
    7: OpenRouterEvaluatorOptimizerWorkflow,
    8: OpenRouterAutonomousAgentWorkflow,
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

def get_openrouter_workflow(workflow_depth: int, worker, tools_registry):
    """
    Get OpenRouter workflow instance for the given depth.
    
    Args:
        workflow_depth: Workflow index (0-8)
        worker: OpenRouterWorker instance
        tools_registry: ToolRegistry instance
        
    Returns:
        OpenRouter Workflow instance
    """
    workflow_class = OPENROUTER_WORKFLOW_REGISTRY.get(workflow_depth)
    if workflow_class is None:
        raise ValueError(f"Unknown workflow depth: {workflow_depth}")
    
    workflow = workflow_class(worker, tools_registry)
    
    # Special handling for workflow 2 (Reason+Verify+Ans)
    if workflow_depth == 2 and isinstance(workflow, OpenRouterPromptChainingWorkflow):
        workflow.use_verifier = True
    
    return workflow

