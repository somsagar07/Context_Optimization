"""
OpenRouter workflow implementations.
These workflows use OpenRouter API via OpenRouterWorker.
"""

from .openrouter_direct import OpenRouterDirectWorkflow
from .openrouter_prompt_chaining import OpenRouterPromptChainingWorkflow
from .openrouter_routing import OpenRouterRoutingWorkflow
from .openrouter_parallel_sectioning import OpenRouterParallelSectioningWorkflow
from .openrouter_parallel_voting import OpenRouterParallelVotingWorkflow
from .openrouter_orchestrator_workers import OpenRouterOrchestratorWorkersWorkflow
from .openrouter_evaluator_optimizer import OpenRouterEvaluatorOptimizerWorkflow
from .openrouter_autonomous_agent import OpenRouterAutonomousAgentWorkflow

__all__ = [
    'OpenRouterDirectWorkflow',
    'OpenRouterPromptChainingWorkflow',
    'OpenRouterRoutingWorkflow',
    'OpenRouterParallelSectioningWorkflow',
    'OpenRouterParallelVotingWorkflow',
    'OpenRouterOrchestratorWorkersWorkflow',
    'OpenRouterEvaluatorOptimizerWorkflow',
    'OpenRouterAutonomousAgentWorkflow',
]

