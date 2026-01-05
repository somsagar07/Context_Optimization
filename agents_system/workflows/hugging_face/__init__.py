"""
HuggingFace workflow implementations.
These workflows use local HuggingFace models via LLMWorker.
"""

from .direct import DirectWorkflow
from .prompt_chaining import PromptChainingWorkflow
from .routing import RoutingWorkflow
from .parallel_sectioning import ParallelSectioningWorkflow
from .parallel_voting import ParallelVotingWorkflow
from .orchestrator_workers import OrchestratorWorkersWorkflow
from .evaluator_optimizer import EvaluatorOptimizerWorkflow
from .autonomous_agent import AutonomousAgentWorkflow

__all__ = [
    'DirectWorkflow',
    'PromptChainingWorkflow',
    'RoutingWorkflow',
    'ParallelSectioningWorkflow',
    'ParallelVotingWorkflow',
    'OrchestratorWorkersWorkflow',
    'EvaluatorOptimizerWorkflow',
    'AutonomousAgentWorkflow',
]

