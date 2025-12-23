"""
Environment exports for Context_Opt.

Available environments:
- GeneralAgentEnv: Single-step environment (all decisions at once)
- MultiStepAgentEnv: Multi-step environment (sequential decisions)
- StructureEnv: High-level policy env for structure/tools/budget (hierarchical)
- PromptEnv: Low-level policy env for prompt selection (hierarchical)
"""
from .general_env import GeneralAgentEnv
from .multistep_env import MultiStepAgentEnv
from .structure_env import StructureEnv
from .prompt_env import PromptEnv

__all__ = [
    'GeneralAgentEnv',      # single_step mode
    'MultiStepAgentEnv',    # multi_step mode
    'StructureEnv',         # hierarchical mode (high-level)
    'PromptEnv',            # hierarchical mode (low-level)
]
