"""Tau2 integration for the Context_Optimization hierarchical RL framework.

Public surface:
- Tau2Dataset:        BaseDataset-shaped wrapper around a tau2 domain split.
- Tau2ToolRegistry:   ToolRegistry-shaped wrapper around a tau2 domain's toolkit.
- ConfiguredAgent:    Wraps any of our 9 workflows + atoms + tools into .respond(history).
- tau2_dialog_rollout: Drives multi-turn interaction with the gym env, returns reward + transcript.
"""
from .dataset import Tau2Dataset
from .tool_registry import Tau2ToolRegistry
from .configured_agent import ConfiguredAgent
from .rollout import tau2_dialog_rollout

__all__ = [
    "Tau2Dataset",
    "Tau2ToolRegistry",
    "ConfiguredAgent",
    "tau2_dialog_rollout",
]
