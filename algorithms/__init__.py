"""
Algorithms module for hierarchical RL training.

Supported algorithms:
- PPO: Proximal Policy Optimization (with value function)
- GRPO: Group Relative Policy Optimization (critic-free)
"""
from algorithms.base import Algorithm, BaseTrainer
from algorithms.ppo import PPOTrainer
from algorithms.grpo import GRPOTrainer

__all__ = [
    "Algorithm",
    "BaseTrainer", 
    "PPOTrainer",
    "GRPOTrainer",
]

