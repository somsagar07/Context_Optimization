"""
Single-Step Environment Configuration.

In this mode, the agent selects ALL action dimensions at once:
- Action Space: [3, 8, 3, 8, 3, 3] = 5,184 combinations
- Episode Length: 1 step (all decisions made simultaneously)
- No temporal credit assignment (gamma is effectively useless)

Use this for:
- Baseline comparison
- Simple/fast training
- When you don't need to learn decision dependencies
"""
from configs.base import *

# --- Environment Mode ---
ENV_MODE = "single_step"

TOTAL_TIMESTEPS = 15000 
N_STEPS = 2048         # More steps per update (single-step episodes are fast)
BATCH_SIZE = 64        # Standard batch size
LEARNING_RATE = 3e-4   # Higher LR is fine for single-step
GAMMA = 0.0            # No future rewards to discount in single-step episodes!

# --- PPO Specific ---
ENT_COEF = 0.02        # Higher entropy for exploration in large action space

