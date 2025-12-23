"""
Multi-Step Environment Configuration.

In this mode, the agent makes SEQUENTIAL decisions:
- Step 0: Choose workflow depth [0, 1, 2]
- Step 1: Choose reasoner config (tools + budget) [24 options]
- Step 2: Choose verifier config (if depth=2) [24 options]
- Step 3: Choose answerer budget [3 options]

Benefits:
- Temporal credit assignment (gamma matters!)
- Smaller per-step action spaces
- Agent learns dependencies between decisions
- Intermediate shaping rewards guide exploration

Use this for:
- Better learning efficiency
- When you want the agent to learn decision structure
- Production training
"""
from configs.base import *

# --- Environment Mode ---
ENV_MODE = "multi_step"

TOTAL_TIMESTEPS = 15000 
N_STEPS = 512          # Fewer steps per rollout (episodes are 2-4 steps)
BATCH_SIZE = 64        # Stable batch size
LEARNING_RATE = 1e-4   # Lower LR for multi-step stability
GAMMA = 0.95           # Discount future rewards (now meaningful!)

# --- PPO Specific ---
ENT_COEF = 0.01        # Lower entropy (smaller action spaces per step)

