"""
Hierarchical Environment Configuration (Dual-Policy)

Uses TWO separate policy networks:
1. STRUCTURE POLICY (High-Level Manager):
   - Single-step, MultiDiscrete action space
   - Selects: workflow, tools, budgets
   - Action: MultiDiscrete([9, 16, 3, 16, 3, 3])  

2. PROMPT POLICY (Low-Level Worker):
   - Multi-step, Discrete action space
   - Sequential prompt selection per agent
   - Action: Discrete(7) where 0=DONE, 1-6=prompt atoms

Benefits:
- Each network specializes in its role
- Structure uses MultiDiscrete (interpretable, no encoding)
- Prompts get proper credit assignment
- Separate hyperparameters per policy
"""
from configs.base import *

# --- Environment Mode ---
ENV_MODE = "hierarchical"

# ==============================================================================
# STRUCTURE POLICY (High-Level)
# ==============================================================================
STRUCTURE_TOTAL_TIMESTEPS = 10000   # Fewer steps needed (single-step env)
STRUCTURE_N_STEPS = 128             # Short rollouts
STRUCTURE_BATCH_SIZE = 32
STRUCTURE_LEARNING_RATE = 3e-4      # Lowered from 5e-4 to reduce loss spikes
STRUCTURE_GAMMA = 0.0               # No future reward (immediate)
STRUCTURE_ENT_COEF = 0.03           # Higher entropy for exploration

# ==============================================================================
# PROMPT POLICY (Low-Level)
# ==============================================================================
PROMPT_TOTAL_TIMESTEPS = 20000      # More steps for sequential learning
PROMPT_N_STEPS = 512                # Longer rollouts for multi-step
PROMPT_BATCH_SIZE = 64
PROMPT_LEARNING_RATE = 5e-5         # Lowered from 1e-4 to reduce loss spikes
PROMPT_GAMMA = 0.95                 # Discount across prompt steps
PROMPT_ENT_COEF = 0.015             # Moderate exploration


# ==============================================================================
# SHARED CONFIG
# ==============================================================================
MAX_PROMPTS_PER_AGENT = 3           # Max prompts each agent can select

# Legacy compatibility (for any scripts that expect these)
TOTAL_TIMESTEPS = STRUCTURE_TOTAL_TIMESTEPS + PROMPT_TOTAL_TIMESTEPS
N_STEPS = PROMPT_N_STEPS
BATCH_SIZE = PROMPT_BATCH_SIZE
LEARNING_RATE = PROMPT_LEARNING_RATE
GAMMA = PROMPT_GAMMA
ENT_COEF = PROMPT_ENT_COEF

# ==============================================================================
# REWARD TUNING
# ==============================================================================
# Reduced penalties to encourage diverse workflow exploration
COST_PER_STEP = 0.02        # Reduced from 0.05 to encourage complex workflows
COST_TOKEN_BUDGET = 0.03    # Reduced to not over-penalize larger budgets
