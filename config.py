import torch

# --- Model Config ---
LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Experiment Config ---
# Options: "gsm8k", "hotpotqa" 
DATASET_NAME = "gsm8k" 

# --- RL Hyperparameters ---
TOTAL_TIMESTEPS = 1000 
N_STEPS = 50          # Collect 20 steps before updating (Default stablebaselines3 1024)
BATCH_SIZE = 5        # Mini-batch size (must be factor of N_STEPS)
LEARNING_RATE = 3e-4
GAMMA = 0.99 

# --- Environment Config ---
# Context Window sizes to optimize over
CONTEXT_WINDOWS = [256, 512, 1024] 

# Penalty weights 
COST_PER_STEP = 0.10      # Cost for each LLM call in the chain
COST_PER_TOKEN = 0.0001   # Legacy (not used in new action space)
COST_TOOL_USAGE = 0.02    # Cost per tool enabled
COST_TOKEN_BUDGET = 0.10  # Max penalty for using all high token budgets