"""
Base configuration shared across all environment modes.
"""
import torch

# --- Model Config --- 
LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct" 
# LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Experiment Config ---
# Options: "gsm8k", "hotpotqa", "gaia", "medqa", "aime25"
DATASET_NAME = "gsm8k" 

# --- Callback Config ---
SAVE_EVERY_EPISODES = 500

# --- Environment Config ---
# Context Window sizes to optimize over
CONTEXT_WINDOWS = [256, 512, 1024] 

# Penalty weights 
COST_PER_STEP = 0.05      # Cost for each LLM call in the chain
COST_PER_TOKEN = 0.0001   
COST_TOOL_USAGE = 0.0     # Set to 0 - use --tool-bonus arg instead
COST_TOKEN_BUDGET = 0.05  # Max penalty for using all high token budgets

