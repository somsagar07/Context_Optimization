"""
Configuration for atom generation ablation experiments using OpenRouter API models.
"""

# OpenRouter API Models to Test
# Format: OpenRouter model ID
API_MODELS_TO_TEST = [
    # OpenAI
    # "openai/gpt-4o-mini",      # Small, fast, cost-effective
    # "openai/gpt-4o",            # Large, high quality
    # "openai/gpt-4-turbo",      # Balanced
    "openai/gpt-5.2",
    
    # # Anthropic
    # "anthropic/claude-3.5-haiku",  # Small, fast
    # "anthropic/claude-3.5-sonnet", # Medium, high quality
    
    # # Meta
    # "meta-llama/llama-3.1-8b-instruct",   # Medium
    # "meta-llama/llama-3.1-70b-instruct",  # Large
    
    # # Mistral AI
    # "mistralai/mistral-large",  # Large
    
    # # Google
    # "google/gemini-2.5-pro",    # Large
    
    # # Qwen
    # "qwen/qwen-2.5-7b-instruct",    # Small-medium
    # "qwen/qwen-2.5-72b-instruct",   # Very large
]

# Datasets to use for experiments
DATASETS = ["gsm8k", "hotpotqa", "gaia", "medqa", "aime25"]

# API Configuration
API_RATE_LIMIT_DELAY = 1.0  # Seconds to wait between API calls
API_MAX_RETRIES = 2         # Maximum retries for failed API calls
API_TIMEOUT = 60           # Timeout in seconds for API calls

# Metrics Configuration
METRICS_CONFIG = {
    "diversity": {
        "uniqueness": {
            "enabled": True,
            "embedding_model": "metaclip",  # Use MetaCLIP-H14 for semantic similarity
        },
        "strategy_coverage": {
            "enabled": True,
            "strategies": ["analytical", "creative", "pedagogical", "critical", "expert_persona", "constraint_focused"],
        },
        "semantic_diversity": {
            "enabled": True,
            "embedding_model": "metaclip",  # Use MetaCLIP-H14 for clustering
            "n_clusters": 5,
            "random_state": 42,
        },
    },
    "quality": {
        "coherence": {
            "enabled": True,
            "evaluator_model": "openai/gpt-5.2",  # Use GPT-5.2 for coherence evaluation
            "scale": (1, 10),  # Rating scale
        },
        "specificity": {
            "enabled": True,
            "embedding_model": "metaclip",  # Use MetaCLIP-H14 for specificity comparison
        },
        "clarity": {
            "enabled": True,
            "evaluator_model": "openai/gpt-5",  # Use GPT-4o for clarity evaluation
            "scale": (1, 10),  # Rating scale
        },
    },
}

# Combined Score Weights (no consensus, only diversity and quality)
SCORE_WEIGHTS = {
    "diversity": 0.40,   # 40% weight
    "quality": 0.60,     # 60% weight (includes GPT-5.2 coherence and clarity)
}

# Output directory for results
RESULTS_DIR = "results"

# Random state for reproducibility
RANDOM_STATE = 42

