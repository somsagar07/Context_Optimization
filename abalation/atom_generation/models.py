"""
API Model metadata and configurations for atom generation ablation study.
"""

API_MODEL_METADATA = {
    # OpenAI Models
    "openai/gpt-4o-mini": {
        "display_name": "GPT-4o Mini",
        "family": "OpenAI",
        "estimated_params": "~7B",
        "type": "chat",
        "cost_tier": "low",
        "capabilities": "Fast, cost-effective, good for high-volume tasks",
    },
    "openai/gpt-4o": {
        "display_name": "GPT-4o",
        "family": "OpenAI",
        "estimated_params": "~1.8T",
        "type": "chat",
        "cost_tier": "high",
        "capabilities": "High quality, multimodal, strong reasoning",
    },
    "openai/gpt-4-turbo": {
        "display_name": "GPT-4 Turbo",
        "family": "OpenAI",
        "estimated_params": "~1.8T",
        "type": "chat",
        "cost_tier": "high",
        "capabilities": "Balanced performance, good reasoning",
    },
    "openai/gpt-5.2": {
        "display_name": "GPT-5.2",
        "family": "OpenAI",
        "estimated_params": "~2T",
        "type": "chat",
        "cost_tier": "high",
        "capabilities": "High quality, multimodal, strong reasoning",
    },
    
    # Anthropic Models
    "anthropic/claude-3.5-haiku": {
        "display_name": "Claude 3.5 Haiku",
        "family": "Anthropic",
        "estimated_params": "~8B",
        "type": "chat",
        "cost_tier": "low",
        "capabilities": "Fast, efficient, good for simple tasks",
    },
    "anthropic/claude-3.5-sonnet": {
        "display_name": "Claude 3.5 Sonnet",
        "family": "Anthropic",
        "estimated_params": "~70B",
        "type": "chat",
        "cost_tier": "medium",
        "capabilities": "High quality, strong reasoning, balanced",
    },

    
    # Meta Models
    "meta-llama/llama-3.1-8b-instruct": {
        "display_name": "Llama 3.1 8B Instruct",
        "family": "Meta",
        "estimated_params": "8B",
        "type": "instruct",
        "cost_tier": "low",
        "capabilities": "Open-source, good performance, efficient",
    },
    "meta-llama/llama-3.1-70b-instruct": {
        "display_name": "Llama 3.1 70B Instruct",
        "family": "Meta",
        "estimated_params": "70B",
        "type": "instruct",
        "cost_tier": "medium",
        "capabilities": "Open-source, high quality, strong reasoning",
    },
    
    # Mistral AI
    "mistralai/mistral-large": {
        "display_name": "Mistral Large",
        "family": "Mistral",
        "estimated_params": "~70B",
        "type": "chat",
        "cost_tier": "medium",
        "capabilities": "High quality, multilingual, strong reasoning",
    },
    
    # Google
    "google/gemini-2.5-pro": {
        "display_name": "Gemini 2.5 Pro",
        "family": "Google",
        "estimated_params": "~1.5T",
        "type": "chat",
        "cost_tier": "medium",
        "capabilities": "Multimodal, long context, strong performance",
    },
    
    # Qwen Models
    "qwen/qwen-2.5-7b-instruct": {
        "display_name": "Qwen 2.5 7B Instruct",
        "family": "Qwen",
        "estimated_params": "7B",
        "type": "instruct",
        "cost_tier": "low",
        "capabilities": "Efficient, good performance, multilingual",
    },
    "qwen/qwen-2.5-72b-instruct": {
        "display_name": "Qwen 2.5 72B Instruct",
        "family": "Qwen",
        "estimated_params": "72B",
        "type": "instruct",
        "cost_tier": "medium",
        "capabilities": "Very high quality, excellent reasoning, multilingual",
    },
}


def get_model_metadata(model_id: str) -> dict:
    """Get metadata for a given model ID."""
    return API_MODEL_METADATA.get(model_id, {
        "display_name": model_id,
        "family": "Unknown",
        "estimated_params": "Unknown",
        "type": "Unknown",
        "cost_tier": "Unknown",
        "capabilities": "Unknown",
    })


def get_models_by_family(family: str) -> list:
    """Get all model IDs for a given family."""
    return [model_id for model_id, metadata in API_MODEL_METADATA.items() 
            if metadata["family"] == family]


def get_models_by_cost_tier(tier: str) -> list:
    """Get all model IDs for a given cost tier."""
    return [model_id for model_id, metadata in API_MODEL_METADATA.items() 
            if metadata["cost_tier"] == tier]

