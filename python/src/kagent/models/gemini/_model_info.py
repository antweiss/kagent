import logging
from typing import Dict, List

from autogen_core.models import ModelInfo

logger = logging.getLogger(__name__)

# Gemini AI available models with their capabilities
# Based on https://ai.google.dev/gemini-api/docs/models
GEMINI_MODELS: Dict[str, ModelInfo] = {
    "gemini-1.5-pro": {
        "vision": True,
        "function_calling": True,
        "json_output": True,
        "family": "gemini-1.5",
        "structured_output": True,
        "multiple_system_messages": False,
    },
    "gemini-1.5-pro-latest": {
        "vision": True,
        "function_calling": True,
        "json_output": True,
        "family": "gemini-1.5",
        "structured_output": True,
        "multiple_system_messages": False,
    },
    "gemini-1.5-flash": {
        "vision": True,
        "function_calling": True,
        "json_output": True,
        "family": "gemini-1.5",
        "structured_output": True,
        "multiple_system_messages": False,
    },
    "gemini-1.5-flash-latest": {
        "vision": True,
        "function_calling": True,
        "json_output": True,
        "family": "gemini-1.5",
        "structured_output": True,
        "multiple_system_messages": False,
    },
    "gemini-1.5-flash-8b": {
        "vision": True,
        "function_calling": True,
        "json_output": True,
        "family": "gemini-1.5",
        "structured_output": True,
        "multiple_system_messages": False,
    },
    "gemini-1.5-flash-8b-latest": {
        "vision": True,
        "function_calling": True,
        "json_output": True,
        "family": "gemini-1.5",
        "structured_output": True,
        "multiple_system_messages": False,
    },
    "gemini-1.0-pro": {
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": "gemini-1.0",
        "structured_output": False,
        "multiple_system_messages": False,
    },
    "gemini-1.0-pro-latest": {
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": "gemini-1.0",
        "structured_output": False,
        "multiple_system_messages": False,
    },
    "gemini-pro": {
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": "gemini-1.0",
        "structured_output": False,
        "multiple_system_messages": False,
    },
    "gemini-pro-vision": {
        "vision": True,
        "function_calling": False,
        "json_output": True,
        "family": "gemini-1.0",
        "structured_output": False,
        "multiple_system_messages": False,
    },
    # Experimental models
    "gemini-exp-1206": {
        "vision": True,
        "function_calling": True,
        "json_output": True,
        "family": "gemini-exp",
        "structured_output": True,
        "multiple_system_messages": False,
    },
    "gemini-exp-1121": {
        "vision": True,
        "function_calling": True,
        "json_output": True,
        "family": "gemini-exp",
        "structured_output": True,
        "multiple_system_messages": False,
    },
    # Gemini 2.0 models
    "gemini-2.0-flash-exp": {
        "vision": True,
        "function_calling": True,
        "json_output": True,
        "family": "gemini-2.0",
        "structured_output": True,
        "multiple_system_messages": False,
    },
}

# Model token limits (context window size)
# Based on https://ai.google.dev/gemini-api/docs/models
GEMINI_MODEL_TOKEN_LIMITS: Dict[str, int] = {
    "gemini-1.5-pro": 2_097_152,
    "gemini-1.5-pro-latest": 2_097_152,
    "gemini-1.5-flash": 1_048_576,
    "gemini-1.5-flash-latest": 1_048_576,
    "gemini-1.5-flash-8b": 1_048_576,
    "gemini-1.5-flash-8b-latest": 1_048_576,
    "gemini-1.0-pro": 32_768,
    "gemini-1.0-pro-latest": 32_768,
    "gemini-pro": 32_768,
    "gemini-pro-vision": 16_384,
    "gemini-exp-1206": 2_097_152,
    "gemini-exp-1121": 2_097_152,
    "gemini-2.0-flash-exp": 1_048_576,
}


def get_info(model: str) -> ModelInfo:
    """Get the model information for a specific model."""
    if not model or not isinstance(model, str):
        raise KeyError(f"Model '{model}' not found in Gemini model info")
    
    # Check for exact match first
    if model in GEMINI_MODELS:
        # Return a copy to prevent external modification
        return GEMINI_MODELS[model].copy()
    
    # Try to match by prefix for versioned models (only if model has more specific parts)
    for model_key in GEMINI_MODELS:
        # Only match if the input model has at least as many parts as the base model
        model_parts = model.split('-')
        key_parts = model_key.split('-')
        if (len(model_parts) >= len(key_parts) and 
            model_parts[0] == key_parts[0] and  # Must start with "gemini"
            len(model_parts) > 1):  # Must have more than just "gemini"
            logger.warning(f"Model '{model}' not found exactly, using closest match '{model_key}'")
            return GEMINI_MODELS[model_key].copy()
    
    # If no match found, raise error
    raise KeyError(f"Model '{model}' not found in Gemini model info")


def get_token_limit(model: str) -> int:
    """Get the token limit for a specific model."""
    # Check for exact match first
    if model in GEMINI_MODEL_TOKEN_LIMITS:
        return GEMINI_MODEL_TOKEN_LIMITS[model]
    
    # Try to match by prefix for versioned models
    for model_key in GEMINI_MODEL_TOKEN_LIMITS:
        if model.startswith(model_key.split('-')[0]):
            return GEMINI_MODEL_TOKEN_LIMITS[model_key]
    
    # Default fallback
    return 32_768


def validate_model(model: str) -> bool:
    """Validate that a model is supported by Gemini AI."""
    try:
        get_info(model)
        return True
    except KeyError:
        return False


def get_vision_capable_models() -> List[str]:
    """Get list of models that support vision/multimodal capabilities."""
    return [model for model, info in GEMINI_MODELS.items() if info.get("vision", False)]


def get_function_calling_models() -> List[str]:
    """Get list of models that support function calling."""
    return [model for model, info in GEMINI_MODELS.items() if info.get("function_calling", False)]


def get_json_output_models() -> List[str]:
    """Get list of models that support JSON output."""
    return [model for model, info in GEMINI_MODELS.items() if info.get("json_output", False)]