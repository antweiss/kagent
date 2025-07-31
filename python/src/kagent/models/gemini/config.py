from typing import Dict, List, Optional

from autogen_core.models import ModelInfo
from pydantic import BaseModel, Field, SecretStr


class GeminiClientConfiguration(BaseModel):
    """Configuration for Gemini AI client."""
    
    model: str = Field(description="Name of the Gemini model to use")
    api_key: SecretStr = Field(description="Google AI API key")
    base_url: Optional[str] = Field(
        default="https://generativelanguage.googleapis.com", 
        description="Base URL for Gemini API"
    )
    temperature: Optional[float] = Field(
        default=None, 
        ge=0.0, 
        le=2.0, 
        description="Temperature for sampling (0.0 to 2.0)"
    )
    max_output_tokens: Optional[int] = Field(
        default=None, 
        ge=1, 
        description="Maximum tokens to generate"
    )
    top_p: Optional[float] = Field(
        default=None, 
        ge=0.0, 
        le=1.0, 
        description="Top-p sampling parameter (0.0 to 1.0)"
    )
    top_k: Optional[int] = Field(
        default=None, 
        ge=0, 
        description="Top-k sampling parameter"
    )
    candidate_count: Optional[int] = Field(
        default=None, 
        ge=1, 
        description="Number of candidate responses to generate"
    )
    stop_sequences: Optional[List[str]] = Field(
        default=None, 
        description="Stop sequences"
    )
    response_mime_type: Optional[str] = Field(
        default=None, 
        description="Response MIME type for structured output"
    )
    safety_settings: Optional[Dict[str, str]] = Field(
        default=None, 
        description="Safety settings for content filtering"
    )
    model_info_override: Optional[ModelInfo] = Field(
        default=None, 
        description="Optional override for model capabilities and information."
    )