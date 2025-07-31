"""Message format conversion utilities for Gemini AI."""

import base64
import json
import logging
from typing import Dict, List, Optional, Union

from autogen_core import FunctionCall
from autogen_core.models import (
    AssistantMessage,
    FunctionExecutionResultMessage,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from google.genai import types as genai_types
from google.genai.types import Content, Part

logger = logging.getLogger(__name__)


def normalize_gemini_name(name: str) -> str:
    """Normalize names by replacing invalid characters with underscore for Gemini tools."""
    import re
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)[:63]  # Gemini limit seems to be 63 chars


def convert_message_to_gemini(message: LLMMessage) -> Optional[Content]:
    """Convert a single LLMMessage to Gemini Content format.
    
    This function handles:
    - Text messages
    - Multimodal messages with images (base64 and URLs)
    - Function calls and responses
    - System messages (returns None as they're handled separately)
    
    Args:
        message: The LLMMessage to convert
        
    Returns:
        Gemini Content object or None for system messages
    """
    parts: List[Part] = []
    role: str = "user"

    if isinstance(message, UserMessage):
        role = "user"
        parts = _convert_user_message_content(message.content)
        return Content(parts=parts, role=role)

    elif isinstance(message, AssistantMessage):
        role = "model"
        parts = _convert_assistant_message_content(message.content)
        return Content(parts=parts, role=role)

    elif isinstance(message, FunctionExecutionResultMessage):
        parts = _convert_function_result_content(message.content)
        return Content(parts=parts, role="user")

    elif isinstance(message, SystemMessage):
        # System messages are handled separately in Gemini via system_instruction
        return None

    return None


def _convert_user_message_content(content: Union[str, List]) -> List[Part]:
    """Convert user message content to Gemini Parts."""
    parts: List[Part] = []
    
    if isinstance(content, str):
        parts.append(Part(text=content if content.strip() else " "))
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                parts.append(Part(text=item if item.strip() else " "))
            elif isinstance(item, dict):
                part = _convert_content_item_to_part(item)
                if part:
                    parts.append(part)
            else:
                logger.warning(f"Unsupported content type in UserMessage: {type(item)}")
    
    return parts


def _convert_content_item_to_part(item: Dict) -> Optional[Part]:
    """Convert a content item dictionary to a Gemini Part."""
    content_type = item.get("type")
    
    if content_type == "text":
        return Part(text=item.get("text", " "))
    
    elif content_type == "image_url":
        return _convert_image_to_part(item.get("image_url", {}))
    
    else:
        logger.warning(f"Unsupported content type: {content_type}")
        return None


def _convert_image_to_part(image_url: Dict) -> Optional[Part]:
    """Convert image URL to Gemini Part with proper format handling."""
    url = image_url.get("url", "")
    
    if url.startswith("data:"):
        # Handle base64 encoded images
        return _convert_base64_image_to_part(url)
    elif url.startswith("http"):
        # Handle URL-based images
        return _convert_url_image_to_part(url)
    else:
        logger.warning(f"Unsupported image URL format: {url}")
        return None


def _convert_base64_image_to_part(data_url: str) -> Optional[Part]:
    """Convert base64 data URL to Gemini Part."""
    try:
        # Parse data URL: data:image/jpeg;base64,<data>
        header, data = data_url.split(",", 1)
        mime_type = header.split(":")[1].split(";")[0]
        image_data = base64.b64decode(data)
        
        return Part(inline_data=genai_types.Blob(
            mime_type=mime_type,
            data=image_data
        ))
    except Exception as e:
        logger.warning(f"Failed to process base64 image: {e}")
        return None


def _convert_url_image_to_part(url: str) -> Part:
    """Convert image URL to Gemini Part."""
    # Detect MIME type from URL extension or default to JPEG
    mime_type = "image/jpeg"  # Default
    if url.lower().endswith(('.png', '.PNG')):
        mime_type = "image/png"
    elif url.lower().endswith(('.gif', '.GIF')):
        mime_type = "image/gif"
    elif url.lower().endswith(('.webp', '.WEBP')):
        mime_type = "image/webp"
    
    return Part(file_data=genai_types.FileData(
        file_uri=url,
        mime_type=mime_type
    ))


def _convert_assistant_message_content(content: Union[str, List]) -> List[Part]:
    """Convert assistant message content to Gemini Parts."""
    parts: List[Part] = []
    
    if isinstance(content, str):
        parts.append(Part(text=content))
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, FunctionCall):
                part = _convert_function_call_to_part(item)
                if part:
                    parts.append(part)
            else:
                logger.warning(f"Unsupported content type in AssistantMessage list: {type(item)}")
    
    return parts


def _convert_function_call_to_part(func_call: FunctionCall) -> Optional[Part]:
    """Convert function call to Gemini Part."""
    try:
        args = func_call.arguments
        args_dict = json.loads(args) if isinstance(args, str) else args
    except json.JSONDecodeError:
        args_dict = {"_raw_arguments": args}
        logger.warning(
            f"Function call arguments for {func_call.name} are not valid JSON. Passing as raw string."
        )

    return Part(
        function_call=genai_types.FunctionCall(
            name=normalize_gemini_name(func_call.name), 
            args=args_dict
        )
    )


def _convert_function_result_content(content: List) -> List[Part]:
    """Convert function execution results to Gemini Parts."""
    gemini_parts: List[Part] = []
    
    for result in content:
        try:
            content_value = json.loads(result.content) if isinstance(result.content, str) else result.content
        except json.JSONDecodeError:
            content_value = {"_raw_content": str(result.content)}

        gemini_parts.append(
            Part(
                function_response=genai_types.FunctionResponse(
                    name=normalize_gemini_name(result.name),
                    response={"content": content_value},
                )
            )
        )
    
    return gemini_parts


def extract_system_instructions(messages: List[LLMMessage]) -> Optional[str]:
    """Extract and merge system messages into a single system instruction.
    
    Args:
        messages: List of LLM messages
        
    Returns:
        Merged system instruction string or None if no system messages
    """
    system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
    
    if not system_messages:
        return None
    
    merged_content = ""
    for msg in system_messages:
        if msg.content and msg.content.strip():
            merged_content += msg.content.strip() + "\n"
    
    return merged_content.strip() if merged_content else None


def filter_non_system_messages(messages: List[LLMMessage]) -> List[LLMMessage]:
    """Filter out system messages from the message list.
    
    Args:
        messages: List of LLM messages
        
    Returns:
        List of messages without system messages
    """
    return [msg for msg in messages if not isinstance(msg, SystemMessage)]


def convert_messages_to_gemini_contents(messages: List[LLMMessage]) -> List[Content]:
    """Convert a list of LLM messages to Gemini Content format.
    
    This function:
    - Filters out system messages (they're handled separately)
    - Converts each message to Gemini Content format
    - Validates message roles
    
    Args:
        messages: List of LLM messages to convert
        
    Returns:
        List of Gemini Content objects
    """
    gemini_contents: List[Content] = []
    regular_messages = filter_non_system_messages(messages)
    
    for msg in regular_messages:
        converted_content = convert_message_to_gemini(msg)
        if converted_content:
            # Validate and fix role if necessary
            if converted_content.role not in ["user", "model"]:
                logger.warning(
                    f"Message role '{converted_content.role}' not 'user' or 'model'. Adjusting to 'user'."
                )
                converted_content.role = "user"
            
            gemini_contents.append(converted_content)
    
    return gemini_contents