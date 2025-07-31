import json
import pytest
from unittest.mock import MagicMock, patch
from typing import List

from autogen_core import FunctionCall
from autogen_core.models import (
    AssistantMessage,
    FunctionExecutionResultMessage,
    SystemMessage,
    UserMessage,
)
from google.genai.types import Content, Part
from google.genai import types as genai_types

from kagent.models.gemini._message_conversion import (
    convert_message_to_gemini,
    convert_messages_to_gemini_contents,
    extract_system_instructions,
    filter_non_system_messages,
    normalize_gemini_name,
    _convert_user_message_content,
    _convert_assistant_message_content,
    _convert_function_result_content,
    _convert_base64_image_to_part,
    _convert_url_image_to_part,
)


class TestGeminiMessageConversion:
    """Test suite for Gemini message conversion utilities."""

    def test_normalize_gemini_name(self):
        """Test name normalization for Gemini tools."""
        # Test valid names (should remain unchanged)
        assert normalize_gemini_name("valid_name") == "valid_name"
        assert normalize_gemini_name("ValidName123") == "ValidName123"
        
        # Test invalid characters (should be replaced with underscores)
        assert normalize_gemini_name("invalid-name") == "invalid_name"
        assert normalize_gemini_name("invalid.name") == "invalid_name"
        assert normalize_gemini_name("invalid name") == "invalid_name"
        assert normalize_gemini_name("invalid@name#") == "invalid_name_"
        
        # Test length truncation (should be limited to 63 characters)
        long_name = "a" * 100
        normalized = normalize_gemini_name(long_name)
        assert len(normalized) == 63
        assert normalized == "a" * 63

    def test_convert_user_message_simple_text(self):
        """Test converting simple text user message."""
        message = UserMessage(content="Hello, world!")
        result = convert_message_to_gemini(message)
        
        assert isinstance(result, Content)
        assert result.role == "user"
        assert len(result.parts) == 1
        assert result.parts[0].text == "Hello, world!"

    def test_convert_user_message_empty_text(self):
        """Test converting user message with empty text."""
        message = UserMessage(content="")
        result = convert_message_to_gemini(message)
        
        assert isinstance(result, Content)
        assert result.role == "user"
        assert len(result.parts) == 1
        assert result.parts[0].text == " "  # Should be converted to space

    def test_convert_user_message_multimodal_text_and_image(self):
        """Test converting multimodal user message with text and image."""
        content = [
            {"type": "text", "text": "What do you see?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
        message = UserMessage(content=content)
        result = convert_message_to_gemini(message)
        
        assert isinstance(result, Content)
        assert result.role == "user"
        assert len(result.parts) == 2
        
        # First part should be text
        assert result.parts[0].text == "What do you see?"
        
        # Second part should be image
        assert hasattr(result.parts[1], 'file_data')

    def test_convert_user_message_base64_image(self):
        """Test converting user message with base64 image."""
        base64_data = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/8A=="
        content = [
            {"type": "image_url", "image_url": {"url": base64_data}}
        ]
        message = UserMessage(content=content)
        
        with patch('kagent.models.gemini._message_conversion.base64.b64decode') as mock_decode:
            mock_decode.return_value = b"fake_image_data"
            result = convert_message_to_gemini(message)
        
        assert isinstance(result, Content)
        assert result.role == "user"
        assert len(result.parts) == 1
        assert hasattr(result.parts[0], 'inline_data')

    def test_convert_assistant_message_text(self):
        """Test converting assistant message with text content."""
        message = AssistantMessage(content="Hello! How can I help you?")
        result = convert_message_to_gemini(message)
        
        assert isinstance(result, Content)
        assert result.role == "model"
        assert len(result.parts) == 1
        assert result.parts[0].text == "Hello! How can I help you?"

    def test_convert_assistant_message_function_calls(self):
        """Test converting assistant message with function calls."""
        function_calls = [
            FunctionCall(id="call_1", name="get_weather", arguments='{"location": "New York"}'),
            FunctionCall(id="call_2", name="get_time", arguments='{"timezone": "UTC"}')
        ]
        message = AssistantMessage(content=function_calls)
        result = convert_message_to_gemini(message)
        
        assert isinstance(result, Content)
        assert result.role == "model"
        assert len(result.parts) == 2
        
        # Check first function call
        assert hasattr(result.parts[0], 'function_call')
        assert result.parts[0].function_call.name == "get_weather"
        assert result.parts[0].function_call.args == {"location": "New York"}
        
        # Check second function call
        assert hasattr(result.parts[1], 'function_call')
        assert result.parts[1].function_call.name == "get_time"
        assert result.parts[1].function_call.args == {"timezone": "UTC"}

    def test_convert_function_execution_result_message(self):
        """Test converting function execution result message."""
        # Mock function execution result
        mock_result = MagicMock()
        mock_result.name = "get_weather"
        mock_result.content = '{"temperature": 72, "condition": "sunny"}'
        
        message = FunctionExecutionResultMessage(content=[mock_result])
        result = convert_message_to_gemini(message)
        
        assert isinstance(result, Content)
        assert result.role == "user"
        assert len(result.parts) == 1
        
        # Check function response
        assert hasattr(result.parts[0], 'function_response')
        assert result.parts[0].function_response.name == "get_weather"
        expected_response = {"content": {"temperature": 72, "condition": "sunny"}}
        assert result.parts[0].function_response.response == expected_response

    def test_convert_system_message(self):
        """Test converting system message (should return None)."""
        message = SystemMessage(content="You are a helpful assistant.")
        result = convert_message_to_gemini(message)
        
        assert result is None

    def test_extract_system_instructions_single(self):
        """Test extracting system instructions from single system message."""
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="Hello")
        ]
        
        result = extract_system_instructions(messages)
        assert result == "You are a helpful assistant."

    def test_extract_system_instructions_multiple(self):
        """Test extracting system instructions from multiple system messages."""
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            SystemMessage(content="Be concise in your responses."),
            UserMessage(content="Hello")
        ]
        
        result = extract_system_instructions(messages)
        expected = "You are a helpful assistant.\nBe concise in your responses."
        assert result == expected

    def test_extract_system_instructions_none(self):
        """Test extracting system instructions when none exist."""
        messages = [
            UserMessage(content="Hello"),
            AssistantMessage(content="Hi there!")
        ]
        
        result = extract_system_instructions(messages)
        assert result is None

    def test_extract_system_instructions_empty_content(self):
        """Test extracting system instructions with empty content."""
        messages = [
            SystemMessage(content=""),
            SystemMessage(content="   "),
            UserMessage(content="Hello")
        ]
        
        result = extract_system_instructions(messages)
        assert result is None

    def test_filter_non_system_messages(self):
        """Test filtering out system messages."""
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="Hello"),
            AssistantMessage(content="Hi there!"),
            SystemMessage(content="Another system message")
        ]
        
        result = filter_non_system_messages(messages)
        
        assert len(result) == 2
        assert isinstance(result[0], UserMessage)
        assert isinstance(result[1], AssistantMessage)
        assert result[0].content == "Hello"
        assert result[1].content == "Hi there!"

    def test_convert_messages_to_gemini_contents(self):
        """Test converting full message list to Gemini contents."""
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="Hello"),
            AssistantMessage(content="Hi there!"),
            UserMessage(content="How are you?")
        ]
        
        result = convert_messages_to_gemini_contents(messages)
        
        # Should have 3 contents (system message filtered out)
        assert len(result) == 3
        
        # Check first message (user)
        assert result[0].role == "user"
        assert result[0].parts[0].text == "Hello"
        
        # Check second message (assistant)
        assert result[1].role == "model"
        assert result[1].parts[0].text == "Hi there!"
        
        # Check third message (user)
        assert result[2].role == "user"
        assert result[2].parts[0].text == "How are you?"

    def test_convert_user_message_content_string_list(self):
        """Test converting user message content with string list."""
        content = ["Hello", "world", "!"]
        result = _convert_user_message_content(content)
        
        assert len(result) == 3
        assert all(isinstance(part, Part) for part in result)
        assert result[0].text == "Hello"
        assert result[1].text == "world"
        assert result[2].text == "!"

    def test_convert_user_message_content_mixed_list(self):
        """Test converting user message content with mixed content types."""
        content = [
            "Hello",
            {"type": "text", "text": "world"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
        result = _convert_user_message_content(content)
        
        assert len(result) == 3
        assert result[0].text == "Hello"
        assert result[1].text == "world"
        assert hasattr(result[2], 'file_data')

    def test_convert_assistant_message_content_invalid_json(self):
        """Test converting assistant message with invalid JSON in function call."""
        function_call = FunctionCall(id="call_1", name="test_func", arguments="invalid json")
        message = AssistantMessage(content=[function_call])
        
        with patch('kagent.models.gemini._message_conversion.logger') as mock_logger:
            result = convert_message_to_gemini(message)
            
            # Should still create the content but log a warning
            assert isinstance(result, Content)
            assert result.role == "model"
            assert len(result.parts) == 1
            assert hasattr(result.parts[0], 'function_call')
            assert result.parts[0].function_call.args == {"_raw_arguments": "invalid json"}
            mock_logger.warning.assert_called()

    def test_convert_function_result_content_invalid_json(self):
        """Test converting function result with invalid JSON content."""
        mock_result = MagicMock()
        mock_result.name = "test_func"
        mock_result.content = "invalid json content"
        
        result = _convert_function_result_content([mock_result])
        
        assert len(result) == 1
        assert hasattr(result[0], 'function_response')
        assert result[0].function_response.name == "test_func"
        expected_response = {"content": {"_raw_content": "invalid json content"}}
        assert result[0].function_response.response == expected_response

    def test_convert_base64_image_to_part_success(self):
        """Test successful base64 image conversion."""
        data_url = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD"
        
        with patch('kagent.models.gemini._message_conversion.base64.b64decode') as mock_decode:
            mock_decode.return_value = b"fake_image_data"
            result = _convert_base64_image_to_part(data_url)
        
        assert result is not None
        assert hasattr(result, 'inline_data')
        mock_decode.assert_called_once()

    def test_convert_base64_image_to_part_failure(self):
        """Test base64 image conversion failure."""
        data_url = "data:image/jpeg;base64,invalid_base64"
        
        with patch('kagent.models.gemini._message_conversion.base64.b64decode', side_effect=Exception("Invalid base64")):
            with patch('kagent.models.gemini._message_conversion.logger') as mock_logger:
                result = _convert_base64_image_to_part(data_url)
                
                assert result is None
                mock_logger.warning.assert_called()

    def test_convert_url_image_to_part_different_formats(self):
        """Test URL image conversion with different image formats."""
        test_cases = [
            ("https://example.com/image.jpg", "image/jpeg"),
            ("https://example.com/image.png", "image/png"),
            ("https://example.com/image.gif", "image/gif"),
            ("https://example.com/image.webp", "image/webp"),
            ("https://example.com/image", "image/jpeg"),  # Default
        ]
        
        for url, expected_mime in test_cases:
            result = _convert_url_image_to_part(url)
            assert hasattr(result, 'file_data')
            assert result.file_data.file_uri == url
            assert result.file_data.mime_type == expected_mime

    def test_convert_message_role_validation(self):
        """Test that invalid roles are corrected."""
        # This test simulates the role validation in convert_messages_to_gemini_contents
        messages = [UserMessage(content="Hello")]
        
        with patch('kagent.models.gemini._message_conversion.logger') as mock_logger:
            # Simulate a content with invalid role
            mock_content = MagicMock()
            mock_content.role = "invalid_role"
            
            with patch('kagent.models.gemini._message_conversion.convert_message_to_gemini', return_value=mock_content):
                result = convert_messages_to_gemini_contents(messages)
                
                # Role should be corrected to "user"
                assert result[0].role == "user"
                mock_logger.warning.assert_called()

    def test_edge_cases_empty_messages(self):
        """Test edge cases with empty message lists."""
        # Empty message list
        result = convert_messages_to_gemini_contents([])
        assert result == []
        
        # Only system messages
        messages = [SystemMessage(content="System only")]
        result = convert_messages_to_gemini_contents([])
        assert result == []
        
        # System instruction extraction from empty list
        result = extract_system_instructions([])
        assert result is None

    def test_function_call_name_normalization(self):
        """Test that function call names are properly normalized."""
        function_call = FunctionCall(id="call_1", name="invalid-function.name", arguments='{}')
        message = AssistantMessage(content=[function_call])
        result = convert_message_to_gemini(message)
        
        assert result.parts[0].function_call.name == "invalid_function_name"

    def test_function_result_name_normalization(self):
        """Test that function result names are properly normalized."""
        mock_result = MagicMock()
        mock_result.name = "invalid-function.name"
        mock_result.content = '{"result": "success"}'
        
        message = FunctionExecutionResultMessage(content=[mock_result])
        result = convert_message_to_gemini(message)
        
        assert result.parts[0].function_response.name == "invalid_function_name"