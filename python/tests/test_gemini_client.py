import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from autogen_core import CancellationToken, FunctionCall
from autogen_core.models import (
    AssistantMessage,
    CreateResult,
    FunctionExecutionResultMessage,
    LLMMessage,
    RequestUsage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import Tool, ToolSchema
from google.genai import types as genai_types
from pydantic import SecretStr

from kagent.models.gemini._gemini_client import GeminiChatCompletionClient
from kagent.models.gemini.config import GeminiClientConfiguration


class TestGeminiChatCompletionClient:
    """Test suite for GeminiChatCompletionClient."""

    @pytest.fixture
    def mock_genai_client(self):
        """Mock genai client for testing."""
        with patch('kagent.models.gemini._gemini_client.genai') as mock_genai:
            mock_client = MagicMock()
            mock_client.close = AsyncMock()  # Make close method async
            mock_genai.Client.return_value = mock_client
            mock_genai.configure = MagicMock()
            yield mock_genai, mock_client

    @pytest.fixture
    def basic_config(self):
        """Basic configuration for testing."""
        return {
            "model": "gemini-1.5-pro",
            "api_key": "test-api-key",
            "temperature": 0.7,
            "max_output_tokens": 1024,
        }

    @pytest.fixture
    def full_config(self):
        """Full configuration with all options for testing."""
        return {
            "model": "gemini-1.5-pro",
            "api_key": "test-api-key",
            "base_url": "https://generativelanguage.googleapis.com",
            "temperature": 0.7,
            "max_output_tokens": 1024,
            "top_p": 0.9,
            "top_k": 40,
            "candidate_count": 1,
            "stop_sequences": ["STOP", "END"],
            "response_mime_type": "application/json",
            "safety_settings": {
                "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_ONLY_HIGH",
            },
        }

    def test_client_initialization_basic(self, mock_genai_client, basic_config):
        """Test basic client initialization."""
        mock_genai, mock_client = mock_genai_client
        
        client = GeminiChatCompletionClient(**basic_config)
        
        # Verify genai configuration
        mock_genai.configure.assert_called_with(api_key="test-api-key")
        mock_genai.Client.assert_called_once()
        
        # Verify client properties
        assert client._model_name == "gemini-1.5-pro"
        assert client._create_args.temperature == 0.7
        assert client._create_args.max_output_tokens == 1024

    def test_client_initialization_full(self, mock_genai_client, full_config):
        """Test full client initialization with all options."""
        mock_genai, mock_client = mock_genai_client
        
        client = GeminiChatCompletionClient(**full_config)
        
        # Verify genai configuration with custom base URL
        mock_genai.configure.assert_any_call(api_key="test-api-key")
        mock_genai.configure.assert_any_call(
            transport="rest", 
            api_endpoint="https://generativelanguage.googleapis.com"
        )
        
        # Verify generation config
        assert client._create_args.temperature == 0.7
        assert client._create_args.max_output_tokens == 1024
        assert client._create_args.top_p == 0.9
        assert client._create_args.top_k == 40
        assert client._create_args.candidate_count == 1
        assert client._create_args.stop_sequences == ["STOP", "END"]
        assert client._create_args.response_mime_type == "application/json"
        
        # Verify safety settings parsing
        assert client._safety_settings is not None
        assert len(client._safety_settings) == 2

    def test_safety_settings_parsing(self, mock_genai_client, basic_config):
        """Test safety settings parsing."""
        mock_genai, mock_client = mock_genai_client
        
        config = basic_config.copy()
        config["safety_settings"] = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_ONLY_HIGH",
            "INVALID_CATEGORY": "INVALID_THRESHOLD",
        }
        
        with patch('kagent.models.gemini._gemini_client.logger') as mock_logger:
            client = GeminiChatCompletionClient(**config)
            
            # Should have parsed valid settings and warned about invalid ones
            assert client._safety_settings is not None
            assert len(client._safety_settings) == 2
            mock_logger.warning.assert_called()

    def test_safety_settings_empty(self, mock_genai_client, basic_config):
        """Test empty safety settings."""
        mock_genai, mock_client = mock_genai_client
        
        client = GeminiChatCompletionClient(**basic_config)
        assert client._safety_settings is None

    @pytest.mark.asyncio
    async def test_create_basic_text(self, mock_genai_client, basic_config):
        """Test basic text completion."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock API response
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = "Hello, world!"
        mock_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5
        mock_response.to_dict.return_value = {"mock": "response"}
        
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        client = GeminiChatCompletionClient(**basic_config)
        
        messages = [UserMessage(content="Hello", source="user")]
        result = await client.create(messages)
        
        assert isinstance(result, CreateResult)
        assert result.content == "Hello, world!"
        assert result.finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5

    @pytest.mark.asyncio
    async def test_create_with_system_message(self, mock_genai_client, basic_config):
        """Test completion with system message."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock API response
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = "Response"
        mock_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
        mock_response.usage_metadata.prompt_token_count = 15
        mock_response.usage_metadata.candidates_token_count = 8
        
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        client = GeminiChatCompletionClient(**basic_config)
        
        messages = [
            SystemMessage(content="You are a helpful assistant.", source="system"),
            UserMessage(content="Hello", source="user")
        ]
        result = await client.create(messages)
        
        # Verify system instruction was extracted and used
        call_args = mock_client.aio.models.generate_content.call_args
        assert call_args[1]['config'].system_instruction == "You are a helpful assistant."
        assert result.content == "Response"

    @pytest.mark.asyncio
    async def test_create_with_function_calls(self, mock_genai_client, basic_config):
        """Test completion with function calling."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock API response with function call
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].function_call = MagicMock()
        mock_response.candidates[0].content.parts[0].function_call.name = "test_function"
        mock_response.candidates[0].content.parts[0].function_call.args = {"param": "value"}
        mock_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
        mock_response.usage_metadata.prompt_token_count = 20
        mock_response.usage_metadata.candidates_token_count = 10
        
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        client = GeminiChatCompletionClient(**basic_config)
        
        # Define a tool
        tool_schema: ToolSchema = {
            "name": "test_function",
            "description": "A test function",
            "parameters": {
                "type": "object",
                "properties": {
                    "param": {"type": "string", "description": "A parameter"}
                },
                "required": ["param"]
            }
        }
        
        messages = [UserMessage(content="Call the test function", source="user")]
        result = await client.create(messages, tools=[tool_schema])
        
        assert isinstance(result.content, list)
        assert len(result.content) == 1
        assert isinstance(result.content[0], FunctionCall)
        assert result.content[0].name == "test_function"
        assert json.loads(result.content[0].arguments) == {"param": "value"}

    @pytest.mark.asyncio
    async def test_create_with_multimodal_content(self, mock_genai_client, basic_config):
        """Test completion with multimodal content (text + image)."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock API response
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = "I can see the image."
        mock_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
        mock_response.usage_metadata.prompt_token_count = 25
        mock_response.usage_metadata.candidates_token_count = 12
        
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        client = GeminiChatCompletionClient(**basic_config)
        
        # Create multimodal message
        multimodal_content = [
            {"type": "text", "text": "What do you see in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/8A=="}}
        ]
        
        messages = [UserMessage(content=multimodal_content, source="user")]
        result = await client.create(messages)
        
        assert result.content == "I can see the image."

    @pytest.mark.asyncio
    async def test_create_with_json_output(self, mock_genai_client, basic_config):
        """Test completion with JSON output mode."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock API response with JSON
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = '{"result": "success"}'
        mock_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
        mock_response.usage_metadata.prompt_token_count = 15
        mock_response.usage_metadata.candidates_token_count = 8
        
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        client = GeminiChatCompletionClient(**basic_config)
        
        messages = [UserMessage(content="Return JSON", source="user")]
        result = await client.create(messages, json_output=True)
        
        # Verify JSON output was requested
        call_args = mock_client.aio.models.generate_content.call_args
        assert call_args[1]['config'].response_mime_type == "application/json"
        assert result.content == '{"result": "success"}'

    @pytest.mark.asyncio
    async def test_create_with_cancellation(self, mock_genai_client, basic_config):
        """Test completion with cancellation token."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock API call that can be cancelled
        mock_future = AsyncMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_future)
        
        client = GeminiChatCompletionClient(**basic_config)
        cancellation_token = CancellationToken()
        
        messages = [UserMessage(content="Hello", source="user")]
        
        # Start the create call (don't await yet)
        create_task = asyncio.create_task(
            client.create(messages, cancellation_token=cancellation_token)
        )
        
        # Cancel the token
        cancellation_token.cancel()
        
        # The task should be cancelled
        with pytest.raises(asyncio.CancelledError):
            await create_task

    @pytest.mark.asyncio
    async def test_create_stream_basic(self, mock_genai_client, basic_config):
        """Test basic streaming completion."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock streaming response
        async def mock_stream():
            chunk1 = MagicMock()
            chunk1.candidates = [MagicMock()]
            chunk1.candidates[0].content.parts = [MagicMock()]
            chunk1.candidates[0].content.parts[0].text = "Hello"
            chunk1.candidates[0].finish_reason = None
            chunk1.usage_metadata = None
            yield chunk1
            
            chunk2 = MagicMock()
            chunk2.candidates = [MagicMock()]
            chunk2.candidates[0].content.parts = [MagicMock()]
            chunk2.candidates[0].content.parts[0].text = " world!"
            chunk2.candidates[0].finish_reason = genai_types.FinishReason.STOP
            chunk2.usage_metadata.prompt_token_count = 10
            chunk2.usage_metadata.candidates_token_count = 5
            yield chunk2
        
        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=mock_stream())
        
        client = GeminiChatCompletionClient(**basic_config)
        
        messages = [UserMessage(content="Hello", source="user")]
        stream = client.create_stream(messages)
        
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        
        # Should have text chunks and final result
        assert len(chunks) == 3  # "Hello", " world!", CreateResult
        assert chunks[0] == "Hello"
        assert chunks[1] == " world!"
        assert isinstance(chunks[2], CreateResult)
        assert chunks[2].content == "Hello world!"
        assert chunks[2].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_create_error_handling(self, mock_genai_client, basic_config):
        """Test error handling in create method."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock API error
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        client = GeminiChatCompletionClient(**basic_config)
        
        messages = [UserMessage(content="Hello", source="user")]
        
        with pytest.raises(Exception, match="API Error"):
            await client.create(messages)

    @pytest.mark.asyncio
    async def test_create_empty_response(self, mock_genai_client, basic_config):
        """Test handling of empty response from API."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock empty response
        mock_response = MagicMock()
        mock_response.candidates = []
        mock_response.prompt_feedback = None
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 0
        
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        client = GeminiChatCompletionClient(**basic_config)
        
        messages = [UserMessage(content="Hello", source="user")]
        result = await client.create(messages)
        
        assert result.content == ""
        assert result.finish_reason == "unknown"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 0

    @pytest.mark.asyncio
    async def test_create_content_filtered(self, mock_genai_client, basic_config):
        """Test handling of content-filtered response."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock response with content filtering
        mock_response = MagicMock()
        mock_response.candidates = []
        mock_response.prompt_feedback = MagicMock()
        mock_response.prompt_feedback.block_reason = "SAFETY"
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 0
        
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        client = GeminiChatCompletionClient(**basic_config)
        
        messages = [UserMessage(content="Inappropriate content", source="user")]
        result = await client.create(messages)
        
        assert result.content == ""
        assert result.finish_reason == "content_filter"

    def test_usage_tracking(self, mock_genai_client, basic_config):
        """Test usage tracking functionality."""
        mock_genai, mock_client = mock_genai_client
        
        client = GeminiChatCompletionClient(**basic_config)
        
        # Initial usage should be zero
        usage = client.actual_usage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0

    @pytest.mark.asyncio
    async def test_close_client(self, mock_genai_client, basic_config):
        """Test client cleanup."""
        mock_genai, mock_client = mock_genai_client
        
        client = GeminiChatCompletionClient(**basic_config)
        
        await client.close()
        
        mock_client.close.assert_called_once()

    def test_model_info_override(self, mock_genai_client, basic_config):
        """Test model info override functionality."""
        mock_genai, mock_client = mock_genai_client
        
        custom_model_info = {
            "vision": False,
            "function_calling": False,
            "json_output": True,
            "family": "custom",
            "structured_output": False,
            "multiple_system_messages": False,
        }
        
        config = basic_config.copy()
        config["model_info_override"] = custom_model_info
        
        client = GeminiChatCompletionClient(**config)
        
        assert client._model_info == custom_model_info

    @pytest.mark.asyncio
    async def test_tools_validation_no_function_calling(self, mock_genai_client, basic_config):
        """Test that tools are rejected when model doesn't support function calling."""
        mock_genai, mock_client = mock_genai_client
        
        # Override model info to disable function calling
        config = basic_config.copy()
        config["model_info_override"] = {
            "vision": True,
            "function_calling": False,  # Disable function calling
            "json_output": True,
            "family": "gemini-1.5",
            "structured_output": True,
            "multiple_system_messages": False,
        }
        
        client = GeminiChatCompletionClient(**config)
        
        tool_schema: ToolSchema = {
            "name": "test_function",
            "description": "A test function",
            "parameters": {"type": "object", "properties": {}}
        }
        
        messages = [UserMessage(content="Hello", source="user")]
        
        with pytest.raises(ValueError, match="Model does not support function calling"):
            await client.create(messages, tools=[tool_schema])

    def test_configuration_validation(self, mock_genai_client):
        """Test configuration validation and edge cases."""
        mock_genai, mock_client = mock_genai_client
        
        # Test with minimal required config
        minimal_config = {
            "model": "gemini-1.5-pro",
            "api_key": "test-key"
        }
        client = GeminiChatCompletionClient(**minimal_config)
        assert client._model_name == "gemini-1.5-pro"
        
        # Test with all optional parameters
        full_config = {
            "model": "gemini-1.5-pro",
            "api_key": "test-key",
            "base_url": "https://custom-endpoint.com",
            "temperature": 0.8,
            "max_output_tokens": 2048,
            "top_p": 0.95,
            "top_k": 50,
            "candidate_count": 2,
            "stop_sequences": ["STOP", "END"],
            "response_mime_type": "application/json",
            "safety_settings": {
                "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE"
            }
        }
        client = GeminiChatCompletionClient(**full_config)
        assert client._create_args.temperature == 0.8
        assert client._create_args.max_output_tokens == 2048

    def test_safety_settings_edge_cases(self, mock_genai_client, basic_config):
        """Test safety settings parsing edge cases."""
        mock_genai, mock_client = mock_genai_client
        
        # Test with empty safety settings
        config = basic_config.copy()
        config["safety_settings"] = {}
        client = GeminiChatCompletionClient(**config)
        assert client._safety_settings is None
        
        # Test with None safety settings
        config = basic_config.copy()
        config["safety_settings"] = None
        client = GeminiChatCompletionClient(**config)
        assert client._safety_settings is None
        
        # Test with mixed valid/invalid settings
        config = basic_config.copy()
        config["safety_settings"] = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",  # Valid
            "INVALID_CATEGORY": "BLOCK_MEDIUM_AND_ABOVE",  # Invalid category
            "HARM_CATEGORY_HATE_SPEECH": "INVALID_THRESHOLD",  # Invalid threshold
        }
        
        with patch('kagent.models.gemini._gemini_client.logger') as mock_logger:
            client = GeminiChatCompletionClient(**config)
            # Should have parsed only the valid setting
            assert client._safety_settings is not None
            assert len(client._safety_settings) == 1
            # Should have logged warnings for invalid settings
            assert mock_logger.warning.call_count == 2

    @pytest.mark.asyncio
    async def test_message_conversion_edge_cases(self, mock_genai_client, basic_config):
        """Test message conversion edge cases."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock API response
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = "Response"
        mock_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5
        
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        client = GeminiChatCompletionClient(**basic_config)
        
        # Test with empty string content
        messages = [UserMessage(content="", source="user")]
        result = await client.create(messages)
        assert result.content == "Response"
        
        # Test with whitespace-only content
        messages = [UserMessage(content="   ", source="user")]
        result = await client.create(messages)
        assert result.content == "Response"
        
        # Test with mixed content types
        mixed_content = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": ""},  # Empty text
            {"type": "image_url", "image_url": {"url": "invalid-url"}},  # Invalid image
        ]
        messages = [UserMessage(content=mixed_content, source="user")]
        
        with patch('kagent.models.gemini._gemini_client.logger') as mock_logger:
            result = await client.create(messages)
            # Should log warning about unsupported image URL
            mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_function_execution_result_handling(self, mock_genai_client, basic_config):
        """Test handling of function execution results."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock API response
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = "Function result processed"
        mock_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
        mock_response.usage_metadata.prompt_token_count = 15
        mock_response.usage_metadata.candidates_token_count = 8
        
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        client = GeminiChatCompletionClient(**basic_config)
        
        # Create function execution result message
        from autogen_core.models import FunctionExecutionResult
        func_result = FunctionExecutionResult(call_id="test_call_id", name="test_function", content='{"result": "success"}'
        )
        
        messages = [
            UserMessage(content="Call function", source="user"),
            FunctionExecutionResultMessage(content=[func_result])
        ]
        
        result = await client.create(messages)
        assert result.content == "Function result processed"

    @pytest.mark.asyncio
    async def test_extra_create_args_handling(self, mock_genai_client, basic_config):
        """Test handling of extra create arguments."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock API response
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = "Response"
        mock_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5
        
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        client = GeminiChatCompletionClient(**basic_config)
        
        # Test with valid extra args
        extra_args = {
            "temperature": 0.9,
            "max_output_tokens": 1500,
            "top_p": 0.8,
        }
        
        messages = [UserMessage(content="Hello", source="user")]
        result = await client.create(messages, extra_create_args=extra_args)
        
        # Verify the call was made with updated parameters
        call_args = mock_client.aio.models.generate_content.call_args
        config = call_args[1]['config']
        assert config.temperature == 0.9
        assert config.max_output_tokens == 1500
        assert config.top_p == 0.8
        
        # Test with invalid extra args
        invalid_extra_args = {
            "temperature": 0.9,
            "invalid_param": "value",
        }
        
        with patch('kagent.models.gemini._gemini_client.logger') as mock_logger:
            result = await client.create(messages, extra_create_args=invalid_extra_args)
            # Should log warning about unsupported parameter
            mock_logger.warning.assert_called_with("Unsupported extra_create_arg: invalid_param")

    @pytest.mark.asyncio
    async def test_json_output_validation(self, mock_genai_client, basic_config):
        """Test JSON output validation and warnings."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock API response with invalid JSON
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = "Not valid JSON"
        mock_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5
        
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        # Test with model that doesn't support JSON output
        config = basic_config.copy()
        config["model_info_override"] = {
            "vision": False,
            "function_calling": True,
            "json_output": False,  # Disable JSON output
            "family": "gemini-1.0",
            "structured_output": False,
            "multiple_system_messages": False,
        }
        
        client = GeminiChatCompletionClient(**config)
        
        messages = [UserMessage(content="Return JSON", source="user")]
        
        with patch('kagent.models.gemini._gemini_client.logger') as mock_logger:
            result = await client.create(messages, json_output=True)
            # Should log warning about JSON capability
            mock_logger.warning.assert_called()
            assert "json_output capability is False" in str(mock_logger.warning.call_args)
        
        # Test with invalid JSON response
        with patch('kagent.models.gemini._gemini_client.logger') as mock_logger:
            result = await client.create(messages, json_output=True)
            # Should log warning about invalid JSON
            mock_logger.warning.assert_called_with("JSON output was requested, but the response is not valid JSON.")

    @pytest.mark.asyncio
    async def test_streaming_edge_cases(self, mock_genai_client, basic_config):
        """Test streaming edge cases and error handling."""
        mock_genai, mock_client = mock_genai_client
        
        # Test streaming with empty chunks
        async def mock_empty_stream():
            chunk = MagicMock()
            chunk.candidates = []
            chunk.usage_metadata = None
            yield chunk
        
        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=mock_empty_stream())
        
        client = GeminiChatCompletionClient(**basic_config)
        
        messages = [UserMessage(content="Hello", source="user")]
        stream = client.create_stream(messages)
        
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        
        # Should have final result even with empty chunks
        assert len(chunks) == 1
        assert isinstance(chunks[0], CreateResult)
        assert chunks[0].content == ""

    @pytest.mark.asyncio
    async def test_usage_tracking_accumulation(self, mock_genai_client, basic_config):
        """Test that usage tracking accumulates correctly across multiple calls."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock API responses with different usage
        def create_mock_response(prompt_tokens, completion_tokens):
            mock_response = MagicMock()
            mock_response.candidates = [MagicMock()]
            mock_response.candidates[0].content.parts = [MagicMock()]
            mock_response.candidates[0].content.parts[0].text = "Response"
            mock_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
            mock_response.usage_metadata.prompt_token_count = prompt_tokens
            mock_response.usage_metadata.candidates_token_count = completion_tokens
            return mock_response
        
        # Set up multiple responses
        responses = [
            create_mock_response(10, 5),
            create_mock_response(15, 8),
            create_mock_response(20, 12),
        ]
        
        mock_client.aio.models.generate_content = AsyncMock(side_effect=responses)
        
        client = GeminiChatCompletionClient(**basic_config)
        
        messages = [UserMessage(content="Hello", source="user")]
        
        # Make multiple calls
        for _ in range(3):
            await client.create(messages)
        
        # Check accumulated usage
        usage = client.actual_usage()
        assert usage.prompt_tokens == 45  # 10 + 15 + 20
        assert usage.completion_tokens == 25  # 5 + 8 + 12

    def test_tool_schema_conversion_edge_cases(self, mock_genai_client, basic_config):
        """Test tool schema conversion edge cases."""
        mock_genai, mock_client = mock_genai_client
        
        client = GeminiChatCompletionClient(**basic_config)
        
        # Test with complex nested schema
        complex_tool: ToolSchema = {
            "name": "complex_function",
            "description": "A complex function with nested parameters",
            "parameters": {
                "type": "object",
                "properties": {
                    "simple_param": {"type": "string", "description": "A simple parameter"},
                    "nested_object": {
                        "type": "object",
                        "properties": {
                            "inner_param": {"type": "number", "description": "Inner parameter"}
                        },
                        "required": ["inner_param"]
                    },
                    "array_param": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of strings"
                    },
                    "enum_param": {
                        "type": "string",
                        "enum": ["option1", "option2", "option3"],
                        "description": "Enum parameter"
                    }
                },
                "required": ["simple_param", "nested_object"]
            }
        }
        
        # Should not raise an exception
        gemini_tools = client._convert_tools_to_gemini([complex_tool])
        assert gemini_tools is not None
        assert len(gemini_tools) == 1
        assert len(gemini_tools[0].function_declarations) == 1
        
        # Test with invalid tool name
        invalid_tool: ToolSchema = {
            "name": "invalid-tool-name-with-special-chars!@#",
            "description": "Tool with invalid name",
            "parameters": {"type": "object", "properties": {}}
        }
        
        with pytest.raises(ValueError, match="Invalid Gemini tool/function name"):
            client._convert_tools_to_gemini([invalid_tool])

    @pytest.mark.asyncio
    async def test_base64_image_processing(self, mock_genai_client, basic_config):
        """Test base64 image processing in multimodal content."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock API response
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = "I can see the image"
        mock_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
        mock_response.usage_metadata.prompt_token_count = 25
        mock_response.usage_metadata.candidates_token_count = 12
        
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        client = GeminiChatCompletionClient(**basic_config)
        
        # Test with valid base64 image
        valid_base64_content = [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url", 
                "image_url": {
                    "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/8A=="
                }
            }
        ]
        
        messages = [UserMessage(content=valid_base64_content, source="user")]
        result = await client.create(messages)
        assert result.content == "I can see the image"
        
        # Test with invalid base64 image
        invalid_base64_content = [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url", 
                "image_url": {
                    "url": "data:image/jpeg;base64,invalid_base64_data"
                }
            }
        ]
        
        messages = [UserMessage(content=invalid_base64_content, source="user")]
        
        with patch('kagent.models.gemini._gemini_client.logger') as mock_logger:
            result = await client.create(messages)
            # Should log warning about failed base64 processing
            mock_logger.warning.assert_called()
            assert "Failed to process base64 image" in str(mock_logger.warning.call_args)

    @pytest.mark.asyncio
    async def test_thought_content_extraction(self, mock_genai_client, basic_config):
        """Test extraction of thought content alongside function calls."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock API response with both text and function call
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        
        # Create parts with both text and function call
        text_part = MagicMock()
        text_part.text = "Let me call a function to help with this."
        text_part.function_call = None
        
        func_part = MagicMock()
        func_part.text = None
        func_part.function_call = MagicMock()
        func_part.function_call.name = "helper_function"
        func_part.function_call.args = {"param": "value"}
        
        mock_response.candidates[0].content.parts = [text_part, func_part]
        mock_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
        mock_response.usage_metadata.prompt_token_count = 20
        mock_response.usage_metadata.candidates_token_count = 15
        
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        client = GeminiChatCompletionClient(**basic_config)
        
        tool_schema: ToolSchema = {
            "name": "helper_function",
            "description": "A helper function",
            "parameters": {"type": "object", "properties": {"param": {"type": "string"}}}
        }
        
        messages = [UserMessage(content="Help me with this task", source="user")]
        result = await client.create(messages, tools=[tool_schema])
        
        # Should have function calls as content
        assert isinstance(result.content, list)
        assert len(result.content) == 1
        assert isinstance(result.content[0], FunctionCall)
        
        # Should have thought content
        assert result.thought == "Let me call a function to help with this."