import asyncio
import os
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
from google.genai.errors import ClientError, ServerError
from pydantic import SecretStr

from kagent.models.gemini._gemini_client import GeminiChatCompletionClient
from kagent.models.gemini.config import GeminiClientConfiguration


class TestGeminiIntegration:
    """Integration test suite for Gemini AI API."""

    @pytest.fixture
    def integration_config(self):
        """Configuration for integration testing."""
        return {
            "model": "gemini-1.5-pro",
            "api_key": os.getenv("GEMINI_API_KEY", "test-api-key"),
            "temperature": 0.1,  # Low temperature for consistent results
            "max_output_tokens": 1024,
        }

    @pytest.fixture
    def mock_genai_client(self):
        """Mock genai client for integration testing."""
        with patch('kagent.models.gemini._gemini_client.genai') as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_genai.configure = MagicMock()
            yield mock_genai, mock_client

    @pytest.mark.asyncio
    async def test_end_to_end_text_completion(self, mock_genai_client, integration_config):
        """Test end-to-end text completion flow."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = "Hello! I'm a helpful AI assistant."
        mock_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
        mock_response.usage_metadata.prompt_token_count = 15
        mock_response.usage_metadata.candidates_token_count = 12
        
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        # Create client and test
        client = GeminiChatCompletionClient(**integration_config)
        
        messages = [
            SystemMessage(content="You are a helpful AI assistant."),
            UserMessage(content="Hello, introduce yourself.")
        ]
        
        result = await client.create(messages)
        
        # Verify result
        assert isinstance(result, CreateResult)
        assert result.content == "Hello! I'm a helpful AI assistant."
        assert result.finish_reason == "stop"
        assert result.usage.prompt_tokens == 15
        assert result.usage.completion_tokens == 12
        
        # Verify API was called correctly
        call_args = mock_client.aio.models.generate_content.call_args
        assert call_args[1]['model'] == "gemini-1.5-pro"
        assert call_args[1]['config'].system_instruction == "You are a helpful AI assistant."
        assert len(call_args[1]['contents']) == 1  # Only user message (system filtered out)

    @pytest.mark.asyncio
    async def test_authentication_scenarios(self, mock_genai_client, integration_config):
        """Test various authentication scenarios."""
        mock_genai, mock_client = mock_genai_client
        
        # Test 1: Valid API key
        client = GeminiChatCompletionClient(**integration_config)
        mock_genai.configure.assert_called_with(api_key="test-api-key")
        
        # Test 2: Invalid API key (simulate auth error)
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=ClientError("Invalid API key")
        )
        
        messages = [UserMessage(content="Hello")]
        
        with pytest.raises(ClientError, match="Invalid API key"):
            await client.create(messages)

    @pytest.mark.asyncio
    async def test_error_handling_scenarios(self, mock_genai_client, integration_config):
        """Test error handling for various API error scenarios."""
        mock_genai, mock_client = mock_genai_client
        client = GeminiChatCompletionClient(**integration_config)
        messages = [UserMessage(content="Hello")]
        
        # Test 1: Server error (500)
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=ServerError("Internal server error")
        )
        
        with pytest.raises(ServerError, match="Internal server error"):
            await client.create(messages)
        
        # Test 2: Rate limiting error
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=ClientError("Rate limit exceeded")
        )
        
        with pytest.raises(ClientError, match="Rate limit exceeded"):
            await client.create(messages)
        
        # Test 3: Network timeout
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=asyncio.TimeoutError("Request timeout")
        )
        
        with pytest.raises(asyncio.TimeoutError):
            await client.create(messages)

    @pytest.mark.asyncio
    async def test_rate_limiting_behavior(self, mock_genai_client, integration_config):
        """Test rate limiting behavior and backoff."""
        mock_genai, mock_client = mock_genai_client
        client = GeminiChatCompletionClient(**integration_config)
        messages = [UserMessage(content="Hello")]
        
        # Simulate rate limit error followed by success
        rate_limit_error = ClientError("Rate limit exceeded")
        success_response = MagicMock()
        success_response.candidates = [MagicMock()]
        success_response.candidates[0].content.parts = [MagicMock()]
        success_response.candidates[0].content.parts[0].text = "Success after rate limit"
        success_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
        success_response.usage_metadata.prompt_token_count = 10
        success_response.usage_metadata.candidates_token_count = 8
        
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=[rate_limit_error, success_response]
        )
        
        # First call should fail with rate limit
        with pytest.raises(ClientError, match="Rate limit exceeded"):
            await client.create(messages)
        
        # Second call should succeed
        result = await client.create(messages)
        assert result.content == "Success after rate limit"

    @pytest.mark.asyncio
    async def test_streaming_response_integration(self, mock_genai_client, integration_config):
        """Test streaming response integration."""
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
        
        client = GeminiChatCompletionClient(**integration_config)
        messages = [UserMessage(content="Hello")]
        
        # Test streaming
        chunks = []
        async for chunk in client.create_stream(messages):
            chunks.append(chunk)
        
        # Verify streaming results
        assert len(chunks) == 3  # "Hello", " world!", CreateResult
        assert chunks[0] == "Hello"
        assert chunks[1] == " world!"
        assert isinstance(chunks[2], CreateResult)
        assert chunks[2].content == "Hello world!"
        assert chunks[2].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_function_calling_integration(self, mock_genai_client, integration_config):
        """Test function calling integration."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock function call response
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].function_call = MagicMock()
        mock_response.candidates[0].content.parts[0].function_call.name = "get_weather"
        mock_response.candidates[0].content.parts[0].function_call.args = {"location": "New York"}
        mock_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
        mock_response.usage_metadata.prompt_token_count = 25
        mock_response.usage_metadata.candidates_token_count = 15
        
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        client = GeminiChatCompletionClient(**integration_config)
        
        # Define tool
        weather_tool: ToolSchema = {
            "name": "get_weather",
            "description": "Get weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The location to get weather for"}
                },
                "required": ["location"]
            }
        }
        
        messages = [UserMessage(content="What's the weather in New York?")]
        result = await client.create(messages, tools=[weather_tool])
        
        # Verify function call result
        assert isinstance(result.content, list)
        assert len(result.content) == 1
        assert isinstance(result.content[0], FunctionCall)
        assert result.content[0].name == "get_weather"
        
        # Verify tool was passed to API
        call_args = mock_client.aio.models.generate_content.call_args
        assert call_args[1]['config'].tools is not None
        assert len(call_args[1]['config'].tools) == 1

    @pytest.mark.asyncio
    async def test_function_execution_result_integration(self, mock_genai_client, integration_config):
        """Test function execution result handling."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock response to function result
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = "The weather in New York is sunny with 72°F."
        mock_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
        mock_response.usage_metadata.prompt_token_count = 30
        mock_response.usage_metadata.candidates_token_count = 18
        
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        client = GeminiChatCompletionClient(**integration_config)
        
        # Simulate function execution result
        mock_result = MagicMock()
        mock_result.name = "get_weather"
        mock_result.content = '{"temperature": 72, "condition": "sunny"}'
        
        messages = [
            UserMessage(content="What's the weather in New York?"),
            AssistantMessage(content=[FunctionCall(id="call_1", name="get_weather", arguments='{"location": "New York"}')]),
            FunctionExecutionResultMessage(content=[mock_result])
        ]
        
        result = await client.create(messages)
        
        # Verify response
        assert result.content == "The weather in New York is sunny with 72°F."
        
        # Verify function result was passed correctly
        call_args = mock_client.aio.models.generate_content.call_args
        contents = call_args[1]['contents']
        
        # Should have user message, assistant message with function call, and function result
        assert len(contents) == 3
        assert contents[0].role == "user"
        assert contents[1].role == "model"
        assert contents[2].role == "user"  # Function results are sent as user messages

    @pytest.mark.asyncio
    async def test_multimodal_content_integration(self, mock_genai_client, integration_config):
        """Test multimodal content handling integration."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock vision response
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = "I can see a beautiful sunset in the image."
        mock_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
        mock_response.usage_metadata.prompt_token_count = 35
        mock_response.usage_metadata.candidates_token_count = 20
        
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        client = GeminiChatCompletionClient(**integration_config)
        
        # Create multimodal message
        multimodal_content = [
            {"type": "text", "text": "What do you see in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/sunset.jpg"}}
        ]
        
        messages = [UserMessage(content=multimodal_content)]
        result = await client.create(messages)
        
        # Verify response
        assert result.content == "I can see a beautiful sunset in the image."
        
        # Verify multimodal content was passed correctly
        call_args = mock_client.aio.models.generate_content.call_args
        contents = call_args[1]['contents']
        assert len(contents) == 1
        assert len(contents[0].parts) == 2  # Text + image parts

    @pytest.mark.asyncio
    async def test_safety_filtering_integration(self, mock_genai_client, integration_config):
        """Test safety filtering integration."""
        mock_genai, mock_client = mock_genai_client
        
        # Add safety settings to config
        config = integration_config.copy()
        config["safety_settings"] = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_ONLY_HIGH",
        }
        
        # Mock safety-filtered response
        mock_response = MagicMock()
        mock_response.candidates = []  # Empty candidates due to safety filtering
        mock_response.prompt_feedback = MagicMock()
        mock_response.prompt_feedback.block_reason = "SAFETY"
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 0
        
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        client = GeminiChatCompletionClient(**config)
        
        messages = [UserMessage(content="Inappropriate content that gets filtered")]
        result = await client.create(messages)
        
        # Verify safety filtering result
        assert result.content == ""
        assert result.finish_reason == "content_filter"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 0
        
        # Verify safety settings were passed to API
        call_args = mock_client.aio.models.generate_content.call_args
        assert call_args[1]['config'].safety_settings is not None

    @pytest.mark.asyncio
    async def test_json_output_integration(self, mock_genai_client, integration_config):
        """Test JSON output mode integration."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock JSON response
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = '{"name": "John", "age": 30, "city": "New York"}'
        mock_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
        mock_response.usage_metadata.prompt_token_count = 20
        mock_response.usage_metadata.candidates_token_count = 15
        
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        client = GeminiChatCompletionClient(**integration_config)
        
        messages = [UserMessage(content="Generate a JSON object with name, age, and city")]
        result = await client.create(messages, json_output=True)
        
        # Verify JSON response
        assert result.content == '{"name": "John", "age": 30, "city": "New York"}'
        
        # Verify JSON output was requested
        call_args = mock_client.aio.models.generate_content.call_args
        assert call_args[1]['config'].response_mime_type == "application/json"

    @pytest.mark.asyncio
    async def test_cancellation_integration(self, mock_genai_client, integration_config):
        """Test cancellation token integration."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock long-running API call
        async def slow_response():
            await asyncio.sleep(1)  # Simulate slow response
            return MagicMock()
        
        mock_client.aio.models.generate_content = AsyncMock(side_effect=slow_response)
        
        client = GeminiChatCompletionClient(**integration_config)
        cancellation_token = CancellationToken()
        
        messages = [UserMessage(content="Hello")]
        
        # Start the create call
        create_task = asyncio.create_task(
            client.create(messages, cancellation_token=cancellation_token)
        )
        
        # Cancel after a short delay
        await asyncio.sleep(0.1)
        cancellation_token.cancel()
        
        # The task should be cancelled
        with pytest.raises(asyncio.CancelledError):
            await create_task

    @pytest.mark.asyncio
    async def test_usage_tracking_integration(self, mock_genai_client, integration_config):
        """Test usage tracking across multiple requests."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock multiple responses with different usage
        responses = []
        for i in range(3):
            mock_response = MagicMock()
            mock_response.candidates = [MagicMock()]
            mock_response.candidates[0].content.parts = [MagicMock()]
            mock_response.candidates[0].content.parts[0].text = f"Response {i+1}"
            mock_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
            mock_response.usage_metadata.prompt_token_count = 10 + i
            mock_response.usage_metadata.candidates_token_count = 5 + i
            responses.append(mock_response)
        
        mock_client.aio.models.generate_content = AsyncMock(side_effect=responses)
        
        client = GeminiChatCompletionClient(**integration_config)
        messages = [UserMessage(content="Hello")]
        
        # Make multiple requests
        for i in range(3):
            result = await client.create(messages)
            assert result.content == f"Response {i+1}"
        
        # Verify cumulative usage tracking
        total_usage = client.actual_usage()
        expected_prompt_tokens = 10 + 11 + 12  # 33
        expected_completion_tokens = 5 + 6 + 7  # 18
        
        assert total_usage.prompt_tokens == expected_prompt_tokens
        assert total_usage.completion_tokens == expected_completion_tokens

    @pytest.mark.asyncio
    async def test_client_lifecycle_integration(self, mock_genai_client, integration_config):
        """Test client lifecycle (initialization, usage, cleanup)."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock response
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = "Hello!"
        mock_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
        mock_response.usage_metadata.prompt_token_count = 5
        mock_response.usage_metadata.candidates_token_count = 3
        
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        # Test client lifecycle
        client = GeminiChatCompletionClient(**integration_config)
        
        # Verify initialization
        mock_genai.configure.assert_called_with(api_key="test-api-key")
        mock_genai.Client.assert_called_once()
        
        # Use client
        messages = [UserMessage(content="Hello")]
        result = await client.create(messages)
        assert result.content == "Hello!"
        
        # Cleanup
        await client.close()
        mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_configuration_validation_integration(self, mock_genai_client):
        """Test configuration validation during client creation."""
        mock_genai, mock_client = mock_genai_client
        
        # Test valid configuration
        valid_config = {
            "model": "gemini-1.5-pro",
            "api_key": "test-api-key",
            "temperature": 0.7,
            "max_output_tokens": 1024,
            "top_p": 0.9,
            "top_k": 40,
        }
        
        client = GeminiChatCompletionClient(**valid_config)
        assert client._model_name == "gemini-1.5-pro"
        assert client._create_args.temperature == 0.7
        assert client._create_args.max_output_tokens == 1024
        assert client._create_args.top_p == 0.9
        assert client._create_args.top_k == 40

    @pytest.mark.asyncio
    async def test_custom_base_url_integration(self, mock_genai_client, integration_config):
        """Test custom base URL configuration."""
        mock_genai, mock_client = mock_genai_client
        
        # Add custom base URL
        config = integration_config.copy()
        config["base_url"] = "https://custom-gemini-endpoint.com"
        
        client = GeminiChatCompletionClient(**config)
        
        # Verify custom base URL was configured
        mock_genai.configure.assert_any_call(api_key="test-api-key")
        mock_genai.configure.assert_any_call(
            transport="rest", 
            api_endpoint="https://custom-gemini-endpoint.com"
        )

    @pytest.mark.asyncio
    async def test_error_recovery_integration(self, mock_genai_client, integration_config):
        """Test error recovery scenarios."""
        mock_genai, mock_client = mock_genai_client
        
        # Simulate transient error followed by success
        error_response = ClientError("Temporary service unavailable")
        success_response = MagicMock()
        success_response.candidates = [MagicMock()]
        success_response.candidates[0].content.parts = [MagicMock()]
        success_response.candidates[0].content.parts[0].text = "Success after error"
        success_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
        success_response.usage_metadata.prompt_token_count = 10
        success_response.usage_metadata.candidates_token_count = 8
        
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=[error_response, success_response]
        )
        
        client = GeminiChatCompletionClient(**integration_config)
        messages = [UserMessage(content="Hello")]
        
        # First call should fail
        with pytest.raises(ClientError, match="Temporary service unavailable"):
            await client.create(messages)
        
        # Second call should succeed
        result = await client.create(messages)
        assert result.content == "Success after error"