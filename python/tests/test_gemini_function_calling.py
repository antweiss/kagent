import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from autogen_core import CancellationToken, FunctionCall
from autogen_core.models import (
    AssistantMessage,
    CreateResult,
    FunctionExecutionResultMessage,
    UserMessage,
    SystemMessage,
)
from autogen_core.tools import Tool, ToolSchema
from google.genai import types as genai_types

from kagent.models.gemini._gemini_client import (
    GeminiChatCompletionClient,
    normalize_gemini_name,
    assert_valid_gemini_name,
)
from kagent.models.gemini._model_info import get_function_calling_models, GEMINI_MODELS


class TestGeminiFunctionCalling:
    """Test suite for Gemini function calling and tool integration."""

    @pytest.fixture
    def function_calling_config(self):
        """Configuration for function calling capable model."""
        return {
            "model": "gemini-1.5-pro",
            "api_key": "test-api-key",
            "temperature": 0.1,
            "max_output_tokens": 1024,
        }

    @pytest.fixture
    def mock_genai_client(self):
        """Mock genai client for testing."""
        with patch('kagent.models.gemini._gemini_client.genai') as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_genai.configure = MagicMock()
            yield mock_genai, mock_client

    @pytest.fixture
    def sample_tool_schema(self):
        """Sample tool schema for testing."""
        return {
            "name": "get_weather",
            "description": "Get current weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }

    @pytest.fixture
    def complex_tool_schema(self):
        """Complex tool schema for testing."""
        return {
            "name": "analyze_data",
            "description": "Analyze data with various parameters",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Array of numbers to analyze"
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": ["mean", "median", "mode", "std"],
                        "description": "Type of analysis to perform"
                    },
                    "options": {
                        "type": "object",
                        "properties": {
                            "precision": {"type": "integer", "minimum": 1, "maximum": 10},
                            "include_outliers": {"type": "boolean"}
                        }
                    }
                },
                "required": ["data", "analysis_type"]
            }
        }

    def test_function_calling_models_detection(self):
        """Test detection of function calling capable models."""
        function_models = get_function_calling_models()
        
        assert isinstance(function_models, list)
        assert len(function_models) > 0
        
        # Verify known function calling models are included
        expected_function_models = [
            "gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"
        ]
        
        for model in expected_function_models:
            if model in GEMINI_MODELS:
                assert model in function_models, f"Expected function calling model {model} not found"
        
        # Verify all returned models actually have function calling capability
        for model in function_models:
            assert GEMINI_MODELS[model]["function_calling"] is True

    def test_normalize_gemini_name_function(self):
        """Test Gemini name normalization for function names."""
        test_cases = [
            ("valid_function", "valid_function"),
            ("ValidFunction123", "ValidFunction123"),
            ("invalid-function", "invalid_function"),
            ("invalid.function", "invalid_function"),
            ("invalid function", "invalid_function"),
            ("invalid@function#", "invalid_function_"),
            ("a" * 100, "a" * 63),  # Truncation test
        ]
        
        for input_name, expected_output in test_cases:
            result = normalize_gemini_name(input_name)
            assert result == expected_output, f"Failed for input: {input_name}"

    def test_assert_valid_gemini_name_function(self):
        """Test Gemini name validation function."""
        # Valid names should pass
        valid_names = ["valid_function", "ValidFunction123", "func_123"]
        for name in valid_names:
            try:
                result = assert_valid_gemini_name(name)
                assert result == name
            except ValueError:
                pytest.fail(f"Valid name {name} was rejected")
        
        # Invalid names should raise ValueError
        invalid_names = ["invalid-function", "invalid.function", "invalid function", "a" * 64]
        for name in invalid_names:
            with pytest.raises(ValueError, match="Invalid Gemini tool/function name"):
                assert_valid_gemini_name(name)

    def test_tool_schema_conversion_simple(self, mock_genai_client, function_calling_config, sample_tool_schema):
        """Test conversion of simple tool schema to Gemini format."""
        mock_genai, mock_client = mock_genai_client
        client = GeminiChatCompletionClient(**function_calling_config)
        
        # Test tool conversion
        gemini_tools = client._convert_tools_to_gemini([sample_tool_schema])
        
        assert gemini_tools is not None
        assert len(gemini_tools) == 1
        assert len(gemini_tools[0].function_declarations) == 1
        
        func_decl = gemini_tools[0].function_declarations[0]
        assert func_decl.name == "get_weather"
        assert func_decl.description == "Get current weather information for a location"
        assert func_decl.parameters is not None
        assert func_decl.parameters.type == genai_types.Type.OBJECT

    def test_tool_schema_conversion_complex(self, mock_genai_client, function_calling_config, complex_tool_schema):
        """Test conversion of complex tool schema to Gemini format."""
        mock_genai, mock_client = mock_genai_client
        client = GeminiChatCompletionClient(**function_calling_config)
        
        # Test complex tool conversion
        gemini_tools = client._convert_tools_to_gemini([complex_tool_schema])
        
        assert gemini_tools is not None
        assert len(gemini_tools) == 1
        
        func_decl = gemini_tools[0].function_declarations[0]
        assert func_decl.name == "analyze_data"
        assert func_decl.parameters.type == genai_types.Type.OBJECT
        
        # Check nested properties
        assert "data" in func_decl.parameters.properties
        assert "analysis_type" in func_decl.parameters.properties
        assert "options" in func_decl.parameters.properties
        
        # Check array type
        data_prop = func_decl.parameters.properties["data"]
        assert data_prop.type == genai_types.Type.ARRAY
        assert data_prop.items is not None

    def test_tool_schema_conversion_multiple_tools(self, mock_genai_client, function_calling_config, sample_tool_schema):
        """Test conversion of multiple tool schemas."""
        mock_genai, mock_client = mock_genai_client
        client = GeminiChatCompletionClient(**function_calling_config)
        
        # Create multiple tools
        tool2 = {
            "name": "get_time",
            "description": "Get current time",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "Timezone"}
                }
            }
        }
        
        gemini_tools = client._convert_tools_to_gemini([sample_tool_schema, tool2])
        
        assert gemini_tools is not None
        assert len(gemini_tools) == 1
        assert len(gemini_tools[0].function_declarations) == 2
        
        func_names = [func.name for func in gemini_tools[0].function_declarations]
        assert "get_weather" in func_names
        assert "get_time" in func_names

    def test_tool_schema_conversion_invalid_name(self, mock_genai_client, function_calling_config):
        """Test tool schema conversion with invalid function name."""
        mock_genai, mock_client = mock_genai_client
        client = GeminiChatCompletionClient(**function_calling_config)
        
        invalid_tool = {
            "name": "invalid-function-name",  # Invalid characters
            "description": "Test function",
            "parameters": {"type": "object", "properties": {}}
        }
        
        with pytest.raises(ValueError, match="Invalid Gemini tool/function name"):
            client._convert_tools_to_gemini([invalid_tool])

    @pytest.mark.asyncio
    async def test_function_call_generation(self, mock_genai_client, function_calling_config, sample_tool_schema):
        """Test function call generation."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock function call response
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].function_call = MagicMock()
        mock_response.candidates[0].content.parts[0].function_call.name = "get_weather"
        mock_response.candidates[0].content.parts[0].function_call.args = {
            "location": "San Francisco, CA",
            "unit": "celsius"
        }
        mock_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
        mock_response.usage_metadata.prompt_token_count = 25
        mock_response.usage_metadata.candidates_token_count = 15
        
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        client = GeminiChatCompletionClient(**function_calling_config)
        
        messages = [UserMessage(content="What's the weather in San Francisco?")]
        result = await client.create(messages, tools=[sample_tool_schema])
        
        # Verify function call result
        assert isinstance(result.content, list)
        assert len(result.content) == 1
        assert isinstance(result.content[0], FunctionCall)
        assert result.content[0].name == "get_weather"
        
        args = json.loads(result.content[0].arguments)
        assert args["location"] == "San Francisco, CA"
        assert args["unit"] == "celsius"

    @pytest.mark.asyncio
    async def test_multiple_function_calls(self, mock_genai_client, function_calling_config, sample_tool_schema):
        """Test multiple function calls in one response."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock response with multiple function calls
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [
            MagicMock(),  # First function call
            MagicMock()   # Second function call
        ]
        
        # First function call
        mock_response.candidates[0].content.parts[0].function_call = MagicMock()
        mock_response.candidates[0].content.parts[0].function_call.name = "get_weather"
        mock_response.candidates[0].content.parts[0].function_call.args = {"location": "New York"}
        
        # Second function call
        mock_response.candidates[0].content.parts[1].function_call = MagicMock()
        mock_response.candidates[0].content.parts[1].function_call.name = "get_weather"
        mock_response.candidates[0].content.parts[1].function_call.args = {"location": "Los Angeles"}
        
        mock_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
        mock_response.usage_metadata.prompt_token_count = 30
        mock_response.usage_metadata.candidates_token_count = 20
        
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        client = GeminiChatCompletionClient(**function_calling_config)
        
        messages = [UserMessage(content="What's the weather in New York and Los Angeles?")]
        result = await client.create(messages, tools=[sample_tool_schema])
        
        # Verify multiple function calls
        assert isinstance(result.content, list)
        assert len(result.content) == 2
        
        for i, func_call in enumerate(result.content):
            assert isinstance(func_call, FunctionCall)
            assert func_call.name == "get_weather"
            args = json.loads(func_call.arguments)
            assert args["location"] in ["New York", "Los Angeles"]

    @pytest.mark.asyncio
    async def test_function_call_with_thought(self, mock_genai_client, function_calling_config, sample_tool_schema):
        """Test function call with accompanying thought/text."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock response with both text and function call
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [
            MagicMock(),  # Text part
            MagicMock()   # Function call part
        ]
        
        # Text part
        mock_response.candidates[0].content.parts[0].text = "I'll check the weather for you."
        
        # Function call part
        mock_response.candidates[0].content.parts[1].function_call = MagicMock()
        mock_response.candidates[0].content.parts[1].function_call.name = "get_weather"
        mock_response.candidates[0].content.parts[1].function_call.args = {"location": "Boston"}
        
        mock_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
        mock_response.usage_metadata.prompt_token_count = 20
        mock_response.usage_metadata.candidates_token_count = 15
        
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        client = GeminiChatCompletionClient(**function_calling_config)
        
        messages = [UserMessage(content="What's the weather in Boston?")]
        result = await client.create(messages, tools=[sample_tool_schema])
        
        # Verify function call with thought
        assert isinstance(result.content, list)
        assert len(result.content) == 1
        assert isinstance(result.content[0], FunctionCall)
        assert result.thought == "I'll check the weather for you."

    @pytest.mark.asyncio
    async def test_function_execution_result_handling(self, mock_genai_client, function_calling_config):
        """Test handling of function execution results."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock response to function result
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = "The weather in San Francisco is 72°F and sunny."
        mock_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
        mock_response.usage_metadata.prompt_token_count = 35
        mock_response.usage_metadata.candidates_token_count = 20
        
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        client = GeminiChatCompletionClient(**function_calling_config)
        
        # Simulate function execution result
        mock_result = MagicMock()
        mock_result.name = "get_weather"
        mock_result.content = '{"temperature": 72, "condition": "sunny", "location": "San Francisco"}'
        
        messages = [
            UserMessage(content="What's the weather in San Francisco?"),
            AssistantMessage(content=[FunctionCall(id="call_1", name="get_weather", arguments='{"location": "San Francisco"}')]),
            FunctionExecutionResultMessage(content=[mock_result])
        ]
        
        result = await client.create(messages)
        
        # Verify response to function result
        assert result.content == "The weather in San Francisco is 72°F and sunny."
        
        # Verify function result was passed correctly
        call_args = mock_client.aio.models.generate_content.call_args
        contents = call_args[1]['contents']
        
        # Should have user message, assistant message with function call, and function result
        assert len(contents) == 3
        assert contents[2].role == "user"  # Function results are sent as user messages
        assert len(contents[2].parts) == 1
        assert hasattr(contents[2].parts[0], 'function_response')

    @pytest.mark.asyncio
    async def test_function_execution_result_invalid_json(self, mock_genai_client, function_calling_config):
        """Test handling of function execution results with invalid JSON."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock response
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = "I received the function result."
        mock_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
        mock_response.usage_metadata.prompt_token_count = 25
        mock_response.usage_metadata.candidates_token_count = 15
        
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        client = GeminiChatCompletionClient(**function_calling_config)
        
        # Function result with invalid JSON
        mock_result = MagicMock()
        mock_result.name = "get_weather"
        mock_result.content = "Invalid JSON content"
        
        messages = [
            UserMessage(content="What's the weather?"),
            AssistantMessage(content=[FunctionCall(id="call_1", name="get_weather", arguments='{}')]),
            FunctionExecutionResultMessage(content=[mock_result])
        ]
        
        result = await client.create(messages)
        
        # Should handle invalid JSON gracefully
        assert result.content == "I received the function result."
        
        # Verify function result was wrapped in _raw_content
        call_args = mock_client.aio.models.generate_content.call_args
        contents = call_args[1]['contents']
        function_response = contents[2].parts[0].function_response
        assert function_response.response["content"]["_raw_content"] == "Invalid JSON content"

    @pytest.mark.asyncio
    async def test_function_calling_validation_no_support(self, mock_genai_client, sample_tool_schema):
        """Test validation when model doesn't support function calling."""
        mock_genai, mock_client = mock_genai_client
        
        # Configure client with model that doesn't support function calling
        config = {
            "model": "gemini-pro-vision",
            "api_key": "test-api-key",
            "model_info_override": {
                "vision": True,
                "function_calling": False,  # Disable function calling
                "json_output": True,
                "family": "gemini-1.0",
                "structured_output": False,
                "multiple_system_messages": False,
            }
        }
        
        client = GeminiChatCompletionClient(**config)
        
        messages = [UserMessage(content="What's the weather?")]
        
        with pytest.raises(ValueError, match="Model does not support function calling"):
            await client.create(messages, tools=[sample_tool_schema])

    @pytest.mark.asyncio
    async def test_function_calling_streaming(self, mock_genai_client, function_calling_config, sample_tool_schema):
        """Test function calling with streaming response."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock streaming response with function call
        async def mock_stream():
            chunk1 = MagicMock()
            chunk1.candidates = [MagicMock()]
            chunk1.candidates[0].content.parts = [MagicMock()]
            chunk1.candidates[0].content.parts[0].text = "I'll check the weather"
            chunk1.candidates[0].finish_reason = None
            chunk1.usage_metadata = None
            yield chunk1
            
            chunk2 = MagicMock()
            chunk2.candidates = [MagicMock()]
            chunk2.candidates[0].content.parts = [MagicMock()]
            chunk2.candidates[0].content.parts[0].function_call = MagicMock()
            chunk2.candidates[0].content.parts[0].function_call.name = "get_weather"
            chunk2.candidates[0].content.parts[0].function_call.args = {"location": "Seattle"}
            chunk2.candidates[0].finish_reason = genai_types.FinishReason.STOP
            chunk2.usage_metadata.prompt_token_count = 20
            chunk2.usage_metadata.candidates_token_count = 10
            yield chunk2
        
        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=mock_stream())
        
        client = GeminiChatCompletionClient(**function_calling_config)
        
        messages = [UserMessage(content="What's the weather in Seattle?")]
        
        # Test streaming with function calls
        chunks = []
        async for chunk in client.create_stream(messages, tools=[sample_tool_schema]):
            chunks.append(chunk)
        
        # Verify streaming results
        assert len(chunks) == 2  # Text chunk + final result
        assert chunks[0] == "I'll check the weather"
        assert isinstance(chunks[1], CreateResult)
        assert isinstance(chunks[1].content, list)
        assert len(chunks[1].content) == 1
        assert isinstance(chunks[1].content[0], FunctionCall)
        assert chunks[1].thought == "I'll check the weather"

    def test_tool_server_integration_schema_format(self, mock_genai_client, function_calling_config):
        """Test integration with tool server schema format."""
        mock_genai, mock_client = mock_genai_client
        client = GeminiChatCompletionClient(**function_calling_config)
        
        # Tool server style schema
        tool_server_schema = {
            "name": "search_database",
            "description": "Search the database for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "filters": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string"},
                            "date_range": {
                                "type": "object",
                                "properties": {
                                    "start": {"type": "string", "format": "date"},
                                    "end": {"type": "string", "format": "date"}
                                }
                            }
                        }
                    }
                },
                "required": ["query"]
            }
        }
        
        # Test conversion
        gemini_tools = client._convert_tools_to_gemini([tool_server_schema])
        
        assert gemini_tools is not None
        func_decl = gemini_tools[0].function_declarations[0]
        assert func_decl.name == "search_database"
        assert "query" in func_decl.parameters.properties
        assert "filters" in func_decl.parameters.properties
        
        # Check nested object structure
        filters_prop = func_decl.parameters.properties["filters"]
        assert filters_prop.type == genai_types.Type.OBJECT
        assert "category" in filters_prop.properties
        assert "date_range" in filters_prop.properties

    @pytest.mark.asyncio
    async def test_tool_execution_results_handling_multiple(self, mock_genai_client, function_calling_config):
        """Test handling of multiple tool execution results."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock response to multiple function results
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = "Based on the weather data from both cities, New York is warmer than Seattle today."
        mock_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
        mock_response.usage_metadata.prompt_token_count = 50
        mock_response.usage_metadata.candidates_token_count = 25
        
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        client = GeminiChatCompletionClient(**function_calling_config)
        
        # Multiple function execution results
        result1 = MagicMock()
        result1.name = "get_weather"
        result1.content = '{"temperature": 75, "location": "New York"}'
        
        result2 = MagicMock()
        result2.name = "get_weather"
        result2.content = '{"temperature": 65, "location": "Seattle"}'
        
        messages = [
            UserMessage(content="Compare weather in New York and Seattle"),
            AssistantMessage(content=[
                FunctionCall(id="call_1", name="get_weather", arguments='{"location": "New York"}'),
                FunctionCall(id="call_2", name="get_weather", arguments='{"location": "Seattle"}')
            ]),
            FunctionExecutionResultMessage(content=[result1, result2])
        ]
        
        result = await client.create(messages)
        
        # Verify response handles multiple results
        assert "New York" in result.content
        assert "Seattle" in result.content
        
        # Verify multiple function results were passed
        call_args = mock_client.aio.models.generate_content.call_args
        contents = call_args[1]['contents']
        function_result_content = contents[2]
        assert len(function_result_content.parts) == 2  # Two function results

    def test_function_call_argument_handling_edge_cases(self, mock_genai_client, function_calling_config):
        """Test edge cases in function call argument handling."""
        mock_genai, mock_client = mock_genai_client
        client = GeminiChatCompletionClient(**function_calling_config)
        
        # Test 1: Empty arguments
        func_call_empty = FunctionCall(id="call_1", name="test_func", arguments="")
        message = AssistantMessage(content=[func_call_empty])
        result = client._convert_message_to_gemini(message)
        
        assert result.parts[0].function_call.args == {}
        
        # Test 2: None arguments
        func_call_none = FunctionCall(id="call_2", name="test_func", arguments=None)
        message = AssistantMessage(content=[func_call_none])
        result = client._convert_message_to_gemini(message)
        
        assert result.parts[0].function_call.args == {}
        
        # Test 3: Already parsed arguments (dict)
        func_call_dict = FunctionCall(id="call_3", name="test_func", arguments={"key": "value"})
        message = AssistantMessage(content=[func_call_dict])
        result = client._convert_message_to_gemini(message)
        
        assert result.parts[0].function_call.args == {"key": "value"}

    def test_function_name_normalization_in_conversion(self, mock_genai_client, function_calling_config):
        """Test function name normalization during message conversion."""
        mock_genai, mock_client = mock_genai_client
        client = GeminiChatCompletionClient(**function_calling_config)
        
        # Function call with invalid name
        func_call = FunctionCall(id="call_1", name="invalid-function.name", arguments='{"param": "value"}')
        message = AssistantMessage(content=[func_call])
        result = client._convert_message_to_gemini(message)
        
        # Name should be normalized
        assert result.parts[0].function_call.name == "invalid_function_name"
        
        # Function execution result with invalid name
        mock_result = MagicMock()
        mock_result.name = "invalid-function.name"
        mock_result.content = '{"result": "success"}'
        
        result_message = FunctionExecutionResultMessage(content=[mock_result])
        result = client._convert_message_to_gemini(result_message)
        
        # Name should be normalized
        assert result.parts[0].function_response.name == "invalid_function_name"

    def test_tool_schema_parameter_type_mapping(self, mock_genai_client, function_calling_config):
        """Test mapping of different parameter types in tool schemas."""
        mock_genai, mock_client = mock_genai_client
        client = GeminiChatCompletionClient(**function_calling_config)
        
        # Tool with various parameter types
        complex_tool = {
            "name": "complex_function",
            "description": "Function with various parameter types",
            "parameters": {
                "type": "object",
                "properties": {
                    "string_param": {"type": "string"},
                    "number_param": {"type": "number"},
                    "integer_param": {"type": "integer"},
                    "boolean_param": {"type": "boolean"},
                    "array_param": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "object_param": {
                        "type": "object",
                        "properties": {
                            "nested_string": {"type": "string"}
                        }
                    },
                    "enum_param": {
                        "type": "string",
                        "enum": ["option1", "option2", "option3"]
                    }
                }
            }
        }
        
        gemini_tools = client._convert_tools_to_gemini([complex_tool])
        func_decl = gemini_tools[0].function_declarations[0]
        props = func_decl.parameters.properties
        
        # Verify type mappings
        assert props["string_param"].type == genai_types.Type.STRING
        assert props["number_param"].type == genai_types.Type.NUMBER
        assert props["integer_param"].type == genai_types.Type.INTEGER
        assert props["boolean_param"].type == genai_types.Type.BOOLEAN
        assert props["array_param"].type == genai_types.Type.ARRAY
        assert props["object_param"].type == genai_types.Type.OBJECT
        assert props["enum_param"].enum == ["option1", "option2", "option3"]

    def test_tool_schema_unsupported_type_handling(self, mock_genai_client, function_calling_config):
        """Test handling of unsupported parameter types in tool schemas."""
        mock_genai, mock_client = mock_genai_client
        client = GeminiChatCompletionClient(**function_calling_config)
        
        # Tool with unsupported type
        tool_with_unsupported = {
            "name": "test_function",
            "description": "Function with unsupported type",
            "parameters": {
                "type": "object",
                "properties": {
                    "unsupported_param": {"type": "unsupported_type"}
                }
            }
        }
        
        with patch('kagent.models.gemini._gemini_client.logger') as mock_logger:
            gemini_tools = client._convert_tools_to_gemini([tool_with_unsupported])
            
            # Should default to STRING type and log warning
            func_decl = gemini_tools[0].function_declarations[0]
            assert func_decl.parameters.properties["unsupported_param"].type == genai_types.Type.STRING
            mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_function_calling_with_json_output(self, mock_genai_client, function_calling_config, sample_tool_schema):
        """Test function calling combined with JSON output mode."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock response with function call in JSON mode
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].function_call = MagicMock()
        mock_response.candidates[0].content.parts[0].function_call.name = "get_weather"
        mock_response.candidates[0].content.parts[0].function_call.args = {"location": "Miami"}
        mock_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
        mock_response.usage_metadata.prompt_token_count = 25
        mock_response.usage_metadata.candidates_token_count = 15
        
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        client = GeminiChatCompletionClient(**function_calling_config)
        
        messages = [UserMessage(content="Get weather for Miami in JSON format")]
        result = await client.create(messages, tools=[sample_tool_schema], json_output=True)
        
        # Verify function call was generated even with JSON output
        assert isinstance(result.content, list)
        assert len(result.content) == 1
        assert isinstance(result.content[0], FunctionCall)
        
        # Verify JSON output was requested
        call_args = mock_client.aio.models.generate_content.call_args
        assert call_args[1]['config'].response_mime_type == "application/json"