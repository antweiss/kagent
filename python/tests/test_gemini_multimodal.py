import base64
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from autogen_core import CancellationToken
from autogen_core.models import (
    CreateResult,
    UserMessage,
    SystemMessage,
)
from google.genai import types as genai_types

from kagent.models.gemini._gemini_client import GeminiChatCompletionClient
from kagent.models.gemini._model_info import get_vision_capable_models, GEMINI_MODELS
from kagent.models.gemini._message_conversion import (
    convert_message_to_gemini,
    _convert_base64_image_to_part,
    _convert_url_image_to_part,
)


class TestGeminiMultimodal:
    """Test suite for Gemini multimodal and vision capabilities."""

    @pytest.fixture
    def vision_config(self):
        """Configuration for vision-capable model testing."""
        return {
            "model": "gemini-1.5-pro",  # Vision-capable model
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
    def sample_base64_image(self):
        """Sample base64 encoded image for testing."""
        # Minimal valid JPEG header in base64
        return "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/8A=="

    def test_vision_capable_models_detection(self):
        """Test detection of vision-capable models."""
        vision_models = get_vision_capable_models()
        
        assert isinstance(vision_models, list)
        assert len(vision_models) > 0
        
        # Verify known vision models are included
        expected_vision_models = [
            "gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro-vision",
            "gemini-1.5-pro-latest", "gemini-1.5-flash-latest"
        ]
        
        for model in expected_vision_models:
            if model in GEMINI_MODELS:
                assert model in vision_models, f"Expected vision model {model} not found"
        
        # Verify all returned models actually have vision capability
        for model in vision_models:
            assert GEMINI_MODELS[model]["vision"] is True

    def test_model_capability_detection(self):
        """Test multimodal capability detection for different models."""
        test_cases = [
            ("gemini-1.5-pro", True, True, True),  # vision, function_calling, json_output
            ("gemini-1.5-flash", True, True, True),
            ("gemini-1.0-pro", False, True, True),  # No vision
            ("gemini-pro-vision", True, False, True),  # Vision but limited function calling
        ]
        
        for model_name, expected_vision, expected_functions, expected_json in test_cases:
            if model_name in GEMINI_MODELS:
                model_info = GEMINI_MODELS[model_name]
                assert model_info["vision"] == expected_vision, f"Vision capability mismatch for {model_name}"
                assert model_info["function_calling"] == expected_functions, f"Function calling mismatch for {model_name}"
                assert model_info["json_output"] == expected_json, f"JSON output mismatch for {model_name}"

    def test_base64_image_conversion(self, sample_base64_image):
        """Test base64 image conversion to Gemini format."""
        with patch('kagent.models.gemini._message_conversion.base64.b64decode') as mock_decode:
            mock_decode.return_value = b"fake_image_data"
            
            result = _convert_base64_image_to_part(sample_base64_image)
            
            assert result is not None
            assert hasattr(result, 'inline_data')
            assert result.inline_data.mime_type == "image/jpeg"
            assert result.inline_data.data == b"fake_image_data"
            mock_decode.assert_called_once()

    def test_base64_image_conversion_different_formats(self):
        """Test base64 image conversion with different image formats."""
        test_cases = [
            ("data:image/jpeg;base64,fake_data", "image/jpeg"),
            ("data:image/png;base64,fake_data", "image/png"),
            ("data:image/gif;base64,fake_data", "image/gif"),
            ("data:image/webp;base64,fake_data", "image/webp"),
        ]
        
        for data_url, expected_mime in test_cases:
            with patch('kagent.models.gemini._message_conversion.base64.b64decode') as mock_decode:
                mock_decode.return_value = b"fake_image_data"
                
                result = _convert_base64_image_to_part(data_url)
                
                assert result is not None
                assert result.inline_data.mime_type == expected_mime

    def test_base64_image_conversion_error_handling(self):
        """Test base64 image conversion error handling."""
        invalid_data_url = "data:image/jpeg;base64,invalid_base64_data"
        
        with patch('kagent.models.gemini._message_conversion.base64.b64decode', side_effect=Exception("Invalid base64")):
            with patch('kagent.models.gemini._message_conversion.logger') as mock_logger:
                result = _convert_base64_image_to_part(invalid_data_url)
                
                assert result is None
                mock_logger.warning.assert_called()

    def test_url_image_conversion(self):
        """Test URL image conversion to Gemini format."""
        test_cases = [
            ("https://example.com/image.jpg", "image/jpeg"),
            ("https://example.com/image.png", "image/png"),
            ("https://example.com/image.gif", "image/gif"),
            ("https://example.com/image.webp", "image/webp"),
            ("https://example.com/image", "image/jpeg"),  # Default
        ]
        
        for url, expected_mime in test_cases:
            result = _convert_url_image_to_part(url)
            
            assert result is not None
            assert hasattr(result, 'file_data')
            assert result.file_data.file_uri == url
            assert result.file_data.mime_type == expected_mime

    def test_multimodal_message_conversion_text_and_image(self, sample_base64_image):
        """Test conversion of multimodal message with text and image."""
        multimodal_content = [
            {"type": "text", "text": "What do you see in this image?"},
            {"type": "image_url", "image_url": {"url": sample_base64_image}}
        ]
        
        message = UserMessage(content=multimodal_content)
        
        with patch('kagent.models.gemini._message_conversion.base64.b64decode') as mock_decode:
            mock_decode.return_value = b"fake_image_data"
            
            result = convert_message_to_gemini(message)
            
            assert result is not None
            assert result.role == "user"
            assert len(result.parts) == 2
            
            # First part should be text
            assert result.parts[0].text == "What do you see in this image?"
            
            # Second part should be image
            assert hasattr(result.parts[1], 'inline_data')

    def test_multimodal_message_conversion_multiple_images(self):
        """Test conversion of message with multiple images."""
        multimodal_content = [
            {"type": "text", "text": "Compare these two images:"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/image2.png"}},
            {"type": "text", "text": "What are the differences?"}
        ]
        
        message = UserMessage(content=multimodal_content)
        result = convert_message_to_gemini(message)
        
        assert result is not None
        assert result.role == "user"
        assert len(result.parts) == 4
        
        # Check content types
        assert result.parts[0].text == "Compare these two images:"
        assert hasattr(result.parts[1], 'file_data')
        assert hasattr(result.parts[2], 'file_data')
        assert result.parts[3].text == "What are the differences?"

    def test_multimodal_message_conversion_mixed_content(self, sample_base64_image):
        """Test conversion of message with mixed content types."""
        multimodal_content = [
            "Initial text as string",
            {"type": "text", "text": "Structured text"},
            {"type": "image_url", "image_url": {"url": sample_base64_image}},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
        ]
        
        message = UserMessage(content=multimodal_content)
        
        with patch('kagent.models.gemini._message_conversion.base64.b64decode') as mock_decode:
            mock_decode.return_value = b"fake_image_data"
            
            result = convert_message_to_gemini(message)
            
            assert result is not None
            assert len(result.parts) == 4
            
            # Check all parts are converted correctly
            assert result.parts[0].text == "Initial text as string"
            assert result.parts[1].text == "Structured text"
            assert hasattr(result.parts[2], 'inline_data')  # Base64 image
            assert hasattr(result.parts[3], 'file_data')    # URL image

    @pytest.mark.asyncio
    async def test_vision_model_image_processing(self, mock_genai_client, vision_config, sample_base64_image):
        """Test vision model processing of image content."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock vision response
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = "I can see a beautiful landscape with mountains and a lake."
        mock_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
        mock_response.usage_metadata.prompt_token_count = 50
        mock_response.usage_metadata.candidates_token_count = 25
        
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        client = GeminiChatCompletionClient(**vision_config)
        
        # Create multimodal message
        multimodal_content = [
            {"type": "text", "text": "Describe what you see in this image."},
            {"type": "image_url", "image_url": {"url": sample_base64_image}}
        ]
        
        messages = [UserMessage(content=multimodal_content)]
        result = await client.create(messages)
        
        # Verify vision processing result
        assert result.content == "I can see a beautiful landscape with mountains and a lake."
        assert result.finish_reason == "stop"
        
        # Verify multimodal content was passed to API
        call_args = mock_client.aio.models.generate_content.call_args
        contents = call_args[1]['contents']
        assert len(contents) == 1
        assert len(contents[0].parts) == 2  # Text + image

    @pytest.mark.asyncio
    async def test_vision_model_multiple_images(self, mock_genai_client, vision_config):
        """Test vision model processing multiple images."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock comparison response
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = "The first image shows a cat, while the second shows a dog. Both are pets but different species."
        mock_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
        mock_response.usage_metadata.prompt_token_count = 75
        mock_response.usage_metadata.candidates_token_count = 35
        
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        client = GeminiChatCompletionClient(**vision_config)
        
        # Create message with multiple images
        multimodal_content = [
            {"type": "text", "text": "Compare these two images:"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/dog.jpg"}},
            {"type": "text", "text": "What are the main differences?"}
        ]
        
        messages = [UserMessage(content=multimodal_content)]
        result = await client.create(messages)
        
        # Verify comparison result
        assert "cat" in result.content.lower()
        assert "dog" in result.content.lower()
        
        # Verify multiple images were processed
        call_args = mock_client.aio.models.generate_content.call_args
        contents = call_args[1]['contents']
        assert len(contents) == 1
        assert len(contents[0].parts) == 4  # Text + image + image + text

    @pytest.mark.asyncio
    async def test_vision_model_with_system_prompt(self, mock_genai_client, vision_config, sample_base64_image):
        """Test vision model with system prompt for specialized analysis."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock specialized analysis response
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = "Medical Analysis: The X-ray shows normal bone structure with no visible fractures or abnormalities."
        mock_response.candidates[0].finish_reason = genai_types.FinishReason.STOP
        mock_response.usage_metadata.prompt_token_count = 60
        mock_response.usage_metadata.candidates_token_count = 30
        
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        client = GeminiChatCompletionClient(**vision_config)
        
        messages = [
            SystemMessage(content="You are a medical imaging specialist. Analyze medical images with professional expertise."),
            UserMessage(content=[
                {"type": "text", "text": "Please analyze this X-ray image."},
                {"type": "image_url", "image_url": {"url": sample_base64_image}}
            ])
        ]
        
        result = await client.create(messages)
        
        # Verify specialized analysis
        assert "Medical Analysis" in result.content
        assert "X-ray" in result.content
        
        # Verify system instruction was used
        call_args = mock_client.aio.models.generate_content.call_args
        assert "medical imaging specialist" in call_args[1]['config'].system_instruction.lower()

    @pytest.mark.asyncio
    async def test_vision_capability_validation(self, mock_genai_client):
        """Test validation of vision capabilities for different models."""
        mock_genai, mock_client = mock_genai_client
        
        # Test vision-capable model
        vision_config = {
            "model": "gemini-1.5-pro",
            "api_key": "test-api-key",
            "model_info_override": {
                "vision": True,
                "function_calling": True,
                "json_output": True,
                "family": "gemini-1.5",
                "structured_output": True,
                "multiple_system_messages": False,
            }
        }
        
        client = GeminiChatCompletionClient(**vision_config)
        assert client._model_info["vision"] is True
        
        # Test non-vision model
        non_vision_config = {
            "model": "gemini-1.0-pro",
            "api_key": "test-api-key",
            "model_info_override": {
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "gemini-1.0",
                "structured_output": False,
                "multiple_system_messages": False,
            }
        }
        
        non_vision_client = GeminiChatCompletionClient(**non_vision_config)
        assert non_vision_client._model_info["vision"] is False

    def test_image_input_processing_through_kagent_interfaces(self, sample_base64_image):
        """Test image input processing through KAgent interfaces."""
        # Test that images are properly processed through the standard KAgent message interface
        
        # Test 1: Base64 image
        base64_message = UserMessage(content=[
            {"type": "text", "text": "Analyze this image"},
            {"type": "image_url", "image_url": {"url": sample_base64_image}}
        ])
        
        with patch('kagent.models.gemini._message_conversion.base64.b64decode') as mock_decode:
            mock_decode.return_value = b"fake_image_data"
            
            result = convert_message_to_gemini(base64_message)
            assert len(result.parts) == 2
            assert hasattr(result.parts[1], 'inline_data')
        
        # Test 2: URL image
        url_message = UserMessage(content=[
            {"type": "text", "text": "Analyze this image"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ])
        
        result = convert_message_to_gemini(url_message)
        assert len(result.parts) == 2
        assert hasattr(result.parts[1], 'file_data')
        
        # Test 3: Mixed content
        mixed_message = UserMessage(content=[
            "Text as string",
            {"type": "text", "text": "Structured text"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ])
        
        result = convert_message_to_gemini(mixed_message)
        assert len(result.parts) == 3
        assert result.parts[0].text == "Text as string"
        assert result.parts[1].text == "Structured text"
        assert hasattr(result.parts[2], 'file_data')

    @pytest.mark.asyncio
    async def test_multimodal_streaming_response(self, mock_genai_client, vision_config, sample_base64_image):
        """Test streaming response with multimodal input."""
        mock_genai, mock_client = mock_genai_client
        
        # Mock streaming vision response
        async def mock_stream():
            chunk1 = MagicMock()
            chunk1.candidates = [MagicMock()]
            chunk1.candidates[0].content.parts = [MagicMock()]
            chunk1.candidates[0].content.parts[0].text = "I can see"
            chunk1.candidates[0].finish_reason = None
            chunk1.usage_metadata = None
            yield chunk1
            
            chunk2 = MagicMock()
            chunk2.candidates = [MagicMock()]
            chunk2.candidates[0].content.parts = [MagicMock()]
            chunk2.candidates[0].content.parts[0].text = " a beautiful sunset"
            chunk2.candidates[0].finish_reason = None
            chunk2.usage_metadata = None
            yield chunk2
            
            chunk3 = MagicMock()
            chunk3.candidates = [MagicMock()]
            chunk3.candidates[0].content.parts = [MagicMock()]
            chunk3.candidates[0].content.parts[0].text = " over the ocean."
            chunk3.candidates[0].finish_reason = genai_types.FinishReason.STOP
            chunk3.usage_metadata.prompt_token_count = 40
            chunk3.usage_metadata.candidates_token_count = 20
            yield chunk3
        
        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=mock_stream())
        
        client = GeminiChatCompletionClient(**vision_config)
        
        multimodal_content = [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": sample_base64_image}}
        ]
        
        messages = [UserMessage(content=multimodal_content)]
        
        # Test streaming with multimodal input
        chunks = []
        async for chunk in client.create_stream(messages):
            chunks.append(chunk)
        
        # Verify streaming results
        assert len(chunks) == 4  # 3 text chunks + final result
        assert chunks[0] == "I can see"
        assert chunks[1] == " a beautiful sunset"
        assert chunks[2] == " over the ocean."
        assert isinstance(chunks[3], CreateResult)
        assert chunks[3].content == "I can see a beautiful sunset over the ocean."

    def test_multimodal_content_handling_edge_cases(self):
        """Test edge cases in multimodal content handling."""
        # Test 1: Empty image URL
        empty_url_content = [
            {"type": "text", "text": "Test"},
            {"type": "image_url", "image_url": {"url": ""}}
        ]
        
        message = UserMessage(content=empty_url_content)
        
        with patch('kagent.models.gemini._message_conversion.logger') as mock_logger:
            result = convert_message_to_gemini(message)
            # Should still create message but may log warning
            assert result is not None
            assert len(result.parts) >= 1  # At least the text part
        
        # Test 2: Invalid image URL format
        invalid_url_content = [
            {"type": "text", "text": "Test"},
            {"type": "image_url", "image_url": {"url": "invalid-url-format"}}
        ]
        
        message = UserMessage(content=invalid_url_content)
        
        with patch('kagent.models.gemini._message_conversion.logger') as mock_logger:
            result = convert_message_to_gemini(message)
            assert result is not None
            mock_logger.warning.assert_called()
        
        # Test 3: Missing image_url key
        missing_key_content = [
            {"type": "text", "text": "Test"},
            {"type": "image_url", "image_url": {}}  # Missing 'url' key
        ]
        
        message = UserMessage(content=missing_key_content)
        result = convert_message_to_gemini(message)
        assert result is not None
        # Should handle gracefully

    def test_vision_model_capability_consistency(self):
        """Test consistency of vision model capabilities."""
        vision_models = get_vision_capable_models()
        
        for model_name in vision_models:
            model_info = GEMINI_MODELS[model_name]
            
            # Vision models should generally support JSON output
            assert model_info["json_output"] is True, f"Vision model {model_name} should support JSON output"
            
            # Most modern vision models should support function calling
            if "1.5" in model_name or "2.0" in model_name:
                assert model_info["function_calling"] is True, f"Modern vision model {model_name} should support function calling"
            
            # Vision models should have reasonable families
            assert model_info["family"] in ["gemini-1.0", "gemini-1.5", "gemini-2.0", "gemini-exp"], f"Unexpected family for {model_name}"

    @pytest.mark.asyncio
    async def test_vision_model_error_handling(self, mock_genai_client, vision_config, sample_base64_image):
        """Test error handling specific to vision processing."""
        mock_genai, mock_client = mock_genai_client
        
        # Test image processing error
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("Image processing failed")
        )
        
        client = GeminiChatCompletionClient(**vision_config)
        
        multimodal_content = [
            {"type": "text", "text": "Analyze this image"},
            {"type": "image_url", "image_url": {"url": sample_base64_image}}
        ]
        
        messages = [UserMessage(content=multimodal_content)]
        
        with pytest.raises(Exception, match="Image processing failed"):
            await client.create(messages)

    def test_multimodal_message_validation(self):
        """Test validation of multimodal message formats."""
        # Test valid multimodal formats
        valid_formats = [
            [{"type": "text", "text": "Hello"}],
            [{"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}],
            [
                {"type": "text", "text": "Analyze this"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
            ],
            ["String content", {"type": "text", "text": "Mixed content"}],
        ]
        
        for content in valid_formats:
            message = UserMessage(content=content)
            result = convert_message_to_gemini(message)
            assert result is not None
            assert result.role == "user"
            assert len(result.parts) > 0
        
        # Test handling of unknown content types
        unknown_type_content = [
            {"type": "unknown_type", "data": "some data"}
        ]
        
        message = UserMessage(content=unknown_type_content)
        
        with patch('kagent.models.gemini._message_conversion.logger') as mock_logger:
            result = convert_message_to_gemini(message)
            assert result is not None
            mock_logger.warning.assert_called()