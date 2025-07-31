import pytest
from unittest.mock import patch, MagicMock
from autogen_core import ComponentModel

from autogenstudio.validation.validation_service import ValidationService, ValidationError


class TestGeminiValidation:
    """Test suite for Gemini provider validation in ValidationService."""

    @pytest.fixture
    def gemini_component(self):
        """Basic Gemini component for testing."""
        return ComponentModel(
            provider="kagent.models.gemini.GeminiChatCompletionClient",
            component_type="model",
            config={
                "model": "gemini-1.5-pro",
                "api_key": "test-api-key",
                "temperature": 0.7,
            }
        )

    def test_validate_provider_gemini_success(self):
        """Test successful Gemini provider validation."""
        with patch('autogenstudio.validation.validation_service.importlib.import_module') as mock_import:
            with patch('autogenstudio.validation.validation_service.getattr') as mock_getattr:
                with patch('autogenstudio.validation.validation_service.is_component_class', return_value=True):
                    mock_module = MagicMock()
                    mock_import.return_value = mock_module
                    mock_component_class = MagicMock()
                    mock_getattr.return_value = mock_component_class
                    
                    result = ValidationService.validate_provider("kagent.models.gemini.GeminiChatCompletionClient")
                    
                    assert result is None  # No error means success
                    mock_import.assert_called_with("kagent.models.gemini")
                    mock_getattr.assert_called_with(mock_module, "GeminiChatCompletionClient")

    def test_validate_provider_gemini_alias(self):
        """Test Gemini provider validation with alias."""
        with patch('autogenstudio.validation.validation_service.importlib.import_module') as mock_import:
            with patch('autogenstudio.validation.validation_service.getattr') as mock_getattr:
                with patch('autogenstudio.validation.validation_service.is_component_class', return_value=True):
                    mock_module = MagicMock()
                    mock_import.return_value = mock_module
                    mock_component_class = MagicMock()
                    mock_getattr.return_value = mock_component_class
                    
                    # Test with alias
                    result = ValidationService.validate_provider("gemini_chat_completion_client")
                    
                    assert result is None
                    mock_import.assert_called_with("kagent.models.gemini")

    def test_validate_provider_gemini_import_error(self):
        """Test Gemini provider validation with import error."""
        with patch('autogenstudio.validation.validation_service.importlib.import_module', side_effect=ImportError("Module not found")):
            result = ValidationService.validate_provider("kagent.models.gemini.GeminiChatCompletionClient")
            
            assert isinstance(result, ValidationError)
            assert result.field == "provider"
            assert "Could not import provider" in result.error
            assert "Module not found" in result.error

    def test_validate_provider_gemini_not_component_class(self):
        """Test Gemini provider validation when class is not a component."""
        with patch('autogenstudio.validation.validation_service.importlib.import_module') as mock_import:
            with patch('autogenstudio.validation.validation_service.getattr') as mock_getattr:
                with patch('autogenstudio.validation.validation_service.is_component_class', return_value=False):
                    mock_module = MagicMock()
                    mock_import.return_value = mock_module
                    mock_component_class = MagicMock()
                    mock_getattr.return_value = mock_component_class
                    
                    result = ValidationService.validate_provider("kagent.models.gemini.GeminiChatCompletionClient")
                    
                    assert isinstance(result, ValidationError)
                    assert result.field == "provider"
                    assert "is not a valid component class" in result.error

    def test_validate_gemini_config_success(self, gemini_component):
        """Test successful Gemini configuration validation."""
        errors = ValidationService.validate_gemini_config(gemini_component)
        assert errors == []

    def test_validate_gemini_config_missing_api_key(self, gemini_component):
        """Test Gemini configuration validation with missing API key."""
        gemini_component.config.pop("api_key")
        
        errors = ValidationService.validate_gemini_config(gemini_component)
        
        assert len(errors) == 1
        assert errors[0].field == "api_key"
        assert "Gemini API key is required" in errors[0].error

    def test_validate_gemini_config_invalid_model(self, gemini_component):
        """Test Gemini configuration validation with invalid model."""
        gemini_component.config["model"] = "invalid-model"
        
        with patch('kagent.models.gemini._model_info.validate_model', return_value=False):
            errors = ValidationService.validate_gemini_config(gemini_component)
            
            assert len(errors) == 1
            assert errors[0].field == "model"
            assert "Unsupported Gemini model" in errors[0].error

    def test_validate_gemini_config_model_validation_import_error(self, gemini_component):
        """Test Gemini configuration validation when model info module is not available."""
        gemini_component.config["model"] = "some-model"
        
        with patch('kagent.models.gemini._model_info.validate_model', side_effect=ImportError("Module not found")):
            errors = ValidationService.validate_gemini_config(gemini_component)
            
            # Should not raise error when model info module is not available
            assert len(errors) == 0

    def test_validate_gemini_config_invalid_temperature_range(self, gemini_component):
        """Test Gemini configuration validation with invalid temperature range."""
        gemini_component.config["temperature"] = 3.0  # Above valid range
        
        errors = ValidationService.validate_gemini_config(gemini_component)
        
        assert len(errors) == 1
        assert errors[0].field == "temperature"
        assert "Temperature must be between 0.0 and 2.0" in errors[0].error

    def test_validate_gemini_config_invalid_temperature_type(self, gemini_component):
        """Test Gemini configuration validation with invalid temperature type."""
        gemini_component.config["temperature"] = "invalid"
        
        errors = ValidationService.validate_gemini_config(gemini_component)
        
        assert len(errors) == 1
        assert errors[0].field == "temperature"
        assert "Temperature must be a valid number" in errors[0].error

    def test_validate_gemini_config_invalid_top_p_range(self, gemini_component):
        """Test Gemini configuration validation with invalid top_p range."""
        gemini_component.config["top_p"] = 1.5  # Above valid range
        
        errors = ValidationService.validate_gemini_config(gemini_component)
        
        assert len(errors) == 1
        assert errors[0].field == "top_p"
        assert "top_p must be between 0.0 and 1.0" in errors[0].error

    def test_validate_gemini_config_invalid_top_p_type(self, gemini_component):
        """Test Gemini configuration validation with invalid top_p type."""
        gemini_component.config["top_p"] = "invalid"
        
        errors = ValidationService.validate_gemini_config(gemini_component)
        
        assert len(errors) == 1
        assert errors[0].field == "top_p"
        assert "top_p must be a valid number" in errors[0].error

    def test_validate_gemini_config_valid_safety_settings(self, gemini_component):
        """Test Gemini configuration validation with valid safety settings."""
        gemini_component.config["safety_settings"] = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_ONLY_HIGH",
        }
        
        errors = ValidationService.validate_gemini_config(gemini_component)
        
        assert len(errors) == 0

    def test_validate_gemini_config_invalid_safety_category(self, gemini_component):
        """Test Gemini configuration validation with invalid safety category."""
        gemini_component.config["safety_settings"] = {
            "INVALID_CATEGORY": "BLOCK_MEDIUM_AND_ABOVE",
        }
        
        errors = ValidationService.validate_gemini_config(gemini_component)
        
        assert len(errors) == 1
        assert errors[0].field == "safety_settings"
        assert "Invalid safety category" in errors[0].error

    def test_validate_gemini_config_invalid_safety_threshold(self, gemini_component):
        """Test Gemini configuration validation with invalid safety threshold."""
        gemini_component.config["safety_settings"] = {
            "HARM_CATEGORY_HARASSMENT": "INVALID_THRESHOLD",
        }
        
        errors = ValidationService.validate_gemini_config(gemini_component)
        
        assert len(errors) == 1
        assert errors[0].field == "safety_settings"
        assert "Invalid safety threshold" in errors[0].error

    def test_validate_gemini_config_multiple_safety_errors(self, gemini_component):
        """Test Gemini configuration validation with multiple safety setting errors."""
        gemini_component.config["safety_settings"] = {
            "INVALID_CATEGORY": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_HARASSMENT": "INVALID_THRESHOLD",
        }
        
        errors = ValidationService.validate_gemini_config(gemini_component)
        
        assert len(errors) == 2
        error_messages = [error.error for error in errors]
        assert any("Invalid safety category" in msg for msg in error_messages)
        assert any("Invalid safety threshold" in msg for msg in error_messages)

    def test_validate_gemini_config_non_gemini_provider(self):
        """Test that Gemini config validation is skipped for non-Gemini providers."""
        non_gemini_component = ComponentModel(
            provider="autogen_ext.models.openai.OpenAIChatCompletionClient",
            component_type="model",
            config={"model": "gpt-4", "api_key": "test-key"}
        )
        
        errors = ValidationService.validate_gemini_config(non_gemini_component)
        
        assert errors == []  # Should return empty list for non-Gemini providers

    def test_validate_gemini_config_all_safety_categories(self, gemini_component):
        """Test Gemini configuration validation with all valid safety categories."""
        gemini_component.config["safety_settings"] = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_LOW_AND_ABOVE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_ONLY_HIGH",
        }
        
        errors = ValidationService.validate_gemini_config(gemini_component)
        
        assert len(errors) == 0

    def test_validate_gemini_config_edge_case_temperature_boundaries(self, gemini_component):
        """Test Gemini configuration validation with temperature boundary values."""
        # Test minimum boundary
        gemini_component.config["temperature"] = 0.0
        errors = ValidationService.validate_gemini_config(gemini_component)
        assert len(errors) == 0
        
        # Test maximum boundary
        gemini_component.config["temperature"] = 2.0
        errors = ValidationService.validate_gemini_config(gemini_component)
        assert len(errors) == 0
        
        # Test just below minimum
        gemini_component.config["temperature"] = -0.1
        errors = ValidationService.validate_gemini_config(gemini_component)
        assert len(errors) == 1
        
        # Test just above maximum
        gemini_component.config["temperature"] = 2.1
        errors = ValidationService.validate_gemini_config(gemini_component)
        assert len(errors) == 1

    def test_validate_gemini_config_edge_case_top_p_boundaries(self, gemini_component):
        """Test Gemini configuration validation with top_p boundary values."""
        # Test minimum boundary
        gemini_component.config["top_p"] = 0.0
        errors = ValidationService.validate_gemini_config(gemini_component)
        assert len(errors) == 0
        
        # Test maximum boundary
        gemini_component.config["top_p"] = 1.0
        errors = ValidationService.validate_gemini_config(gemini_component)
        assert len(errors) == 0
        
        # Test just below minimum
        gemini_component.config["top_p"] = -0.1
        errors = ValidationService.validate_gemini_config(gemini_component)
        assert len(errors) == 1
        
        # Test just above maximum
        gemini_component.config["top_p"] = 1.1
        errors = ValidationService.validate_gemini_config(gemini_component)
        assert len(errors) == 1

    def test_validate_gemini_config_none_values(self, gemini_component):
        """Test Gemini configuration validation with None values."""
        gemini_component.config["temperature"] = None
        gemini_component.config["top_p"] = None
        
        errors = ValidationService.validate_gemini_config(gemini_component)
        
        # None values should be valid (optional parameters)
        assert len(errors) == 0

    def test_validate_gemini_config_empty_safety_settings(self, gemini_component):
        """Test Gemini configuration validation with empty safety settings."""
        gemini_component.config["safety_settings"] = {}
        
        errors = ValidationService.validate_gemini_config(gemini_component)
        
        assert len(errors) == 0

    def test_validate_gemini_config_non_dict_safety_settings(self, gemini_component):
        """Test Gemini configuration validation with non-dict safety settings."""
        gemini_component.config["safety_settings"] = "not_a_dict"
        
        errors = ValidationService.validate_gemini_config(gemini_component)
        
        # Should not crash, but also won't validate the contents
        assert len(errors) == 0

    def test_full_validation_gemini_success(self, gemini_component):
        """Test full validation pipeline for successful Gemini component."""
        with patch('autogenstudio.validation.validation_service.importlib.import_module') as mock_import:
            with patch('autogenstudio.validation.validation_service.getattr') as mock_getattr:
                with patch('autogenstudio.validation.validation_service.is_component_class', return_value=True):
                    # Mock successful component loading
                    mock_module = MagicMock()
                    mock_import.return_value = mock_module
                    mock_component_class = MagicMock()
                    mock_component_class.component_config_schema = MagicMock()
                    mock_component_class.component_config_schema.model_validate = MagicMock()
                    mock_component_class.load_component = MagicMock()
                    mock_getattr.return_value = mock_component_class
                    
                    result = ValidationService.validate(gemini_component)
                    
                    assert result.is_valid is True
                    assert len(result.errors) == 0

    def test_full_validation_gemini_with_errors(self):
        """Test full validation pipeline for Gemini component with errors."""
        invalid_component = ComponentModel(
            provider="kagent.models.gemini.GeminiChatCompletionClient",
            component_type="model",
            config={
                "model": "gemini-1.5-pro",
                # Missing API key
                "temperature": 3.0,  # Invalid temperature
            }
        )
        
        with patch('autogenstudio.validation.validation_service.importlib.import_module') as mock_import:
            with patch('autogenstudio.validation.validation_service.getattr') as mock_getattr:
                with patch('autogenstudio.validation.validation_service.is_component_class', return_value=True):
                    mock_module = MagicMock()
                    mock_import.return_value = mock_module
                    mock_component_class = MagicMock()
                    mock_component_class.component_config_schema = MagicMock()
                    mock_component_class.component_config_schema.model_validate = MagicMock()
                    mock_getattr.return_value = mock_component_class
                    
                    result = ValidationService.validate(invalid_component)
                    
                    assert result.is_valid is False
                    assert len(result.errors) >= 2  # API key + temperature errors

    def test_validate_gemini_config_comprehensive_parameter_validation(self, gemini_component):
        """Test comprehensive parameter validation for Gemini configuration."""
        # Test max_output_tokens validation
        gemini_component.config["max_output_tokens"] = -1
        errors = ValidationService.validate_gemini_config(gemini_component)
        assert any("max_output_tokens must be positive" in error.error for error in errors)
        
        # Test valid max_output_tokens
        gemini_component.config["max_output_tokens"] = 1024
        errors = ValidationService.validate_gemini_config(gemini_component)
        assert not any("max_output_tokens" in error.field for error in errors)
        
        # Test top_k validation
        gemini_component.config["top_k"] = -1
        errors = ValidationService.validate_gemini_config(gemini_component)
        assert any("top_k must be positive" in error.error for error in errors)
        
        # Test valid top_k
        gemini_component.config["top_k"] = 40
        errors = ValidationService.validate_gemini_config(gemini_component)
        assert not any("top_k" in error.field for error in errors)
        
        # Test candidate_count validation
        gemini_component.config["candidate_count"] = 0
        errors = ValidationService.validate_gemini_config(gemini_component)
        assert any("candidate_count must be positive" in error.error for error in errors)
        
        # Test valid candidate_count
        gemini_component.config["candidate_count"] = 1
        errors = ValidationService.validate_gemini_config(gemini_component)
        assert not any("candidate_count" in error.field for error in errors)

    def test_validate_gemini_config_base_url_validation(self, gemini_component):
        """Test base URL validation for Gemini configuration."""
        # Test valid URLs
        valid_urls = [
            "https://generativelanguage.googleapis.com",
            "https://custom-endpoint.example.com",
            "http://localhost:8080",
            "https://api.example.com/v1",
        ]
        
        for url in valid_urls:
            gemini_component.config["base_url"] = url
            errors = ValidationService.validate_gemini_config(gemini_component)
            assert not any("base_url" in error.field for error in errors), f"Valid URL {url} should not cause errors"
        
        # Test invalid URLs
        invalid_urls = [
            "not-a-url",
            "ftp://invalid-protocol.com",
            "https://",
            "",
        ]
        
        for url in invalid_urls:
            gemini_component.config["base_url"] = url
            errors = ValidationService.validate_gemini_config(gemini_component)
            if url:  # Empty string might be valid (use default)
                assert any("base_url" in error.field for error in errors), f"Invalid URL {url} should cause errors"

    def test_validate_gemini_config_response_mime_type_validation(self, gemini_component):
        """Test response MIME type validation for Gemini configuration."""
        # Test valid MIME types
        valid_mime_types = [
            "application/json",
            "text/plain",
            "text/x.enum",
        ]
        
        for mime_type in valid_mime_types:
            gemini_component.config["response_mime_type"] = mime_type
            errors = ValidationService.validate_gemini_config(gemini_component)
            assert not any("response_mime_type" in error.field for error in errors)
        
        # Test invalid MIME types
        invalid_mime_types = [
            "invalid-mime-type",
            "application/",
            "/json",
            "",
        ]
        
        for mime_type in invalid_mime_types:
            gemini_component.config["response_mime_type"] = mime_type
            errors = ValidationService.validate_gemini_config(gemini_component)
            if mime_type:  # Empty string might be valid (use default)
                assert any("response_mime_type" in error.field for error in errors)

    def test_validate_gemini_config_stop_sequences_validation(self, gemini_component):
        """Test stop sequences validation for Gemini configuration."""
        # Test valid stop sequences
        gemini_component.config["stop_sequences"] = ["STOP", "END", "FINISH"]
        errors = ValidationService.validate_gemini_config(gemini_component)
        assert not any("stop_sequences" in error.field for error in errors)
        
        # Test empty stop sequences
        gemini_component.config["stop_sequences"] = []
        errors = ValidationService.validate_gemini_config(gemini_component)
        assert not any("stop_sequences" in error.field for error in errors)
        
        # Test non-list stop sequences
        gemini_component.config["stop_sequences"] = "not-a-list"
        errors = ValidationService.validate_gemini_config(gemini_component)
        assert any("stop_sequences must be a list" in error.error for error in errors)
        
        # Test stop sequences with non-string elements
        gemini_component.config["stop_sequences"] = ["STOP", 123, "END"]
        errors = ValidationService.validate_gemini_config(gemini_component)
        assert any("stop_sequences must contain only strings" in error.error for error in errors)

    def test_validate_gemini_config_model_specific_validation(self, gemini_component):
        """Test model-specific validation for Gemini configuration."""
        # Test with vision model
        gemini_component.config["model"] = "gemini-pro-vision"
        
        with patch('kagent.models.gemini._model_info.validate_model', return_value=True):
            with patch('kagent.models.gemini._model_info.get_info', return_value={
                "vision": True,
                "function_calling": False,
                "json_output": True,
                "family": "gemini-1.0",
                "structured_output": False,
                "multiple_system_messages": False,
            }):
                errors = ValidationService.validate_gemini_config(gemini_component)
                assert len(errors) == 0
        
        # Test with non-vision model
        gemini_component.config["model"] = "gemini-1.0-pro"
        
        with patch('kagent.models.gemini._model_info.validate_model', return_value=True):
            with patch('kagent.models.gemini._model_info.get_info', return_value={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "gemini-1.0",
                "structured_output": False,
                "multiple_system_messages": False,
            }):
                errors = ValidationService.validate_gemini_config(gemini_component)
                assert len(errors) == 0

    def test_validate_gemini_config_concurrent_validation(self, gemini_component):
        """Test that validation works correctly with concurrent access."""
        import threading
        import time
        
        results = []
        errors_list = []
        
        def validate_config():
            try:
                errors = ValidationService.validate_gemini_config(gemini_component)
                results.append(len(errors))
            except Exception as e:
                errors_list.append(e)
        
        # Run multiple validations concurrently
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=validate_config)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check that all validations completed successfully
        assert len(errors_list) == 0, f"Concurrent validation errors: {errors_list}"
        assert len(results) == 10
        assert all(result == 0 for result in results), "All validations should succeed"

    def test_validate_gemini_config_memory_usage(self, gemini_component):
        """Test that validation doesn't cause memory leaks."""
        import gc
        import sys
        
        # Get initial memory usage
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Run validation many times
        for _ in range(100):
            ValidationService.validate_gemini_config(gemini_component)
        
        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory usage should not grow significantly
        object_growth = final_objects - initial_objects
        assert object_growth < 1000, f"Memory usage grew by {object_growth} objects"

    def test_validate_provider_gemini_case_sensitivity(self):
        """Test that provider validation is case sensitive."""
        # Test correct case
        with patch('autogenstudio.validation.validation_service.importlib.import_module') as mock_import:
            with patch('autogenstudio.validation.validation_service.getattr') as mock_getattr:
                with patch('autogenstudio.validation.validation_service.is_component_class', return_value=True):
                    mock_module = MagicMock()
                    mock_import.return_value = mock_module
                    mock_component_class = MagicMock()
                    mock_getattr.return_value = mock_component_class
                    
                    result = ValidationService.validate_provider("kagent.models.gemini.GeminiChatCompletionClient")
                    assert result is None
        
        # Test incorrect case
        result = ValidationService.validate_provider("kagent.models.gemini.geminichatcompletionclient")
        assert isinstance(result, ValidationError)

    def test_validate_gemini_config_with_model_info_override(self, gemini_component):
        """Test validation with model info override."""
        gemini_component.config["model_info_override"] = {
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": "custom",
            "structured_output": True,
            "multiple_system_messages": False,
        }
        
        errors = ValidationService.validate_gemini_config(gemini_component)
        
        # Should not validate model name when override is provided
        assert not any("Unsupported Gemini model" in error.error for error in errors)

    def test_validate_gemini_config_performance(self, gemini_component):
        """Test validation performance."""
        import time
        
        # Measure validation time
        start_time = time.time()
        for _ in range(100):
            ValidationService.validate_gemini_config(gemini_component)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / 100
        
        # Validation should be fast (less than 10ms per validation)
        assert avg_time < 0.01, f"Validation too slow: {avg_time:.4f}s per validation"

    def test_validate_gemini_config_error_message_quality(self, gemini_component):
        """Test that error messages are helpful and specific."""
        # Test with multiple errors
        gemini_component.config.pop("api_key")
        gemini_component.config["temperature"] = 3.0
        gemini_component.config["top_p"] = 1.5
        gemini_component.config["model"] = "invalid-model"
        
        with patch('kagent.models.gemini._model_info.validate_model', return_value=False):
            errors = ValidationService.validate_gemini_config(gemini_component)
        
        # Check that all errors are present and specific
        error_fields = [error.field for error in errors]
        error_messages = [error.error for error in errors]
        
        assert "api_key" in error_fields
        assert "temperature" in error_fields
        assert "top_p" in error_fields
        assert "model" in error_fields
        
        # Check that error messages are specific
        assert any("API key is required" in msg for msg in error_messages)
        assert any("Temperature must be between 0.0 and 2.0" in msg for msg in error_messages)
        assert any("top_p must be between 0.0 and 1.0" in msg for msg in error_messages)
        assert any("Unsupported Gemini model" in msg for msg in error_messages)

    def test_validate_gemini_config_with_all_optional_parameters(self, gemini_component):
        """Test validation with all optional parameters set."""
        gemini_component.config.update({
            "base_url": "https://custom-endpoint.com",
            "temperature": 0.8,
            "max_output_tokens": 2048,
            "top_p": 0.9,
            "top_k": 40,
            "candidate_count": 1,
            "stop_sequences": ["STOP", "END", "FINISH"],
            "response_mime_type": "application/json",
            "safety_settings": {
                "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_ONLY_HIGH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_LOW_AND_ABOVE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            }
        })
        
        errors = ValidationService.validate_gemini_config(gemini_component)
        
        assert len(errors) == 0, f"All valid parameters should not cause errors: {[e.error for e in errors]}"

    def test_validate_gemini_config_partial_safety_settings(self, gemini_component):
        """Test validation with partial safety settings."""
        # Test with only some safety categories
        gemini_component.config["safety_settings"] = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_ONLY_HIGH",
        }
        
        errors = ValidationService.validate_gemini_config(gemini_component)
        
        assert len(errors) == 0, "Partial safety settings should be valid"

    def test_validate_gemini_config_boundary_conditions(self, gemini_component):
        """Test validation with boundary conditions."""
        # Test with very large values
        gemini_component.config["max_output_tokens"] = 1000000
        gemini_component.config["top_k"] = 1000
        gemini_component.config["candidate_count"] = 8  # Gemini's maximum
        
        errors = ValidationService.validate_gemini_config(gemini_component)
        
        # Should handle large but valid values
        assert not any("max_output_tokens" in error.field for error in errors)
        assert not any("top_k" in error.field for error in errors)
        assert not any("candidate_count" in error.field for error in errors)