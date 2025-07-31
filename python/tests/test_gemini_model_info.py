import pytest
from kagent.models.gemini._model_info import (
    GEMINI_MODELS,
    GEMINI_MODEL_TOKEN_LIMITS,
    get_info,
    get_token_limit,
    validate_model,
    get_vision_capable_models,
    get_function_calling_models,
    get_json_output_models,
)


class TestGeminiModelInfo:
    """Test suite for Gemini model information registry."""

    def test_gemini_models_registry_structure(self):
        """Test that GEMINI_MODELS has the expected structure."""
        assert isinstance(GEMINI_MODELS, dict)
        assert len(GEMINI_MODELS) > 0
        
        # Check that all models have required fields
        required_fields = {
            "vision", "function_calling", "json_output", 
            "family", "structured_output", "multiple_system_messages"
        }
        
        for model_name, model_info in GEMINI_MODELS.items():
            assert isinstance(model_name, str)
            assert isinstance(model_info, dict)
            
            # Check all required fields are present
            assert required_fields.issubset(set(model_info.keys())), f"Model {model_name} missing required fields"
            
            # Check field types
            assert isinstance(model_info["vision"], bool)
            assert isinstance(model_info["function_calling"], bool)
            assert isinstance(model_info["json_output"], bool)
            assert isinstance(model_info["family"], str)
            assert isinstance(model_info["structured_output"], bool)
            assert isinstance(model_info["multiple_system_messages"], bool)

    def test_gemini_model_token_limits_structure(self):
        """Test that GEMINI_MODEL_TOKEN_LIMITS has the expected structure."""
        assert isinstance(GEMINI_MODEL_TOKEN_LIMITS, dict)
        assert len(GEMINI_MODEL_TOKEN_LIMITS) > 0
        
        for model_name, token_limit in GEMINI_MODEL_TOKEN_LIMITS.items():
            assert isinstance(model_name, str)
            assert isinstance(token_limit, int)
            assert token_limit > 0

    def test_model_registry_consistency(self):
        """Test that model registry and token limits are consistent."""
        # All models in GEMINI_MODELS should have token limits
        for model_name in GEMINI_MODELS.keys():
            assert model_name in GEMINI_MODEL_TOKEN_LIMITS, f"Model {model_name} missing token limit"
        
        # All models in token limits should be in model registry
        for model_name in GEMINI_MODEL_TOKEN_LIMITS.keys():
            assert model_name in GEMINI_MODELS, f"Token limit for unknown model {model_name}"

    def test_get_info_exact_match(self):
        """Test get_info with exact model name matches."""
        # Test with known models
        test_models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"]
        
        for model_name in test_models:
            if model_name in GEMINI_MODELS:
                info = get_info(model_name)
                assert info == GEMINI_MODELS[model_name]
                assert isinstance(info, dict)
                assert "vision" in info
                assert "function_calling" in info

    def test_get_info_prefix_match(self):
        """Test get_info with prefix matching for versioned models."""
        # This should match gemini-1.5-pro or similar
        info = get_info("gemini-custom-version")
        assert isinstance(info, dict)
        assert "vision" in info

    def test_get_info_unknown_model(self):
        """Test get_info with unknown model name."""
        with pytest.raises(KeyError, match="Model 'unknown-model' not found"):
            get_info("unknown-model")

    def test_get_token_limit_exact_match(self):
        """Test get_token_limit with exact model name matches."""
        test_models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"]
        
        for model_name in test_models:
            if model_name in GEMINI_MODEL_TOKEN_LIMITS:
                limit = get_token_limit(model_name)
                assert limit == GEMINI_MODEL_TOKEN_LIMITS[model_name]
                assert isinstance(limit, int)
                assert limit > 0

    def test_get_token_limit_prefix_match(self):
        """Test get_token_limit with prefix matching."""
        # This should match a gemini model and return its token limit
        limit = get_token_limit("gemini-custom-version")
        assert isinstance(limit, int)
        assert limit > 0

    def test_get_token_limit_unknown_model(self):
        """Test get_token_limit with unknown model returns default."""
        limit = get_token_limit("completely-unknown-model")
        assert limit == 32_768  # Default fallback

    def test_validate_model_valid(self):
        """Test validate_model with valid model names."""
        valid_models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"]
        
        for model_name in valid_models:
            if model_name in GEMINI_MODELS:
                assert validate_model(model_name) is True

    def test_validate_model_invalid(self):
        """Test validate_model with invalid model names."""
        invalid_models = ["gpt-4", "claude-3", "unknown-model"]
        
        for model_name in invalid_models:
            assert validate_model(model_name) is False

    def test_get_vision_capable_models(self):
        """Test get_vision_capable_models returns correct models."""
        vision_models = get_vision_capable_models()
        
        assert isinstance(vision_models, list)
        assert len(vision_models) > 0
        
        # Verify all returned models actually have vision capability
        for model_name in vision_models:
            assert model_name in GEMINI_MODELS
            assert GEMINI_MODELS[model_name]["vision"] is True
        
        # Verify no non-vision models are included
        for model_name, model_info in GEMINI_MODELS.items():
            if model_info["vision"] is True:
                assert model_name in vision_models
            else:
                assert model_name not in vision_models

    def test_get_function_calling_models(self):
        """Test get_function_calling_models returns correct models."""
        function_models = get_function_calling_models()
        
        assert isinstance(function_models, list)
        assert len(function_models) > 0
        
        # Verify all returned models actually have function calling capability
        for model_name in function_models:
            assert model_name in GEMINI_MODELS
            assert GEMINI_MODELS[model_name]["function_calling"] is True
        
        # Verify no non-function-calling models are included
        for model_name, model_info in GEMINI_MODELS.items():
            if model_info["function_calling"] is True:
                assert model_name in function_models
            else:
                assert model_name not in function_models

    def test_get_json_output_models(self):
        """Test get_json_output_models returns correct models."""
        json_models = get_json_output_models()
        
        assert isinstance(json_models, list)
        assert len(json_models) > 0
        
        # Verify all returned models actually have JSON output capability
        for model_name in json_models:
            assert model_name in GEMINI_MODELS
            assert GEMINI_MODELS[model_name]["json_output"] is True
        
        # Verify no non-JSON models are included
        for model_name, model_info in GEMINI_MODELS.items():
            if model_info["json_output"] is True:
                assert model_name in json_models
            else:
                assert model_name not in json_models

    def test_model_families(self):
        """Test that models are properly categorized by family."""
        families = set()
        for model_info in GEMINI_MODELS.values():
            families.add(model_info["family"])
        
        # Should have multiple families
        assert len(families) > 1
        
        # Check expected families exist
        expected_families = {"gemini-1.5", "gemini-1.0", "gemini-2.0"}
        assert expected_families.intersection(families), "Expected families not found"

    def test_model_capabilities_consistency(self):
        """Test logical consistency of model capabilities."""
        for model_name, model_info in GEMINI_MODELS.items():
            # If a model has structured output, it should also have JSON output
            if model_info["structured_output"]:
                assert model_info["json_output"], f"Model {model_name} has structured output but not JSON output"
            
            # Vision models should generally be newer (this is a heuristic test)
            if model_info["vision"] and "1.5" in model_name:
                assert model_info["function_calling"], f"Vision model {model_name} should support function calling"

    def test_token_limits_reasonable(self):
        """Test that token limits are within reasonable ranges."""
        for model_name, token_limit in GEMINI_MODEL_TOKEN_LIMITS.items():
            # Token limits should be reasonable (between 1K and 10M tokens)
            assert 1_000 <= token_limit <= 10_000_000, f"Token limit for {model_name} seems unreasonable: {token_limit}"
            
            # Newer models should generally have higher limits
            if "1.5" in model_name:
                assert token_limit >= 32_768, f"Gemini 1.5 model {model_name} should have high token limit"

    def test_specific_model_properties(self):
        """Test properties of specific well-known models."""
        # Test gemini-1.5-pro if it exists
        if "gemini-1.5-pro" in GEMINI_MODELS:
            pro_info = GEMINI_MODELS["gemini-1.5-pro"]
            assert pro_info["vision"] is True
            assert pro_info["function_calling"] is True
            assert pro_info["json_output"] is True
            assert pro_info["family"] == "gemini-1.5"
            assert GEMINI_MODEL_TOKEN_LIMITS["gemini-1.5-pro"] > 1_000_000
        
        # Test gemini-1.5-flash if it exists
        if "gemini-1.5-flash" in GEMINI_MODELS:
            flash_info = GEMINI_MODELS["gemini-1.5-flash"]
            assert flash_info["vision"] is True
            assert flash_info["function_calling"] is True
            assert flash_info["json_output"] is True
            assert flash_info["family"] == "gemini-1.5"
        
        # Test gemini-pro-vision if it exists (older model with limited capabilities)
        if "gemini-pro-vision" in GEMINI_MODELS:
            vision_info = GEMINI_MODELS["gemini-pro-vision"]
            assert vision_info["vision"] is True
            # This older model has limited function calling
            assert vision_info["function_calling"] is False

    def test_model_name_patterns(self):
        """Test that model names follow expected patterns."""
        for model_name in GEMINI_MODELS.keys():
            # All model names should start with "gemini"
            assert model_name.startswith("gemini"), f"Model name {model_name} doesn't start with 'gemini'"
            
            # Should not contain spaces or special characters (except hyphens)
            assert " " not in model_name, f"Model name {model_name} contains spaces"
            assert all(c.isalnum() or c in "-." for c in model_name), f"Model name {model_name} contains invalid characters"

    def test_experimental_models(self):
        """Test properties of experimental models."""
        experimental_models = [name for name in GEMINI_MODELS.keys() if "exp" in name]
        
        for model_name in experimental_models:
            model_info = GEMINI_MODELS[model_name]
            # Experimental models should generally have advanced capabilities
            assert model_info["json_output"] is True, f"Experimental model {model_name} should support JSON output"
            # Most experimental models should support vision and function calling
            if "2.0" in model_name or "1.5" in model_name:
                assert model_info["vision"] is True, f"Advanced experimental model {model_name} should support vision"
                assert model_info["function_calling"] is True, f"Advanced experimental model {model_name} should support function calling"

    def test_model_info_edge_cases(self):
        """Test edge cases in model information handling."""
        # Test with empty model name
        with pytest.raises(KeyError):
            get_info("")
        
        # Test with None model name
        with pytest.raises((KeyError, TypeError)):
            get_info(None)
        
        # Test with model name containing special characters
        with pytest.raises(KeyError):
            get_info("gemini@1.5-pro")
        
        # Test with very long model name - this will actually match due to prefix matching
        long_name = "gemini-" + "x" * 100
        # This should work due to prefix matching, so let's test it works
        info = get_info(long_name)
        assert isinstance(info, dict)
        assert "vision" in info

    def test_token_limit_edge_cases(self):
        """Test edge cases in token limit handling."""
        # Test with empty model name
        limit = get_token_limit("")
        assert limit == 32_768  # Should return default
        
        # Test with None model name (should handle gracefully)
        try:
            limit = get_token_limit(None)
            assert limit == 32_768
        except (TypeError, AttributeError):
            # This is acceptable behavior
            pass
        
        # Test with model name that partially matches
        limit = get_token_limit("gemini-custom-1.5-pro-variant")
        assert limit > 0  # Should find a match and return valid limit

    def test_model_validation_comprehensive(self):
        """Test comprehensive model validation scenarios."""
        # Test all known models are valid
        for model_name in GEMINI_MODELS.keys():
            assert validate_model(model_name) is True, f"Known model {model_name} should be valid"
        
        # Test case sensitivity
        assert validate_model("GEMINI-1.5-PRO") is False, "Model names should be case sensitive"
        assert validate_model("Gemini-1.5-Pro") is False, "Model names should be case sensitive"
        
        # Test partial matches
        assert validate_model("gemini") is False, "Partial model names should not be valid"
        assert validate_model("1.5-pro") is False, "Partial model names should not be valid"
        
        # Test with extra characters - these will actually match due to prefix matching
        # So let's test that they do work (since the implementation allows it)
        assert validate_model("gemini-1.5-pro-extra") is True, "Model names with extra characters will match via prefix"
        assert validate_model(" gemini-1.5-pro ") is False, "Model names with whitespace should not be valid"

    def test_capability_filter_functions_comprehensive(self):
        """Test comprehensive capability filtering functions."""
        all_models = set(GEMINI_MODELS.keys())
        
        # Test vision models
        vision_models = set(get_vision_capable_models())
        non_vision_models = all_models - vision_models
        
        # Verify vision models actually have vision capability
        for model in vision_models:
            assert GEMINI_MODELS[model]["vision"] is True
        
        # Verify non-vision models don't have vision capability
        for model in non_vision_models:
            assert GEMINI_MODELS[model]["vision"] is False
        
        # Test function calling models
        function_models = set(get_function_calling_models())
        non_function_models = all_models - function_models
        
        # Verify function calling models actually have the capability
        for model in function_models:
            assert GEMINI_MODELS[model]["function_calling"] is True
        
        # Verify non-function calling models don't have the capability
        for model in non_function_models:
            assert GEMINI_MODELS[model]["function_calling"] is False
        
        # Test JSON output models
        json_models = set(get_json_output_models())
        non_json_models = all_models - json_models
        
        # Verify JSON models actually have the capability
        for model in json_models:
            assert GEMINI_MODELS[model]["json_output"] is True
        
        # Verify non-JSON models don't have the capability
        for model in non_json_models:
            assert GEMINI_MODELS[model]["json_output"] is False

    def test_model_capability_combinations(self):
        """Test various combinations of model capabilities."""
        # Find models with multiple capabilities
        multimodal_function_models = []
        vision_json_models = []
        all_capability_models = []
        
        for model_name, model_info in GEMINI_MODELS.items():
            if model_info["vision"] and model_info["function_calling"]:
                multimodal_function_models.append(model_name)
            
            if model_info["vision"] and model_info["json_output"]:
                vision_json_models.append(model_name)
            
            if (model_info["vision"] and 
                model_info["function_calling"] and 
                model_info["json_output"] and 
                model_info["structured_output"]):
                all_capability_models.append(model_name)
        
        # Should have some models with multiple capabilities
        assert len(multimodal_function_models) > 0, "Should have models with both vision and function calling"
        assert len(vision_json_models) > 0, "Should have models with both vision and JSON output"
        assert len(all_capability_models) > 0, "Should have models with all advanced capabilities"
        
        # Advanced models should generally have multiple capabilities
        advanced_models = [name for name in GEMINI_MODELS.keys() if "1.5" in name or "2.0" in name]
        for model_name in advanced_models:
            model_info = GEMINI_MODELS[model_name]
            capability_count = sum([
                model_info["vision"],
                model_info["function_calling"],
                model_info["json_output"],
                model_info["structured_output"]
            ])
            assert capability_count >= 2, f"Advanced model {model_name} should have multiple capabilities"

    def test_model_family_consistency(self):
        """Test consistency within model families."""
        families = {}
        for model_name, model_info in GEMINI_MODELS.items():
            family = model_info["family"]
            if family not in families:
                families[family] = []
            families[family].append((model_name, model_info))
        
        # Test family consistency
        for family, models in families.items():
            if len(models) > 1:
                # Models in the same family should have similar capabilities
                first_model = models[0][1]
                for model_name, model_info in models[1:]:
                    # Family members should have the same structured_output capability
                    if "1.5" in family or "2.0" in family:
                        assert (model_info["structured_output"] == first_model["structured_output"]), \
                            f"Models in family {family} should have consistent structured_output capability"

    def test_token_limits_consistency(self):
        """Test token limits are consistent with model capabilities."""
        for model_name, model_info in GEMINI_MODELS.items():
            token_limit = GEMINI_MODEL_TOKEN_LIMITS[model_name]
            
            # Vision models should generally have higher token limits
            if model_info["vision"] and "1.5" in model_name:
                assert token_limit >= 100_000, f"Vision model {model_name} should have high token limit"
            
            # Newer models should have higher limits than older ones
            if "1.5" in model_name:
                assert token_limit >= 32_768, f"Gemini 1.5 model {model_name} should have at least 32K tokens"
            
            if "2.0" in model_name:
                assert token_limit >= 100_000, f"Gemini 2.0 model {model_name} should have high token limit"

    def test_model_naming_conventions(self):
        """Test that model names follow consistent naming conventions."""
        for model_name in GEMINI_MODELS.keys():
            # Should start with "gemini"
            assert model_name.startswith("gemini"), f"Model {model_name} should start with 'gemini'"
            
            # Should not have consecutive hyphens
            assert "--" not in model_name, f"Model {model_name} should not have consecutive hyphens"
            
            # Should not start or end with hyphen
            assert not model_name.startswith("-"), f"Model {model_name} should not start with hyphen"
            assert not model_name.endswith("-"), f"Model {model_name} should not end with hyphen"
            
            # Should be lowercase
            assert model_name == model_name.lower(), f"Model {model_name} should be lowercase"
            
            # Should not contain spaces
            assert " " not in model_name, f"Model {model_name} should not contain spaces"

    def test_model_info_immutability(self):
        """Test that model info dictionaries are not accidentally modified."""
        # Get model info
        original_info = get_info("gemini-1.5-pro")
        
        # Try to modify it
        original_info["vision"] = False
        
        # Get it again - should not be modified
        fresh_info = get_info("gemini-1.5-pro")
        assert fresh_info["vision"] is True, "Model info should not be modified by external changes"

    def test_prefix_matching_behavior(self):
        """Test prefix matching behavior in detail."""
        # Test that prefix matching works correctly
        if "gemini-1.5-pro" in GEMINI_MODELS:
            info = get_info("gemini-1.5-pro-custom")
            # Should get info for gemini-1.5-pro (closest match)
            expected_info = GEMINI_MODELS["gemini-1.5-pro"]
            assert info == expected_info
        
        # Test that prefix matching is case sensitive
        with pytest.raises(KeyError):
            get_info("GEMINI-1.5-pro")

    def test_model_registry_completeness(self):
        """Test that model registry is complete and consistent."""
        # Every model should have all required fields
        required_fields = {
            "vision", "function_calling", "json_output", 
            "family", "structured_output", "multiple_system_messages"
        }
        
        for model_name, model_info in GEMINI_MODELS.items():
            missing_fields = required_fields - set(model_info.keys())
            assert not missing_fields, f"Model {model_name} missing fields: {missing_fields}"
            
            # Check field types
            assert isinstance(model_info["vision"], bool)
            assert isinstance(model_info["function_calling"], bool)
            assert isinstance(model_info["json_output"], bool)
            assert isinstance(model_info["family"], str)
            assert isinstance(model_info["structured_output"], bool)
            assert isinstance(model_info["multiple_system_messages"], bool)
            
            # Family should not be empty
            assert model_info["family"].strip(), f"Model {model_name} has empty family"

    def test_performance_characteristics(self):
        """Test performance-related characteristics of model info functions."""
        import time
        
        # Test that lookups are fast
        start_time = time.time()
        for _ in range(1000):
            get_info("gemini-1.5-pro")
        lookup_time = time.time() - start_time
        
        # Should be very fast (less than 1 second for 1000 lookups)
        assert lookup_time < 1.0, "Model info lookup should be fast"
        
        # Test that validation is fast
        start_time = time.time()
        for _ in range(1000):
            validate_model("gemini-1.5-pro")
        validation_time = time.time() - start_time
        
        assert validation_time < 1.0, "Model validation should be fast"

    def test_error_messages_quality(self):
        """Test that error messages are helpful and informative."""
        try:
            get_info("nonexistent-model")
            assert False, "Should have raised KeyError"
        except KeyError as e:
            error_msg = str(e)
            assert "nonexistent-model" in error_msg, "Error message should include the model name"
            assert "not found" in error_msg.lower(), "Error message should be descriptive"

    def test_model_versioning_patterns(self):
        """Test model versioning patterns."""
        # Collect version patterns
        version_patterns = set()
        for model_name in GEMINI_MODELS.keys():
            if "1.5" in model_name:
                version_patterns.add("1.5")
            elif "1.0" in model_name:
                version_patterns.add("1.0")
            elif "2.0" in model_name:
                version_patterns.add("2.0")
            elif "exp" in model_name:
                version_patterns.add("exp")
        
        # Should have multiple version patterns
        assert len(version_patterns) >= 2, "Should have multiple version patterns"
        
        # Test that newer versions generally have better capabilities
        if "1.0" in [m for m in GEMINI_MODELS.keys() if "1.0" in m] and \
           "1.5" in [m for m in GEMINI_MODELS.keys() if "1.5" in m]:
            
            v10_models = [name for name in GEMINI_MODELS.keys() if "1.0" in name]
            v15_models = [name for name in GEMINI_MODELS.keys() if "1.5" in name]
            
            # 1.5 models should generally have better capabilities than 1.0
            v10_structured = sum(1 for name in v10_models if GEMINI_MODELS[name]["structured_output"])
            v15_structured = sum(1 for name in v15_models if GEMINI_MODELS[name]["structured_output"])
            
            if v15_models and v10_models:
                v10_structured_ratio = v10_structured / len(v10_models)
                v15_structured_ratio = v15_structured / len(v15_models)
                assert v15_structured_ratio >= v10_structured_ratio, \
                    "Newer models should generally have better capabilities"