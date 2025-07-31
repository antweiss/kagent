# validation/validation_service.py
import importlib
from calendar import c
from typing import Any, Dict, List, Optional

from autogen_core import ComponentModel, is_component_class
from pydantic import BaseModel


class ValidationRequest(BaseModel):
    component: ComponentModel


class ValidationError(BaseModel):
    field: str
    error: str
    suggestion: Optional[str] = None


class ValidationResponse(BaseModel):
    is_valid: bool
    errors: List[ValidationError] = []
    warnings: List[ValidationError] = []


class ValidationService:
    @staticmethod
    def validate_provider(provider: str) -> Optional[ValidationError]:
        """Validate that the provider exists and can be imported"""
        try:
            if provider in ["azure_openai_chat_completion_client", "AzureOpenAIChatCompletionClient"]:
                provider = "autogen_ext.models.openai.AzureOpenAIChatCompletionClient"
            elif provider in ["openai_chat_completion_client", "OpenAIChatCompletionClient"]:
                provider = "autogen_ext.models.openai.OpenAIChatCompletionClient"
            elif provider in ["gemini_chat_completion_client", "GeminiChatCompletionClient"]:
                provider = "kagent.models.gemini.GeminiChatCompletionClient"
            elif provider in ["gemini_vertexai_chat_completion_client", "GeminiVertexAIChatCompletionClient"]:
                provider = "kagent.models.vertexai.GeminiVertexAIChatCompletionClient"

            module_path, class_name = provider.rsplit(".", maxsplit=1)
            module = importlib.import_module(module_path)
            component_class = getattr(module, class_name)

            if not is_component_class(component_class):
                return ValidationError(
                    field="provider",
                    error=f"Class {provider} is not a valid component class",
                    suggestion="Ensure the class inherits from Component and implements required methods",
                )
            return None
        except ImportError:
            return ValidationError(
                field="provider",
                error=f"Could not import provider {provider}",
                suggestion="Check that the provider module is installed and the path is correct",
            )
        except Exception as e:
            return ValidationError(
                field="provider",
                error=f"Error validating provider: {str(e)}",
                suggestion="Check the provider string format and class implementation",
            )

    @staticmethod
    def validate_component_type(component: ComponentModel) -> Optional[ValidationError]:
        """Validate the component type"""
        if not component.component_type:
            return ValidationError(
                field="component_type",
                error="Component type is missing",
                suggestion="Add a component_type field to the component configuration",
            )

    @staticmethod
    def validate_gemini_config(component: ComponentModel) -> List[ValidationError]:
        """Validate Gemini-specific configuration parameters"""
        errors = []
        
        if component.provider not in ["kagent.models.gemini.GeminiChatCompletionClient", 
                                     "gemini_chat_completion_client", "GeminiChatCompletionClient"]:
            return errors
        
        config = component.config or {}
        
        # Validate API key is present
        if not config.get("api_key"):
            errors.append(ValidationError(
                field="api_key",
                error="Gemini API key is required",
                suggestion="Provide a valid Google AI API key for Gemini authentication"
            ))
        
        # Validate model name
        model = config.get("model")
        if model:
            try:
                from kagent.models.gemini._model_info import validate_model
                if not validate_model(model):
                    errors.append(ValidationError(
                        field="model",
                        error=f"Unsupported Gemini model: {model}",
                        suggestion="Use a supported Gemini model like 'gemini-1.5-pro' or 'gemini-1.5-flash'"
                    ))
            except ImportError:
                # If model info module is not available, skip validation
                pass
        
        # Validate temperature range
        temperature = config.get("temperature")
        if temperature is not None:
            try:
                temp_val = float(temperature)
                if not (0.0 <= temp_val <= 2.0):
                    errors.append(ValidationError(
                        field="temperature",
                        error="Temperature must be between 0.0 and 2.0",
                        suggestion="Set temperature to a value between 0.0 (deterministic) and 2.0 (creative)"
                    ))
            except (ValueError, TypeError):
                errors.append(ValidationError(
                    field="temperature",
                    error="Temperature must be a valid number",
                    suggestion="Provide temperature as a number between 0.0 and 2.0"
                ))
        
        # Validate top_p range
        top_p = config.get("top_p")
        if top_p is not None:
            try:
                top_p_val = float(top_p)
                if not (0.0 <= top_p_val <= 1.0):
                    errors.append(ValidationError(
                        field="top_p",
                        error="top_p must be between 0.0 and 1.0",
                        suggestion="Set top_p to a value between 0.0 and 1.0"
                    ))
            except (ValueError, TypeError):
                errors.append(ValidationError(
                    field="top_p",
                    error="top_p must be a valid number",
                    suggestion="Provide top_p as a number between 0.0 and 1.0"
                ))
        
        # Validate safety settings
        safety_settings = config.get("safety_settings")
        if safety_settings:
            valid_categories = {
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH", 
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT"
            }
            valid_thresholds = {
                "BLOCK_NONE",
                "BLOCK_LOW_AND_ABOVE",
                "BLOCK_MEDIUM_AND_ABOVE", 
                "BLOCK_ONLY_HIGH"
            }
            
            if isinstance(safety_settings, dict):
                for category, threshold in safety_settings.items():
                    if category not in valid_categories:
                        errors.append(ValidationError(
                            field="safety_settings",
                            error=f"Invalid safety category: {category}",
                            suggestion=f"Use one of: {', '.join(valid_categories)}"
                        ))
                    if threshold not in valid_thresholds:
                        errors.append(ValidationError(
                            field="safety_settings",
                            error=f"Invalid safety threshold: {threshold}",
                            suggestion=f"Use one of: {', '.join(valid_thresholds)}"
                        ))
        
        return errors

    @staticmethod
    def validate_config_schema(component: ComponentModel) -> List[ValidationError]:
        """Validate the component configuration against its schema"""
        errors = []
        try:
            # Convert to ComponentModel for initial validation
            model = component.model_copy(deep=True)

            # Get the component class
            provider = model.provider
            module_path, class_name = provider.rsplit(".", maxsplit=1)
            module = importlib.import_module(module_path)
            component_class = getattr(module, class_name)

            # Validate against component's schema
            if hasattr(component_class, "component_config_schema"):
                try:
                    component_class.component_config_schema.model_validate(model.config)
                except Exception as e:
                    errors.append(
                        ValidationError(
                            field="config",
                            error=f"Config validation failed: {str(e)}",
                            suggestion="Check that the config matches the component's schema",
                        )
                    )
            else:
                errors.append(
                    ValidationError(
                        field="config",
                        error="Component class missing config schema",
                        suggestion="Implement component_config_schema in the component class",
                    )
                )
        except Exception as e:
            errors.append(
                ValidationError(
                    field="config",
                    error=f"Schema validation error: {str(e)}",
                    suggestion="Check the component configuration format",
                )
            )
        return errors

    @staticmethod
    def validate_instantiation(component: ComponentModel) -> Optional[ValidationError]:
        """Validate that the component can be instantiated"""
        try:
            model = component.model_copy(deep=True)
            # Attempt to load the component
            module_path, class_name = model.provider.rsplit(".", maxsplit=1)
            module = importlib.import_module(module_path)
            component_class = getattr(module, class_name)
            component_class.load_component(model)
            return None
        except Exception as e:
            return ValidationError(
                field="instantiation",
                error=f"Failed to instantiate component: {str(e)}",
                suggestion="Check that the component can be properly instantiated with the given config",
            )

    @classmethod
    def validate(cls, component: ComponentModel) -> ValidationResponse:
        """Validate a component configuration"""
        errors = []
        warnings = []

        # Check provider
        if provider_error := cls.validate_provider(component.provider):
            errors.append(provider_error)

        # Check component type
        if type_error := cls.validate_component_type(component):
            errors.append(type_error)

        # Validate schema
        schema_errors = cls.validate_config_schema(component)
        errors.extend(schema_errors)
        
        # Validate Gemini-specific configuration
        gemini_errors = cls.validate_gemini_config(component)
        errors.extend(gemini_errors)

        # Only attempt instantiation if no errors so far
        if not errors:
            if inst_error := cls.validate_instantiation(component):
                errors.append(inst_error)

        # Check for version warnings
        if not component.version:
            warnings.append(
                ValidationError(
                    field="version",
                    error="Component version not specified",
                    suggestion="Consider adding a version to ensure compatibility",
                )
            )

        return ValidationResponse(is_valid=len(errors) == 0, errors=errors, warnings=warnings)
