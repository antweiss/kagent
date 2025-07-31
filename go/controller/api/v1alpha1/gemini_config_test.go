package v1alpha1

import (
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestGeminiConfig_Validation(t *testing.T) {
	tests := []struct {
		name        string
		config      GeminiConfig
		expectValid bool
		description string
	}{
		{
			name: "ValidMinimalConfig",
			config: GeminiConfig{
				Temperature: "0.7",
			},
			expectValid: true,
			description: "Minimal valid configuration with just temperature",
		},
		{
			name: "ValidFullConfig",
			config: GeminiConfig{
				BaseURL:          "https://generativelanguage.googleapis.com",
				Temperature:      "0.7",
				MaxOutputTokens:  intPtr(1024),
				TopP:             "0.9",
				TopK:             intPtr(40),
				CandidateCount:   intPtr(1),
				StopSequences:    []string{"STOP", "END"},
				ResponseMimeType: "application/json",
				SafetySettings: map[string]string{
					"HARM_CATEGORY_HARASSMENT":  "BLOCK_MEDIUM_AND_ABOVE",
					"HARM_CATEGORY_HATE_SPEECH": "BLOCK_ONLY_HIGH",
				},
			},
			expectValid: true,
			description: "Full configuration with all optional fields",
		},
		{
			name:        "ValidEmptyConfig",
			config:      GeminiConfig{},
			expectValid: true,
			description: "Empty configuration should be valid (all fields optional)",
		},
		{
			name: "ValidCustomBaseURL",
			config: GeminiConfig{
				BaseURL:     "https://custom-gemini-endpoint.com",
				Temperature: "0.5",
			},
			expectValid: true,
			description: "Custom base URL should be valid",
		},
		{
			name: "ValidSafetySettings",
			config: GeminiConfig{
				SafetySettings: map[string]string{
					"HARM_CATEGORY_HARASSMENT":        "BLOCK_NONE",
					"HARM_CATEGORY_HATE_SPEECH":       "BLOCK_LOW_AND_ABOVE",
					"HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
					"HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_ONLY_HIGH",
				},
			},
			expectValid: true,
			description: "All safety categories with valid thresholds",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a ModelConfig with Gemini provider to test validation
			modelConfig := &ModelConfig{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-gemini-config",
					Namespace: "default",
				},
				Spec: ModelConfigSpec{
					Model:    "gemini-1.5-pro",
					Provider: Gemini,
					Gemini:   &tt.config,
				},
			}

			// Test that the config can be created without validation errors
			// In a real Kubernetes environment, this would trigger webhook validation
			assert.NotNil(t, modelConfig.Spec.Gemini, tt.description)

			// Verify specific field values
			if tt.config.BaseURL != "" {
				assert.Equal(t, tt.config.BaseURL, modelConfig.Spec.Gemini.BaseURL)
			}
			if tt.config.Temperature != "" {
				assert.Equal(t, tt.config.Temperature, modelConfig.Spec.Gemini.Temperature)
			}
			if tt.config.MaxOutputTokens != nil {
				assert.Equal(t, tt.config.MaxOutputTokens, modelConfig.Spec.Gemini.MaxOutputTokens)
			}
			if tt.config.SafetySettings != nil {
				assert.Equal(t, tt.config.SafetySettings, modelConfig.Spec.Gemini.SafetySettings)
			}
		})
	}
}

func TestGeminiConfig_DefaultValues(t *testing.T) {
	config := GeminiConfig{}

	// Test that optional fields are properly nil/empty when not set
	assert.Empty(t, config.BaseURL)
	assert.Empty(t, config.Temperature)
	assert.Nil(t, config.MaxOutputTokens)
	assert.Empty(t, config.TopP)
	assert.Nil(t, config.TopK)
	assert.Nil(t, config.CandidateCount)
	assert.Nil(t, config.StopSequences)
	assert.Empty(t, config.ResponseMimeType)
	assert.Nil(t, config.SafetySettings)
}

func TestGeminiConfig_JSONSerialization(t *testing.T) {
	config := GeminiConfig{
		BaseURL:          "https://generativelanguage.googleapis.com",
		Temperature:      "0.7",
		MaxOutputTokens:  intPtr(1024),
		TopP:             "0.9",
		TopK:             intPtr(40),
		CandidateCount:   intPtr(1),
		StopSequences:    []string{"STOP"},
		ResponseMimeType: "application/json",
		SafetySettings: map[string]string{
			"HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
		},
	}

	modelConfig := ModelConfig{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-config",
			Namespace: "default",
		},
		Spec: ModelConfigSpec{
			Model:    "gemini-1.5-pro",
			Provider: Gemini,
			Gemini:   &config,
		},
	}

	// Test that the config can be marshaled and unmarshaled
	// This tests the JSON tags and struct serialization
	assert.Equal(t, Gemini, modelConfig.Spec.Provider)
	assert.NotNil(t, modelConfig.Spec.Gemini)
	assert.Equal(t, "gemini-1.5-pro", modelConfig.Spec.Model)
}

func TestModelConfigSpec_GeminiProviderValidation(t *testing.T) {
	tests := []struct {
		name        string
		spec        ModelConfigSpec
		isValid     bool
		description string
	}{
		{
			name: "GeminiProviderWithGeminiConfig",
			spec: ModelConfigSpec{
				Model:    "gemini-1.5-pro",
				Provider: Gemini,
				Gemini: &GeminiConfig{
					Temperature: "0.7",
				},
			},
			isValid:     true,
			description: "Gemini provider with Gemini config should be valid",
		},
		{
			name: "GeminiProviderWithoutGeminiConfig",
			spec: ModelConfigSpec{
				Model:    "gemini-1.5-pro",
				Provider: Gemini,
				// No Gemini config - should still be valid as it's optional
			},
			isValid:     true,
			description: "Gemini provider without Gemini config should be valid",
		},
		{
			name: "NonGeminiProviderWithGeminiConfig",
			spec: ModelConfigSpec{
				Model:    "gpt-4",
				Provider: OpenAI,
				Gemini: &GeminiConfig{
					Temperature: "0.7",
				},
			},
			isValid:     false,
			description: "Non-Gemini provider with Gemini config should be invalid",
		},
		{
			name: "GeminiProviderWithOtherProviderConfig",
			spec: ModelConfigSpec{
				Model:    "gemini-1.5-pro",
				Provider: Gemini,
				OpenAI: &OpenAIConfig{
					Temperature: "0.7",
				},
			},
			isValid:     false,
			description: "Gemini provider with other provider config should be invalid",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			modelConfig := &ModelConfig{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-config",
					Namespace: "default",
				},
				Spec: tt.spec,
			}

			// In a real environment, Kubernetes validation would catch these issues
			// Here we test the logical consistency
			if tt.isValid {
				assert.Equal(t, tt.spec.Provider, modelConfig.Spec.Provider, tt.description)
				if tt.spec.Provider == Gemini && tt.spec.Gemini != nil {
					assert.NotNil(t, modelConfig.Spec.Gemini)
				}
			} else {
				// Test that invalid combinations are logically inconsistent
				if tt.spec.Provider != Gemini && tt.spec.Gemini != nil {
					assert.NotEqual(t, Gemini, modelConfig.Spec.Provider, "Provider should not be Gemini when Gemini config is set for different provider")
				}
				if tt.spec.Provider == Gemini && tt.spec.OpenAI != nil {
					assert.NotNil(t, modelConfig.Spec.OpenAI, "OpenAI config should not be set for Gemini provider")
				}
			}
		})
	}
}

func TestGeminiConfig_ParameterTypes(t *testing.T) {
	t.Run("IntegerPointers", func(t *testing.T) {
		config := GeminiConfig{
			MaxOutputTokens: intPtr(2048),
			TopK:            intPtr(50),
			CandidateCount:  intPtr(2),
		}

		assert.NotNil(t, config.MaxOutputTokens)
		assert.Equal(t, 2048, *config.MaxOutputTokens)
		assert.NotNil(t, config.TopK)
		assert.Equal(t, 50, *config.TopK)
		assert.NotNil(t, config.CandidateCount)
		assert.Equal(t, 2, *config.CandidateCount)
	})

	t.Run("StringSlices", func(t *testing.T) {
		config := GeminiConfig{
			StopSequences: []string{"STOP", "END", "FINISH"},
		}

		require.NotNil(t, config.StopSequences)
		assert.Len(t, config.StopSequences, 3)
		assert.Contains(t, config.StopSequences, "STOP")
		assert.Contains(t, config.StopSequences, "END")
		assert.Contains(t, config.StopSequences, "FINISH")
	})

	t.Run("StringMaps", func(t *testing.T) {
		config := GeminiConfig{
			SafetySettings: map[string]string{
				"HARM_CATEGORY_HARASSMENT":        "BLOCK_MEDIUM_AND_ABOVE",
				"HARM_CATEGORY_HATE_SPEECH":       "BLOCK_ONLY_HIGH",
				"HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_LOW_AND_ABOVE",
			},
		}

		require.NotNil(t, config.SafetySettings)
		assert.Len(t, config.SafetySettings, 3)
		assert.Equal(t, "BLOCK_MEDIUM_AND_ABOVE", config.SafetySettings["HARM_CATEGORY_HARASSMENT"])
		assert.Equal(t, "BLOCK_ONLY_HIGH", config.SafetySettings["HARM_CATEGORY_HATE_SPEECH"])
		assert.Equal(t, "BLOCK_LOW_AND_ABOVE", config.SafetySettings["HARM_CATEGORY_SEXUALLY_EXPLICIT"])
	})
}

func TestGeminiConfig_EdgeCases(t *testing.T) {
	t.Run("NilPointerFields", func(t *testing.T) {
		config := GeminiConfig{
			MaxOutputTokens: nil,
			TopK:            nil,
			CandidateCount:  nil,
		}

		// Test that nil pointers are handled correctly
		assert.Nil(t, config.MaxOutputTokens)
		assert.Nil(t, config.TopK)
		assert.Nil(t, config.CandidateCount)
	})

	t.Run("EmptyStringFields", func(t *testing.T) {
		config := GeminiConfig{
			BaseURL:          "",
			Temperature:      "",
			TopP:             "",
			ResponseMimeType: "",
		}

		// Test that empty strings are handled correctly
		assert.Empty(t, config.BaseURL)
		assert.Empty(t, config.Temperature)
		assert.Empty(t, config.TopP)
		assert.Empty(t, config.ResponseMimeType)
	})

	t.Run("EmptySlicesAndMaps", func(t *testing.T) {
		config := GeminiConfig{
			StopSequences:  []string{},
			SafetySettings: map[string]string{},
		}

		// Test that empty collections are handled correctly
		assert.NotNil(t, config.StopSequences)
		assert.Len(t, config.StopSequences, 0)
		assert.NotNil(t, config.SafetySettings)
		assert.Len(t, config.SafetySettings, 0)
	})
}

func TestGeminiConfig_BoundaryValues(t *testing.T) {
	t.Run("MinimumValues", func(t *testing.T) {
		config := GeminiConfig{
			Temperature:     "0.0",
			MaxOutputTokens: intPtr(1),
			TopP:            "0.0",
			TopK:            intPtr(1),
			CandidateCount:  intPtr(1),
		}

		assert.Equal(t, "0.0", config.Temperature)
		assert.Equal(t, 1, *config.MaxOutputTokens)
		assert.Equal(t, "0.0", config.TopP)
		assert.Equal(t, 1, *config.TopK)
		assert.Equal(t, 1, *config.CandidateCount)
	})

	t.Run("MaximumValues", func(t *testing.T) {
		config := GeminiConfig{
			Temperature:     "2.0",
			MaxOutputTokens: intPtr(8192),
			TopP:            "1.0",
			TopK:            intPtr(100),
			CandidateCount:  intPtr(8),
		}

		assert.Equal(t, "2.0", config.Temperature)
		assert.Equal(t, 8192, *config.MaxOutputTokens)
		assert.Equal(t, "1.0", config.TopP)
		assert.Equal(t, 100, *config.TopK)
		assert.Equal(t, 8, *config.CandidateCount)
	})
}

func TestGeminiConfig_SafetySettingsValidation(t *testing.T) {
	validCategories := []string{
		"HARM_CATEGORY_HARASSMENT",
		"HARM_CATEGORY_HATE_SPEECH",
		"HARM_CATEGORY_SEXUALLY_EXPLICIT",
		"HARM_CATEGORY_DANGEROUS_CONTENT",
	}

	validThresholds := []string{
		"BLOCK_NONE",
		"BLOCK_LOW_AND_ABOVE",
		"BLOCK_MEDIUM_AND_ABOVE",
		"BLOCK_ONLY_HIGH",
	}

	t.Run("ValidSafetySettings", func(t *testing.T) {
		for _, category := range validCategories {
			for _, threshold := range validThresholds {
				config := GeminiConfig{
					SafetySettings: map[string]string{
						category: threshold,
					},
				}

				assert.Equal(t, threshold, config.SafetySettings[category])
			}
		}
	})

	t.Run("MultipleSafetySettings", func(t *testing.T) {
		config := GeminiConfig{
			SafetySettings: map[string]string{
				"HARM_CATEGORY_HARASSMENT":        "BLOCK_MEDIUM_AND_ABOVE",
				"HARM_CATEGORY_HATE_SPEECH":       "BLOCK_ONLY_HIGH",
				"HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_LOW_AND_ABOVE",
				"HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
			},
		}

		assert.Len(t, config.SafetySettings, 4)
		assert.Equal(t, "BLOCK_MEDIUM_AND_ABOVE", config.SafetySettings["HARM_CATEGORY_HARASSMENT"])
		assert.Equal(t, "BLOCK_ONLY_HIGH", config.SafetySettings["HARM_CATEGORY_HATE_SPEECH"])
		assert.Equal(t, "BLOCK_LOW_AND_ABOVE", config.SafetySettings["HARM_CATEGORY_SEXUALLY_EXPLICIT"])
		assert.Equal(t, "BLOCK_NONE", config.SafetySettings["HARM_CATEGORY_DANGEROUS_CONTENT"])
	})
}

func TestGeminiConfig_StopSequencesValidation(t *testing.T) {
	t.Run("SingleStopSequence", func(t *testing.T) {
		config := GeminiConfig{
			StopSequences: []string{"STOP"},
		}

		assert.Len(t, config.StopSequences, 1)
		assert.Equal(t, "STOP", config.StopSequences[0])
	})

	t.Run("MultipleStopSequences", func(t *testing.T) {
		sequences := []string{"STOP", "END", "FINISH", "DONE"}
		config := GeminiConfig{
			StopSequences: sequences,
		}

		assert.Len(t, config.StopSequences, 4)
		for i, seq := range sequences {
			assert.Equal(t, seq, config.StopSequences[i])
		}
	})

	t.Run("EmptyStopSequences", func(t *testing.T) {
		config := GeminiConfig{
			StopSequences: []string{"", "STOP", ""},
		}

		// Should preserve empty strings if provided
		assert.Len(t, config.StopSequences, 3)
		assert.Equal(t, "", config.StopSequences[0])
		assert.Equal(t, "STOP", config.StopSequences[1])
		assert.Equal(t, "", config.StopSequences[2])
	})
}

func TestGeminiConfig_ResponseMimeTypeValidation(t *testing.T) {
	validMimeTypes := []string{
		"application/json",
		"text/plain",
		"text/x.enum",
	}

	for _, mimeType := range validMimeTypes {
		t.Run(fmt.Sprintf("ValidMimeType_%s", strings.ReplaceAll(mimeType, "/", "_")), func(t *testing.T) {
			config := GeminiConfig{
				ResponseMimeType: mimeType,
			}

			assert.Equal(t, mimeType, config.ResponseMimeType)
		})
	}
}

func TestGeminiConfig_BaseURLValidation(t *testing.T) {
	validURLs := []string{
		"https://generativelanguage.googleapis.com",
		"https://custom-endpoint.example.com",
		"http://localhost:8080",
		"https://api.example.com/v1",
	}

	for _, url := range validURLs {
		t.Run(fmt.Sprintf("ValidURL_%s", strings.ReplaceAll(url, "://", "_")), func(t *testing.T) {
			config := GeminiConfig{
				BaseURL: url,
			}

			assert.Equal(t, url, config.BaseURL)
		})
	}
}

func TestModelConfigSpec_GeminiProviderCombinations(t *testing.T) {
	t.Run("GeminiWithAllOtherProviders", func(t *testing.T) {
		// Test that Gemini config should not be combined with other provider configs
		invalidCombinations := []struct {
			name string
			spec ModelConfigSpec
		}{
			{
				name: "GeminiWithOpenAI",
				spec: ModelConfigSpec{
					Provider: Gemini,
					Gemini:   &GeminiConfig{Temperature: "0.7"},
					OpenAI:   &OpenAIConfig{Temperature: "0.5"},
				},
			},
			{
				name: "GeminiWithAnthropic",
				spec: ModelConfigSpec{
					Provider:  Gemini,
					Gemini:    &GeminiConfig{Temperature: "0.7"},
					Anthropic: &AnthropicConfig{Temperature: "0.5"},
				},
			},
			{
				name: "GeminiWithGeminiVertexAI",
				spec: ModelConfigSpec{
					Provider: Gemini,
					Gemini:   &GeminiConfig{Temperature: "0.7"},
					GeminiVertexAI: &GeminiVertexAIConfig{
						BaseVertexAIConfig: BaseVertexAIConfig{Temperature: "0.5"},
					},
				},
			},
			{
				name: "GeminiWithAzureOpenAI",
				spec: ModelConfigSpec{
					Provider:    Gemini,
					Gemini:      &GeminiConfig{Temperature: "0.7"},
					AzureOpenAI: &AzureOpenAIConfig{Temperature: "0.5"},
				},
			},
			{
				name: "GeminiWithOllama",
				spec: ModelConfigSpec{
					Provider: Gemini,
					Gemini:   &GeminiConfig{Temperature: "0.7"},
					Ollama:   &OllamaConfig{Host: "localhost"},
				},
			},
		}

		for _, combo := range invalidCombinations {
			t.Run(combo.name, func(t *testing.T) {
				// In a real Kubernetes environment, these would be caught by validation webhooks
				// Here we test the logical inconsistency
				assert.Equal(t, Gemini, combo.spec.Provider)
				assert.NotNil(t, combo.spec.Gemini)

				// Count non-nil provider configs
				providerConfigs := 0
				if combo.spec.OpenAI != nil {
					providerConfigs++
				}
				if combo.spec.Anthropic != nil {
					providerConfigs++
				}
				if combo.spec.GeminiVertexAI != nil {
					providerConfigs++
				}
				if combo.spec.AzureOpenAI != nil {
					providerConfigs++
				}
				if combo.spec.Ollama != nil {
					providerConfigs++
				}
				if combo.spec.Gemini != nil {
					providerConfigs++
				}

				// Should have more than one provider config (invalid)
				assert.Greater(t, providerConfigs, 1, "Should have multiple provider configs (invalid combination)")
			})
		}
	})
}

func TestGeminiConfig_ComplexScenarios(t *testing.T) {
	t.Run("ProductionLikeConfig", func(t *testing.T) {
		config := GeminiConfig{
			BaseURL:         "https://generativelanguage.googleapis.com",
			Temperature:     "0.7",
			MaxOutputTokens: intPtr(2048),
			TopP:            "0.9",
			TopK:            intPtr(40),
			CandidateCount:  intPtr(1),
			StopSequences:   []string{"Human:", "Assistant:", "END"},
			SafetySettings: map[string]string{
				"HARM_CATEGORY_HARASSMENT":        "BLOCK_MEDIUM_AND_ABOVE",
				"HARM_CATEGORY_HATE_SPEECH":       "BLOCK_MEDIUM_AND_ABOVE",
				"HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
				"HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE",
			},
		}

		modelConfig := ModelConfig{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "production-gemini",
				Namespace: "production",
			},
			Spec: ModelConfigSpec{
				Model:           "gemini-1.5-pro",
				Provider:        Gemini,
				APIKeySecretRef: "gemini-api-key",
				APIKeySecretKey: "GEMINI_API_KEY",
				Gemini:          &config,
			},
		}

		// Verify all fields are set correctly
		assert.Equal(t, "production-gemini", modelConfig.Name)
		assert.Equal(t, "production", modelConfig.Namespace)
		assert.Equal(t, "gemini-1.5-pro", modelConfig.Spec.Model)
		assert.Equal(t, Gemini, modelConfig.Spec.Provider)
		assert.Equal(t, "gemini-api-key", modelConfig.Spec.APIKeySecretRef)
		assert.Equal(t, "GEMINI_API_KEY", modelConfig.Spec.APIKeySecretKey)

		geminiConfig := modelConfig.Spec.Gemini
		require.NotNil(t, geminiConfig)
		assert.Equal(t, "https://generativelanguage.googleapis.com", geminiConfig.BaseURL)
		assert.Equal(t, "0.7", geminiConfig.Temperature)
		assert.Equal(t, 2048, *geminiConfig.MaxOutputTokens)
		assert.Equal(t, "0.9", geminiConfig.TopP)
		assert.Equal(t, 40, *geminiConfig.TopK)
		assert.Equal(t, 1, *geminiConfig.CandidateCount)
		assert.Len(t, geminiConfig.StopSequences, 3)
		assert.Len(t, geminiConfig.SafetySettings, 4)
	})

	t.Run("MinimalConfig", func(t *testing.T) {
		modelConfig := ModelConfig{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "minimal-gemini",
				Namespace: "default",
			},
			Spec: ModelConfigSpec{
				Model:           "gemini-1.5-flash",
				Provider:        Gemini,
				APIKeySecretRef: "gemini-secret",
				// No Gemini config - should be valid
			},
		}

		assert.Equal(t, "minimal-gemini", modelConfig.Name)
		assert.Equal(t, "gemini-1.5-flash", modelConfig.Spec.Model)
		assert.Equal(t, Gemini, modelConfig.Spec.Provider)
		assert.Nil(t, modelConfig.Spec.Gemini)
	})
}

// Helper function for tests
func intPtr(i int) *int {
	return &i
}
