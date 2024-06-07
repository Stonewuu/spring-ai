/*
 * Copyright 2023 - 2024 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.springframework.ai.googleai.gemini;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonInclude.Include;
import com.fasterxml.jackson.annotation.JsonProperty;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.googleai.gemini.api.GeminiAiApi;
import org.springframework.ai.model.function.FunctionCallback;
import org.springframework.ai.model.function.FunctionCallingOptions;
import org.springframework.boot.context.properties.NestedConfigurationProperty;
import org.springframework.util.Assert;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * @author Christian Tzolov
 * @since 0.8.1
 */
@JsonInclude(Include.NON_NULL)
public class GoogleAiGeminiChatOptions implements FunctionCallingOptions, ChatOptions {

	private @JsonProperty("safetySettings") List<GeminiAiApi.ChatCompletionRequest.SafetySetting> safetySettings;

	private @JsonProperty("generationConfig") GeminiAiApi.ChatCompletionRequest.GenerationConfig generationConfig;

	/**
	 * Gemini model name.
	 */
	private @JsonProperty("model") String model;

	// https://cloud.google.com/vertex-ai/docs/reference/rest/v1/GenerationConfig
	/**
	 * Tool Function Callbacks to register with the ChatModel. For Prompt Options the
	 * functionCallbacks are automatically enabled for the duration of the prompt
	 * execution. For Default Options the functionCallbacks are registered but disabled by
	 * default. Use the enableFunctions to set the functions from the registry to be used
	 * by the ChatModel chat completion requests.
	 */
	@NestedConfigurationProperty
	@JsonIgnore
	private List<FunctionCallback> functionCallbacks = new ArrayList<>();

	/**
	 * List of functions, identified by their names, to configure for function calling in
	 * the chat completion requests. Functions with those names must exist in the
	 * functionCallbacks registry. The {@link #functionCallbacks} from the PromptOptions
	 * are automatically enabled for the duration of the prompt execution.
	 *
	 * <p>
	 * Note that function enabled with the default options are enabled for all chat
	 * completion requests. This could impact the token count and the billing. If the
	 * functions is set in a prompt options, then the enabled functions are only active
	 * for the duration of this prompt execution.
	 */
	@NestedConfigurationProperty
	@JsonIgnore
	private Set<String> functions = new HashSet<>();

	public static Builder builder() {
		return new Builder();
	}

	public static GoogleAiGeminiChatOptions fromOptions(GoogleAiGeminiChatOptions fromOptions) {
		GoogleAiGeminiChatOptions options = new GoogleAiGeminiChatOptions();
		options.setSafetySettings(fromOptions.getSafetySettings());
		options.setGenerationConfig(fromOptions.getGenerationConfig());
		options.setModel(fromOptions.getModel());
		options.setFunctionCallbacks(fromOptions.getFunctionCallbacks());
		options.setFunctions(fromOptions.getFunctions());
		return options;
	}

	@Override
	public Float getTemperature() {
		return generationConfig == null ? 0 : generationConfig.temperature();
	}

	@Override
	public Float getTopP() {
		return generationConfig == null ? 0 : generationConfig.topP();
	}

	// @formatter:on

	@Override
	public Integer getTopK() {
		return generationConfig == null ? 0 : generationConfig.topK();
	}

	public List<GeminiAiApi.ChatCompletionRequest.SafetySetting> getSafetySettings() {
		return safetySettings;
	}

	public void setSafetySettings(List<GeminiAiApi.ChatCompletionRequest.SafetySetting> safetySettings) {
		this.safetySettings = safetySettings;
	}

	public GeminiAiApi.ChatCompletionRequest.GenerationConfig getGenerationConfig() {
		return generationConfig;
	}

	public void setGenerationConfig(GeminiAiApi.ChatCompletionRequest.GenerationConfig generationConfig) {
		this.generationConfig = generationConfig;
	}

	public String getModel() {
		return model;
	}

	public void setModel(String model) {
		this.model = model;
	}

	@Override
	public List<FunctionCallback> getFunctionCallbacks() {
		return functionCallbacks;
	}

	@Override
	public void setFunctionCallbacks(List<FunctionCallback> functionCallbacks) {
		this.functionCallbacks = functionCallbacks;
	}

	@Override
	public Set<String> getFunctions() {
		return functions;
	}

	@Override
	public void setFunctions(Set<String> functions) {
		this.functions = functions;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((safetySettings == null) ? 0 : safetySettings.hashCode());
		result = prime * result + ((generationConfig == null) ? 0 : generationConfig.hashCode());
		result = prime * result + ((model == null) ? 0 : model.hashCode());
		result = prime * result + ((functionCallbacks == null) ? 0 : functionCallbacks.hashCode());
		result = prime * result + ((functions == null) ? 0 : functions.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		GoogleAiGeminiChatOptions other = (GoogleAiGeminiChatOptions) obj;
		if (safetySettings == null) {
			if (other.safetySettings != null)
				return false;
		}
		else if (!safetySettings.equals(other.safetySettings))
			return false;
		if (generationConfig == null) {
			if (other.generationConfig != null)
				return false;
		}
		else if (!generationConfig.equals(other.generationConfig))
			return false;
		if (model == null) {
			if (other.model != null)
				return false;
		}
		else if (!model.equals(other.model))
			return false;
		if (functionCallbacks == null) {
			if (other.functionCallbacks != null)
				return false;
		}
		else if (!functionCallbacks.equals(other.functionCallbacks))
			return false;
		if (functions == null) {
			if (other.functions != null)
				return false;
		}
		else if (!functions.equals(other.functions))
			return false;
		return true;
	}

	@Override
	public String toString() {
		return "VertexAiGeminiChatOptions [safetySettings=" + safetySettings + ", generationConfig=" + generationConfig
				+ ", model=" + model + ", functionCallbacks=" + functionCallbacks + ", functions=" + functions
				+ ", getClass()=" + getClass() + ", getModel()=" + getModel() + ", getFunctionCallbacks()="
				+ getFunctionCallbacks() + ", getFunctions()=" + getFunctions() + ", hashCode()=" + hashCode()
				+ ", toString()=" + super.toString() + "]";
	}

	public enum TransportType {

		GRPC, REST

	}

	public static class Builder {

		private GoogleAiGeminiChatOptions options = new GoogleAiGeminiChatOptions();

		public Builder withSafetySettings(List<GeminiAiApi.ChatCompletionRequest.SafetySetting> safetySettings) {
			this.options.safetySettings = safetySettings;
			return this;
		}

		public Builder withGenerationConfig(GeminiAiApi.ChatCompletionRequest.GenerationConfig generationConfig) {
			this.options.generationConfig = generationConfig;
			return this;
		}

		public Builder withModel(String modelName) {
			this.options.setModel(modelName);
			return this;
		}

		public Builder withModel(GeminiAiApi.ChatModel model) {
			this.options.setModel(model.getValue());
			return this;
		}

		public Builder withFunctionCallbacks(List<FunctionCallback> functionCallbacks) {
			this.options.functionCallbacks = functionCallbacks;
			return this;
		}

		public Builder withFunctions(Set<String> functionNames) {
			Assert.notNull(functionNames, "Function names must not be null");
			this.options.functions = functionNames;
			return this;
		}

		public Builder withFunction(String functionName) {
			Assert.hasText(functionName, "Function name must not be empty");
			this.options.functions.add(functionName);
			return this;
		}

		public GoogleAiGeminiChatOptions build() {
			return this.options;
		}

	}

}
