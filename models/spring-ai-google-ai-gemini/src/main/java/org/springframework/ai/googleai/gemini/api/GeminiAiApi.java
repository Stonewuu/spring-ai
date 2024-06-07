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
package org.springframework.ai.googleai.gemini.api;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonInclude.Include;
import com.fasterxml.jackson.annotation.JsonProperty;
import org.springframework.ai.model.ModelDescription;
import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.ai.retry.RetryUtils;
import org.springframework.boot.context.properties.bind.ConstructorBinding;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.ResponseEntity;
import org.springframework.util.Assert;
import org.springframework.util.CollectionUtils;
import org.springframework.web.client.ResponseErrorHandler;
import org.springframework.web.client.RestClient;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.util.List;
import java.util.Map;

/**
 * @author Stone Wu
 */
public class GeminiAiApi {

	public static final String DEFAULT_CHAT_MODEL = ChatModel.GEMINI_PRO.getValue();

	public static final String DEFAULT_EMBEDDING_MODEL = EmbeddingModel.TEXT_EMBEDDING_004.getValue();

	private final RestClient restClient;

	private final WebClient webClient;

	/**
	 * Create an new chat completion api with base URL set to
	 * https://generativelanguage.googleapis.com
	 * @param geminiAiToken GeminiAI apiKey.
	 */
	public GeminiAiApi(String geminiAiToken) {
		this(ApiUtils.DEFAULT_BASE_URL, geminiAiToken);
	}

	/**
	 * Create a new chat completion api.
	 * @param baseUrl api base URL.
	 * @param geminiAiToken GeminiAI apiKey.
	 */
	public GeminiAiApi(String baseUrl, String geminiAiToken) {
		this(baseUrl, geminiAiToken, RestClient.builder(), WebClient.builder());
	}

	/**
	 * Create a new chat completion api.
	 * @param baseUrl api base URL.
	 * @param geminiAiToken GeminiAI apiKey.
	 * @param restClientBuilder RestClient builder.
	 */
	public GeminiAiApi(String baseUrl, String geminiAiToken, RestClient.Builder restClientBuilder,
			WebClient.Builder webClientBuilder) {
		this(baseUrl, geminiAiToken, restClientBuilder, webClientBuilder, RetryUtils.DEFAULT_RESPONSE_ERROR_HANDLER);
	}

	/**
	 * Create a new chat completion api.
	 * @param baseUrl api base URL.
	 * @param geminiAiToken GeminiAI apiKey.
	 * @param restClientBuilder RestClient builder.
	 * @param responseErrorHandler Response error handler.
	 */
	public GeminiAiApi(String baseUrl, String geminiAiToken, RestClient.Builder restClientBuilder,
			WebClient.Builder webClientBuilder, ResponseErrorHandler responseErrorHandler) {

		this.restClient = restClientBuilder.baseUrl(baseUrl)
			.defaultHeaders(ApiUtils.getJsonContentHeaders(geminiAiToken))
			.defaultStatusHandler(responseErrorHandler)
			.build();

		this.webClient = webClientBuilder.baseUrl(baseUrl)
			.defaultHeaders(ApiUtils.getJsonContentHeaders(geminiAiToken))
			.build();
	}

	/**
	 * Creates a model response for the given chat conversation.
	 * @param chatRequest The chat completion request.
	 * @return Entity response with {@link ChatCompletion} as a body and HTTP status code
	 * and headers.
	 */
	public ResponseEntity<ChatCompletion> chatCompletionEntity(ChatCompletionRequest chatRequest) {

		Assert.notNull(chatRequest, "The request body can not be null.");

		return this.restClient.post()
			.uri("/v1/models/" + chatRequest.model() + ":generateContent")
			.body(chatRequest)
			.retrieve()
			.toEntity(ChatCompletion.class);
	}

	/**
	 * Creates a streaming chat response for the given chat conversation.
	 * @param chatRequest The chat completion request. Must have the stream property set
	 * to true.
	 * @return Returns a {@link Flux} stream from chat completion chunks.
	 */
	public Flux<ChatCompletion> chatCompletionStream(ChatCompletionRequest chatRequest) {

		Assert.notNull(chatRequest, "The request body can not be null.");

		return this.webClient.post()
			.uri("/v1/models/" + chatRequest.model + ":streamGenerateContent")
			.body(Mono.just(chatRequest), ChatCompletionRequest.class)
			.retrieve()
			.bodyToFlux(ChatCompletion.class);
	}

	/**
	 * Creates an embedding vector representing the input text or token array.
	 * @param embeddingRequest The embedding request.
	 * @return Returns list of {@link Embedding} wrapped in {@link EmbeddingList}.
	 * @param <T> Type of the entity in the data list. Can be a {@link String} or
	 * {@link List} of tokens (e.g. Integers). For embedding multiple inputs in a single
	 * request, You can pass a {@link List} of {@link String} or {@link List} of
	 * {@link List} of tokens. For example: <pre>
	 *     {@code List.of("text1", "text2", "text3") or List.of(List.of(1, 2, 3), List.of(3, 4, 5))}
	 *     </pre>
	 */
	public <T> ResponseEntity<EmbeddingList<Embedding>> embeddings(EmbeddingRequest embeddingRequest) {

		Assert.notNull(embeddingRequest, "The request body can not be null.");

		// Input text to embed, encoded as a string or array of tokens. To embed multiple
		// inputs in a
		// single
		// request, pass an array of strings or array of token arrays.
		Assert.notNull(embeddingRequest.input(), "The input can not be null.");

		// The input must not exceed the max input tokens for the model (8192 tokens for
		// text-embedding-ada-002), cannot
		// be an empty string, and any array must be 2048 title or less.
		Assert.isTrue(!CollectionUtils.isEmpty(embeddingRequest.input().parts), "The input list can not be empty.");

		return this.restClient.post()
			.uri("/v1/models/" + embeddingRequest.model + ":embedContent")
			.body(embeddingRequest)
			.retrieve()
			.toEntity(new ParameterizedTypeReference<>() {
			});
	}

	public enum ChatModel implements ModelDescription {

		GEMINI_PRO_VISION("gemini-pro-vision"),

		GEMINI_PRO("gemini-pro"),

		GEMINI_PRO_1_5_PRO("gemini-1.5-pro"),

		GEMINI_PRO_1_5_FLASH("gemini-1.5-flash-preview-0514");

		public final String value;

		ChatModel(String value) {
			this.value = value;
		}

		public String getValue() {
			return this.value;
		}

		@Override
		public String getModelName() {
			return this.value;
		}

	}

	public enum EmbeddingModel {

		/**
		 * Most capable embedding model for both english and non-english tasks. DIMENSION:
		 * 3072
		 */
		TEXT_EMBEDDING_004("text-embedding-004"),

		/**
		 * Increased performance over 2nd generation ada embedding model. DIMENSION: 1536
		 */
		EMBEDDING_001("embedding-001");

		public final String value;

		EmbeddingModel(String value) {
			this.value = value;
		}

		public String getValue() {
			return value;
		}

	}

	/**
	 * Represents a tool the model may call. Currently, only functions are supported as a
	 * tool.
	 *
	 * @param type The type of the tool. Currently, only 'function' is supported.
	 * @param function The function definition.
	 */
	@JsonInclude(Include.NON_NULL)
	public record FunctionTool(@JsonProperty("type") Type type, @JsonProperty("function") Function function) {

		/**
		 * Create a tool of type 'function' and the given function definition.
		 * @param function function definition.
		 */
		@ConstructorBinding
		public FunctionTool(Function function) {
			this(Type.FUNCTION, function);
		}

		/** Create a tool of type 'function' and the given function definition. */
		public enum Type {

			/** Function tool type. */
			@JsonProperty("function")
			FUNCTION

		}

		/**
		 * Function definition.
		 *
		 * @param description A description of what the function does, used by the model
		 * to choose when and how to call the function.
		 * @param name The name of the function to be called. Must be a-z, A-Z, 0-9, or
		 * contain underscores and dashes, with a maximum length of 64.
		 * @param parameters The parameters the functions accepts, described as a JSON
		 * Schema object. To describe a function that accepts no parameters, provide the
		 * value {"type": "object", "properties": {}}.
		 */
		public record Function(@JsonProperty("description") String description, @JsonProperty("name") String name,
				@JsonProperty("parameters") Map<String, Object> parameters) {

			/**
			 * Create tool function definition.
			 * @param description tool function description.
			 * @param name tool function name.
			 * @param jsonSchema tool function schema as json.
			 */
			@ConstructorBinding
			public Function(String description, String name, String jsonSchema) {
				this(description, name, ModelOptionsUtils.jsonToMap(jsonSchema));
			}
		}
	}

	/**
	 * Creates a model response for the given chat conversation.
	 *
	 * @param contents A list of messages comprising the conversation so far.
	 */
	@JsonInclude(Include.NON_NULL)
	public record ChatCompletionRequest(@JsonProperty("contents") List<Content> contents,
			@JsonProperty("safetySettings") List<SafetySetting> safetySettings,
			@JsonProperty("generationConfig") GenerationConfig generationConfig, @JsonProperty("model") String model) {

		/**
		 * Shortcut constructor for a chat completion request with the given messages and
		 * model.
		 * @param messages A list of messages comprising the conversation so far.
		 */
		public ChatCompletionRequest(List<Content> messages) {
			this(messages, null, null, null);
		}

		public ChatCompletionRequest(List<Content> messages, String model) {
			this(messages, null, null, model);
		}

		@JsonInclude(Include.NON_NULL)
		public record SafetySetting(@JsonProperty("category") String category,
				@JsonProperty("threshold") String threshold) {
		}

		@JsonInclude(Include.NON_NULL)
		public record GenerationConfig(@JsonProperty("stopSequences") List<String> stopSequences,
				@JsonProperty("candidateCount") Integer candidateCount,
				@JsonProperty("maxOutputTokens") Integer maxOutputTokens,
				@JsonProperty("temperature") Float temperature, @JsonProperty("topP") Float topP,
				@JsonProperty("topK") Integer topK) {
		}

		/**
		 * Helper factory that creates a tool_choice of type 'none', 'auto' or selected
		 * function by name.
		 */
		public static class ToolChoiceBuilder {

			/** Model can pick between generating a message or calling a function. */
			public static final String AUTO = "auto";

			/** Model will not call a function and instead generates a message */
			public static final String NONE = "none";

			/**
			 * Specifying a particular function forces the model to call that function.
			 */
			public static Object FUNCTION(String functionName) {
				return Map.of("type", "function", "function", Map.of("name", functionName));
			}

		}

		/**
		 * An object specifying the format that the model must output.
		 *
		 * @param type Must be one of 'text' or 'json_object'.
		 */
		@JsonInclude(Include.NON_NULL)
		public record ResponseFormat(@JsonProperty("type") String type) {
		}
	}

	/** Message comprising the conversation. */
	@JsonInclude(Include.NON_NULL)
	public record Content(@JsonProperty("parts") List<ChatCompletion.Part> parts, @JsonProperty("role") String role) {

		/** Get message content as String. */
		/*
		 * public String content() { if (this.parts == null) { return null; } if
		 * (this.parts instanceof String text) { return text; } throw new
		 * IllegalStateException("The content is not a string!"); }
		 */

		/** The role of the author of this message. */
		public enum Role {

			/** User message. */
			@JsonProperty("user")
			USER("user"),
			/** Assistant message. */
			@JsonProperty("model")
			MODEL("model");

			public final String value;

			Role(String value) {
				this.value = value;
			}

			public String getValue() {
				return this.value;
			}

		}

		/**
		 * The relevant tool call.
		 *
		 * @param id The ID of the tool call. This ID must be referenced when you submit
		 * the tool outputs in using the Submit tool outputs to run endpoint.
		 * @param type The type of tool call the output is required for. For now, this is
		 * always function.
		 * @param function The function definition.
		 */
		@JsonInclude(Include.NON_NULL)
		public record ToolCall(@JsonProperty("id") String id, @JsonProperty("type") String type,
				@JsonProperty("function") ChatCompletionFunction function) {
		}

		/**
		 * The function definition.
		 *
		 * @param name The name of the function.
		 * @param arguments The arguments that the model expects you to pass to the
		 * function.
		 */
		@JsonInclude(Include.NON_NULL)
		public record ChatCompletionFunction(@JsonProperty("name") String name,
				@JsonProperty("arguments") String arguments) {
		}
	}

	/**
	 * Represents a chat completion response returned by model, based on the provided
	 * input.
	 */
	@JsonInclude(Include.NON_NULL)
	public record ChatCompletion(@JsonProperty("candidates") List<Candidate> candidates,
			@JsonProperty("promptFeedback") PromptFeedback promptFeedback,
			@JsonProperty("usageMetadata") UsageMetadata usageMetadata) {

		@JsonInclude(Include.NON_NULL)
		public record Candidate(@JsonProperty("content") Content content,
				@JsonProperty("finishReason") String finishReason,
				@JsonProperty("safetyRatings") List<SafetyRating> safetyRatings,
				@JsonProperty("citationMetadata") CitationMetadata citationMetadata,
				@JsonProperty("tokenCount") Integer tokenCount, @JsonProperty("index") Integer index) {
		}

		@JsonInclude(Include.NON_NULL)
		public record PromptFeedback(@JsonProperty("blockReason") String blockReason,
				@JsonProperty("safetyRatings") List<SafetyRating> safetyRatings) {
		}

		@JsonInclude(Include.NON_NULL)
		public record UsageMetadata(@JsonProperty("promptTokenCount") Integer promptTokenCount,
				@JsonProperty("candidatesTokenCount") Integer candidatesTokenCount,
				@JsonProperty("totalTokenCount") Integer totalTokenCount) {
		}

		@JsonInclude(Include.NON_NULL)
		public record Part(@JsonProperty("text") String text, @JsonProperty("inlineData") MediaContent inlineData) {
			public Part(String text) {
				this(text, null);
			}

			public Part(MediaContent inlineData) {
				this(null, inlineData);
			}
		}

		@JsonInclude(Include.NON_NULL)
		public record MediaContent(@JsonProperty("mimeType") String mimeType, @JsonProperty("data") String data) {
		}

		@JsonInclude(Include.NON_NULL)
		public record SafetyRating(@JsonProperty("category") String category,
				@JsonProperty("probability") String probability, @JsonProperty("blocked") Boolean blocked) {
		}

		@JsonInclude(Include.NON_NULL)
		public record CitationMetadata(@JsonProperty("citationSource") List<CitationSource> citationSource) {
		}

		@JsonInclude(Include.NON_NULL)
		public record CitationSource(@JsonProperty("startIndex") Integer startIndex,
				@JsonProperty("endIndex") Integer endIndex, @JsonProperty("uri") String uri,
				@JsonProperty("license") String license) {
		}
	}

	/**
	 * Log probability information for the choice.
	 *
	 * @param content A list of message content tokens with log probability information.
	 */
	@JsonInclude(Include.NON_NULL)
	public record LogProbs(@JsonProperty("content") List<Content> content) {

		/**
		 * Message content tokens with log probability information.
		 *
		 * @param token The token.
		 * @param logprob The log probability of the token.
		 * @param probBytes A list of integers representing the UTF-8 bytes representation
		 * of the token. Useful in instances where characters are represented by multiple
		 * tokens and their byte representations must be combined to generate the correct
		 * text representation. Can be null if there is no bytes representation for the
		 * token.
		 * @param topLogprobs List of the most likely tokens and their log probability, at
		 * this token position. In rare cases, there may be fewer than the number of
		 * requested top_logprobs returned.
		 */
		@JsonInclude(Include.NON_NULL)
		public record Content(@JsonProperty("token") String token, @JsonProperty("logprob") Float logprob,
				@JsonProperty("bytes") List<Integer> probBytes,
				@JsonProperty("top_logprobs") List<TopLogProbs> topLogprobs) {

			/**
			 * The most likely tokens and their log probability, at this token position.
			 *
			 * @param token The token.
			 * @param logprob The log probability of the token.
			 * @param probBytes A list of integers representing the UTF-8 bytes
			 * representation of the token. Useful in instances where characters are
			 * represented by multiple tokens and their byte representations must be
			 * combined to generate the correct text representation. Can be null if there
			 * is no bytes representation for the token.
			 */
			@JsonInclude(Include.NON_NULL)
			public record TopLogProbs(@JsonProperty("token") String token, @JsonProperty("logprob") Float logprob,
					@JsonProperty("bytes") List<Integer> probBytes) {
			}
		}
	}

	// Embeddings API

	/**
	 * Usage statistics for the completion request.
	 *
	 * @param completionTokens Number of tokens in the generated completion. Only
	 * applicable for completion requests.
	 * @param promptTokens Number of tokens in the prompt.
	 * @param totalTokens Total number of tokens used in the request (prompt +
	 * completion).
	 */
	@JsonInclude(Include.NON_NULL)
	public record Usage(@JsonProperty("completion_tokens") Integer completionTokens,
			@JsonProperty("prompt_tokens") Integer promptTokens, @JsonProperty("total_tokens") Integer totalTokens) {
	}

	@JsonInclude(Include.NON_NULL)
	public record ChatCompletionChunk(@JsonProperty("candidates") ChatCompletion.Candidate candidates,
			@JsonProperty("promptFeedback") ChatCompletion.PromptFeedback promptFeedback,
			@JsonProperty("usageMetadata") ChatCompletion.UsageMetadata usageMetadata) {
	}

	/**
	 * Represents an embedding vector returned by embedding endpoint.
	 *
	 * @param index The index of the embedding in the list of embeddings.
	 * @param embedding The embedding vector, which is a list of floats. The length of
	 * vector depends on the model.
	 * @param object The object type, which is always 'embedding'.
	 */
	@JsonInclude(Include.NON_NULL)
	public record Embedding(@JsonProperty("index") Integer index, @JsonProperty("embedding") List<Double> embedding,
			@JsonProperty("object") String object) {

		/**
		 * Create an embedding with the given index, embedding and object type set to
		 * 'embedding'.
		 * @param index The index of the embedding in the list of embeddings.
		 * @param embedding The embedding vector, which is a list of floats. The length of
		 * vector depends on the model.
		 */
		public Embedding(Integer index, List<Double> embedding) {
			this(index, embedding, "embedding");
		}
	}

	/**
	 * Creates an embedding vector representing the input text.
	 *
	 */
	@JsonInclude(Include.NON_NULL)
	public record EmbeddingRequest(@JsonProperty("content") Content input, @JsonProperty("model") String model,
			@JsonProperty("taskType") String taskType, @JsonProperty("title") String title,
			@JsonProperty("outputDimensionality") Integer outputDimensionality) {

		/**
		 * Create an embedding request with the given input, model and encoding format set
		 * to float.
		 * @param input Input text to embed.
		 * @param model ID of the model to use.
		 */
		public EmbeddingRequest(Content input, String model) {
			this(input, model, "float", null, null);
		}

		/**
		 * Create an embedding request with the given input. Encoding format is set to
		 * float and user is null and the model is set to 'text-embedding-ada-002'.
		 * @param input Input text to embed.
		 */
		public EmbeddingRequest(Content input) {
			this(input, DEFAULT_EMBEDDING_MODEL);
		}
	}

	/**
	 * List of multiple embedding responses.
	 *
	 * @param <T> Type of the entities in the data list.
	 * @param object Must have value "list".
	 * @param data List of entities.
	 * @param model ID of the model to use.
	 * @param usage Usage statistics for the completion request.
	 */
	@JsonInclude(Include.NON_NULL)
	public record EmbeddingList<T>(@JsonProperty("object") String object, @JsonProperty("data") List<T> data,
			@JsonProperty("model") String model, @JsonProperty("usage") Usage usage) {
	}

}
// @formatter:on
