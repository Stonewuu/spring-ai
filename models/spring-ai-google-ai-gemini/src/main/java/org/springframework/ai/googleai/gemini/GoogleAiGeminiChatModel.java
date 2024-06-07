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

import java.util.*;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.MessageType;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.model.StreamingChatModel;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.googleai.gemini.api.GeminiAiApi;
import org.springframework.ai.googleai.gemini.api.GeminiAiApi.ChatCompletion;
import org.springframework.ai.googleai.gemini.api.GeminiAiApi.ChatCompletion.Part;
import org.springframework.ai.googleai.gemini.api.GeminiAiApi.ChatCompletionRequest;
import org.springframework.ai.googleai.gemini.api.GeminiAiApi.Content;
import org.springframework.ai.googleai.gemini.metadata.GeminiAiChatResponseMetadata;
import org.springframework.ai.googleai.gemini.metadata.GeminiAiUsage;
import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.ai.model.function.AbstractFunctionCallSupport;
import org.springframework.ai.model.function.FunctionCallbackContext;
import org.springframework.ai.retry.RetryUtils;
import org.springframework.http.ResponseEntity;
import org.springframework.lang.NonNull;
import org.springframework.retry.support.RetryTemplate;
import org.springframework.util.Assert;
import org.springframework.util.CollectionUtils;
import org.springframework.util.MimeType;
import org.springframework.util.StringUtils;
import reactor.core.publisher.Flux;

/**
 * {@link ChatModel} and {@link StreamingChatModel} implementation for {@literal GeminiAI}
 * backed by {@link GeminiAiApi}.
 *
 * @author Stone Wu
 * @see ChatModel
 * @see StreamingChatModel
 * @see GeminiAiApi
 */
public class GoogleAiGeminiChatModel
		extends AbstractFunctionCallSupport<Content, GeminiAiApi.ChatCompletionRequest, ResponseEntity<ChatCompletion>>
		implements ChatModel, StreamingChatModel {

	private static final Logger logger = LoggerFactory.getLogger(GoogleAiGeminiChatModel.class);

	/** The retry template used to retry the GeminiAI API calls. */
	private final RetryTemplate retryTemplate;

	/** Low-level access to the GeminiAI API. */
	private final GeminiAiApi geminiAiApi;

	/** The default options used for the chat completion requests. */
	private final GoogleAiGeminiChatOptions defaultOptions;

	/**
	 * Creates an instance of the GeminiAiChatModel.
	 * @param geminiAiApi The GeminiAiApi instance to be used for interacting with the
	 * GeminiAI Chat API.
	 * @throws IllegalArgumentException if geminiAiApi is null
	 */
	public GoogleAiGeminiChatModel(GeminiAiApi geminiAiApi) {
		this(geminiAiApi, GoogleAiGeminiChatOptions.builder().withModel(GeminiAiApi.DEFAULT_CHAT_MODEL).build());
	}

	/**
	 * Initializes an instance of the GeminiAiChatModel.
	 * @param geminiAiApi The GeminiAiApi instance to be used for interacting with the
	 * GeminiAI Chat API.
	 * @param options The GoogleAiGeminiChatOptions to configure the chat model.
	 */
	public GoogleAiGeminiChatModel(GeminiAiApi geminiAiApi, GoogleAiGeminiChatOptions options) {
		this(geminiAiApi, options, null, RetryUtils.DEFAULT_RETRY_TEMPLATE);
	}

	/**
	 * Initializes a new instance of the GeminiAiChatModel.
	 * @param geminiAiApi The GeminiAiApi instance to be used for interacting with the
	 * GeminiAI Chat API.
	 * @param options The GoogleAiGeminiChatOptions to configure the chat model.
	 * @param functionCallbackContext The function callback context.
	 * @param retryTemplate The retry template.
	 */
	public GoogleAiGeminiChatModel(GeminiAiApi geminiAiApi, GoogleAiGeminiChatOptions options,
			FunctionCallbackContext functionCallbackContext, RetryTemplate retryTemplate) {
		super(functionCallbackContext);
		Assert.notNull(geminiAiApi, "GeminiAiApi must not be null");
		Assert.notNull(options, "Options must not be null");
		Assert.notNull(retryTemplate, "RetryTemplate must not be null");
		this.geminiAiApi = geminiAiApi;
		this.defaultOptions = options;
		this.retryTemplate = retryTemplate;
	}

	private static Content.Role toGeminiMessageType(@NonNull MessageType type) {

		Assert.notNull(type, "Message type must not be null");

		switch (type) {
			case USER:
				return Content.Role.USER;
			case ASSISTANT:
				return Content.Role.MODEL;
			default:
				throw new IllegalArgumentException("Unsupported message type: " + type);
		}
	}

	List<Part> messageToGeminiParts(Message message, String systemContext) {

		if (message instanceof UserMessage userMessage) {

			String messageTextContent = (userMessage.getContent() == null) ? "null" : userMessage.getContent();
			if (StringUtils.hasText(systemContext)) {
				messageTextContent = systemContext + "\n\n" + messageTextContent;
			}
			Part textPart = new Part(messageTextContent);

			List<Part> parts = new ArrayList<>(List.of(textPart));
			List<Part> mediaParts = userMessage.getMedia()
				.stream()
				.map(mediaData -> new Part(new ChatCompletion.MediaContent(mediaData.getMimeType().toString(),
						this.fromMediaData(mediaData.getMimeType(), mediaData.getData()))))
				.toList();

			if (!CollectionUtils.isEmpty(mediaParts)) {
				parts.addAll(mediaParts);
			}

			return parts;
		}
		else if (message instanceof AssistantMessage assistantMessage) {
			return List.of(new Part(assistantMessage.getContent()));
		}
		else {
			throw new IllegalArgumentException("Gemini doesn't support message type: " + message.getClass());
		}
	}

	@Override
	public ChatResponse call(Prompt prompt) {

		ChatCompletionRequest request = createRequest(prompt);

		var response = this.callWithFunctionSupport(request);

		List<Generation> generations = response.getBody()
			.candidates()
			.stream()
			.map(candidate -> candidate.content().parts())
			.flatMap(List::stream)
			.map(Part::text)
			.map(t -> new Generation(t))
			.toList();

		return new ChatResponse(generations, toChatResponseMetadata(response.getBody()));
	}

	@Override
	public Flux<ChatResponse> stream(Prompt prompt) {

		ChatCompletionRequest request = createRequest(prompt);

		return this.retryTemplate.execute(ctx -> {
			Flux<ChatCompletion> completions = this.geminiAiApi.chatCompletionStream(request);

			return completions
				.switchMap(r -> handleFunctionCallOrReturnStream(request, Flux.just(ResponseEntity.of(Optional.of(r)))))
				.map(response -> {
					List<Generation> generations = response.getBody()
						.candidates()
						.stream()
						.map(candidate -> candidate.content().parts())
						.flatMap(List::stream)
						.map(Part::text)
						.map(Generation::new)
						.toList();

					return new ChatResponse(generations, toChatResponseMetadata(response.getBody()));
				});
		});
	}

	private GeminiAiChatResponseMetadata toChatResponseMetadata(ChatCompletion response) {
		return new GeminiAiChatResponseMetadata(new GeminiAiUsage(response.usageMetadata()));
	}

	/** Accessible for testing. */
	ChatCompletionRequest createRequest(Prompt prompt) {

		Set<String> functionsForThisRequest = new HashSet<>();

		List<Content> geminiContent = toGeminiContent(prompt);

		ChatCompletionRequest request = new ChatCompletionRequest(geminiContent);

		if (prompt.getOptions() != null) {
			if (prompt.getOptions() instanceof ChatOptions runtimeOptions) {
				GoogleAiGeminiChatOptions updatedRuntimeOptions = ModelOptionsUtils.copyToTarget(runtimeOptions,
						ChatOptions.class, GoogleAiGeminiChatOptions.class);

				Set<String> promptEnabledFunctions = this.handleFunctionCallbackConfigurations(updatedRuntimeOptions,
						IS_RUNTIME_CALL);
				functionsForThisRequest.addAll(promptEnabledFunctions);

				request = ModelOptionsUtils.merge(updatedRuntimeOptions, request, ChatCompletionRequest.class);
			}
			else {
				throw new IllegalArgumentException("Prompt options are not of type ChatOptions: "
						+ prompt.getOptions().getClass().getSimpleName());
			}
		}

		if (this.defaultOptions != null) {

			Set<String> defaultEnabledFunctions = this.handleFunctionCallbackConfigurations(this.defaultOptions,
					!IS_RUNTIME_CALL);

			functionsForThisRequest.addAll(defaultEnabledFunctions);

			request = ModelOptionsUtils.merge(request, this.defaultOptions, ChatCompletionRequest.class);
		}

		return request;
	}

	private List<Content> toGeminiContent(Prompt prompt) {

		String systemContext = prompt.getInstructions()
			.stream()
			.filter(m -> m.getMessageType() == MessageType.SYSTEM)
			.map(m -> m.getContent())
			.collect(Collectors.joining(System.lineSeparator()));

		List<Content> contents = prompt.getInstructions()
			.stream()
			.filter(m -> m.getMessageType() == MessageType.USER || m.getMessageType() == MessageType.ASSISTANT)
			.map(message -> new Content(messageToGeminiParts(message, systemContext),
					toGeminiMessageType(message.getMessageType()).name()))
			.toList();

		return contents;
	}

	private String fromMediaData(MimeType mimeType, Object mediaContentData) {
		if (mediaContentData instanceof byte[] bytes) {
			// Assume the bytes are an image. So, convert the bytes to a base64 encoded
			// following the prefix pattern.
			return Base64.getEncoder().encodeToString(bytes);
			// return String.format("data:%s;base64,%s", mimeType.toString(),
			// Base64.getEncoder().encodeToString(bytes));
		}
		else if (mediaContentData instanceof String text) {
			// Assume the text is a URLs or a base64 encoded image prefixed by the user.
			return text;
		}
		else {
			throw new IllegalArgumentException(
					"Unsupported media data type: " + mediaContentData.getClass().getSimpleName());
		}
	}

	private List<GeminiAiApi.FunctionTool> getFunctionTools(Set<String> functionNames) {
		return this.resolveFunctionCallbacks(functionNames).stream().map(functionCallback -> {
			var function = new GeminiAiApi.FunctionTool.Function(functionCallback.getDescription(),
					functionCallback.getName(), functionCallback.getInputTypeSchema());
			return new GeminiAiApi.FunctionTool(function);
		}).toList();
	}

	@Override
	protected ChatCompletionRequest doCreateToolResponseRequest(ChatCompletionRequest previousRequest,
			Content responseMessage, List<Content> conversationHistory) {
		return previousRequest;
	}

	@Override
	protected List<Content> doGetUserMessages(ChatCompletionRequest request) {
		return request.contents();
	}

	@Override
	protected Content doGetToolResponseMessage(ResponseEntity<ChatCompletion> chatCompletion) {
		return chatCompletion.getBody().candidates().iterator().next().content();
	}

	@Override
	protected ResponseEntity<ChatCompletion> doChatCompletion(ChatCompletionRequest request) {
		return this.geminiAiApi.chatCompletionEntity(request);
	}

	@Override
	protected Flux<ResponseEntity<ChatCompletion>> doChatCompletionStream(ChatCompletionRequest request) {
		return this.geminiAiApi.chatCompletionStream(request).map(Optional::ofNullable).map(ResponseEntity::of);
	}

	@Override
	protected boolean isToolFunctionCall(ResponseEntity<ChatCompletion> chatCompletion) {
		return false;
	}

	@Override
	public ChatOptions getDefaultOptions() {
		return GoogleAiGeminiChatOptions.fromOptions(this.defaultOptions);
	}

}
