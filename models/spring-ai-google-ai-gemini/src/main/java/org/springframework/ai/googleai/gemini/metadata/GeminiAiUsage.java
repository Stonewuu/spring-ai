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
package org.springframework.ai.googleai.gemini.metadata;

import org.springframework.ai.chat.metadata.Usage;
import org.springframework.ai.googleai.gemini.api.GeminiAiApi;
import org.springframework.util.Assert;

/**
 * @author Christian Tzolov
 * @since 0.8.1
 * 
 */
public class GeminiAiUsage implements Usage {

	private final GeminiAiApi.ChatCompletion.UsageMetadata usageMetadata;

	public GeminiAiUsage(GeminiAiApi.ChatCompletion.UsageMetadata usageMetadata) {
		Assert.notNull(usageMetadata, "UsageMetadata must not be null");
		this.usageMetadata = usageMetadata;
	}

	@Override
	public Long getPromptTokens() {
		return Long.valueOf(usageMetadata.promptTokenCount());
	}

	@Override
	public Long getGenerationTokens() {
		return Long.valueOf(usageMetadata.candidatesTokenCount());
	}

}
