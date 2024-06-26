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

import org.springframework.ai.chat.metadata.ChatResponseMetadata;
import org.springframework.ai.chat.metadata.Usage;

import java.util.HashMap;

/**
 * @author Christian Tzolov
 * @since 0.8.1
 */
public class GeminiAiChatResponseMetadata extends HashMap<String, Object> implements ChatResponseMetadata {

	private final GeminiAiUsage usage;

	public GeminiAiChatResponseMetadata(GeminiAiUsage usage) {
		this.usage = usage;
	}

	@Override
	public Usage getUsage() {
		return this.usage;
	}

}
