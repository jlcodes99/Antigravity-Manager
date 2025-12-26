// Claude 非流式响应转换 (Gemini → Claude)
// 对应 NonStreamingProcessor

use super::models::*;
use super::utils::to_claude_usage;

use std::sync::Arc;
use tokio::sync::Mutex;
use std::collections::HashMap;

/// 非流式响应处理器
pub struct NonStreamingProcessor {
    content_blocks: Vec<ContentBlock>,
    text_builder: String,
    thinking_builder: String,
    thinking_signature: Option<String>,
    trailing_signature: Option<String>,
    has_tool_call: bool,
    signature_map: Option<Arc<Mutex<HashMap<String, String>>>>,
}

impl NonStreamingProcessor {
    pub fn new(signature_map: Option<Arc<Mutex<HashMap<String, String>>>>) -> Self {
        Self {
            content_blocks: Vec::new(),
            text_builder: String::new(),
            thinking_builder: String::new(),
            thinking_signature: None,
            trailing_signature: None,
            has_tool_call: false,
            signature_map,
        }
    }

    /// 处理 Gemini 响应并转换为 Claude 响应
    pub fn process(&mut self, gemini_response: &GeminiResponse) -> ClaudeResponse {
        // 获取 parts
        let empty_parts = vec![];
        let parts = gemini_response
            .candidates
            .as_ref()
            .and_then(|c| c.get(0))
            .and_then(|candidate| candidate.content.as_ref())
            .map(|content| &content.parts)
            .unwrap_or(&empty_parts);

        // 处理所有 parts
        for part in parts {
            self.process_part(part);
        }

        // 刷新剩余内容
        self.flush_thinking();
        self.flush_text();

        // 处理 trailingSignature (空 text 带签名)
        if let Some(signature) = self.trailing_signature.take() {
            self.content_blocks.push(ContentBlock::Thinking {
                thinking: String::new(),
                signature: Some(signature),
            });
        }

        // 构建响应
        self.build_response(gemini_response)
    }

    /// 处理单个 part
    fn process_part(&mut self, part: &GeminiPart) {
        let signature = part.thought_signature.clone();

        // 1. FunctionCall 处理
        if let Some(fc) = &part.function_call {
            self.flush_thinking();
            self.flush_text();

            // 处理 trailingSignature (B4/C3 场景)
            if let Some(trailing_sig) = self.trailing_signature.take() {
                self.content_blocks.push(ContentBlock::Thinking {
                    thinking: String::new(),
                    signature: Some(trailing_sig),
                });
            }

            self.has_tool_call = true;

            // 生成 tool_use id
            let tool_id = fc.id.clone().unwrap_or_else(|| {
                format!(
                    "{}-{}",
                    fc.name,
                    crate::proxy::common::utils::generate_random_id()
                )
            });

            // 核心修复：存储签名到服务端的全局 map
            if let (Some(sig), Some(map)) = (signature.clone(), &self.signature_map) {
                let tid = tool_id.clone();
                let m_clone = map.clone();
                tokio::spawn(async move {
                    let mut m = m_clone.lock().await;
                    m.insert(tid, sig);
                });
            }

            let mut tool_use = ContentBlock::ToolUse {
                id: tool_id,
                name: fc.name.clone(),
                input: fc.args.clone().unwrap_or(serde_json::json!({})),
                signature: None,
            };

            // 只使用 FC 自己的签名
            if let ContentBlock::ToolUse { signature: sig, .. } = &mut tool_use {
                *sig = signature;
            }

            self.content_blocks.push(tool_use);
            return;
        }

        // 2. Text 处理
        if let Some(text) = &part.text {
            if part.thought.unwrap_or(false) {
                // Thinking part
                self.flush_text();

                // 处理 trailingSignature
                if let Some(trailing_sig) = self.trailing_signature.take() {
                    self.flush_thinking();
                    self.content_blocks.push(ContentBlock::Thinking {
                        thinking: String::new(),
                        signature: Some(trailing_sig),
                    });
                }

                self.thinking_builder.push_str(text);
                if signature.is_some() {
                    self.thinking_signature = signature;
                }
            } else {
                // 普通 Text
                if text.is_empty() {
                    // 空 text 带签名 - 暂存到 trailingSignature
                    if signature.is_some() {
                        self.trailing_signature = signature;
                    }
                    return;
                }

                self.flush_thinking();

                // 处理之前的 trailingSignature
                if let Some(trailing_sig) = self.trailing_signature.take() {
                    self.flush_text();
                    self.content_blocks.push(ContentBlock::Thinking {
                        thinking: String::new(),
                        signature: Some(trailing_sig),
                    });
                }

                self.text_builder.push_str(text);

                // 非空 text 带签名 - 立即刷新并输出空 thinking 块
                if let Some(sig) = signature {
                    self.flush_text();
                    self.content_blocks.push(ContentBlock::Thinking {
                        thinking: String::new(),
                        signature: Some(sig),
                    });
                }
            }
        }

        // 3. InlineData (Image) 处理
        if let Some(img) = &part.inline_data {
            self.flush_thinking();
            
            let mime_type = &img.mime_type;
            let data = &img.data;
            if !data.is_empty() {
                let markdown_img = format!("![image](data:{};base64,{})", mime_type, data);
                self.text_builder.push_str(&markdown_img);
                self.flush_text();
            }
        }
    }

    /// 刷新 text builder
    fn flush_text(&mut self) {
        if self.text_builder.is_empty() {
            return;
        }

        self.content_blocks.push(ContentBlock::Text {
            text: self.text_builder.clone(),
        });
        self.text_builder.clear();
    }

    /// 刷新 thinking builder
    fn flush_thinking(&mut self) {
        // 如果既没有内容也没有签名，直接返回
        if self.thinking_builder.is_empty() && self.thinking_signature.is_none() {
            return;
        }

        let thinking = self.thinking_builder.clone();
        let signature = self.thinking_signature.take();

        self.content_blocks.push(ContentBlock::Thinking { thinking, signature });
        self.thinking_builder.clear();
    }

    /// 构建最终响应
    fn build_response(&self, gemini_response: &GeminiResponse) -> ClaudeResponse {
        let finish_reason = gemini_response
            .candidates
            .as_ref()
            .and_then(|c| c.get(0))
            .and_then(|candidate| candidate.finish_reason.as_deref());

        let stop_reason = if self.has_tool_call {
            "tool_use"
        } else if finish_reason == Some("MAX_TOKENS") {
            "max_tokens"
        } else {
            "end_turn"
        };

        let usage = gemini_response
            .usage_metadata
            .as_ref()
            .map(|u| to_claude_usage(u))
            .unwrap_or(Usage {
                input_tokens: 0,
                output_tokens: 0,
            });

        ClaudeResponse {
            id: gemini_response
                .response_id
                .clone()
                .unwrap_or_else(|| format!("msg_{}", crate::proxy::common::utils::generate_random_id())),
            type_: "message".to_string(),
            role: "assistant".to_string(),
            model: gemini_response
                .model_version
                .clone()
                .unwrap_or_default(),
            content: self.content_blocks.clone(),
            stop_reason: stop_reason.to_string(),
            stop_sequence: None,
            usage,
        }
    }
}

/// 转换 Gemini 响应为 Claude 响应 (公共接口)
pub fn transform_response(
    gemini_response: &GeminiResponse,
    signature_map: Option<Arc<Mutex<HashMap<String, String>>>>,
) -> Result<ClaudeResponse, String> {
    let mut processor = NonStreamingProcessor::new(signature_map);
    Ok(processor.process(gemini_response))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_text_response() {
        let gemini_resp = GeminiResponse {
            candidates: Some(vec![Candidate {
                content: Some(GeminiContent {
                    role: "model".to_string(),
                    parts: vec![GeminiPart {
                        text: Some("Hello, world!".to_string()),
                        thought: None,
                        thought_signature: None,
                        function_call: None,
                        function_response: None,
                        inline_data: None,
                    }],
                }),
                finish_reason: Some("STOP".to_string()),
                index: Some(0),
            }]),
            usage_metadata: Some(UsageMetadata {
                prompt_token_count: Some(10),
                candidates_token_count: Some(5),
                total_token_count: Some(15),
            }),
            model_version: Some("gemini-2.5-pro".to_string()),
            response_id: Some("resp_123".to_string()),
        };

        let result = transform_response(&gemini_resp, None);
        assert!(result.is_ok());

        let claude_resp = result.unwrap();
        assert_eq!(claude_resp.role, "assistant");
        assert_eq!(claude_resp.stop_reason, "end_turn");
        assert_eq!(claude_resp.content.len(), 1);

        match &claude_resp.content[0] {
            ContentBlock::Text { text } => {
                assert_eq!(text, "Hello, world!");
            }
            _ => panic!("Expected Text block"),
        }
    }

    #[test]
    fn test_thinking_with_signature() {
        let gemini_resp = GeminiResponse {
            candidates: Some(vec![Candidate {
                content: Some(GeminiContent {
                    role: "model".to_string(),
                    parts: vec![
                        GeminiPart {
                            text: Some("Let me think...".to_string()),
                            thought: Some(true),
                            thought_signature: Some("sig123".to_string()),
                            function_call: None,
                            function_response: None,
                            inline_data: None,
                        },
                        GeminiPart {
                            text: Some("The answer is 42".to_string()),
                            thought: None,
                            thought_signature: None,
                            function_call: None,
                            function_response: None,
                            inline_data: None,
                        },
                    ],
                }),
                finish_reason: Some("STOP".to_string()),
                index: Some(0),
            }]),
            usage_metadata: None,
            model_version: Some("gemini-2.5-pro".to_string()),
            response_id: Some("resp_456".to_string()),
        };

        let result = transform_response(&gemini_resp, None);
        assert!(result.is_ok());

        let claude_resp = result.unwrap();
        assert_eq!(claude_resp.content.len(), 2);

        match &claude_resp.content[0] {
            ContentBlock::Thinking { thinking, signature } => {
                assert_eq!(thinking, "Let me think...");
                assert_eq!(signature.as_deref(), Some("sig123"));
            }
            _ => panic!("Expected Thinking block"),
        }

        match &claude_resp.content[1] {
            ContentBlock::Text { text } => {
                assert_eq!(text, "The answer is 42");
            }
            _ => panic!("Expected Text block"),
        }
    }

    #[tokio::test]
    async fn test_signature_caching() {
        use serde_json::json;
        let signature_map = Arc::new(Mutex::new(HashMap::new()));
        let tool_id = "call_cache_test".to_string();
        let sig = "cache_sig_789".to_string();

        let gemini_resp = GeminiResponse {
            candidates: Some(vec![Candidate {
                content: Some(GeminiContent {
                    role: "model".to_string(),
                    parts: vec![GeminiPart {
                        text: None,
                        thought: Some(true),
                        thought_signature: Some(sig.clone()),
                        function_call: Some(FunctionCall {
                            name: "get_weather".to_string(),
                            id: Some(tool_id.clone()),
                            args: Some(json!({"location": "Chicago"})),
                        }),
                        function_response: None,
                        inline_data: None,
                    }],
                }),
                finish_reason: Some("STOP".to_string()),
                index: Some(0),
            }]),
            usage_metadata: None,
            model_version: Some("gemini-2.0-flash".to_string()),
            response_id: Some("resp_789".to_string()),
        };

        // 执行转换，并传入 map
        let result = transform_response(&gemini_resp, Some(signature_map.clone()));
        assert!(result.is_ok());

        // 给异步任务一点时间执行
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        // 验证 map 中是否已经缓存了签名
        let locked_map = signature_map.lock().await;
        assert_eq!(locked_map.get(&tool_id), Some(&sig));
    }
}
