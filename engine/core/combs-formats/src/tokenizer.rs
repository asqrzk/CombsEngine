//! Tokenizer specification returned by [`crate::ModelSource::tokenizer`].

use std::collections::HashMap;
use std::path::PathBuf;

/// Where to find the tokenizer and which special tokens it defines.
#[derive(Debug, Clone)]
pub struct TokenizerSpec {
    /// Path to the HuggingFace `tokenizer.json`.
    pub tokenizer_json: PathBuf,
    /// Added special tokens parsed from `tokenizer_config.json`
    /// (`added_tokens_decoder`): token id → token string (e.g. `<|im_end|>`).
    pub added_tokens: HashMap<u32, String>,
    /// Raw chat template string (Jinja) if present in `tokenizer_config.json`.
    /// Phase 1 uses a built-in ChatML wrap instead of evaluating Jinja.
    pub chat_template: Option<String>,
}

impl TokenizerSpec {
    /// Dummy spec used by weight-only safetensors sources that have no
    /// tokenizer. `tokenizer()` on those sources errors before this is used.
    pub(crate) fn placeholder() -> Self {
        Self {
            tokenizer_json: PathBuf::new(),
            added_tokens: HashMap::new(),
            chat_template: None,
        }
    }

    /// Looks up the id of an added special token by its string, e.g.
    /// `spec.special_token_id("<|im_end|>")`.
    pub fn special_token_id(&self, token: &str) -> Option<u32> {
        self.added_tokens
            .iter()
            .find(|(_, s)| s.as_str() == token)
            .map(|(id, _)| *id)
    }

    /// Wraps a user prompt in the ChatML template used by SmolLM2-style
    /// instruction models. Returns `None` if the tokenizer does not define
    /// `<|im_start|>` / `<|im_end|>` tokens.
    pub fn chatml_wrap(&self, user_prompt: &str) -> Option<String> {
        let start = self.special_token_id("<|im_start|>")?;
        self.special_token_id("<|im_end|>")?;
        let _ = start;
        Some(format!(
            "<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        ))
    }

    /// Which chat template this tokenizer's special tokens imply.
    pub fn chat_template_kind(&self) -> ChatTemplate {
        if self.special_token_id("<start_of_turn>").is_some() {
            ChatTemplate::Gemma
        } else {
            ChatTemplate::Chatml
        }
    }

    /// Wraps (role, content) message pairs into the model's chat format,
    /// ending with the assistant turn left open for generation. Unknown
    /// roles are coerced to `user` (same convention as serve).
    pub fn wrap_messages(&self, messages: &[(String, String)]) -> String {
        match self.chat_template_kind() {
            ChatTemplate::Chatml => {
                let mut out = String::new();
                for (role, content) in messages {
                    let role = match role.as_str() {
                        "system" | "user" | "assistant" => role.as_str(),
                        _ => "user",
                    };
                    out.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
                }
                out.push_str("<|im_start|>assistant\n");
                out
            }
            ChatTemplate::Gemma => {
                // Gemma-3 template: <bos> prefix, turns are
                // `<start_of_turn>{user|model}\n{content}<end_of_turn>\n`,
                // and system content is folded into the first user turn
                // (the HF template prepends it with a blank line).
                let mut out = String::from("<bos>");
                let mut system_prefix = String::new();
                let mut first_user_seen = false;
                for (role, content) in messages {
                    match role.as_str() {
                        "system" => {
                            system_prefix.push_str(content);
                            system_prefix.push_str("\n\n");
                        }
                        _ => {
                            let turn_role = match role.as_str() {
                                "assistant" | "model" => "model",
                                _ => "user",
                            };
                            let mut body = String::new();
                            if turn_role == "user" && !first_user_seen {
                                body.push_str(&system_prefix);
                                first_user_seen = true;
                            }
                            body.push_str(content);
                            out.push_str(&format!(
                                "<start_of_turn>{turn_role}\n{body}<end_of_turn>\n"
                            ));
                        }
                    }
                }
                out.push_str("<start_of_turn>model\n");
                out
            }
        }
    }
}

/// Chat template flavor detected from the tokenizer's special tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    /// `<|im_start|>role\n…<|im_end|>` (SmolLM2 / Qwen style).
    Chatml,
    /// `<start_of_turn>user\n…<end_of_turn>` (Gemma style).
    Gemma,
}
