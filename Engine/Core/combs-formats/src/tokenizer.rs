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
}
