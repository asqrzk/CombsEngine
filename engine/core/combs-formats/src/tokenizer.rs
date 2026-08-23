//! Tokenizer specification returned by [`crate::ModelSource::tokenizer`].

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Where a tokenizer's `tokenizer.json` comes from.
///
/// A door rather than a flag: an adapter names the one origin its
/// tokenizer has, and consumers read it through
/// [`TokenizerSpec::json_bytes`] without caring which. A model delivered
/// as bytes over a wire — the browser case — has no path to hand out, and
/// a model on disk should not be forced to copy itself into memory to be
/// read. A third origin later (an OPFS handle, a ranged fetch) is a
/// variant here and a change nowhere else.
#[derive(Debug, Clone)]
pub enum TokenizerSource {
    /// A file on disk. Its directory is also where sibling artifacts are
    /// looked up (`1_Pooling/config.json`, tokenizer companions).
    Path(PathBuf),
    /// The JSON itself, held in memory: synthesized from container
    /// metadata, or received without a filesystem behind it.
    Bytes(Vec<u8>),
}

/// Where to find the tokenizer and which special tokens it defines.
#[derive(Debug, Clone)]
pub struct TokenizerSpec {
    /// Origin of the HuggingFace `tokenizer.json`.
    pub tokenizer: TokenizerSource,
    /// Added special tokens parsed from `tokenizer_config.json`
    /// (`added_tokens_decoder`): token id → token string (e.g. `<|im_end|>`).
    pub added_tokens: HashMap<u32, String>,
    /// Raw chat template string (Jinja) if present in `tokenizer_config.json`.
    /// Phase 1 uses a built-in ChatML wrap instead of evaluating Jinja.
    pub chat_template: Option<String>,
    /// Whether prompts should be prefixed with BOS (HF `add_bos_token` /
    /// GGUF `tokenizer.ggml.add_bos_token`). `None` = unspecified, which the
    /// engine treats as "prepend when the model declares a BOS id". Qwen2
    /// declares a BOS id but sets this to `false`; ignoring it prepends
    /// `<|endoftext|>` to every prompt.
    pub add_bos: Option<bool>,
}

impl TokenizerSpec {
    /// Dummy spec used by weight-only safetensors sources that have no
    /// tokenizer. `tokenizer()` on those sources errors before this is used.
    pub(crate) fn placeholder() -> Self {
        Self {
            tokenizer: TokenizerSource::Bytes(Vec::new()),
            added_tokens: HashMap::new(),
            chat_template: None,
            add_bos: None,
        }
    }

    /// Builds a spec around a `tokenizer.json` on disk.
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        Self {
            tokenizer: TokenizerSource::Path(path.into()),
            added_tokens: HashMap::new(),
            chat_template: None,
            add_bos: None,
        }
    }

    /// True for the weights-only placeholder — a source that carries no
    /// tokenizer at all, as distinct from one whose tokenizer failed to
    /// load.
    pub fn is_placeholder(&self) -> bool {
        match &self.tokenizer {
            TokenizerSource::Path(p) => p.as_os_str().is_empty(),
            TokenizerSource::Bytes(b) => b.is_empty(),
        }
    }

    /// The `tokenizer.json` bytes, read from disk on demand or borrowed
    /// from memory. This is what every consumer should call:
    /// `Tokenizer::from_bytes(spec.json_bytes()?)` works on every target,
    /// where `from_file` needs a filesystem that a browser does not have.
    pub fn json_bytes(&self) -> crate::Result<Cow<'_, [u8]>> {
        match &self.tokenizer {
            TokenizerSource::Path(p) => Ok(Cow::Owned(std::fs::read(p)?)),
            TokenizerSource::Bytes(b) => Ok(Cow::Borrowed(b)),
        }
    }

    /// The `tokenizer.json` path, when the tokenizer has one. `None` for
    /// an in-memory tokenizer — callers that need a real file (an external
    /// tool, a cache) must handle its absence rather than fabricate a path.
    pub fn json_path(&self) -> Option<&Path> {
        match &self.tokenizer {
            TokenizerSource::Path(p) => Some(p),
            TokenizerSource::Bytes(_) => None,
        }
    }

    /// The directory sibling artifacts live in, when there is one.
    pub fn json_dir(&self) -> Option<&Path> {
        self.json_path().and_then(Path::parent)
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
