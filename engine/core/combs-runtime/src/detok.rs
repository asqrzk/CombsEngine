//! Incremental detokenization for streaming output.
//!
//! Decodes the full generated sequence each step and emits the newly appended
//! suffix. O(n²) in generated length, which is irrelevant at Phase 1 scales,
//! and robust to multi-token UTF-8 sequences (no partial-byte artifacts).

use tokenizers::Tokenizer;

use crate::{EngineError, Result};

/// Streaming detokenizer state.
pub struct IncrementalDetokenizer {
    ids: Vec<u32>,
    emitted: String,
}

impl IncrementalDetokenizer {
    /// Creates an empty detokenizer.
    pub fn new() -> Self {
        IncrementalDetokenizer {
            ids: Vec::new(),
            emitted: String::new(),
        }
    }

    /// Appends a token and returns the newly decoded text piece (may be
    /// empty when the token completes a not-yet-decodable sequence).
    ///
    /// Decodes WITH special tokens (`skip_special_tokens = false`), the
    /// transformers convention for tool-parsing pipelines: qwen marks
    /// `<tool_call>`/`<think>` as control tokens in GGUF vocabs, and
    /// skipping them silently strips the very markers the tool-call parser
    /// (and the user-visible thinking trace) depend on. Stop tokens never
    /// reach the detokenizer — generation halts before they are pushed —
    /// so eos/turn markers still never appear in output.
    pub fn push(&mut self, tokenizer: &Tokenizer, id: u32) -> Result<String> {
        self.ids.push(id);
        let full = tokenizer
            .decode(&self.ids, false)
            .map_err(|e| EngineError::Tokenizer(e.to_string()))?;
        let piece = full
            .strip_prefix(self.emitted.as_str())
            .unwrap_or("")
            .to_string();
        self.emitted = full;
        Ok(piece)
    }
}

impl Default for IncrementalDetokenizer {
    fn default() -> Self {
        Self::new()
    }
}
