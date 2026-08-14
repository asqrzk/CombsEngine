//! Incremental detokenization for streaming output.
//!
//! Decodes the full generated sequence each step and emits the newly appended
//! suffix. O(n²) in generated length, which is irrelevant at Phase 1 scales.
//!
//! Multi-token UTF-8 needs care. The ByteLevel decoder converts bytes with
//! `from_utf8_lossy`, so a character split across tokens (every emoji, and
//! CJK on byte-level vocabs) decodes as a replacement char on the first
//! token and resolves on the next. Emitting that is doubly wrong: the client
//! sees the replacement char, and the completed character then fails the
//! prefix check and is dropped entirely. So a TRAILING replacement char is
//! held back until the sequence completes.

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
        // Hold back a trailing replacement char — it is an incomplete
        // UTF-8 sequence the next token completes. A genuine replacement
        // char in the model's own output is released as soon as anything
        // follows it, and `flush` covers the end-of-generation case.
        let stable = full.trim_end_matches(char::REPLACEMENT_CHARACTER);
        let piece = self.advance(stable);
        Ok(piece)
    }

    /// Emits anything still held back (call once when generation ends), so
    /// a reply that legitimately ends in a replacement char is not
    /// silently truncated.
    pub fn flush(&mut self, tokenizer: &Tokenizer) -> Result<String> {
        if self.ids.is_empty() {
            return Ok(String::new());
        }
        let full = tokenizer
            .decode(&self.ids, false)
            .map_err(|e| EngineError::Tokenizer(e.to_string()))?;
        Ok(self.advance(&full))
    }

    /// Emits the part of `text` not yet sent, and records it as emitted.
    /// The decoder can re-normalize earlier text, in which case the new
    /// decode is not a suffix of what was emitted; falling back to the
    /// longest common prefix keeps output flowing instead of dropping it
    /// (the old `unwrap_or("")` lost the rest of the reply).
    fn advance(&mut self, text: &str) -> String {
        let piece = match text.strip_prefix(self.emitted.as_str()) {
            Some(rest) => rest.to_string(),
            None => {
                let common = self
                    .emitted
                    .char_indices()
                    .zip(text.char_indices())
                    .take_while(|((_, a), (_, b))| a == b)
                    .map(|((i, c), _)| i + c.len_utf8())
                    .last()
                    .unwrap_or(0);
                text[common..].to_string()
            }
        };
        self.emitted = text.to_string();
        piece
    }
}

impl Default for IncrementalDetokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drives `advance` directly with the decode results a ByteLevel
    /// decoder actually produces, so the emoji case is testable without a
    /// tokenizer fixture.
    fn feed(d: &mut IncrementalDetokenizer, full_decode: &str) -> String {
        let stable = full_decode.trim_end_matches(char::REPLACEMENT_CHARACTER);
        d.advance(stable)
    }

    #[test]
    fn holds_back_partial_utf8_then_emits_the_whole_character() {
        let mut d = IncrementalDetokenizer::new();
        assert_eq!(feed(&mut d, "hi "), "hi ");
        // First half of a 4-byte emoji: lossy-decoded to a replacement char.
        assert_eq!(feed(&mut d, "hi \u{FFFD}"), "", "nothing leaks mid-character");
        // Second half completes it.
        assert_eq!(feed(&mut d, "hi 😀"), "😀");
        assert_eq!(feed(&mut d, "hi 😀!"), "!");
    }

    #[test]
    fn genuine_replacement_char_is_released_once_text_follows() {
        let mut d = IncrementalDetokenizer::new();
        // The 'a' is stable and goes out immediately; only the trailing
        // replacement char waits.
        assert_eq!(feed(&mut d, "a\u{FFFD}"), "a");
        assert_eq!(feed(&mut d, "a\u{FFFD}b"), "\u{FFFD}b");
    }

    #[test]
    fn renormalized_decode_emits_the_tail_instead_of_dropping_it() {
        // The decoder rewrote earlier text, so the new decode is not a
        // suffix of what was emitted. Text already streamed cannot be
        // retracted, so the best available answer is everything past the
        // divergence point — the old code returned "" and lost the rest of
        // the reply outright.
        let mut d = IncrementalDetokenizer::new();
        assert_eq!(feed(&mut d, "colour"), "colour");
        assert_eq!(feed(&mut d, "color me"), "r me");
    }

    #[test]
    fn multi_token_cjk_streams_cleanly() {
        let mut d = IncrementalDetokenizer::new();
        assert_eq!(feed(&mut d, "\u{FFFD}"), "");
        assert_eq!(feed(&mut d, "蜂"), "蜂");
        assert_eq!(feed(&mut d, "蜂\u{FFFD}"), "");
        assert_eq!(feed(&mut d, "蜂蜜"), "蜜");
    }
}
