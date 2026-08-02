//! `txt` — free text: a name, a description, and key/value specs.

use serde::{Deserialize, Serialize};

/// Free-text block. Every emoji built with
/// [`crate::EmojiBuilder`] carries one: the emoji's `name` round-trips
/// through this block (the binary container has no name field of its own).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextBlock {
    /// Display name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Arbitrary key/value specs (kept ordered).
    #[serde(default)]
    pub specs: Vec<(String, String)>,
}
