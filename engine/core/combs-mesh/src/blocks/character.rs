//! `chr` — character traits + backstory.

use serde::{Deserialize, Serialize};

/// Character block: weighted traits plus a free-text backstory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CharacterBlock {
    /// Trait name → weight pairs (weights are free-form, typically 0..=1).
    #[serde(default)]
    pub traits: Vec<(String, f32)>,
    /// Free-text backstory.
    #[serde(default)]
    pub backstory: String,
}
