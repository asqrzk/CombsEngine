//! `emo` — emotion states with intensities.

use serde::{Deserialize, Serialize};

/// Emotion block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmotionBlock {
    /// The emotion states.
    #[serde(default)]
    pub states: Vec<EmotionState>,
}

/// A single emotion state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmotionState {
    /// Emotion name.
    pub name: String,
    /// Intensity in 0.0..=1.0.
    pub intensity: f32,
}
