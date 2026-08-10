//! `lfc` — an agent lifecycle state machine.

use serde::{Deserialize, Serialize};

/// Lifecycle block: states + event-driven transitions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LifecycleBlock {
    /// The states (exactly one should have `initial: true`).
    #[serde(default)]
    pub states: Vec<LifecycleState>,
    /// The transitions.
    #[serde(default)]
    pub transitions: Vec<LifecycleTransition>,
}

/// A lifecycle state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LifecycleState {
    /// State name.
    pub name: String,
    /// Whether this is the entry state.
    #[serde(default)]
    pub initial: bool,
}

/// An event-driven transition between states.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LifecycleTransition {
    /// Source state name.
    pub from: String,
    /// Target state name.
    pub to: String,
    /// Event that triggers the transition.
    pub event: String,
}
