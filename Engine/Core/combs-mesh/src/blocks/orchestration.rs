//! `orc` — orchestration directives for runtimes.

use serde::{Deserialize, Serialize};

/// Orchestration block: an ordered list of key/value directives.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrchestrationBlock {
    /// The directives.
    #[serde(default)]
    pub directives: Vec<OrchestrationDirective>,
}

/// A single directive.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrchestrationDirective {
    /// What the directive does.
    pub kind: DirectiveKind,
    /// Key (meaning depends on `kind`).
    pub key: String,
    /// Value (meaning depends on `kind`).
    #[serde(default)]
    pub value: String,
}

/// The kind of orchestration directive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DirectiveKind {
    /// Runtime should schedule work.
    Todo,
    /// Runtime should wait.
    Wait,
    /// Informational note.
    Note,
    /// Warning.
    Warning,
    /// Attention sink hint for inference runtimes.
    AttentionSink,
    /// Address map entry for mesh routing.
    AddressMap,
}
