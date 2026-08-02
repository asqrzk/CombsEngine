//! `tdo` — a task list with statuses and dependencies.

use serde::{Deserialize, Serialize};

/// Task list block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TodoBlock {
    /// The tasks.
    #[serde(default)]
    pub items: Vec<TodoItem>,
}

/// A single task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TodoItem {
    /// Stable task key (referenced by `depends_on`).
    pub key: String,
    /// Human-readable task text.
    pub value: String,
    /// Current status.
    pub status: TodoStatus,
    /// Keys of tasks that must complete first.
    #[serde(default)]
    pub depends_on: Vec<String>,
}

/// Task status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TodoStatus {
    /// Not started.
    Pending,
    /// Being worked on.
    InProgress,
    /// Finished.
    Done,
    /// Cannot proceed.
    Blocked,
}
