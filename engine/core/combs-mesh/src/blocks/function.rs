//! `fnc` — callable function definitions carried by the emoji.

use serde::{Deserialize, Serialize};

/// Function definitions block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionBlock {
    /// The functions.
    #[serde(default)]
    pub definitions: Vec<FunctionDef>,
}

/// A single function definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionDef {
    /// Function name.
    pub name: String,
    /// CRUD-ish kind (or a custom one).
    pub kind: FunctionKind,
    /// Parameter names.
    #[serde(default)]
    pub params: Vec<String>,
    /// Implementation body (script/expression; interpreted by runtimes).
    #[serde(default)]
    pub body: String,
}

/// The kind of function.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FunctionKind {
    /// Creates something.
    Add,
    /// Reads something.
    Get,
    /// Mutates something.
    Update,
    /// Anything else.
    Custom(String),
}
