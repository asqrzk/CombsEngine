//! `api` — HTTP endpoint descriptions.

use serde::{Deserialize, Serialize};

/// API endpoints block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApiBlock {
    /// The endpoints.
    #[serde(default)]
    pub endpoints: Vec<ApiEndpoint>,
}

/// A single HTTP endpoint description.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApiEndpoint {
    /// Endpoint name.
    pub name: String,
    /// HTTP method (GET/POST/...).
    pub method: String,
    /// URL path.
    pub path: String,
    /// Human-readable description.
    #[serde(default)]
    pub description: String,
}
