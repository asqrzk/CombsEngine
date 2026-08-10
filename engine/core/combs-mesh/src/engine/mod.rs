//! The emoji model + the three ways to move it around: build
//! ([`EmojiBuilder`]), serialize ([`EmojiExporter`]), store
//! ([`Registry`]), plus sprite helpers.

mod builder;
mod exporter;
mod registry;
pub mod sprites;

pub use builder::{Emoji, EmojiBuilder};
pub use exporter::EmojiExporter;
pub use registry::{Registry, RegistryEntry};
