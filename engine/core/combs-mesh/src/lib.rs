//! # combs-mesh — CombsMesh emoji engine core
//!
//! Pure-Rust, wasm32-friendly implementation of the CombsMesh emoji spec:
//! an emoji is a typed bag of blocks (text, sprite atlas, todos, functions,
//! APIs, lifecycle, character, emotion, encryption, orchestration) that can
//! be serialized two ways —
//!
//! - **`.cmse` binary** ([`EmojiExporter::to_binary`]): `CMSE` container
//!   with a block directory, CRC32 integrity, and optional per-block AEAD
//!   encryption at rest driven by the `Enc` block + a [`KeyRing`].
//! - **Unicode envelope** ([`EmojiExporter::to_unicode`]): self-describing
//!   PUA plane 15/16 + tag-char encoding for text-only channels.
//!
//! Plus: a content-addressed [`Registry`], CPU sprite rendering behind the
//! [`Renderer`] trait, and the [`CombsEngineCore`] trait the C ABI crate
//! binds to (with [`DefaultEngine`] as the standalone implementation).
//!
//! Deliberate constraints: no GPU/inference deps (the optional `engine`
//! feature pulls combs-runtime only for `infer`), no C dependencies
//! (RustCrypto everywhere, algorithm-compatible with `@combs/zerotrust`),
//! and every reader of external data is panic-free.

pub mod blocks;
pub mod codepoint;
pub mod crypto;
pub mod engine;
pub mod error;
pub mod ffi_trait;
pub mod render;

mod binary;
#[cfg(feature = "wasm")]
pub mod wasm;

pub use blocks::{
    ApiBlock, ApiEndpoint, Block, BlockTag, CharacterBlock, DirectiveKind, EmotionBlock,
    EmotionState, EncryptionAlgorithm, EncryptionBlock, FunctionBlock, FunctionDef, FunctionKind,
    ImageBlock, LifecycleBlock, LifecycleState, LifecycleTransition, OrchestrationBlock,
    OrchestrationDirective, SpriteAtlas, TextBlock, TodoBlock, TodoItem, TodoStatus,
};
pub use codepoint::{decode_blocks, encode_blocks, tag_char, tag_from_char};
pub use crypto::{DEFAULT_HKDF_INFO, KeyRing};
pub use engine::{Emoji, EmojiBuilder, EmojiExporter, Registry, RegistryEntry, sprites};
pub use error::{MeshError, Result};
pub use ffi_trait::{CombsEngineCore, DefaultEngine, EngineError};
pub use render::{CpuRenderer, Renderer};
