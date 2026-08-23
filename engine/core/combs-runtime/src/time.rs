//! Monotonic time that exists on every target the engine runs on.
//!
//! `std::time::Instant::now()` does not merely return something useless on
//! `wasm32-unknown-unknown` — it panics, because there is no clock behind
//! it. `web-time` presents the same API over `performance.now()` in a
//! browser and re-exports `std::time` everywhere else, so the engine
//! measures itself identically in both places and `Duration` stays the std
//! type `GenerationStats` already exposes to callers.

#[cfg(not(target_family = "wasm"))]
pub(crate) use std::time::Instant;

#[cfg(target_family = "wasm")]
pub(crate) use web_time::Instant;
