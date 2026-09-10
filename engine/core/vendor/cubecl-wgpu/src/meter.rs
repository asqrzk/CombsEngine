//! Dispatch and submit counters — the instrument the decode road lacked.
//!
//! A decode step is bound by per-dispatch overhead long before it is bound
//! by bandwidth, and nothing in this stack could say how many dispatches a
//! token actually costs: the figure was always derived from a published
//! microsecond and an operation count on paper. These two relaxed atomics
//! are read at the top and the bottom of a decode step, so the number is
//! measured instead.
//!
//! What is counted: one increment per `dispatch_workgroups` that reaches
//! the encoder (compute only — buffer copies and timestamp writes are
//! tasks in the same stream but are not dispatches), and one per
//! `queue.submit`. Counts are process-wide and monotone; the engine is
//! single-flight, so a difference across one step is that step's cost.
//!
//! Cost: a relaxed increment against a dispatch that costs tens of
//! microseconds. There is no door because there is nothing worth
//! switching off.
//!
//! This is a house patch on the vendored crate, in the manner of the
//! others (plane-kernel compile, safe-tanh): one module and two call
//! sites, no upstream behaviour touched.

use core::sync::atomic::{AtomicU64, Ordering};

static DISPATCHES: AtomicU64 = AtomicU64::new(0);
static SUBMITS: AtomicU64 = AtomicU64::new(0);

/// One `dispatch_workgroups` reached the encoder.
#[inline]
pub(crate) fn count_dispatch() {
    DISPATCHES.fetch_add(1, Ordering::Relaxed);
}

/// One `queue.submit` left for the GPU.
#[inline]
pub(crate) fn count_submit() {
    SUBMITS.fetch_add(1, Ordering::Relaxed);
}

/// `(dispatches, submits)` since the process began. Zero on a backend
/// that never dispatches (the cpu floor).
pub fn counts() -> (u64, u64) {
    (
        DISPATCHES.load(Ordering::Relaxed),
        SUBMITS.load(Ordering::Relaxed),
    )
}
