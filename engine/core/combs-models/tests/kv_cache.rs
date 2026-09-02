//! CPU (NdArray) cross-validation of the two KV cache implementations:
//! identical attention outputs for identical append patterns, and correct
//! rollback (`popn`) semantics for the paged cache.

use burn::backend::NdArray;
use burn::tensor::{Distribution, Tensor};

use combs_models::{CacheConfig, CacheKind, ContiguousKVCache, KVCache, PagedKVCache};

type B = NdArray<f32>;

const N_Q: usize = 4;
const N_KV: usize = 2;
const HEAD_DIM: usize = 8;
const SCALE: f64 = 0.35;

fn rand_qkv(seq: usize) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
    let device = Default::default();
    let q = Tensor::random(
        [1, N_Q, seq, HEAD_DIM],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let k = Tensor::random(
        [1, N_KV, seq, HEAD_DIM],
        Distribution::Normal(0.37, 0.8),
        &device,
    );
    let v = Tensor::random(
        [1, N_KV, seq, HEAD_DIM],
        Distribution::Normal(-0.11, 1.3),
        &device,
    );
    (q, k, v)
}

fn assert_close(a: Tensor<B, 4>, b: Tensor<B, 4>, ctx: &str) {
    assert_eq!(a.dims(), b.dims(), "{ctx}: shape mismatch");
    let a: Vec<f32> = a.into_data().to_vec().unwrap();
    let b: Vec<f32> = b.into_data().to_vec().unwrap();
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            x.is_finite() && y.is_finite(),
            "{ctx}: non-finite at flat index {i}: {x} vs {y}"
        );
        assert!(
            (x - y).abs() < 1e-4,
            "{ctx}: mismatch at flat index {i}: {x} vs {y}"
        );
    }
}

fn paged(layers: usize, max_seq_len: usize, page_size: usize) -> PagedKVCache<B> {
    PagedKVCache::new(
        layers,
        CacheConfig {
            max_seq_len,
            page_size,
            kind: CacheKind::Paged,
            quantize_kv: false,
        },
    )
}

/// Feeds the same chunk pattern through both caches and compares every
/// layer's attention output.
#[test]
fn paged_matches_contiguous_across_chunks_and_decodes() {
    let mut contiguous = ContiguousKVCache::<B>::new(2);
    let mut paged = paged(2, 64, 4);

    // Chunked prefill (5 + 7 tokens), then three single-token decode steps.
    let steps = [5usize, 7, 1, 1, 1];
    let mut pos = 0;
    for (step, &seq) in steps.iter().enumerate() {
        let (q, k, v) = rand_qkv(seq);
        for layer in 0..2 {
            let out_c = contiguous.attention(
                layer,
                q.clone(),
                k.clone(),
                v.clone(),
                pos,
                SCALE,
            );
            let out_p = paged.attention(layer, q.clone(), k.clone(), v.clone(), pos, SCALE);
            assert_close(out_c, out_p, &format!("step {step} layer {layer}"));
        }
        pos += seq;
        assert_eq!(contiguous.seq_len(), pos);
        assert_eq!(paged.seq_len(), pos);
    }
    // 15 positions over 4-token pages = 4 pages.
    assert_eq!(paged.pages_used(), Some(4));
}

/// After `popn`, re-appending different K/V for the rolled-back positions
/// must produce the same result as a fresh cache that never saw them.
#[test]
fn paged_popn_rollback_matches_fresh_history() {
    let mut rolled = paged(2, 64, 4);
    let mut fresh = ContiguousKVCache::<B>::new(2);

    // Original history: 10 tokens (3 pages), then roll back 6 -> keep 4.
    let (q0, k0, v0) = rand_qkv(10);
    for layer in 0..2 {
        rolled.attention(layer, q0.clone(), k0.clone(), v0.clone(), 0, SCALE);
    }
    rolled.popn(6);
    assert_eq!(rolled.seq_len(), 4);
    assert_eq!(rolled.pages_used(), Some(1));

    // Fresh cache gets only the kept prefix.
    let qk = q0.narrow(2, 0, 4);
    let kk = k0.narrow(2, 0, 4);
    let vk = v0.narrow(2, 0, 4);
    for layer in 0..2 {
        fresh.attention(layer, qk.clone(), kk.clone(), vk.clone(), 0, SCALE);
    }

    // Both then see the same 3 new tokens; outputs must match.
    let (q1, k1, v1) = rand_qkv(3);
    for layer in 0..2 {
        let out_r = rolled.attention(layer, q1.clone(), k1.clone(), v1.clone(), 4, SCALE);
        let out_f = fresh.attention(layer, q1.clone(), k1.clone(), v1.clone(), 4, SCALE);
        assert_close(out_r, out_f, &format!("rollback layer {layer}"));
    }
    assert_eq!(rolled.seq_len(), 7);
    assert_eq!(fresh.seq_len(), 7);
}

/// Writes that straddle page boundaries (chunk not aligned to page_size)
/// land in the right slots.
#[test]
fn paged_unaligned_chunks_match_contiguous() {
    let mut contiguous = ContiguousKVCache::<B>::new(1);
    let mut paged = paged(1, 64, 4);

    let steps = [3usize, 6, 5, 2];
    let mut pos = 0;
    for (step, &seq) in steps.iter().enumerate() {
        let (q, k, v) = rand_qkv(seq);
        let out_c = contiguous.attention(0, q.clone(), k.clone(), v.clone(), pos, SCALE);
        let out_p = paged.attention(0, q, k, v, pos, SCALE);
        assert_close(out_c, out_p, &format!("unaligned step {step}"));
        pos += seq;
    }
    assert_eq!(paged.seq_len(), 16);
    assert_eq!(paged.pages_used(), Some(4));
}

/// `reset` returns the paged cache to a clean, reusable state.
#[test]
fn paged_reset_allows_fresh_reuse() {
    let mut c = paged(1, 64, 4);
    let (q, k, v) = rand_qkv(9);
    c.attention(0, q, k, v, 0, SCALE);
    assert_eq!(c.seq_len(), 9);
    c.reset();
    assert_eq!(c.seq_len(), 0);
    assert_eq!(c.pages_used(), Some(0));

    let mut reference = ContiguousKVCache::<B>::new(1);
    let (q, k, v) = rand_qkv(5);
    let out_c = c.attention(0, q.clone(), k.clone(), v.clone(), 0, SCALE);
    let out_r = reference.attention(0, q, k, v, 0, SCALE);
    assert_close(out_c, out_r, "post-reset reuse");
}

/// Appending past capacity panics (the engine guards this earlier).
#[test]
#[should_panic(expected = "capacity exceeded")]
fn paged_capacity_overflow_panics() {
    let mut c = paged(1, 8, 4);
    let (q, k, v) = rand_qkv(9);
    c.attention(0, q, k, v, 0, SCALE);
}
