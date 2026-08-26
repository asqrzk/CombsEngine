//! KV cache abstraction.
//!
//! Phase 2 moves attention *behind* the cache: [`KVCache::attention`] appends
//! new K/V for a layer and computes causal attention against the full cached
//! window in one call, so the cache implementation owns the K/V layout.
//!
//! Two implementations ship:
//! - [`ContiguousKVCache`] — per-layer contiguous K/V tensors, `cat`-extended
//!   each step (Phase 1 behavior, kept as the cross-validation baseline).
//! - [`PagedKVCache`] — MLC-style paged arena: fixed-size pages per layer, a
//!   page table and a free-page allocator. Steady-state decode writes one
//!   page slot and gathers the active pages; no per-token O(seq) rewrite of
//!   the whole cache.

use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{Bool, Device, Int, Tensor, TensorData, activation::softmax, backend::Backend};

use crate::matmul::safe_matmul;
use crate::precision::{to_f32, to_float};

/// Whether to prefer burn's fused (flash) attention kernel over the manual
/// scores→mask→softmax→matmul path. Controlled by `COMBS_ATTN=flash|manual`
/// (default `flash`); read once per process.
fn flash_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("COMBS_ATTN").map(|v| v != "manual").unwrap_or(true)
    })
}

/// Which [`KVCache`] implementation to instantiate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheKind {
    /// Per-layer contiguous K/V, `cat` per step (baseline).
    Contiguous,
    /// Paged arena with a page table (default).
    Paged,
}

/// Configuration for a KV cache instance.
#[derive(Debug, Clone, Copy)]
pub struct CacheConfig {
    /// Maximum number of cached positions (arena capacity).
    pub max_seq_len: usize,
    /// Tokens per page (paged cache only).
    pub page_size: usize,
    /// Implementation to use.
    pub kind: CacheKind,
    /// Store global-layer KV pages int8-quantized (group-32 along
    /// head_dim, packed 4 bytes per int element) — ~3.5× less KV memory
    /// on the f32 build at near-lossless int8 fidelity. Sliding-window
    /// layers stay in float (they hold at most `w-1` tokens). Set from
    /// `COMBS_KV_QUANT=1` by the engine.
    pub quantize_kv: bool,
}

impl CacheConfig {
    /// Default page size (MLC uses 16 as well).
    pub const DEFAULT_PAGE_SIZE: usize = 16;

    /// Paged cache with the default page size.
    pub fn paged(max_seq_len: usize) -> Self {
        CacheConfig {
            max_seq_len,
            page_size: Self::DEFAULT_PAGE_SIZE,
            kind: CacheKind::Paged,
            quantize_kv: false,
        }
    }

    /// Contiguous (baseline) cache.
    pub fn contiguous(max_seq_len: usize) -> Self {
        CacheConfig {
            max_seq_len,
            page_size: Self::DEFAULT_PAGE_SIZE,
            kind: CacheKind::Contiguous,
            quantize_kv: false,
        }
    }

    /// Number of pages in the arena.
    pub fn num_pages(&self) -> usize {
        self.max_seq_len.div_ceil(self.page_size)
    }
}

/// Per-layer key/value storage that owns the attention computation.
///
/// Tensors are 4-D `[batch=1, heads, seq, head_dim]`; `q` has `n_q` heads
/// while `k`/`v` have `n_kv` heads (GQA expansion happens inside the
/// implementation, as does causal masking).
pub trait KVCache<B: Backend>: Send {
    /// Appends `seq` new positions of K/V for `layer` and computes attention
    /// of `q` against the full cached window (past + new).
    ///
    /// `pos` is the absolute position of the first new token and must equal
    /// [`KVCache::seq_len`] on entry (dense contiguous appends). `scale` is
    /// the attention logit scale (`1/sqrt(head_dim)`). Returns the attention
    /// output `[1, n_q, seq, head_dim]`.
    fn attention(
        &mut self,
        layer: usize,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        pos: usize,
        scale: f64,
    ) -> Tensor<B, 4> {
        self.attention_opts(layer, q, k, v, pos, scale, None)
    }

    /// [`KVCache::attention`] with an optional sliding-window span (Gemma
    /// local layers): when `Some(w)`, query at absolute position p attends
    /// only keys in `(p - w, p]` — older keys stay cached but are masked
    /// out. `None` = full causal attention (Llama-family behavior).
    fn attention_opts(
        &mut self,
        layer: usize,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        pos: usize,
        scale: f64,
        window: Option<usize>,
    ) -> Tensor<B, 4>;

    /// Total cached sequence length.
    fn seq_len(&self) -> usize;

    /// Rolls back the last `n` cached tokens, returning how many were
    /// actually dropped. Caches that cannot roll back (the contiguous
    /// baseline) return 0 — callers gate prefix reuse on a nonzero result.
    fn popn(&mut self, n: usize) -> usize {
        let _ = n;
        0
    }

    /// Drops all cached state (session reset).
    fn reset(&mut self);

    /// Pages currently allocated to the sequence (paged cache only).
    fn pages_used(&self) -> Option<usize> {
        None
    }

    /// Page-table snapshot for observability (paged cache only). Cheap:
    /// reads counters, never touches device memory.
    fn page_stats(&self) -> Option<PageStats> {
        None
    }
}

/// A paged cache's page-table state at a point in time (for monitoring).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageStats {
    /// Pages holding live KV entries for this sequence.
    pub pages_used: usize,
    /// Pages still available to this sequence — free ids in the
    /// materialized arena plus capacity not yet materialized.
    pub pages_free: usize,
    /// Total page capacity (`pages_used + pages_free` when healthy).
    pub num_pages: usize,
    /// Pages every materialized layer arena is currently sized to. Grows
    /// geometrically toward `num_pages` with the session's high-water
    /// mark; memory cost tracks this, not capacity.
    pub pages_materialized: usize,
    /// Tokens per page.
    pub page_size: usize,
    /// Cached sequence length in tokens.
    pub seq_len: usize,
    /// Layers whose arena tensor is actually allocated. Arenas are lazy —
    /// a layer materializes on its first write — so this rises through
    /// the first prefill and stays below `layers_total` for models whose
    /// sliding layers never touch the paged arena at all.
    pub layers_materialized: usize,
    /// Layers in this cache.
    pub layers_total: usize,
    /// Layers holding a rolling sliding-window store instead of pages.
    pub layers_sliding: usize,
}

/// Repeats each KV head `n_rep` times consecutively (GQA → MHA expansion):
/// `[b, nkv, s, d] -> [b, nkv * n_rep, s, d]`.
fn repeat_kv<B: Backend>(x: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
    if n_rep == 1 {
        return x;
    }
    let [b, nkv, s, d] = x.dims();
    x.unsqueeze_dim::<5>(2)
        .expand([b, nkv, n_rep, s, d])
        .reshape([b, nkv * n_rep, s, d])
}

/// Standard scaled dot-product causal attention over a fully materialized
/// K/V window.
///
/// `q`: `[1, n_q, seq, d]`; `k`/`v`: `[1, n_kv, total, d]`; `pos` is the
/// absolute position of the first query token. Returns `[1, n_q, seq, d]`.
///
/// Prefers burn's fused flash-attention kernel (one kernel, no materialized
/// `[seq, total]` scores matrix) when the scale is the default
/// `1/sqrt(head_dim)`; the causal mode is bottom-right aligned, which is
/// exactly the `pos`-offset masking the manual path applies, so chunked
/// prefill (`pos > 0`) is covered as well. Set `COMBS_ATTN=manual` to force
/// the reference path.
pub(crate) fn attend<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    pos: usize,
    scale: f64,
    window: Option<usize>,
) -> Tensor<B, 4> {
    let device = q.device();
    // Attention scores + softmax run in f32 for f16 stability (exp overflows
    // f16); no-op in f32 builds. The heavy weight matmuls + KV stay f16.
    let out_dtype = q.dtype();
    let q = to_f32(q);
    let k = to_f32(k);
    let v = to_f32(v);
    let [_, n_q, seq, d] = q.dims();
    let [_, n_kv, total, _] = k.dims();
    let n_rep = n_q / n_kv;
    let k = repeat_kv(k, n_rep);
    let v = repeat_kv(v, n_rep);

    let default_scale = 1.0 / (d as f64).sqrt();
    if flash_enabled() && window.is_none() && (scale - default_scale).abs() < 1e-12 {
        let out = burn::tensor::module::attention(
            q,
            k,
            v,
            None,
            None,
            AttentionModuleOptions {
                scale: None,
                softcap: None,
                // Decode (seq == 1) needs no mask: a single query at the end
                // of the window attends to everything cached.
                is_causal: seq > 1,
            },
        );
        return to_float(out, out_dtype);
    }

    // Reference path: explicit scores, causal (+ optional sliding-window)
    // mask, softmax, P@V.
    let scores = q.matmul(k.transpose()).mul_scalar(scale);
    let scores = if seq > 1 || window.is_some() {
        // Causal mask: query at global position p attends keys <= p; a
        // sliding window w further forbids keys <= p - w.
        let q_pos =
            Tensor::<B, 1, Int>::arange((pos as i64)..((pos + seq) as i64), &device)
                .reshape([seq, 1]);
        let k_pos = Tensor::<B, 1, Int>::arange(0..(total as i64), &device).reshape([1, total]);
        let mut forbidden: Tensor<B, 2, Bool> = k_pos.clone().greater(q_pos.clone());
        if let Some(w) = window {
            let too_old = k_pos
                .add_scalar(w as i64 - 1)
                .lower(q_pos);
            forbidden = forbidden.bool_or(too_old);
        }
        let mask = forbidden
            .unsqueeze_dims::<4>(&[0, 1])
            .expand([1, n_q, seq, total]);
        scores.mask_fill(mask, -1e30f32)
    } else {
        scores // single query attends to everything cached
    };

    // `safe_matmul` for the P@V product: with a >= 512-token window this
    // shape (M = seq, K = total) enters the broken wgpu/Metal matmul region.
    to_float(safe_matmul(softmax(scores, 3), v), out_dtype)
}

/// Values per KV quantization group (along head_dim; one f32 scale each).
const KV_QUANT_GROUP: usize = 32;

/// Quantizes `x: [b, h, s, d]` to int8 with a per-32-value group scale
/// (absmax/127), packing 4 signed bytes per int element.
///
/// Packing scheme (exact i32 fit): lanes 0..2 are stored offset-binary
/// (`q+128 ∈ [1, 255]`, since q is clamped to ±127), lane 3 signed as-is:
/// `p = l0 + l1·2⁸ + l2·2¹⁶ + q3·2²⁴`. The extreme corner
/// (l0=l1=l2=255, q3=127) is exactly `i32::MAX` — no overflow — and the
/// backend Int element is at least i32, so pack/unpack round-trips are
/// integer-exact on every backend.
///
/// Returns `(packed [b,h,s,d/4], scales [b,h,s,d/32])`.
fn kv_quantize<B: Backend>(x: Tensor<B, 4>) -> (Tensor<B, 4, Int>, Tensor<B, 4>) {
    let [b, h, s, d] = x.dims();
    debug_assert_eq!(d % KV_QUANT_GROUP, 0);
    let groups = d / KV_QUANT_GROUP;
    // Scales are computed in f32 for a stable absmax, but returned in the
    // input's native dtype: the arena stores them next to native-dtype
    // tensors, and the unfused f16 backend miscomputes mixed-dtype ops
    // (an f16 × f32-broadcast mul returns power-of-two-wrong values).
    let native = x.dtype();
    let g = to_f32(x).reshape([b, h, s, groups, KV_QUANT_GROUP]);
    let scale = g
        .clone()
        .abs()
        .max_dim(4) // [b, h, s, groups, 1]
        .div_scalar(127.0)
        .clamp_min(1e-8);
    let q = g
        .div(scale.clone().expand([b, h, s, groups, KV_QUANT_GROUP]))
        .round()
        .clamp(-127.0, 127.0)
        .int()
        .reshape([b, h, s, d / 4, 4]);
    let lane = |i: usize| q.clone().narrow(4, i, 1).reshape([b, h, s, d / 4]);
    let packed = lane(0).add_scalar(128)
        + lane(1).add_scalar(128).mul_scalar(256)
        + lane(2).add_scalar(128).mul_scalar(65536)
        + lane(3).mul_scalar(16777216);
    (packed, to_float(scale.reshape([b, h, s, groups]), native))
}

/// Inverse of [`kv_quantize`]: unpack the four lanes (floor-division
/// emulated exactly on the truncating int div: the high lane's remainder
/// is sign-corrected, the rest are non-negative) and apply the group
/// scales. Returns `[b, h, s, d]` float.
fn kv_dequantize<B: Backend>(
    packed: Tensor<B, 4, Int>,
    scales: Tensor<B, 4>,
    d: usize,
) -> Tensor<B, 4> {
    let [b, h, s, _] = packed.dims();
    let groups = d / KV_QUANT_GROUP;
    let t = packed.clone().div_scalar(16777216);
    let r3 = packed - t.clone().mul_scalar(16777216);
    let neg = r3.clone().lower_elem(0);
    let q3 = t.clone().mask_where(neg.clone(), t.sub_scalar(1));
    let r = r3.clone().mask_where(neg, r3.add_scalar(16777216));
    let l2 = r.clone().div_scalar(65536);
    let r = r - l2.clone().mul_scalar(65536);
    let l1 = r.clone().div_scalar(256);
    let l0 = r - l1.clone().mul_scalar(256);
    // Stack along a new trailing lane axis → [b, h, s, d/4, 4]; a plain
    // reshape then restores the original d-ordering (lane i holds value
    // 4·g+i, exactly how the pack side split them).
    let q = Tensor::stack::<5>(
        vec![
            l0.sub_scalar(128),
            l1.sub_scalar(128),
            l2.sub_scalar(128),
            q3,
        ],
        4,
    )
    .reshape([b, h, s, d]);
    let g = q.float().reshape([b, h, s, groups, KV_QUANT_GROUP]);
    // Force the scales onto g's dtype before the broadcast multiply — the
    // unfused f16 backend silently corrupts mixed-dtype ops.
    let scales = to_float(scales, g.dtype());
    g.mul(
        scales
            .reshape([b, h, s, groups, 1])
            .expand([b, h, s, groups, KV_QUANT_GROUP]),
    )
    .reshape([b, h, s, d])
}

/// Simple contiguous cache: stores one K and one V tensor per layer and
/// concatenates along the sequence dimension every step.
///
/// Cost: an O(seq) copy per token per layer — kept as the correctness
/// baseline; the paged arena is the production default.
pub struct ContiguousKVCache<B: Backend> {
    layers: Vec<Option<(Tensor<B, 4>, Tensor<B, 4>)>>,
    seq_len: usize,
}

impl<B: Backend> ContiguousKVCache<B> {
    /// Creates an empty cache for `num_layers` layers.
    pub fn new(num_layers: usize) -> Self {
        ContiguousKVCache {
            layers: (0..num_layers).map(|_| None).collect(),
            seq_len: 0,
        }
    }
}

impl<B: Backend> KVCache<B> for ContiguousKVCache<B> {
    fn attention_opts(
        &mut self,
        layer: usize,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        pos: usize,
        scale: f64,
        window: Option<usize>,
    ) -> Tensor<B, 4> {
        let slot = &mut self.layers[layer];
        let (k_full, v_full) = match slot.take() {
            Some((k_old, v_old)) => (
                Tensor::cat(vec![k_old, k], 2),
                Tensor::cat(vec![v_old, v], 2),
            ),
            None => (k, v),
        };
        self.seq_len = k_full.dims()[2];
        let out = attend(q, k_full.clone(), v_full.clone(), pos, scale, window);
        *slot = Some((k_full, v_full));
        out
    }

    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn reset(&mut self) {
        for slot in &mut self.layers {
            *slot = None;
        }
        self.seq_len = 0;
    }
}

/// Free-page allocator: a stack of physical page ids.
#[derive(Debug)]
struct PageAllocator {
    free: Vec<usize>,
}

impl PageAllocator {
    fn new(num_pages: usize) -> Self {
        // Reversed so page 0 is allocated first (deterministic tests).
        PageAllocator {
            free: (0..num_pages).rev().collect(),
        }
    }

    fn alloc(&mut self) -> Option<usize> {
        self.free.pop()
    }

    fn free_page(&mut self, id: usize) {
        self.free.push(id);
    }

    fn num_free(&self) -> usize {
        self.free.len()
    }

    /// Adds the ids `from..to` to the free stack, lowest-popping-first.
    /// Only ever called with the free stack empty (the growth trigger in
    /// `ensure_pages`), so ordering with previously freed ids never
    /// arises and page-id determinism is preserved.
    fn grow(&mut self, from: usize, to: usize) {
        self.free.extend((from..to).rev());
    }

    fn reset(&mut self, num_pages: usize) {
        *self = PageAllocator::new(num_pages);
    }
}

/// MLC-style paged KV cache.
///
/// Per layer, K and V live in fixed arena tensors of shape
/// `[num_pages, n_kv, page_size, head_dim]`, allocated lazily on the layer's
/// first use. A single-sequence page table maps logical pages to physical
/// page ids drawn from a free-page allocator (the struct is shaped so
/// per-sequence tables can be added later).
///
/// `attention()` writes the new K/V into page slots (one `slice_assign` per
/// touched page), gathers the active pages into a contiguous
/// `[1, n_kv, total, head_dim]` window and runs the standard matmul path.
/// Steady-state decode therefore writes a single slot and gathers — the
/// Phase 1 O(seq) `cat`-rewrite per token is gone. (A fused no-gather
/// CubeCL kernel is a later task.)
/// A global layer's page arena: float, or int8-quantized (packed 4/int +
/// per-group scales) when `CacheConfig::quantize_kv` is set.
enum Arena<B: Backend> {
    Fp {
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
    },
    Quant {
        k_packed: Tensor<B, 4, Int>,
        k_scales: Tensor<B, 4>,
        v_packed: Tensor<B, 4, Int>,
        v_scales: Tensor<B, 4>,
    },
}

impl<B: Backend> Arena<B> {
    /// Pages this arena is currently sized to (dim 0).
    fn num_pages(&self) -> usize {
        match self {
            Arena::Fp { k, .. } => k.dims()[0],
            Arena::Quant { k_packed, .. } => k_packed.dims()[0],
        }
    }
}

/// Pages a fresh arena starts with (512 tokens at the default page
/// size). Arenas double from here toward `CacheConfig::num_pages()` as
/// the session actually grows — the capacity is a bound, not a bill.
const INITIAL_ARENA_PAGES: usize = 32;

pub struct PagedKVCache<B: Backend> {
    config: CacheConfig,
    allocator: PageAllocator,
    /// Page count every materialized layer arena is sized to right now.
    /// Shared across layers; grows geometrically in `ensure_pages` and
    /// each layer's arena catches up on its next touch.
    materialized_pages: usize,
    /// Page table: logical page index -> physical page id (single sequence).
    table: Vec<usize>,
    seq_len: usize,
    arenas: Vec<Option<Arena<B>>>,
    /// Per-layer sliding-window override (transformers' layer-typed cache):
    /// `Some(w)` layers keep only the most recent `w-1` tokens in a rolling
    /// tensor and never touch the paged arena — for gemma's 5 local : 1
    /// global pattern this skips the arena allocation for ~5/6 of layers.
    /// Empty ⇒ every layer is global (the llama case).
    layer_windows: Vec<Option<usize>>,
    /// Rolling K/V for sliding layers, `[1, n_kv, ≤w-1, head_dim]`. Keys
    /// are stored post-RoPE with their absolute positions baked in; only
    /// the attention mask is re-based when old tokens are evicted.
    sliding: Vec<Option<(Tensor<B, 4>, Tensor<B, 4>)>>,
    device: Option<Device<B>>,
    /// Device-resident mirror of `table`, rebuilt only when the table
    /// changes. Without it every attention call re-uploaded the table —
    /// twice per layer per token, ~56 host->device round trips per decode
    /// step on a 28-layer model, for a table that changes once every
    /// `page_size` tokens.
    table_device: Option<Tensor<B, 1, Int>>,
    /// Set by every mutation of `table`; cleared on rebuild. The host
    /// `table` stays the single source of truth — the device tensor is
    /// derived state and must never be read while this is set.
    table_dirty: bool,
}

impl<B: Backend> PagedKVCache<B> {
    /// Creates an empty paged cache for `num_layers` layers (all global).
    /// Arena tensors are allocated lazily on first use of each layer.
    pub fn new(num_layers: usize, config: CacheConfig) -> Self {
        Self::new_with_windows(num_layers, config, vec![None; num_layers])
    }

    /// Creates a paged cache with a per-layer sliding-window assignment
    /// (`windows[i] = Some(w)` ⇒ layer `i` stores at most `w-1` past
    /// tokens). `w >= 2` — a window of 1 would leave decode steps with no
    /// past context at all.
    pub fn new_with_windows(
        num_layers: usize,
        config: CacheConfig,
        windows: Vec<Option<usize>>,
    ) -> Self {
        assert_eq!(windows.len(), num_layers, "one window entry per layer");
        for w in windows.iter().flatten() {
            assert!(*w >= 2, "sliding window must be >= 2, got {w}");
        }
        let initial_pages = config.num_pages().min(INITIAL_ARENA_PAGES);
        PagedKVCache {
            allocator: PageAllocator::new(initial_pages),
            materialized_pages: initial_pages,
            config,
            table: Vec::new(),
            seq_len: 0,
            arenas: (0..num_layers).map(|_| None).collect(),
            layer_windows: windows,
            sliding: (0..num_layers).map(|_| None).collect(),
            device: None,
            table_device: None,
            table_dirty: true,
        }
    }

    /// Pages still available to this sequence: free ids in the
    /// materialized arena plus capacity not yet materialized.
    pub fn num_free_pages(&self) -> usize {
        self.allocator.num_free() + (self.config.num_pages() - self.materialized_pages)
    }

    /// Page-table snapshot (see [`KVCache::page_stats`]).
    pub fn page_stats_inner(&self) -> PageStats {
        PageStats {
            pages_used: self.table.len(),
            pages_free: self.allocator.num_free()
                + (self.config.num_pages() - self.materialized_pages),
            num_pages: self.config.num_pages(),
            pages_materialized: self.materialized_pages,
            page_size: self.config.page_size,
            seq_len: self.seq_len,
            layers_materialized: self.arenas.iter().filter(|a| a.is_some()).count(),
            layers_total: self.arenas.len(),
            layers_sliding: self.sliding.iter().filter(|s| s.is_some()).count(),
        }
    }

    /// Ensures the page table covers `total` positions, doubling the
    /// materialized arena size when the free list runs dry and capacity
    /// remains. `.max(pages_needed)` makes a long prefill chunk a single
    /// growth event; the capacity assert in `attention_opts` fires before
    /// true exhaustion, so the expect below stays a leak detector.
    fn ensure_pages(&mut self, total: usize) -> usize {
        let pages_needed = total.div_ceil(self.config.page_size);
        while self.table.len() < pages_needed {
            if self.allocator.num_free() == 0
                && self.materialized_pages < self.config.num_pages()
            {
                let grown = (self.materialized_pages * 2)
                    .max(pages_needed)
                    .min(self.config.num_pages());
                self.allocator.grow(self.materialized_pages, grown);
                self.materialized_pages = grown;
            }
            let page = self
                .allocator
                .alloc()
                .expect("page allocator exhausted (max_seq_len exceeded)");
            self.table.push(page);
            self.table_dirty = true;
        }
        pages_needed
    }

    /// Sliding-layer attention (transformers `DynamicSlidingWindowLayer`):
    /// concat the rolling store with the new K/V, attend over the full
    /// concat, persist only the trailing `w-1` tokens.
    ///
    /// The stored keys carry their original RoPE (absolute positions);
    /// after eviction the first stored key is at absolute position
    /// `kv_offset = pos + seq - full_len`, so the causal/window mask is
    /// evaluated with the queries re-based by that offset — RoPE is never
    /// re-applied or re-indexed.
    fn sliding_attention(
        &mut self,
        layer: usize,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        pos: usize,
        scale: f64,
        w: usize,
    ) -> Tensor<B, 4> {
        let seq = k.dims()[2];
        let slot = &mut self.sliding[layer];
        let (k_full, v_full) = match slot.take() {
            Some((k_old, v_old)) => (
                Tensor::cat(vec![k_old, k], 2),
                Tensor::cat(vec![v_old, v], 2),
            ),
            None => (k, v),
        };
        let full_len = k_full.dims()[2];
        let kv_offset = pos + seq - full_len;
        // K3b: the decode step attends over the rolling window in one
        // dispatch (contiguous mode of the decode-attention kernel);
        // chunks and refusals take the materialized path below.
        let fast = if seq == 1 {
            crate::wgsl::try_sliding_decode_attention(
                q.clone(),
                k_full.clone(),
                v_full.clone(),
                w,
                scale,
            )
        } else {
            None
        };
        let out = match fast {
            Some(out) => out,
            None => attend(
                q,
                k_full.clone(),
                v_full.clone(),
                pos - kv_offset,
                scale,
                Some(w),
            ),
        };
        // Keep `w-1` past tokens: the next decode step appends one, giving
        // exactly `w` visible keys (transformers' `-window + 1` rule). A
        // prefill chunk longer than the window attends over its full concat
        // first (context within the chunk is never lost), then truncates.
        let keep = full_len.min(w - 1);
        *slot = Some((
            k_full.narrow(2, full_len - keep, keep),
            v_full.narrow(2, full_len - keep, keep),
        ));
        out
    }

    /// Page-table indices for the first `pages` logical pages, on-device.
    ///
    /// The full table is uploaded once per mutation and narrowed per call;
    /// steady-state decode reads a resident tensor instead of paying a
    /// host->device trip per layer.
    fn page_indices(&mut self, pages: usize) -> Tensor<B, 1, Int> {
        if self.table_dirty || self.table_device.is_none() {
            let ids: Vec<i32> = self.table.iter().map(|&p| p as i32).collect();
            let len = ids.len();
            let device = self
                .device
                .as_ref()
                .expect("device set on first attention call");
            self.table_device = Some(Tensor::<B, 1, Int>::from_data(
                TensorData::new(ids, [len]),
                device,
            ));
            self.table_dirty = false;
        }
        self.table_device
            .as_ref()
            .expect("rebuilt above")
            .clone()
            .narrow(0, 0, pages)
    }

    /// Gathers the first `pages` page-table entries of `arena`
    /// (`[num_pages, n_kv, page_size, last]`) into a contiguous
    /// `[1, n_kv, total, last]` window.
    fn gather_window(
        &mut self,
        arena: Tensor<B, 4>,
        pages: usize,
        total: usize,
    ) -> Tensor<B, 4> {
        let [_, n_kv, page_size, last] = arena.dims();
        arena
            .select(0, self.page_indices(pages)) // [pages, n_kv, page_size, last]
            .swap_dims(0, 1) // [n_kv, pages, page_size, last]
            .reshape([1, n_kv, pages * page_size, last])
            .narrow(2, 0, total)
    }

    /// [`Self::gather_window`] for the packed int arenas.
    fn gather_window_int(
        &mut self,
        arena: Tensor<B, 4, Int>,
        pages: usize,
        total: usize,
    ) -> Tensor<B, 4, Int> {
        let [_, n_kv, page_size, last] = arena.dims();
        arena
            .select(0, self.page_indices(pages))
            .swap_dims(0, 1)
            .reshape([1, n_kv, pages * page_size, last])
            .narrow(2, 0, total)
    }
}

impl<B: Backend> KVCache<B> for PagedKVCache<B> {
    fn attention_opts(
        &mut self,
        layer: usize,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        pos: usize,
        scale: f64,
        window: Option<usize>,
    ) -> Tensor<B, 4> {
        let [_, n_kv, seq, head_dim] = k.dims();
        let total = pos + seq;
        // Layer 0 of each forward pass advances the sequence; all layers of
        // the pass see the same pos/seq, so later layers find seq_len
        // already at `total`.
        if layer == 0 {
            assert_eq!(
                pos, self.seq_len,
                "paged cache expects dense contiguous appends (pos == seq_len)"
            );
            self.seq_len = total;
        } else {
            debug_assert_eq!(total, self.seq_len);
        }
        assert!(
            total <= self.config.max_seq_len,
            "paged cache capacity exceeded: {total} > {}",
            self.config.max_seq_len
        );

        if self.device.is_none() {
            self.device = Some(k.device());
        }
        // Sliding layers bypass the paged arena entirely (the caller's
        // `window` argument and this cache's per-layer assignment come from
        // the same AttentionPattern, so the cache's own value is used).
        if let Some(w) = self.layer_windows.get(layer).copied().flatten() {
            return self.sliding_attention(layer, q, k, v, pos, scale, w);
        }
        let quant = self.config.quantize_kv && head_dim % KV_QUANT_GROUP == 0;
        // Growth first: ensure_pages may raise materialized_pages, and
        // both the first-touch allocation and the catch-up below size to
        // the raised value.
        let pages = self.ensure_pages(total);
        let np = self.materialized_pages;
        if self.arenas[layer].is_none() {
            let device = k.device();
            let ps = self.config.page_size;
            self.arenas[layer] = Some(if quant {
                Arena::Quant {
                    k_packed: Tensor::zeros([np, n_kv, ps, head_dim / 4], &device),
                    k_scales: Tensor::zeros([np, n_kv, ps, head_dim / KV_QUANT_GROUP], &device),
                    v_packed: Tensor::zeros([np, n_kv, ps, head_dim / 4], &device),
                    v_scales: Tensor::zeros([np, n_kv, ps, head_dim / KV_QUANT_GROUP], &device),
                }
            } else {
                let shape = [np, n_kv, ps, head_dim];
                Arena::Fp {
                    k: Tensor::zeros(shape, &device),
                    v: Tensor::zeros(shape, &device),
                }
            });
        } else if self.arenas[layer].as_ref().expect("checked").num_pages() < np {
            // This layer's arena lags a growth event: reallocate at the
            // new size and copy the old pages over. Physical page ids are
            // dim-0 indices, so growth appends capacity without touching
            // the table; the zero tail matches first-touch semantics. The
            // copy rides the same stream as every other arena op.
            let device = k.device();
            self.arenas[layer] = Some(match self.arenas[layer].take().expect("checked") {
                Arena::Fp { k: ak, v: av } => {
                    let [old_np, nk, ps, d] = ak.dims();
                    let span = [0..old_np, 0..nk, 0..ps, 0..d];
                    Arena::Fp {
                        k: Tensor::zeros([np, nk, ps, d], &device)
                            .slice_assign(span.clone(), ak),
                        v: Tensor::zeros([np, nk, ps, d], &device).slice_assign(span, av),
                    }
                }
                Arena::Quant {
                    k_packed,
                    k_scales,
                    v_packed,
                    v_scales,
                } => {
                    let [old_np, nk, ps, dp] = k_packed.dims();
                    let dg = k_scales.dims()[3];
                    let span_p = [0..old_np, 0..nk, 0..ps, 0..dp];
                    let span_s = [0..old_np, 0..nk, 0..ps, 0..dg];
                    Arena::Quant {
                        k_packed: Tensor::zeros([np, nk, ps, dp], &device)
                            .slice_assign(span_p.clone(), k_packed),
                        k_scales: Tensor::zeros([np, nk, ps, dg], &device)
                            .slice_assign(span_s.clone(), k_scales),
                        v_packed: Tensor::zeros([np, nk, ps, dp], &device)
                            .slice_assign(span_p, v_packed),
                        v_scales: Tensor::zeros([np, nk, ps, dg], &device)
                            .slice_assign(span_s, v_scales),
                    }
                }
            });
        }

        let page_size = self.config.page_size;

        // Write the new K/V into page slots (one slice_assign per touched
        // page: 1 per steady-state decode step, seq/page_size per chunk),
        // then gather the full logical window and attend.
        let (k_full, v_full) = match self.arenas[layer].take().expect("arena initialized") {
            Arena::Fp { k: mut arena_k, v: mut arena_v } => {
                let mut written = 0;
                while written < seq {
                    let global = pos + written;
                    let slot = global % page_size;
                    let run = (page_size - slot).min(seq - written);
                    let phys = self.table[global / page_size];
                    let range = [phys..phys + 1, 0..n_kv, slot..slot + run, 0..head_dim];
                    arena_k =
                        arena_k.slice_assign(range.clone(), k.clone().narrow(2, written, run));
                    arena_v = arena_v.slice_assign(range, v.clone().narrow(2, written, run));
                    written += run;
                }
                // K3a: the single-token step attends in place — one
                // dispatch reading K/V through the page table; no gather,
                // no repeat_kv, no materialized scores. The slice_assign
                // above is already enqueued on the same stream, so the
                // kernel sees this token's K/V.
                if seq == 1 && window.is_none() {
                    let table = self.page_indices(pages);
                    if let Some(out) = crate::wgsl::try_decode_attention(
                        q.clone(),
                        arena_k.clone(),
                        arena_v.clone(),
                        table,
                        total,
                        0,
                        scale,
                    ) {
                        self.arenas[layer] = Some(Arena::Fp { k: arena_k, v: arena_v });
                        return out;
                    }
                }
                let k_full = self.gather_window(arena_k.clone(), pages, total);
                let v_full = self.gather_window(arena_v.clone(), pages, total);
                self.arenas[layer] = Some(Arena::Fp { k: arena_k, v: arena_v });
                (k_full, v_full)
            }
            Arena::Quant {
                mut k_packed,
                mut k_scales,
                mut v_packed,
                mut v_scales,
            } => {
                // Quantize the incoming chunk once, then place the packed
                // bytes + scales with the same page-slot arithmetic. Each
                // token's groups quantize independently, so nothing is ever
                // re-quantized.
                let (kq, ks) = kv_quantize(k);
                let (vq, vs) = kv_quantize(v);
                let dp = head_dim / 4;
                let dg = head_dim / KV_QUANT_GROUP;
                let mut written = 0;
                while written < seq {
                    let global = pos + written;
                    let slot = global % page_size;
                    let run = (page_size - slot).min(seq - written);
                    let phys = self.table[global / page_size];
                    let rp = [phys..phys + 1, 0..n_kv, slot..slot + run, 0..dp];
                    let rs = [phys..phys + 1, 0..n_kv, slot..slot + run, 0..dg];
                    k_packed =
                        k_packed.slice_assign(rp.clone(), kq.clone().narrow(2, written, run));
                    k_scales =
                        k_scales.slice_assign(rs.clone(), ks.clone().narrow(2, written, run));
                    v_packed = v_packed.slice_assign(rp, vq.clone().narrow(2, written, run));
                    v_scales = v_scales.slice_assign(rs, vs.clone().narrow(2, written, run));
                    written += run;
                }
                // K3c: the single-token step attends over the packed
                // arena in place, dequantizing in-register — the
                // materialized gather + kv_dequantize below become the
                // prefill/fallback path only.
                if seq == 1 && window.is_none() {
                    let table = self.page_indices(pages);
                    if let Some(out) = crate::wgsl::try_decode_attention_q8(
                        q.clone(),
                        crate::wgsl::QuantArena {
                            packed: k_packed.clone(),
                            scales: k_scales.clone(),
                        },
                        crate::wgsl::QuantArena {
                            packed: v_packed.clone(),
                            scales: v_scales.clone(),
                        },
                        table,
                        total,
                        scale,
                    ) {
                        self.arenas[layer] = Some(Arena::Quant {
                            k_packed,
                            k_scales,
                            v_packed,
                            v_scales,
                        });
                        return out;
                    }
                }
                let k_full = kv_dequantize(
                    self.gather_window_int(k_packed.clone(), pages, total),
                    self.gather_window(k_scales.clone(), pages, total),
                    head_dim,
                );
                let v_full = kv_dequantize(
                    self.gather_window_int(v_packed.clone(), pages, total),
                    self.gather_window(v_scales.clone(), pages, total),
                    head_dim,
                );
                self.arenas[layer] = Some(Arena::Quant {
                    k_packed,
                    k_scales,
                    v_packed,
                    v_scales,
                });
                (k_full, v_full)
            }
        };

        attend(q, k_full, v_full, pos, scale, window)
    }

    fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Rolls back the last `n` cached tokens, freeing trailing pages that
    /// become fully unused. K/V content of popped positions is left in the
    /// arena but is never read (writes always cover `seq_len..` densely).
    ///
    /// Sliding layers can only roll back while nothing has been evicted
    /// from their window: once eviction starts, the tokens a rollback
    /// would re-expose are gone, so the whole cache refuses (`0`) and the
    /// caller rebuilds from scratch — the same "prefix caching disables
    /// under sliding windows" rule HF applies. All-or-nothing: state is
    /// only mutated when the full rollback is possible.
    fn popn(&mut self, n: usize) -> usize {
        let n = n.min(self.seq_len);
        if n == 0 {
            return 0;
        }
        for w in self.layer_windows.iter().flatten() {
            if self.seq_len > w - 1 {
                return 0; // eviction already happened in this layer
            }
        }
        for slot in self.sliding.iter_mut() {
            if let Some((k, v)) = slot.take() {
                // Un-evicted invariant: stored length == seq_len, so the
                // rollback is a plain tail truncation.
                let len = k.dims()[2];
                let keep = len.saturating_sub(n);
                if keep > 0 {
                    *slot = Some((k.narrow(2, 0, keep), v.narrow(2, 0, keep)));
                }
            }
        }
        self.seq_len -= n;
        let keep = self.seq_len.div_ceil(self.config.page_size);
        while self.table.len() > keep {
            let page = self.table.pop().expect("table nonempty");
            self.allocator.free_page(page);
            self.table_dirty = true;
        }
        n
    }

    fn reset(&mut self) {
        self.table.clear();
        self.table_device = None;
        self.table_dirty = true;
        // The arenas are KEPT at their grown size (capacity reuse), so
        // the allocator must reset to the materialized count — ids past
        // it would be out-of-bounds slice_assigns on the next write.
        self.allocator.reset(self.materialized_pages);
        self.seq_len = 0;
        for slot in &mut self.sliding {
            *slot = None;
        }
        // Arena tensors are kept (capacity reuse); stale content is never
        // read because writes always cover seq_len.. densely.
    }

    fn pages_used(&self) -> Option<usize> {
        Some(self.table.len())
    }

    fn page_stats(&self) -> Option<PageStats> {
        Some(self.page_stats_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TB = burn::backend::NdArray<f32>;

    /// Deterministic non-degenerate K/V for token `i`: `[1, n_kv, 1, d]`.
    fn kv_tok(i: usize, n_kv: usize, d: usize) -> (Tensor<TB, 4>, Tensor<TB, 4>) {
        let dev = Default::default();
        let mk = |salt: usize| {
            let data: Vec<f32> = (0..n_kv * d)
                .map(|j| ((i * 7 + j * 3 + salt) % 13) as f32 / 13.0 - 0.5)
                .collect();
            Tensor::<TB, 4>::from_data(TensorData::new(data, [1, n_kv, 1, d]), &dev)
        };
        (mk(0), mk(5))
    }

    /// Deterministic query for token `i`: `[1, n_q, 1, d]`.
    fn q_tok(i: usize, n_q: usize, d: usize) -> Tensor<TB, 4> {
        let dev = Default::default();
        let data: Vec<f32> = (0..n_q * d)
            .map(|j| ((i * 11 + j * 5) % 17) as f32 / 17.0 - 0.5)
            .collect();
        Tensor::<TB, 4>::from_data(TensorData::new(data, [1, n_q, 1, d]), &dev)
    }

    fn assert_close4(a: Tensor<TB, 4>, b: Tensor<TB, 4>, what: &str) {
        let av: Vec<f32> = a.into_data().to_vec().unwrap();
        let bv: Vec<f32> = b.into_data().to_vec().unwrap();
        assert_eq!(av.len(), bv.len(), "{what}: shape");
        for (i, (x, y)) in av.iter().zip(bv.iter()).enumerate() {
            assert!((x - y).abs() < 1e-5, "{what}[{i}]: {x} vs {y}");
        }
    }

    /// The int8 pack/unpack must be integer-exact: craft values that are
    /// exact multiples of the group scale (absmax = 63.5 ⇒ scale = 0.5) so
    /// the quantize → dequantize round-trip reproduces the input bit-for-
    /// bit, across all four byte lanes including negative extremes.
    #[test]
    fn kv_quant_roundtrip_exact_on_grid_values() {
        let dev = Default::default();
        let d = 64usize;
        // q values sweep [-127, 127] across positions; x = q * 0.5.
        let data: Vec<f32> = (0..2 * 3 * d)
            .map(|i| {
                let q = ((i * 37) % 255) as i64 - 127; // covers all lanes
                q as f32 * 0.5
            })
            .collect();
        // Force absmax = 63.5 per 32-group: overwrite one slot per group.
        let mut data = data;
        for g in 0..(2 * 3 * d) / 32 {
            data[g * 32] = 63.5;
        }
        let x = Tensor::<TB, 4>::from_data(TensorData::new(data.clone(), [1, 2, 3, d]), &dev);
        let (packed, scales) = kv_quantize(x);
        let back: Vec<f32> = kv_dequantize(packed, scales, d)
            .into_data()
            .to_vec()
            .unwrap();
        for (i, (a, b)) in data.iter().zip(back.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "[{i}]: {a} vs {b} (must be exact)");
        }
    }

    /// On arbitrary values the dequantization error is bounded by half a
    /// quantization step (scale/2 = absmax/254) per element.
    #[test]
    fn kv_quant_error_bounded_by_half_step() {
        let dev = Default::default();
        let d = 64usize;
        let data: Vec<f32> = (0..1 * 2 * 5 * d)
            .map(|i| ((i * 7919) % 1000) as f32 / 250.0 - 2.0)
            .collect();
        let x = Tensor::<TB, 4>::from_data(TensorData::new(data.clone(), [1, 2, 5, d]), &dev);
        let (packed, scales) = kv_quantize(x);
        let back: Vec<f32> = kv_dequantize(packed, scales, d)
            .into_data()
            .to_vec()
            .unwrap();
        for (g, chunk) in data.chunks(32).enumerate() {
            let absmax = chunk.iter().fold(0f32, |m, v| m.max(v.abs()));
            let half_step = absmax / 254.0 + 1e-6;
            for (j, (a, b)) in chunk
                .iter()
                .zip(back[g * 32..g * 32 + 32].iter())
                .enumerate()
            {
                assert!(
                    (a - b).abs() <= half_step,
                    "group {g} elem {j}: |{a} - {b}| > {half_step}"
                );
            }
        }
    }

    /// A quantized paged cache must track the fp cache within int8 noise
    /// across prefill chunks, page boundaries, and decode steps — and its
    /// popn/page semantics are unchanged.
    #[test]
    fn quantized_paged_cache_matches_fp_within_tolerance() {
        let (n_kv, n_q, d) = (2usize, 4usize, 32usize);
        let mut cfg_q = CacheConfig::paged(64);
        cfg_q.quantize_kv = true;
        let cfg_f = CacheConfig::paged(64);
        let scale = 1.0 / (d as f64).sqrt() * 0.9;
        let mut fp = PagedKVCache::<TB>::new(1, cfg_f);
        let mut qn = PagedKVCache::<TB>::new(1, cfg_q);

        // Prefill 20 (crosses a page boundary at 16), then decode to 30.
        let ks: Vec<_> = (0..20).map(|i| kv_tok(i, n_kv, d)).collect();
        let k20 = Tensor::cat(ks.iter().map(|(k, _)| k.clone()).collect(), 2);
        let v20 = Tensor::cat(ks.iter().map(|(_, v)| v.clone()).collect(), 2);
        let q20 = Tensor::cat((0..20).map(|i| q_tok(i, n_q, d)).collect(), 2);
        let a = fp.attention_opts(0, q20.clone(), k20.clone(), v20.clone(), 0, scale, None);
        let b = qn.attention_opts(0, q20, k20, v20, 0, scale, None);
        let av: Vec<f32> = a.into_data().to_vec().unwrap();
        let bv: Vec<f32> = b.into_data().to_vec().unwrap();
        for (i, (x, y)) in av.iter().zip(bv.iter()).enumerate() {
            assert!((x - y).abs() < 1e-2, "prefill[{i}]: {x} vs {y}");
        }
        for i in 20..30 {
            let (k, v) = kv_tok(i, n_kv, d);
            let q = q_tok(i, n_q, d);
            let a = fp.attention_opts(0, q.clone(), k.clone(), v.clone(), i, scale, None);
            let b = qn.attention_opts(0, q, k, v, i, scale, None);
            let av: Vec<f32> = a.into_data().to_vec().unwrap();
            let bv: Vec<f32> = b.into_data().to_vec().unwrap();
            for (j, (x, y)) in av.iter().zip(bv.iter()).enumerate() {
                assert!((x - y).abs() < 1e-2, "decode {i}[{j}]: {x} vs {y}");
            }
        }
        assert_eq!(qn.pages_used(), fp.pages_used());
        assert_eq!(qn.popn(5), 5, "quantized rollback works (no sliding)");
        assert_eq!(qn.seq_len(), 25);
    }

    /// The divine invariant: a sliding layer (evict + offset-rebased mask)
    /// must produce the same attention output as a global layer that keeps
    /// everything and enforces the window purely via masking. Exercises
    /// pre-eviction, eviction, GQA expansion, and a prefill chunk longer
    /// than the window.
    #[test]
    fn sliding_layer_matches_masked_global() {
        let (n_kv, n_q, d, w) = (2usize, 4usize, 4usize, 5usize);
        let cfg = CacheConfig::paged(64);
        let scale = 1.0 / (d as f64).sqrt() * 0.9; // off-default: force mask path
        let mut global = PagedKVCache::<TB>::new(1, cfg);
        let mut sliding = PagedKVCache::<TB>::new_with_windows(1, cfg, vec![Some(w)]);

        // Prefill chunk of 7 (> window) at pos 0.
        let ks: Vec<_> = (0..7).map(|i| kv_tok(i, n_kv, d)).collect();
        let k7 = Tensor::cat(ks.iter().map(|(k, _)| k.clone()).collect(), 2);
        let v7 = Tensor::cat(ks.iter().map(|(_, v)| v.clone()).collect(), 2);
        let q7 = Tensor::cat((0..7).map(|i| q_tok(i, n_q, d)).collect(), 2);
        let a = global.attention_opts(0, q7.clone(), k7.clone(), v7.clone(), 0, scale, Some(w));
        let b = sliding.attention_opts(0, q7, k7, v7, 0, scale, Some(w));
        assert_close4(a, b, "prefill chunk");

        // Decode steps 7..14 — eviction active well past the window.
        for i in 7..14 {
            let (k, v) = kv_tok(i, n_kv, d);
            let q = q_tok(i, n_q, d);
            let a = global.attention_opts(0, q.clone(), k.clone(), v.clone(), i, scale, Some(w));
            let b = sliding.attention_opts(0, q, k, v, i, scale, Some(w));
            assert_close4(a, b, &format!("decode step {i}"));
        }

        // The sliding cache never touched the paged arena.
        assert_eq!(sliding.pages_used(), Some(0), "sliding layers use no pages");
        assert!(global.pages_used().unwrap() > 0);
    }

    /// Rollback before any eviction behaves exactly like a fresh cache
    /// replaying the truncated stream.
    #[test]
    fn sliding_popn_before_eviction_matches_replay() {
        let (n_kv, n_q, d, w) = (2usize, 2usize, 4usize, 8usize);
        let cfg = CacheConfig::paged(64);
        let scale = 0.4;
        let mut cache = PagedKVCache::<TB>::new_with_windows(1, cfg, vec![Some(w)]);
        for i in 0..4 {
            let (k, v) = kv_tok(i, n_kv, d);
            cache.attention_opts(0, q_tok(i, n_q, d), k, v, i, scale, Some(w));
        }
        assert_eq!(cache.popn(2), 2, "un-evicted rollback succeeds");
        assert_eq!(cache.seq_len(), 2);

        let mut fresh = PagedKVCache::<TB>::new_with_windows(1, cfg, vec![Some(w)]);
        for i in 0..2 {
            let (k, v) = kv_tok(i, n_kv, d);
            fresh.attention_opts(0, q_tok(i, n_q, d), k, v, i, scale, Some(w));
        }
        // Same next token after rollback vs replay must agree.
        let (k, v) = kv_tok(9, n_kv, d);
        let a = cache.attention_opts(0, q_tok(9, n_q, d), k.clone(), v.clone(), 2, scale, Some(w));
        let b = fresh.attention_opts(0, q_tok(9, n_q, d), k, v, 2, scale, Some(w));
        assert_close4(a, b, "post-rollback step");
    }

    /// Once a sliding layer has evicted, rollback must refuse entirely
    /// (all-or-nothing) and leave state untouched.
    #[test]
    fn sliding_popn_after_eviction_refuses() {
        let (n_kv, n_q, d, w) = (1usize, 1usize, 4usize, 4usize);
        let cfg = CacheConfig::paged(64);
        let mut cache = PagedKVCache::<TB>::new_with_windows(1, cfg, vec![Some(w)]);
        for i in 0..6 {
            let (k, v) = kv_tok(i, n_kv, d);
            cache.attention_opts(0, q_tok(i, n_q, d), k, v, i, 0.5, Some(w));
        }
        assert_eq!(cache.popn(1), 0, "evicted sliding layer refuses rollback");
        assert_eq!(cache.seq_len(), 6, "refused rollback leaves state intact");
        assert_eq!(cache.popn(0), 0);
    }

    #[test]
    fn allocator_alloc_in_order_and_exhaust() {
        let mut a = PageAllocator::new(3);
        assert_eq!(a.num_free(), 3);
        assert_eq!(a.alloc(), Some(0));
        assert_eq!(a.alloc(), Some(1));
        assert_eq!(a.alloc(), Some(2));
        assert_eq!(a.alloc(), None);
        assert_eq!(a.num_free(), 0);
    }

    #[test]
    fn allocator_free_and_realloc_lifo() {
        let mut a = PageAllocator::new(2);
        let p0 = a.alloc().unwrap();
        let p1 = a.alloc().unwrap();
        a.free_page(p1);
        a.free_page(p0);
        assert_eq!(a.num_free(), 2);
        // LIFO stack: most recently freed page comes back first.
        assert_eq!(a.alloc(), Some(p0));
        assert_eq!(a.alloc(), Some(p1));
    }

    #[test]
    fn allocator_reset_restores_all_pages() {
        let mut a = PageAllocator::new(4);
        a.alloc();
        a.alloc();
        a.reset(4);
        assert_eq!(a.num_free(), 4);
        assert_eq!(a.alloc(), Some(0));
    }

    #[test]
    fn cache_config_num_pages_rounds_up() {
        assert_eq!(CacheConfig::paged(16).num_pages(), 1);
        assert_eq!(CacheConfig::paged(17).num_pages(), 2);
        assert_eq!(CacheConfig::paged(1).num_pages(), 1);
    }

    // Page-table bookkeeping without touching tensors: PagedKVCache only
    // allocates arena tensors lazily inside `attention`, so popn/reset paths
    // can be exercised on the NdArray backend without a GPU.
    type TestBackend = burn::backend::NdArray<f32>;

    fn cache(max_seq_len: usize, page_size: usize) -> PagedKVCache<TestBackend> {
        PagedKVCache::new(
            2,
            CacheConfig {
                max_seq_len,
                page_size,
                kind: CacheKind::Paged,
                quantize_kv: false,
            },
        )
    }

    /// Simulates page-table growth without tensors (mirrors ensure_pages).
    fn grow(c: &mut PagedKVCache<TestBackend>, total: usize) {
        c.ensure_pages(total);
        c.seq_len = total;
    }

    #[test]
    fn popn_frees_only_fully_unused_pages() {
        let mut c = cache(64, 16);
        grow(&mut c, 40); // pages 0,1,2 (page 2 holds slots 32..39)
        assert_eq!(c.pages_used(), Some(3));
        assert_eq!(c.num_free_pages(), 1);

        c.popn(9); // seq 31 -> page 2 fully unused, freed
        assert_eq!(c.seq_len(), 31);
        assert_eq!(c.pages_used(), Some(2));
        assert_eq!(c.num_free_pages(), 2);

        c.popn(15); // seq 16 -> page 1 still needed (slots 16..31)
        assert_eq!(c.pages_used(), Some(1));
        c.popn(1); // seq 15 -> page 0 still needed
        assert_eq!(c.pages_used(), Some(1));

        c.popn(1000); // clamps to seq_len
        assert_eq!(c.seq_len(), 0);
        assert_eq!(c.pages_used(), Some(0));
        assert_eq!(c.num_free_pages(), 4);
    }

    #[test]
    fn popn_boundary_exact_page_edge() {
        let mut c = cache(64, 16);
        grow(&mut c, 32); // exactly 2 pages
        c.popn(16); // seq 16 -> 1 page
        assert_eq!(c.pages_used(), Some(1));
        assert_eq!(c.num_free_pages(), 3);
        c.popn(16);
        assert_eq!(c.pages_used(), Some(0));
        assert_eq!(c.num_free_pages(), 4);
    }

    #[test]
    fn regrowth_after_popn_reuses_freed_pages() {
        let mut c = cache(64, 16);
        grow(&mut c, 40);
        c.popn(9); // frees page for slots 32..48
        grow(&mut c, 33); // needs a page again -> reuses the freed one
        assert_eq!(c.pages_used(), Some(3));
        assert_eq!(c.num_free_pages(), 1);
    }

    #[test]
    fn reset_releases_all_pages() {
        let mut c = cache(64, 16);
        grow(&mut c, 40);
        c.reset();
        assert_eq!(c.seq_len(), 0);
        assert_eq!(c.pages_used(), Some(0));
        assert_eq!(c.num_free_pages(), 4);
    }

    // ---- GPU parity (ignored by default; run with `-- --ignored`) ----
    //
    // The NdArray<f32> tests above cannot see dtype-mismatch bugs: on the
    // f16 build the arena tensors are f16 while kv_quantize computes its
    // scales in f32. These run the same checks on the production
    // `combs_core::CombsBackend` of the current build.

    /// Deterministic K/V/Q builders for any backend.
    fn kv_tok_on<B: Backend>(
        dev: &B::Device,
        i: usize,
        n_kv: usize,
        d: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let mk = |salt: usize| {
            let data: Vec<f32> = (0..n_kv * d)
                .map(|j| ((i * 7 + j * 3 + salt) % 13) as f32 / 13.0 - 0.5)
                .collect();
            Tensor::<B, 4>::from_data(TensorData::new(data, [1, n_kv, 1, d]), dev)
        };
        (mk(0), mk(5))
    }

    fn q_tok_on<B: Backend>(dev: &B::Device, i: usize, n_q: usize, d: usize) -> Tensor<B, 4> {
        let data: Vec<f32> = (0..n_q * d)
            .map(|j| ((i * 11 + j * 5) % 17) as f32 / 17.0 - 0.5)
            .collect();
        Tensor::<B, 4>::from_data(TensorData::new(data, [1, n_q, 1, d]), dev)
    }

    /// Quantize→dequantize on the given backend must return finite values
    /// near the input — the f16-build corruption shows up as NaN/garbage.
    fn quant_roundtrip_on<B: Backend>(dev: &B::Device) {
        let d = 64usize;
        let data: Vec<f32> = (0..2 * d)
            .map(|j| ((j * 5) % 251) as f32 / 251.0 - 0.5)
            .collect();
        let x = Tensor::<B, 4>::from_data(TensorData::new(data.clone(), [1, 2, 1, d]), dev);
        let (packed, scales) = kv_quantize(x);
        let y = kv_dequantize(packed, scales, d);
        let yv: Vec<f32> = y.into_data().convert::<f32>().to_vec().unwrap();
        for (i, (orig, got)) in data.iter().zip(yv.iter()).enumerate() {
            assert!(got.is_finite(), "dequant[{i}] not finite: {got}");
            assert!(
                (orig - got).abs() < 0.01,
                "dequant[{i}]: {orig} vs {got}"
            );
        }
    }

    /// The fp-vs-quantized attend parity check on the given backend.
    fn quant_parity_on<B: Backend>(dev: &B::Device, tol: f32) {
        let (n_kv, n_q, d) = (2usize, 4usize, 32usize);
        let mut cfg_q = CacheConfig::paged(64);
        cfg_q.quantize_kv = true;
        let cfg_f = CacheConfig::paged(64);
        let scale = 1.0 / (d as f64).sqrt() * 0.9;
        let mut fp = PagedKVCache::<B>::new(1, cfg_f);
        let mut qn = PagedKVCache::<B>::new(1, cfg_q);
        for i in 0..24 {
            let (k, v) = kv_tok_on::<B>(dev, i, n_kv, d);
            let q = q_tok_on::<B>(dev, i, n_q, d);
            let a = fp.attention_opts(0, q.clone(), k.clone(), v.clone(), i, scale, None);
            let b = qn.attention_opts(0, q, k, v, i, scale, None);
            let av: Vec<f32> = a.into_data().convert::<f32>().to_vec().unwrap();
            let bv: Vec<f32> = b.into_data().convert::<f32>().to_vec().unwrap();
            for (j, (x, y)) in av.iter().zip(bv.iter()).enumerate() {
                assert!(y.is_finite(), "step {i}[{j}] not finite: {y}");
                assert!((x - y).abs() < tol, "step {i}[{j}]: {x} vs {y}");
            }
        }
    }

    /// Growth parity: a cache that starts small and doubles must produce
    /// bitwise-identical attention outputs to one born at full size —
    /// growth is allocation policy, never arithmetic. Uses page_size 2 so
    /// the 32-page floor is crossed within cheap CPU steps, covering two
    /// doubling events and the popn/regrow interaction.
    fn growth_parity_on(quantize: bool, tol: Option<f32>) {
        let (n_kv, n_q, d) = (2usize, 2usize, 32usize);
        let scale = 0.3;
        let config = CacheConfig {
            max_seq_len: 512,
            page_size: 2,
            kind: CacheKind::Paged,
            quantize_kv: quantize,
        };
        // 256-page capacity, 32-page floor: growth at 33, 65, 129 pages.
        let mut growing = PagedKVCache::<TestBackend>::new(1, config);
        let mut control = PagedKVCache::<TestBackend>::new(1, config);
        control.materialized_pages = config.num_pages();
        control.allocator = PageAllocator::new(config.num_pages());

        for i in 0..300 {
            let (k, v) = kv_tok(i, n_kv, d);
            let q = q_tok(i, n_q, d);
            let a = growing.attention_opts(0, q.clone(), k.clone(), v.clone(), i, scale, None);
            let b = control.attention_opts(0, q, k, v, i, scale, None);
            match tol {
                None => assert_close4(a, b, &format!("step {i}")),
                Some(t) => {
                    let av: Vec<f32> = a.into_data().to_vec().unwrap();
                    let bv: Vec<f32> = b.into_data().to_vec().unwrap();
                    for (j, (x, y)) in av.iter().zip(bv.iter()).enumerate() {
                        assert!((x - y).abs() < t, "step {i}[{j}]: {x} vs {y}");
                    }
                }
            }
            if i == 100 {
                // Roll back across nothing special, regrow — the freed
                // ids stay inside the materialized span.
                assert_eq!(growing.popn(3), 3);
                assert_eq!(control.popn(3), 3);
                let replay: Vec<_> = (98..101).collect();
                for &r in &replay {
                    let (k, v) = kv_tok(r, n_kv, d);
                    let q = q_tok(r, n_q, d);
                    let a = growing.attention_opts(0, q.clone(), k.clone(), v.clone(), r, scale, None);
                    let b = control.attention_opts(0, q, k, v, r, scale, None);
                    match tol {
                        None => assert_close4(a, b, &format!("replay {r}")),
                        Some(t) => {
                            let av: Vec<f32> = a.into_data().to_vec().unwrap();
                            let bv: Vec<f32> = b.into_data().to_vec().unwrap();
                            for (j, (x, y)) in av.iter().zip(bv.iter()).enumerate() {
                                assert!((x - y).abs() < t, "replay {r}[{j}]: {x} vs {y}");
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(
            growing.page_stats_inner().pages_materialized,
            256,
            "300 tokens at page_size 2 crosses 128 pages -> doubled to 256"
        );
        assert_eq!(control.page_stats_inner().pages_materialized, 256);
    }

    #[test]
    fn grown_cache_matches_full_size_control_bitwise() {
        growth_parity_on(false, None);
    }

    #[test]
    fn grown_quant_cache_matches_full_size_control() {
        // int8 KV round-trips are not bitwise across different write
        // batching, but growth itself must not add error: same tolerance
        // class as the quant parity test.
        growth_parity_on(true, Some(1e-5));
    }

    /// The doubling sequence, popn interaction, single-event chunk jumps
    /// and the reset landmine, on the allocator/table alone (no tensors).
    #[test]
    fn arena_growth_doubles_and_reset_keeps_the_materialized_size() {
        let mut c = cache(4096, 16); // 256-page capacity, 32-page floor
        assert_eq!(c.page_stats_inner().pages_materialized, 32);
        assert_eq!(c.num_free_pages(), 256);

        grow(&mut c, 32 * 16); // exactly the floor: no growth
        assert_eq!(c.page_stats_inner().pages_materialized, 32);
        grow(&mut c, 32 * 16 + 1); // one past: double
        assert_eq!(c.page_stats_inner().pages_materialized, 64);
        grow(&mut c, 200 * 16); // a chunk jump: straight to 200, doubled = 128 < 200
        assert_eq!(c.page_stats_inner().pages_materialized, 200);
        assert_eq!(c.num_free_pages(), 256 - 200);

        c.reset();
        assert_eq!(
            c.page_stats_inner().pages_materialized,
            200,
            "arenas are kept at their grown size across reset"
        );
        assert_eq!(c.num_free_pages(), 256, "reset frees every page");
        grow(&mut c, 200 * 16); // regrow to exactly the kept size: no event
        assert_eq!(c.page_stats_inner().pages_materialized, 200);
        grow(&mut c, 256 * 16); // to capacity
        assert_eq!(c.page_stats_inner().pages_materialized, 256);
        assert_eq!(c.num_free_pages(), 0);
    }

    /// The hard cap still rules: growth never mints capacity.
    #[test]
    #[should_panic(expected = "paged cache capacity exceeded")]
    fn growth_never_exceeds_the_configured_capacity() {
        let mut c = cache(64, 16);
        for i in 0..=64 {
            let (k, v) = kv_tok(i, 2, 4);
            let q = q_tok(i, 2, 4);
            // The 65th append asks for total = 65 > max_seq_len = 64.
            c.attention_opts(0, q, k, v, i, 0.3, None);
        }
    }

    #[test]
    #[ignore = "gpu"]
    fn kv_quant_roundtrip_on_production_backend() {
        quant_roundtrip_on::<combs_core::CombsBackend>(&Default::default());
    }

    #[test]
    #[ignore = "gpu"]
    fn quantized_paged_cache_matches_fp_on_production_backend() {
        quant_parity_on::<combs_core::CombsBackend>(&Default::default(), 5e-2);
    }

    /// The K3a interleave proof: paged decode (the in-place kernel on the
    /// production backend) against the contiguous baseline through
    /// prefill → decode → popn → regrow. The popn leg replays a fresh
    /// contiguous cache, since the baseline cannot roll back. Also the
    /// stress test for the fused custom op's read-only arena inputs: the
    /// same arenas are registered step after step and must survive every
    /// registration intact.
    fn paged_decode_parity_on<B: Backend>(dev: &B::Device, tol: f32) {
        let (n_kv, n_q, d) = (2usize, 4usize, 64usize);
        let scale = 1.0 / (d as f64).sqrt();
        let cmp = |a: Tensor<B, 4>, b: Tensor<B, 4>, what: &str| {
            let av: Vec<f32> = a.into_data().convert::<f32>().to_vec().unwrap();
            let bv: Vec<f32> = b.into_data().convert::<f32>().to_vec().unwrap();
            for (j, (x, y)) in av.iter().zip(bv.iter()).enumerate() {
                assert!(x.is_finite(), "{what}[{j}] not finite: {x}");
                assert!((x - y).abs() < tol, "{what}[{j}]: {x} vs {y}");
            }
        };
        let feed = |cache: &mut dyn KVCache<B>, from: usize, to: usize| {
            // One chunk covering [from, to) — the prefill shape.
            let (ks, vs): (Vec<_>, Vec<_>) =
                (from..to).map(|i| kv_tok_on::<B>(dev, i, n_kv, d)).unzip();
            let k = Tensor::cat(ks, 2);
            let v = Tensor::cat(vs, 2);
            let (qs, _): (Vec<_>, Vec<_>) =
                (from..to).map(|i| (q_tok_on::<B>(dev, i, n_q, d), ())).unzip();
            let q = Tensor::cat(qs, 2);
            cache.attention_opts(0, q, k, v, from, scale, None)
        };

        let mut paged = PagedKVCache::<B>::new(1, CacheConfig::paged(64));
        let mut cont = ContiguousKVCache::<B>::new(1);

        // Prefill 21 (mid-page), then decode to 28 — every decode step on
        // the paged side is one kernel dispatch.
        cmp(
            feed(&mut paged, 0, 21),
            feed(&mut cont, 0, 21),
            "prefill",
        );
        for i in 21..28 {
            let (k, v) = kv_tok_on::<B>(dev, i, n_kv, d);
            let q = q_tok_on::<B>(dev, i, n_q, d);
            let a = paged.attention_opts(0, q.clone(), k.clone(), v.clone(), i, scale, None);
            let b = cont.attention_opts(0, q, k, v, i, scale, None);
            cmp(a, b, &format!("decode {i}"));
        }

        // Roll back 3, then regrow over fresh replay.
        assert_eq!(paged.popn(3), 3, "paged cache rolls back");
        let mut replay = ContiguousKVCache::<B>::new(1);
        let _ = feed(&mut replay, 0, 25);
        for i in 25..28 {
            let (k, v) = kv_tok_on::<B>(dev, i, n_kv, d);
            let q = q_tok_on::<B>(dev, i, n_q, d);
            let a = paged.attention_opts(0, q.clone(), k.clone(), v.clone(), i, scale, None);
            let b = replay.attention_opts(0, q, k, v, i, scale, None);
            cmp(a, b, &format!("regrow {i}"));
        }
    }

    #[test]
    #[ignore = "gpu"]
    fn paged_decode_matches_contiguous_on_production_backend() {
        paged_decode_parity_on::<combs_core::CombsBackend>(&Default::default(), 1e-3);
    }

    /// The sliding invariant on the production backend: a sliding layer
    /// (K3b decode kernel over the rolling store) must match a global
    /// layer that keeps everything and enforces the window purely via
    /// masking (the materialized path — the K3a kernel excludes windowed
    /// calls). Same structure as `sliding_layer_matches_masked_global`.
    fn sliding_parity_on<B: Backend>(dev: &B::Device, tol: f32) {
        let (n_kv, n_q, d, w) = (2usize, 4usize, 64usize, 5usize);
        let cfg = CacheConfig::paged(64);
        let scale = 1.0 / (d as f64).sqrt() * 0.9;
        let mut global = PagedKVCache::<B>::new(1, cfg);
        let mut sliding = PagedKVCache::<B>::new_with_windows(1, cfg, vec![Some(w)]);
        let cmp = |a: Tensor<B, 4>, b: Tensor<B, 4>, what: &str| {
            let av: Vec<f32> = a.into_data().convert::<f32>().to_vec().unwrap();
            let bv: Vec<f32> = b.into_data().convert::<f32>().to_vec().unwrap();
            for (j, (x, y)) in av.iter().zip(bv.iter()).enumerate() {
                assert!(y.is_finite(), "{what}[{j}] not finite: {y}");
                assert!((x - y).abs() < tol, "{what}[{j}]: {x} vs {y}");
            }
        };

        let ks: Vec<_> = (0..7).map(|i| kv_tok_on::<B>(dev, i, n_kv, d)).collect();
        let k7 = Tensor::cat(ks.iter().map(|(k, _)| k.clone()).collect(), 2);
        let v7 = Tensor::cat(ks.iter().map(|(_, v)| v.clone()).collect(), 2);
        let q7 = Tensor::cat((0..7).map(|i| q_tok_on::<B>(dev, i, n_q, d)).collect(), 2);
        let a = global.attention_opts(0, q7.clone(), k7.clone(), v7.clone(), 0, scale, Some(w));
        let b = sliding.attention_opts(0, q7, k7, v7, 0, scale, Some(w));
        cmp(a, b, "prefill chunk");

        for i in 7..14 {
            let (k, v) = kv_tok_on::<B>(dev, i, n_kv, d);
            let q = q_tok_on::<B>(dev, i, n_q, d);
            let a = global.attention_opts(0, q.clone(), k.clone(), v.clone(), i, scale, Some(w));
            let b = sliding.attention_opts(0, q, k, v, i, scale, Some(w));
            cmp(a, b, &format!("decode step {i}"));
        }
    }

    #[test]
    #[ignore = "gpu"]
    fn sliding_layer_matches_masked_global_on_production_backend() {
        sliding_parity_on::<combs_core::CombsBackend>(&Default::default(), 1e-3);
    }

    /// Arena growth under the fused decode kernel: a growing cache and a
    /// full-size control run the same kernels over the same values — the
    /// arena's dim-0 is invisible to the arithmetic, so outputs must stay
    /// within readback tolerance across two doubling events on the
    /// production backend (the guard that matters for the wasm path too).
    fn growth_parity_gpu_on<B: Backend>(dev: &B::Device, tol: f32) {
        let (n_kv, n_q, d) = (2usize, 4usize, 64usize);
        let scale = 1.0 / (d as f64).sqrt();
        let config = CacheConfig {
            max_seq_len: 512,
            page_size: 2,
            kind: CacheKind::Paged,
            quantize_kv: false,
        };
        let mut growing = PagedKVCache::<B>::new(1, config);
        let mut control = PagedKVCache::<B>::new(1, config);
        control.materialized_pages = config.num_pages();
        control.allocator = PageAllocator::new(config.num_pages());
        for i in 0..140 {
            let (k, v) = kv_tok_on::<B>(dev, i, n_kv, d);
            let q = q_tok_on::<B>(dev, i, n_q, d);
            let a = growing.attention_opts(0, q.clone(), k.clone(), v.clone(), i, scale, None);
            let b = control.attention_opts(0, q, k, v, i, scale, None);
            let av: Vec<f32> = a.into_data().convert::<f32>().to_vec().unwrap();
            let bv: Vec<f32> = b.into_data().convert::<f32>().to_vec().unwrap();
            for (j, (x, y)) in av.iter().zip(bv.iter()).enumerate() {
                assert!(x.is_finite(), "step {i}[{j}] not finite: {x}");
                assert!((x - y).abs() < tol, "step {i}[{j}]: {x} vs {y}");
            }
        }
        assert!(
            growing.page_stats_inner().pages_materialized >= 128,
            "140 tokens at page_size 2 must have crossed two growth events"
        );
    }

    #[test]
    #[ignore = "gpu"]
    fn grown_cache_matches_control_under_the_decode_kernel() {
        growth_parity_gpu_on::<combs_core::CombsBackend>(&Default::default(), 1e-5);
    }
}
