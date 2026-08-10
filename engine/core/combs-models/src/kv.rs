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
        }
    }

    /// Contiguous (baseline) cache.
    pub fn contiguous(max_seq_len: usize) -> Self {
        CacheConfig {
            max_seq_len,
            page_size: Self::DEFAULT_PAGE_SIZE,
            kind: CacheKind::Contiguous,
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
    /// Pages still available in this cache's arena.
    pub pages_free: usize,
    /// Total pages in the arena (`pages_used + pages_free` when healthy).
    pub num_pages: usize,
    /// Tokens per page.
    pub page_size: usize,
    /// Cached sequence length in tokens.
    pub seq_len: usize,
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
fn attend<B: Backend>(
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
pub struct PagedKVCache<B: Backend> {
    config: CacheConfig,
    allocator: PageAllocator,
    /// Page table: logical page index -> physical page id (single sequence).
    table: Vec<usize>,
    seq_len: usize,
    arenas: Vec<Option<(Tensor<B, 4>, Tensor<B, 4>)>>,
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
        PagedKVCache {
            allocator: PageAllocator::new(config.num_pages()),
            config,
            table: Vec::new(),
            seq_len: 0,
            arenas: (0..num_layers).map(|_| None).collect(),
            layer_windows: windows,
            sliding: (0..num_layers).map(|_| None).collect(),
            device: None,
        }
    }

    /// Number of free pages in the arena.
    pub fn num_free_pages(&self) -> usize {
        self.allocator.num_free()
    }

    /// Page-table snapshot (see [`KVCache::page_stats`]).
    pub fn page_stats_inner(&self) -> PageStats {
        PageStats {
            pages_used: self.table.len(),
            pages_free: self.allocator.num_free(),
            num_pages: self.config.num_pages(),
            page_size: self.config.page_size,
            seq_len: self.seq_len,
        }
    }

    /// Ensures the page table covers `total` positions.
    fn ensure_pages(&mut self, total: usize) -> usize {
        let pages_needed = total.div_ceil(self.config.page_size);
        while self.table.len() < pages_needed {
            let page = self
                .allocator
                .alloc()
                .expect("page allocator exhausted (max_seq_len exceeded)");
            self.table.push(page);
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
        let out = attend(
            q,
            k_full.clone(),
            v_full.clone(),
            pos - kv_offset,
            scale,
            Some(w),
        );
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

    /// Gathers the first `pages` page-table entries of `arena`
    /// (`[num_pages, n_kv, page_size, head_dim]`) into a contiguous
    /// `[1, n_kv, total, head_dim]` window.
    fn gather_window(
        &self,
        arena: Tensor<B, 4>,
        pages: usize,
        total: usize,
    ) -> Tensor<B, 4> {
        let [_, n_kv, page_size, head_dim] = arena.dims();
        let ids: Vec<i32> = self.table[..pages].iter().map(|&p| p as i32).collect();
        let device = self
            .device
            .as_ref()
            .expect("device set on first attention call");
        let indices = Tensor::<B, 1, Int>::from_data(TensorData::new(ids, [pages]), device);
        arena
            .select(0, indices) // [pages, n_kv, page_size, head_dim]
            .swap_dims(0, 1) // [n_kv, pages, page_size, head_dim]
            .reshape([1, n_kv, pages * page_size, head_dim])
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
        if self.arenas[layer].is_none() {
            let device = k.device();
            let shape = [self.config.num_pages(), n_kv, self.config.page_size, head_dim];
            self.arenas[layer] = Some((
                Tensor::zeros(shape, &device),
                Tensor::zeros(shape, &device),
            ));
        }

        let pages = self.ensure_pages(total);
        let page_size = self.config.page_size;

        // Write the new K/V into page slots: one slice_assign per touched
        // page (1 per steady-state decode step, seq/page_size per chunk).
        let (mut arena_k, mut arena_v) = self.arenas[layer].take().expect("arena initialized");
        let mut written = 0;
        while written < seq {
            let global = pos + written;
            let slot = global % page_size;
            let run = (page_size - slot).min(seq - written);
            let phys = self.table[global / page_size];
            let range = [phys..phys + 1, 0..n_kv, slot..slot + run, 0..head_dim];
            arena_k = arena_k.slice_assign(range.clone(), k.clone().narrow(2, written, run));
            arena_v = arena_v.slice_assign(range, v.clone().narrow(2, written, run));
            written += run;
        }

        let k_full = self.gather_window(arena_k.clone(), pages, total);
        let v_full = self.gather_window(arena_v.clone(), pages, total);
        self.arenas[layer] = Some((arena_k, arena_v));

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
        }
        n
    }

    fn reset(&mut self) {
        self.table.clear();
        self.allocator.reset(self.config.num_pages());
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
}
