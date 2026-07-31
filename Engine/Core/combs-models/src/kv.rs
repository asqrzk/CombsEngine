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
) -> Tensor<B, 4> {
    let device = q.device();
    let [_, n_q, seq, d] = q.dims();
    let [_, n_kv, total, _] = k.dims();
    let n_rep = n_q / n_kv;
    let k = repeat_kv(k, n_rep);
    let v = repeat_kv(v, n_rep);

    let default_scale = 1.0 / (d as f64).sqrt();
    if flash_enabled() && (scale - default_scale).abs() < 1e-12 {
        return burn::tensor::module::attention(
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
    }

    // Reference path: explicit scores, causal mask, softmax, P@V.
    let scores = q.matmul(k.transpose()).mul_scalar(scale);
    let scores = if seq > 1 {
        // Causal mask: query at global position p attends keys <= p.
        let q_pos =
            Tensor::<B, 1, Int>::arange((pos as i64)..((pos + seq) as i64), &device)
                .reshape([seq, 1]);
        let k_pos = Tensor::<B, 1, Int>::arange(0..(total as i64), &device).reshape([1, total]);
        let forbidden: Tensor<B, 2, Bool> = k_pos.greater(q_pos);
        let mask = forbidden
            .unsqueeze_dims::<4>(&[0, 1])
            .expand([1, n_q, seq, total]);
        scores.mask_fill(mask, -1e30f32)
    } else {
        scores // single query attends to everything cached
    };

    // `safe_matmul` for the P@V product: with a >= 512-token window this
    // shape (M = seq, K = total) enters the broken wgpu/Metal matmul region.
    safe_matmul(softmax(scores, 3), v)
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
    fn attention(
        &mut self,
        layer: usize,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        pos: usize,
        scale: f64,
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
        let out = attend(q, k_full.clone(), v_full.clone(), pos, scale);
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
    device: Option<Device<B>>,
}

impl<B: Backend> PagedKVCache<B> {
    /// Creates an empty paged cache for `num_layers` layers. Arena tensors
    /// are allocated lazily on first use of each layer.
    pub fn new(num_layers: usize, config: CacheConfig) -> Self {
        PagedKVCache {
            allocator: PageAllocator::new(config.num_pages()),
            config,
            table: Vec::new(),
            seq_len: 0,
            arenas: (0..num_layers).map(|_| None).collect(),
            device: None,
        }
    }

    /// Number of free pages in the arena.
    pub fn num_free_pages(&self) -> usize {
        self.allocator.num_free()
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
    fn attention(
        &mut self,
        layer: usize,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        pos: usize,
        scale: f64,
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

        attend(q, k_full, v_full, pos, scale)
    }

    fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Rolls back the last `n` cached tokens, freeing trailing pages that
    /// become fully unused. K/V content of popped positions is left in the
    /// arena but is never read (writes always cover `seq_len..` densely).
    fn popn(&mut self, n: usize) -> usize {
        let n = n.min(self.seq_len);
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
        // Arena tensors are kept (capacity reuse); stale content is never
        // read because writes always cover seq_len.. densely.
    }

    fn pages_used(&self) -> Option<usize> {
        Some(self.table.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
