//! Looped graph algorithms over a CSR graph: spreading activation
//! (personalized-PageRank-shaped, damped, convergent) and decayed k-hop
//! reach. The CPU path is the reference; the `gpu` feature adds the
//! same activation loop on the estate's Burn/wgpu backend using a dense
//! row-stochastic adjacency (dense is deliberate: memory is N², so the
//! GPU path caps N — callers fall back to CPU above the cap).

use serde::{Deserialize, Serialize};

/// Compressed sparse row over u32 node ids (out-edges).
pub struct Graph {
    pub n: usize,
    row_ptr: Vec<u32>,
    col: Vec<u32>,
}

impl Graph {
    pub fn from_edges(n: usize, edges: &[(u32, u32)]) -> Self {
        let mut deg = vec![0u32; n];
        for &(u, v) in edges {
            // Both ends must be in range — counting a slot for an edge
            // the fill loop later skips would leave a phantom edge to 0.
            if (u as usize) < n && (v as usize) < n {
                deg[u as usize] += 1;
            }
        }
        let mut row_ptr = vec![0u32; n + 1];
        for i in 0..n {
            row_ptr[i + 1] = row_ptr[i] + deg[i];
        }
        let mut col = vec![0u32; edges.len()];
        let mut next = row_ptr.clone();
        for &(u, v) in edges {
            if (u as usize) < n && (v as usize) < n {
                col[next[u as usize] as usize] = v;
                next[u as usize] += 1;
            }
        }
        Self { n, row_ptr, col }
    }

    pub fn out_degree(&self, u: usize) -> usize {
        (self.row_ptr[u + 1] - self.row_ptr[u]) as usize
    }

    pub fn out_edges(&self, u: usize) -> &[u32] {
        &self.col[self.row_ptr[u] as usize..self.row_ptr[u + 1] as usize]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivateParams {
    /// Damping: fraction of mass that flows along edges each step; the
    /// remainder returns to the seeds (personalization).
    pub damping: f32,
    /// Hard iteration cap.
    pub max_steps: usize,
    /// L1 convergence threshold.
    pub eps: f32,
}

impl Default for ActivateParams {
    fn default() -> Self {
        // Convergence is geometric at rate `damping`: 0.85^k < 1e-5
        // needs k ≈ 71, so a 30-step cap could never converge on
        // cycles — 100 keeps the loop honest and still trivial.
        Self { damping: 0.85, max_steps: 100, eps: 1e-5 }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ActivateResult {
    pub scores: Vec<f32>,
    pub steps_run: usize,
    pub converged: bool,
}

fn seed_vector(n: usize, seeds: &[u32]) -> Vec<f32> {
    let mut s = vec![0.0f32; n];
    if seeds.is_empty() {
        return s;
    }
    let w = 1.0 / seeds.len() as f32;
    for &i in seeds {
        if (i as usize) < n {
            s[i as usize] += w;
        }
    }
    s
}

/// Spreading activation: x_{k+1} = (1-d)·seed + d·(P^T x_k), with P the
/// out-degree row-stochastic transition and dangling mass returned to
/// the seeds. Converges on L1 delta < eps.
pub fn activate_cpu(graph: &Graph, seeds: &[u32], params: &ActivateParams) -> ActivateResult {
    let n = graph.n;
    let seed = seed_vector(n, seeds);
    let mut x = seed.clone();
    let mut steps_run = 0;
    let mut converged = false;
    for _ in 0..params.max_steps {
        steps_run += 1;
        let mut next = vec![0.0f32; n];
        let mut dangling = 0.0f32;
        for u in 0..n {
            let deg = graph.out_degree(u);
            if deg == 0 {
                dangling += x[u];
                continue;
            }
            let share = x[u] / deg as f32;
            for &v in graph.out_edges(u) {
                next[v as usize] += share;
            }
        }
        let mut delta = 0.0f32;
        for i in 0..n {
            let v = (1.0 - params.damping) * seed[i]
                + params.damping * (next[i] + dangling * seed[i]);
            delta += (v - x[i]).abs();
            x[i] = v;
        }
        if delta < params.eps {
            converged = true;
            break;
        }
    }
    ActivateResult { scores: x, steps_run, converged }
}

/// Decayed k-hop reach: frontier BFS where a node reached at depth d
/// scores decay^d (best depth wins). Seeds score 1.
pub fn khop_cpu(graph: &Graph, seeds: &[u32], max_depth: usize, decay: f32) -> Vec<f32> {
    let n = graph.n;
    let mut score = vec![0.0f32; n];
    let mut frontier: Vec<u32> = Vec::new();
    for &s in seeds {
        if (s as usize) < n && score[s as usize] == 0.0 {
            score[s as usize] = 1.0;
            frontier.push(s);
        }
    }
    let mut gain = 1.0f32;
    for _ in 0..max_depth {
        gain *= decay;
        let mut next: Vec<u32> = Vec::new();
        for &u in &frontier {
            for &v in graph.out_edges(u as usize) {
                if score[v as usize] == 0.0 {
                    score[v as usize] = gain;
                    next.push(v);
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    score
}

#[cfg(feature = "gpu")]
pub mod gpu {
    //! The same activation loop on Burn/wgpu. Dense P^T (N×N) built
    //! row-stochastic on the host; the loop runs device-side with one
    //! scalar readback per step for the convergence check.

    use super::{ActivateParams, ActivateResult, Graph};
    use burn::tensor::{Tensor, TensorData};

    /// Dense-adjacency cap: N² f32 at 4096 is 64 MB device-side.
    pub const MAX_DENSE_N: usize = 4096;

    pub fn activate_gpu(
        graph: &Graph,
        seeds: &[u32],
        params: &ActivateParams,
    ) -> Result<ActivateResult, String> {
        type B = combs_core::CombsBackendF32;
        let n = graph.n;
        if n == 0 {
            return Ok(ActivateResult { scores: vec![], steps_run: 0, converged: true });
        }
        if n > MAX_DENSE_N {
            return Err(format!("gpu dense path capped at {MAX_DENSE_N} nodes, got {n}"));
        }
        let device = combs_core::init_device();

        // P^T dense: pt[v][u] = 1/deg(u) when u→v. Dangling columns are
        // handled by folding their mass back to the seeds in the loop.
        let mut pt = vec![0.0f32; n * n];
        let mut dangling_mask = vec![0.0f32; n];
        for u in 0..n {
            let deg = graph.out_degree(u);
            if deg == 0 {
                dangling_mask[u] = 1.0;
                continue;
            }
            let share = 1.0 / deg as f32;
            for &v in graph.out_edges(u) {
                pt[(v as usize) * n + u] = share;
            }
        }
        let seed = super::seed_vector(n, seeds);
        let pt_t = Tensor::<B, 2>::from_data(TensorData::new(pt, [n, n]), &device);
        let seed_t = Tensor::<B, 2>::from_data(TensorData::new(seed.clone(), [n, 1]), &device);
        let dang_t = Tensor::<B, 2>::from_data(TensorData::new(dangling_mask, [1, n]), &device);
        let mut x = seed_t.clone();

        let mut steps_run = 0;
        let mut converged = false;
        for _ in 0..params.max_steps {
            steps_run += 1;
            let flowed = pt_t.clone().matmul(x.clone());
            let dangling = dang_t.clone().matmul(x.clone()); // [1,1]
            let next = seed_t.clone() * (1.0 - params.damping)
                + (flowed + seed_t.clone() * dangling) * params.damping;
            let delta: f32 = (next.clone() - x.clone())
                .abs()
                .sum()
                .into_scalar();
            x = next;
            if delta < params.eps {
                converged = true;
                break;
            }
        }
        let scores: Vec<f32> = x
            .into_data()
            .to_vec()
            .map_err(|e| format!("readback: {e:?}"))?;
        Ok(ActivateResult { scores, steps_run, converged })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(n: usize) -> Graph {
        let edges: Vec<(u32, u32)> = (0..n as u32).map(|i| (i, (i + 1) % n as u32)).collect();
        Graph::from_edges(n, &edges)
    }

    #[test]
    fn activation_conserves_mass_and_converges() {
        let g = ring(8);
        let res = activate_cpu(&g, &[0], &ActivateParams::default());
        assert!(res.converged, "ring should converge within the default cap");
        let total: f32 = res.scores.iter().sum();
        assert!((total - 1.0).abs() < 1e-3, "mass {total}");
    }

    #[test]
    fn star_center_dominates() {
        // Star: leaves all point at the hub; hub points back at leaf 1.
        let edges = vec![(1u32, 0u32), (2, 0), (3, 0), (4, 0), (0, 1)];
        let g = Graph::from_edges(5, &edges);
        let res = activate_cpu(&g, &[2], &ActivateParams::default());
        assert!(res.scores[0] > res.scores[3], "hub should outrank a far leaf");
    }

    #[test]
    fn khop_decays_by_depth() {
        let g = ring(6);
        let s = khop_cpu(&g, &[0], 3, 0.5);
        assert_eq!(s[0], 1.0);
        assert_eq!(s[1], 0.5);
        assert_eq!(s[2], 0.25);
        assert_eq!(s[3], 0.125);
        assert_eq!(s[4], 0.0);
    }

    #[test]
    fn out_of_range_edges_are_dropped_not_phantomed() {
        let g = Graph::from_edges(2, &[(0, 1), (0, 99), (99, 1)]);
        assert_eq!(g.out_degree(0), 1);
        assert_eq!(g.out_edges(0), &[1]);
    }

    #[test]
    fn dangling_mass_returns_to_seeds() {
        // 0→1, 1 dangles. Without the dangling fold, mass would leak.
        let g = Graph::from_edges(2, &[(0, 1)]);
        let res = activate_cpu(&g, &[0], &ActivateParams::default());
        let total: f32 = res.scores.iter().sum();
        assert!((total - 1.0).abs() < 1e-3, "mass {total}");
    }
}
