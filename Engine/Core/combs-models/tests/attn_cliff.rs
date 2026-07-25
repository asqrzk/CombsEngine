//! Minimal repro + op bisection for the seq>=512 wgpu/Metal prefill bug.
//!
//! Runs the exact `attend()` op chain from `combs_models::kv` (matmul scores
//! -> causal mask_fill -> softmax(dim=3) -> matmul with V) on random
//! fixed-seed tensors, on three backends:
//!   - `Gpu`        = fused wgpu/CubeCL (the production `CombsBackend`)
//!   - `GpuUnfused` = plain wgpu/CubeCL backend (no fusion decorator)
//!   - `Cpu`        = NdArray (reference)
//!
//! GPU tests are ignored by default; run with:
//! `cargo test --release -p combs-models --test attn_cliff -- --ignored --nocapture`

use burn::tensor::{Bool, Int, Tensor, TensorData, activation::softmax, backend::Backend};

type Cpu = burn::backend::NdArray<f32>;
/// Production backend: fused wgpu.
type Gpu = burn::backend::Wgpu<f32, i32, u32>;
/// Same wgpu runtime without the fusion decorator.
type GpuUnfused =
    burn::backend::wgpu::CubeBackend<burn::backend::wgpu::WgpuRuntime, f32, i32, u32>;

// SmolLM2-135M attention geometry.
const N_Q: usize = 9;
const N_KV: usize = 3;
const HEAD_DIM: usize = 64;
const SCALE: f64 = 0.125; // 1/sqrt(64)

/// Deterministic xorshift RNG -> f32 in [-1, 1).
struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x >> 40) as u32) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
    }
}

fn rand_data(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed | 1);
    (0..len).map(|_| rng.next_f32()).collect()
}

/// Replica of `combs_models::kv::attend` (contiguous window, pos = 0).
fn attend<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    scale: f64,
) -> Tensor<B, 4> {
    let device = q.device();
    let [_, n_q, seq, _] = q.dims();
    let [_, n_kv, total, _] = k.dims();
    let n_rep = n_q / n_kv;
    let repeat = |x: Tensor<B, 4>| {
        if n_rep == 1 {
            return x;
        }
        let [b, nkv, s, d] = x.dims();
        x.unsqueeze_dim::<5>(2)
            .expand([b, nkv, n_rep, s, d])
            .reshape([b, nkv * n_rep, s, d])
    };
    let k = repeat(k);
    let v = repeat(v);

    let scores = q.matmul(k.transpose()).mul_scalar(scale);
    let scores = if seq > 1 {
        let q_pos = Tensor::<B, 1, Int>::arange(0..(seq as i64), &device).reshape([seq, 1]);
        let k_pos = Tensor::<B, 1, Int>::arange(0..(total as i64), &device).reshape([1, total]);
        let forbidden: Tensor<B, 2, Bool> = k_pos.greater(q_pos);
        let mask = forbidden
            .unsqueeze_dims::<4>(&[0, 1])
            .expand([1, n_q, seq, total]);
        scores.mask_fill(mask, -1e30f32)
    } else {
        scores
    };

    softmax(scores, 3).matmul(v)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn vec_of<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

/// Full-attention divergence of one GPU backend vs the NdArray reference.
fn attend_diff<Ref: Backend, Dev: Backend>(name: &str, seqs: &[usize]) {
    let cpu_dev = Default::default();
    let gpu_dev = Default::default();
    for &s in seqs {
        let q = rand_data(N_Q * s * HEAD_DIM, 11);
        let k = rand_data(N_KV * s * HEAD_DIM, 22);
        let v = rand_data(N_KV * s * HEAD_DIM, 33);

        let reference = attend(
            Tensor::<Ref, 4>::from_data(TensorData::new(q.clone(), [1, N_Q, s, HEAD_DIM]), &cpu_dev),
            Tensor::<Ref, 4>::from_data(TensorData::new(k.clone(), [1, N_KV, s, HEAD_DIM]), &cpu_dev),
            Tensor::<Ref, 4>::from_data(TensorData::new(v.clone(), [1, N_KV, s, HEAD_DIM]), &cpu_dev),
            SCALE,
        );
        let candidate = attend(
            Tensor::<Dev, 4>::from_data(TensorData::new(q, [1, N_Q, s, HEAD_DIM]), &gpu_dev),
            Tensor::<Dev, 4>::from_data(TensorData::new(k, [1, N_KV, s, HEAD_DIM]), &gpu_dev),
            Tensor::<Dev, 4>::from_data(TensorData::new(v, [1, N_KV, s, HEAD_DIM]), &gpu_dev),
            SCALE,
        );
        let diff = max_abs_diff(&vec_of(reference), &vec_of(candidate));
        println!("attend [{name}] seq={s}: max|diff| = {diff:.6}");
        assert!(diff < 1e-3, "[{name}] attend diverged at seq={s}: {diff}");
    }
}

#[test]
#[ignore = "gpu"]
fn attend_fused_matches_cpu_below_and_above_cliff() {
    attend_diff::<Cpu, Gpu>("fused", &[256, 511, 512, 513, 1024]);
}

#[test]
#[ignore = "gpu"]
fn attend_unfused_matches_cpu_below_and_above_cliff() {
    attend_diff::<Cpu, GpuUnfused>("unfused", &[256, 511, 512, 513, 1024]);
}

/// Stage bisection on the fused backend: scores matmul, mask, softmax alone,
/// softmax->v matmul, all at square scores shapes [1, N_Q, S, S].
#[test]
#[ignore = "gpu"]
fn bisect_stages_at_cliff() {
    let cpu_dev: burn::tensor::Device<Cpu> = Default::default();
    let gpu_dev: burn::tensor::Device<Gpu> = Default::default();

    for s in [511usize, 512, 513] {
        let q = rand_data(N_Q * s * HEAD_DIM, 11);
        let k = rand_data(N_KV * s * HEAD_DIM, 22);
        let v = rand_data(N_KV * s * HEAD_DIM, 33);

        // ---- stage 1: scores matmul (incl. GQA repeat + transpose) --------
        let stage1 = |q: Tensor<Cpu, 4>, k: Tensor<Cpu, 4>| {
            let [b, nkv, sq, d] = k.dims();
            let k = k
                .unsqueeze_dim::<5>(2)
                .expand([b, nkv, N_Q / nkv, sq, d])
                .reshape([b, N_Q, sq, d]);
            q.matmul(k.transpose()).mul_scalar(SCALE)
        };
        let ref_scores = stage1(
            Tensor::from_data(TensorData::new(q.clone(), [1, N_Q, s, HEAD_DIM]), &cpu_dev),
            Tensor::from_data(TensorData::new(k.clone(), [1, N_KV, s, HEAD_DIM]), &cpu_dev),
        );
        let gpu_scores = {
            let stage1 = |q: Tensor<Gpu, 4>, k: Tensor<Gpu, 4>| {
                let [b, nkv, sq, d] = k.dims();
                let k = k
                    .unsqueeze_dim::<5>(2)
                    .expand([b, nkv, N_Q / nkv, sq, d])
                    .reshape([b, N_Q, sq, d]);
                q.matmul(k.transpose()).mul_scalar(SCALE)
            };
            stage1(
                Tensor::from_data(TensorData::new(q.clone(), [1, N_Q, s, HEAD_DIM]), &gpu_dev),
                Tensor::from_data(TensorData::new(k.clone(), [1, N_KV, s, HEAD_DIM]), &gpu_dev),
            )
        };
        let d_scores = max_abs_diff(&vec_of(ref_scores.clone()), &vec_of(gpu_scores.clone()));
        println!("seq={s}: scores matmul max|diff| = {d_scores:.6}");

        // ---- stage 2: mask_fill -------------------------------------------
        let masked = |scores: Tensor<Cpu, 4>, dev: &burn::tensor::Device<Cpu>| {
            let q_pos = Tensor::<Cpu, 1, Int>::arange(0..(s as i64), dev).reshape([s, 1]);
            let k_pos = Tensor::<Cpu, 1, Int>::arange(0..(s as i64), dev).reshape([1, s]);
            let forbidden: Tensor<Cpu, 2, Bool> = k_pos.greater(q_pos);
            let mask = forbidden.unsqueeze_dims::<4>(&[0, 1]).expand([1, N_Q, s, s]);
            scores.mask_fill(mask, -1e30f32)
        };
        let masked_g = |scores: Tensor<Gpu, 4>, dev: &burn::tensor::Device<Gpu>| {
            let q_pos = Tensor::<Gpu, 1, Int>::arange(0..(s as i64), dev).reshape([s, 1]);
            let k_pos = Tensor::<Gpu, 1, Int>::arange(0..(s as i64), dev).reshape([1, s]);
            let forbidden: Tensor<Gpu, 2, Bool> = k_pos.greater(q_pos);
            let mask = forbidden.unsqueeze_dims::<4>(&[0, 1]).expand([1, N_Q, s, s]);
            scores.mask_fill(mask, -1e30f32)
        };
        let ref_masked = masked(ref_scores.clone(), &cpu_dev);
        let gpu_masked = masked_g(gpu_scores.clone(), &gpu_dev);
        let d_mask = max_abs_diff(&vec_of(ref_masked.clone()), &vec_of(gpu_masked.clone()));
        println!("seq={s}: mask_fill   max|diff| = {d_mask:.6}");

        // ---- stage 3: softmax(dim=3) alone on random data ------------------
        let ref_sm = softmax(ref_masked.clone(), 3);
        let gpu_sm = softmax(gpu_masked.clone(), 3);
        let d_sm = max_abs_diff(&vec_of(ref_sm.clone()), &vec_of(gpu_sm.clone()));
        println!("seq={s}: softmax     max|diff| = {d_sm:.6}");

        // ---- stage 4: softmax(x).matmul(v) --------------------------------
        let ref_out = ref_sm.matmul(
            Tensor::<Cpu, 4>::from_data(
                TensorData::new(v.clone(), [1, N_KV, s, HEAD_DIM]),
                &cpu_dev,
            )
            .unsqueeze_dim::<5>(2)
            .expand([1, N_KV, N_Q / N_KV, s, HEAD_DIM])
            .reshape([1, N_Q, s, HEAD_DIM]),
        );
        let gpu_out = gpu_sm.matmul(
            Tensor::<Gpu, 4>::from_data(
                TensorData::new(v.clone(), [1, N_KV, s, HEAD_DIM]),
                &gpu_dev,
            )
            .unsqueeze_dim::<5>(2)
            .expand([1, N_KV, N_Q / N_KV, s, HEAD_DIM])
            .reshape([1, N_Q, s, HEAD_DIM]),
        );
        let d_out = max_abs_diff(&vec_of(ref_out), &vec_of(gpu_out));
        println!("seq={s}: full chain  max|diff| = {d_out:.6}");
    }
}

/// Softmax alone over dim 3 on unmasked random data — pure reduce stress.
#[test]
#[ignore = "gpu"]
fn softmax_alone_at_cliff() {
    let cpu_dev: burn::tensor::Device<Cpu> = Default::default();
    let gpu_dev: burn::tensor::Device<Gpu> = Default::default();
    for s in [256usize, 511, 512, 513, 1024, 2048] {
        let x = rand_data(N_Q * s * s, 7);
        let ref_sm = softmax(
            Tensor::<Cpu, 4>::from_data(TensorData::new(x.clone(), [1, N_Q, s, s]), &cpu_dev),
            3,
        );
        let gpu_sm = softmax(
            Tensor::<Gpu, 4>::from_data(TensorData::new(x, [1, N_Q, s, s]), &gpu_dev),
            3,
        );
        let diff = max_abs_diff(&vec_of(ref_sm), &vec_of(gpu_sm));
        println!("softmax alone [1,{N_Q},{s},{s}]: max|diff| = {diff:.8}");
    }
}

/// Raw matmul [1, N_Q, S, D] x [1, N_Q, D, S] — matmul-kernel stress.
#[test]
#[ignore = "gpu"]
fn matmul_alone_at_cliff() {
    let cpu_dev: burn::tensor::Device<Cpu> = Default::default();
    let gpu_dev: burn::tensor::Device<Gpu> = Default::default();
    for s in [511usize, 512, 513, 1024] {
        let a = rand_data(N_Q * s * HEAD_DIM, 5);
        let b = rand_data(N_Q * HEAD_DIM * s, 6);
        let ref_out = Tensor::<Cpu, 4>::from_data(
            TensorData::new(a.clone(), [1, N_Q, s, HEAD_DIM]),
            &cpu_dev,
        )
        .matmul(Tensor::<Cpu, 4>::from_data(
            TensorData::new(b.clone(), [1, N_Q, HEAD_DIM, s]),
            &cpu_dev,
        ));
        let gpu_out = Tensor::<Gpu, 4>::from_data(
            TensorData::new(a, [1, N_Q, s, HEAD_DIM]),
            &gpu_dev,
        )
        .matmul(Tensor::<Gpu, 4>::from_data(
            TensorData::new(b, [1, N_Q, HEAD_DIM, s]),
            &gpu_dev,
        ));
        let diff = max_abs_diff(&vec_of(ref_out), &vec_of(gpu_out));
        println!("matmul [1,{N_Q},{s},{HEAD_DIM}]x[..,{HEAD_DIM},{s}]: max|diff| = {diff:.6}");
    }
}

/// Matmul with the *reduction* dim = S (the P@V shape in attention):
/// [1, N_Q, S, S] x [1, N_Q, S, HEAD_DIM].
#[test]
#[ignore = "gpu"]
fn matmul_reduction_dim_at_cliff() {
    let cpu_dev: burn::tensor::Device<Cpu> = Default::default();
    let gpu_dev: burn::tensor::Device<Gpu> = Default::default();
    for s in [256usize, 384, 448, 480, 496, 504, 508, 510, 511, 512, 513, 514, 516, 520, 528, 544, 576, 640, 768, 1024] {
        let a = rand_data(N_Q * s * s, 5);
        let b = rand_data(N_Q * s * HEAD_DIM, 6);
        let ref_out = Tensor::<Cpu, 4>::from_data(
            TensorData::new(a.clone(), [1, N_Q, s, s]),
            &cpu_dev,
        )
        .matmul(Tensor::<Cpu, 4>::from_data(
            TensorData::new(b.clone(), [1, N_Q, s, HEAD_DIM]),
            &cpu_dev,
        ));
        let gpu_out = Tensor::<Gpu, 4>::from_data(TensorData::new(a, [1, N_Q, s, s]), &gpu_dev)
            .matmul(Tensor::<Gpu, 4>::from_data(
                TensorData::new(b, [1, N_Q, s, HEAD_DIM]),
                &gpu_dev,
            ));
        let diff = max_abs_diff(&vec_of(ref_out), &vec_of(gpu_out));
        println!("P@V matmul K={s}: max|diff| = {diff:.6}");
    }
}

/// Sweep the reduction dim K at fixed M/N to find the exact kernel boundary,
/// for both 4D batched and plain 2D matmuls.
#[test]
#[ignore = "gpu"]
fn matmul_k_sweep_boundary() {
    let cpu_dev: burn::tensor::Device<Cpu> = Default::default();
    let gpu_dev: burn::tensor::Device<Gpu> = Default::default();
    for k in [500usize, 504, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 520] {
        let m = 128usize;
        let n = 64usize;
        let a = rand_data(m * k, 5);
        let b = rand_data(k * n, 6);
        let ref_out = Tensor::<Cpu, 2>::from_data(TensorData::new(a.clone(), [m, k]), &cpu_dev)
            .matmul(Tensor::<Cpu, 2>::from_data(
                TensorData::new(b.clone(), [k, n]),
                &cpu_dev,
            ));
        let gpu_out = Tensor::<Gpu, 2>::from_data(TensorData::new(a, [m, k]), &gpu_dev).matmul(
            Tensor::<Gpu, 2>::from_data(TensorData::new(b, [k, n]), &gpu_dev),
        );
        let diff = max_abs_diff(&vec_of(ref_out), &vec_of(gpu_out));
        println!("2D matmul [128,{k}]x[{k},64]: max|diff| = {diff:.6}");
    }
}

/// Which dimension/rank combination of the P@V matmul is broken?
#[test]
#[ignore = "gpu"]
fn matmul_shape_narrowing() {
    let cpu_dev: burn::tensor::Device<Cpu> = Default::default();
    let gpu_dev: burn::tensor::Device<Gpu> = Default::default();

    fn run<const D: usize>(label: &str, a_shape: [usize; D], b_shape: [usize; D]) {
        let cpu_dev: burn::tensor::Device<Cpu> = Default::default();
        let gpu_dev: burn::tensor::Device<Gpu> = Default::default();
        let a_len: usize = a_shape.iter().product();
        let b_len: usize = b_shape.iter().product();
        let a = rand_data(a_len, 5);
        let b = rand_data(b_len, 6);
        let ref_out = Tensor::<Cpu, D>::from_data(TensorData::new(a.clone(), a_shape), &cpu_dev)
            .matmul(Tensor::<Cpu, D>::from_data(
                TensorData::new(b.clone(), b_shape),
                &cpu_dev,
            ));
        let gpu_out = Tensor::<Gpu, D>::from_data(TensorData::new(a, a_shape), &gpu_dev).matmul(
            Tensor::<Gpu, D>::from_data(TensorData::new(b, b_shape), &gpu_dev),
        );
        let diff = max_abs_diff(&vec_of(ref_out), &vec_of(gpu_out));
        println!("{label}: max|diff| = {diff:.6}");
    }

    let _ = (cpu_dev, gpu_dev);
    run("2D   [512,512]x[512,64]      ", [512, 512], [512, 64]);
    run("2D   [512,512]x[512,128]     ", [512, 512], [512, 128]);
    run("2D   [128,512]x[512,64]      ", [128, 512], [512, 64]);
    run("2D   [1,512]x[512,64]        ", [1, 512], [512, 64]);
    run("3D   [9,512,512]x[9,512,64]  ", [9, 512, 512], [9, 512, 64]);
    run("4D   [1,1,512,512]x[..,64]   ", [1, 1, 512, 512], [1, 1, 512, 64]);
    run("4D   [1,9,512,512]x[..,64]   ", [1, 9, 512, 512], [1, 9, 512, 64]);
    run("4D   [1,9,512,512]x[..,128]  ", [1, 9, 512, 512], [1, 9, 512, 128]);
    run("4D   [1,9,64,512]x[..,64]    ", [1, 9, 64, 512], [1, 9, 512, 64]);
    run("4D   [1,9,128,512]x[..,64]   ", [1, 9, 128, 512], [1, 9, 512, 64]);
    run("4D   [1,9,1,512]x[..,64] dec ", [1, 9, 1, 512], [1, 9, 512, 64]);
    run("4D   [1,9,1,1024]x[..,64] dec", [1, 9, 1, 1024], [1, 9, 1024, 64]);
    run("4D   [1,9,511,511]x[..,64]   ", [1, 9, 511, 511], [1, 9, 511, 64]);
    run("4D   [1,9,1024,1024]x[..,64] ", [1, 9, 1024, 1024], [1, 9, 1024, 64]);
    // M=512 rows but K below cliff, and M below cliff with K=512:
    run("4D   [1,9,512,511]x[..,64]   ", [1, 9, 512, 511], [1, 9, 511, 64]);
    run("4D   [1,9,511,512]x[..,64]   ", [1, 9, 511, 512], [1, 9, 512, 64]);
}

/// Same wrong matmul twice in one process: is the wrong result stable?
#[test]
#[ignore = "gpu"]
fn matmul_determinism_at_512() {
    let gpu_dev: burn::tensor::Device<Gpu> = Default::default();
    let a = rand_data(9 * 512 * 512, 5);
    let b = rand_data(9 * 512 * 64, 6);
    let go = |a: &[f32], b: &[f32]| {
        vec_of(
            Tensor::<Gpu, 4>::from_data(TensorData::new(a.to_vec(), [1, 9, 512, 512]), &gpu_dev)
                .matmul(Tensor::<Gpu, 4>::from_data(
                    TensorData::new(b.to_vec(), [1, 9, 512, 64]),
                    &gpu_dev,
                )),
        )
    };
    let r1 = go(&a, &b);
    let r2 = go(&a, &b);
    let drift = max_abs_diff(&r1, &r2);
    println!("same-input repeat drift: {drift:.8}");
}

/// A = identity, B random: correct output IS B. If the GPU output equals
/// neither B nor anything derived from the inputs, the kernel dispatch is
/// being skipped (stale output buffer) rather than miscomputing.
#[test]
#[ignore = "gpu"]
fn matmul_identity_output_at_512() {
    let cpu_dev: burn::tensor::Device<Cpu> = Default::default();
    let gpu_dev: burn::tensor::Device<Gpu> = Default::default();
    let s = 512usize;
    let mut eye = vec![0.0f32; s * s];
    for i in 0..s {
        eye[i * s + i] = 1.0;
    }
    let b = rand_data(s * 64, 6);

    let out = Tensor::<Gpu, 2>::from_data(TensorData::new(eye.clone(), [s, s]), &gpu_dev).matmul(
        Tensor::<Gpu, 2>::from_data(TensorData::new(b.clone(), [s, 64]), &gpu_dev),
    );
    let out = vec_of(out);
    let diff_vs_b = max_abs_diff(&out, &b);
    let all_zero = out.iter().all(|&x| x == 0.0);
    println!("I@B vs B: max|diff| = {diff_vs_b:.6}; all-zero output: {all_zero}");
    println!("out[0..8]   = {:?}", &out[..8]);
    println!("b[0..8]     = {:?}", &b[..8]);

    // Control: CPU must match B exactly.
    let out_cpu = Tensor::<Cpu, 2>::from_data(TensorData::new(eye, [s, s]), &cpu_dev).matmul(
        Tensor::<Cpu, 2>::from_data(TensorData::new(b.clone(), [s, 64]), &cpu_dev),
    );
    println!("cpu I@B vs B: max|diff| = {:.8}", max_abs_diff(&vec_of(out_cpu), &b));
}

/// SmolLM2-135M linear-layer matmul shapes at chunk=512 (M=512): are they
/// in the broken region too?
#[test]
#[ignore = "gpu"]
fn matmul_linear_shapes_at_512() {
    let cpu_dev: burn::tensor::Device<Cpu> = Default::default();
    let gpu_dev: burn::tensor::Device<Gpu> = Default::default();
    let run = |label: &str, m: usize, k: usize, n: usize| {
        let a = rand_data(m * k, 5);
        let b = rand_data(k * n, 6);
        let ref_out = Tensor::<Cpu, 2>::from_data(TensorData::new(a.clone(), [m, k]), &cpu_dev)
            .matmul(Tensor::<Cpu, 2>::from_data(
                TensorData::new(b.clone(), [k, n]),
                &cpu_dev,
            ));
        let gpu_out = Tensor::<Gpu, 2>::from_data(TensorData::new(a, [m, k]), &gpu_dev).matmul(
            Tensor::<Gpu, 2>::from_data(TensorData::new(b, [k, n]), &gpu_dev),
        );
        let diff = max_abs_diff(&vec_of(ref_out), &vec_of(gpu_out));
        println!("{label} [{m},{k}]x[{k},{n}]: max|diff| = {diff:.6}");
    };
    run("qkv-ish  ", 512, 576, 576);
    run("gate/up  ", 512, 576, 1536);
    run("down     ", 512, 1536, 576);
    run("big-n    ", 512, 512, 2304);
    run("small-n  ", 512, 512, 32);
    run("k-only   ", 512, 511, 1536);
    run("m-only   ", 511, 576, 1536);
    run("m=513    ", 513, 576, 1536);
}

/// The exact `linear()` shape: 3D [1, seq, in] x [1, in, out] where the rhs
/// is a *transposed view* of an [out, in] weight (non-contiguous).
#[test]
#[ignore = "gpu"]
fn matmul_transposed_rhs_linear_shape() {
    let cpu_dev: burn::tensor::Device<Cpu> = Default::default();
    let gpu_dev: burn::tensor::Device<Gpu> = Default::default();
    let run = |label: &str, seq: usize, inn: usize, out: usize| {
        let x = rand_data(seq * inn, 5);
        let w = rand_data(out * inn, 6); // [out, in] like an HF weight
        let ref_out = Tensor::<Cpu, 3>::from_data(TensorData::new(x.clone(), [1, seq, inn]), &cpu_dev)
            .matmul(
                Tensor::<Cpu, 2>::from_data(TensorData::new(w.clone(), [out, inn]), &cpu_dev)
                    .transpose()
                    .unsqueeze_dim::<3>(0),
            );
        let gpu_out = Tensor::<Gpu, 3>::from_data(TensorData::new(x, [1, seq, inn]), &gpu_dev)
            .matmul(
                Tensor::<Gpu, 2>::from_data(TensorData::new(w, [out, inn]), &gpu_dev)
                    .transpose()
                    .unsqueeze_dim::<3>(0),
            );
        let diff = max_abs_diff(&vec_of(ref_out), &vec_of(gpu_out));
        println!("{label} [1,{seq},{inn}]x[1,{inn},{out}]T: max|diff| = {diff:.6}");
    };
    run("seq=511", 511, 576, 1536);
    run("seq=512", 512, 576, 1536);
    run("seq=600", 600, 576, 1536);
}

/// End-to-end regression: with the `safe_matmul` workaround in place, the
/// public `KVCache::attention` path must match the NdArray CPU reference at
/// and above the 512 cliff, for both cache implementations.
#[test]
#[ignore = "gpu"]
fn kv_cache_attention_matches_cpu_above_cliff() {
    use combs_models::{CacheConfig, KVCache};

    let cpu_dev: burn::tensor::Device<Cpu> = Default::default();
    let gpu_dev: burn::tensor::Device<Gpu> = Default::default();

    for s in [511usize, 512, 513, 600, 1024] {
        let q = rand_data(N_Q * s * HEAD_DIM, 11);
        let k = rand_data(N_KV * s * HEAD_DIM, 22);
        let v = rand_data(N_KV * s * HEAD_DIM, 33);

        let mut cpu_cache: Box<dyn KVCache<Cpu>> = Box::new(
            combs_models::ContiguousKVCache::<Cpu>::new(1),
        );
        let cpu_out = cpu_cache.attention(
            0,
            Tensor::from_data(TensorData::new(q.clone(), [1, N_Q, s, HEAD_DIM]), &cpu_dev),
            Tensor::from_data(TensorData::new(k.clone(), [1, N_KV, s, HEAD_DIM]), &cpu_dev),
            Tensor::from_data(TensorData::new(v.clone(), [1, N_KV, s, HEAD_DIM]), &cpu_dev),
            0,
            SCALE,
        );

        for kind in ["contiguous", "paged"] {
            let mut gpu_cache: Box<dyn KVCache<Gpu>> = match kind {
                "contiguous" => Box::new(combs_models::ContiguousKVCache::<Gpu>::new(1)),
                _ => Box::new(combs_models::PagedKVCache::<Gpu>::new(
                    1,
                    CacheConfig::paged(2048),
                )),
            };
            let gpu_out = gpu_cache.attention(
                0,
                Tensor::from_data(TensorData::new(q.clone(), [1, N_Q, s, HEAD_DIM]), &gpu_dev),
                Tensor::from_data(TensorData::new(k.clone(), [1, N_KV, s, HEAD_DIM]), &gpu_dev),
                Tensor::from_data(TensorData::new(v.clone(), [1, N_KV, s, HEAD_DIM]), &gpu_dev),
                0,
                SCALE,
            );
            let diff = max_abs_diff(&vec_of(cpu_out.clone()), &vec_of(gpu_out));
            println!("KVCache[{kind}] attention seq={s}: max|diff| = {diff:.6}");
            assert!(diff < 1e-3, "[{kind}] attention diverged at seq={s}: {diff}");
        }
    }
}
