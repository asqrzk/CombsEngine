//! Flux2 transformer parity vs torch-cpu reference activations.
//!
//! Gated on `COMBS_FLUX2_PARITY_DIR` (a dump produced by
//! `tools/harmony/gen_flux2_reference.py`). Inputs per stage are READ
//! from the dump — no cross-language math — so the first failing stage
//! localizes the defect. NdArray backend.
//!
//! Run:
//! ```sh
//! COMBS_FLUX2_PARITY_DIR=$HOME/.cache/combs/flux2-parity \
//!   cargo test --release -p combs-diffusion --test flux2_parity -- --ignored --nocapture
//! ```

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use combs_diffusion::flux2::{
    image_ids, rope_tables, split_mods, text_ids, Flux2Config, Flux2DoubleBlock,
    Flux2SingleBlock, Flux2Transformer,
};
use combs_formats::{ModelSource, SafetensorsSource};

type B = NdArray<f32>;

struct Dump {
    dir: std::path::PathBuf,
    manifest: serde_json::Value,
}

impl Dump {
    fn open() -> Option<Dump> {
        let dir = std::path::PathBuf::from(std::env::var_os("COMBS_FLUX2_PARITY_DIR")?);
        let manifest = serde_json::from_str(
            &std::fs::read_to_string(dir.join("manifest.json")).ok()?,
        )
        .ok()?;
        Some(Dump { dir, manifest })
    }

    fn shape(&self, name: &str) -> Vec<usize> {
        self.manifest["tensors"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["name"] == name)
            .unwrap_or_else(|| panic!("tensor {name} not in manifest"))["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect()
    }

    fn floats(&self, name: &str) -> Vec<f32> {
        let bytes = std::fs::read(self.dir.join(format!("{name}.bin"))).expect(name);
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn tensor3(&self, name: &str) -> Tensor<B, 3> {
        let shape = self.shape(name);
        assert_eq!(shape.len(), 3, "{name}");
        Tensor::from_data(
            TensorData::new(self.floats(name), [shape[0], shape[1], shape[2]]),
            &Default::default(),
        )
    }

    fn config(&self) -> Flux2Config {
        let c = &self.manifest["config"];
        Flux2Config {
            in_channels: c["in_channels"].as_u64().unwrap() as usize,
            num_layers: c["num_layers"].as_u64().unwrap() as usize,
            num_single_layers: c["num_single_layers"].as_u64().unwrap() as usize,
            attention_head_dim: c["attention_head_dim"].as_u64().unwrap() as usize,
            num_attention_heads: c["num_attention_heads"].as_u64().unwrap() as usize,
            joint_attention_dim: c["joint_attention_dim"].as_u64().unwrap() as usize,
            mlp_ratio: c["mlp_ratio"].as_f64().unwrap() as usize,
            axes_dims_rope: c["axes_dims_rope"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect(),
            rope_theta: c["rope_theta"].as_f64().unwrap(),
            eps: 1e-6,
            timestep_channels: 256,
        }
    }
}

fn max_diff(got: &Tensor<B, 3>, want: &[f32]) -> f32 {
    let g: Vec<f32> = got.clone().into_data().to_vec().unwrap();
    assert_eq!(g.len(), want.len(), "element count");
    g.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn check(stage: &str, got: &Tensor<B, 3>, want: &[f32], tol: f32) {
    let d = max_diff(got, want);
    println!("[flux2-parity] {stage}: max |diff| {d:e}");
    assert!(d < tol, "{stage} drifted: {d} (tol {tol})");
}

/// The whole localizing chain in ONE test: stages run in dependency
/// order, each fed reference inputs, so the first panic names the
/// defective stage.
#[test]
#[ignore = "requires COMBS_FLUX2_PARITY_DIR (gen_flux2_reference.py dump)"]
fn flux2_stages_match_reference() {
    let Some(dump) = Dump::open() else {
        eprintln!("skipping: set COMBS_FLUX2_PARITY_DIR");
        return;
    };
    let cfg = dump.config();
    let dim = cfg.inner_dim();
    let device: burn::tensor::Device<B> = Default::default();
    let source =
        SafetensorsSource::load_weights_only(dump.dir.to_str().unwrap(), "flux2")
            .expect("dump weights");
    let src: &dyn ModelSource = &source;

    // The dump geometry: 3x4 latent grid, 5 text tokens, t = 0.75.
    let (grid_h, grid_w, s_txt, timestep) = (3usize, 4usize, 5usize, 0.75f32);
    let img_ids = image_ids(grid_h, grid_w);
    let txt_ids = text_ids(s_txt);

    // Stage 1: rope tables vs the reference pos_embed output for the
    // TEXT ids (the hook captured the last pos_embed call). Reference
    // is repeat-interleaved [s, d]; ours is half-width [s, d/2].
    let (cos, sin) = rope_tables::<B>(&txt_ids, &cfg.axes_dims_rope, cfg.rope_theta, &device);
    let ref_cos = dump.floats("pos_embed_out0");
    let ref_sin = dump.floats("pos_embed_out1");
    let half: usize = cfg.axes_dims_rope.iter().map(|d| d / 2).sum();
    let got_cos: Vec<f32> = cos.clone().into_data().to_vec().unwrap();
    let got_sin: Vec<f32> = sin.clone().into_data().to_vec().unwrap();
    let mut worst = 0.0f32;
    for s in 0..s_txt {
        for j in 0..half {
            worst = worst.max((got_cos[s * half + j] - ref_cos[s * 2 * half + 2 * j]).abs());
            worst = worst.max((got_cos[s * half + j] - ref_cos[s * 2 * half + 2 * j + 1]).abs());
            worst = worst.max((got_sin[s * half + j] - ref_sin[s * 2 * half + 2 * j]).abs());
            worst = worst.max((got_sin[s * half + j] - ref_sin[s * 2 * half + 2 * j + 1]).abs());
        }
    }
    println!("[flux2-parity] rope: max |diff| {worst:e}");
    assert!(worst < 1e-6, "rope tables drifted: {worst}");

    // Stage 2: the full model — loaded once, used for embedder and
    // end-to-end checks.
    let model = Flux2Transformer::<B>::load(src, cfg.clone(), &device).expect("load");
    let input_img = dump.tensor3("input_img");
    let input_txt = dump.tensor3("input_txt");

    // Stage 3: per-block chain, each block fed REFERENCE inputs.
    // Mods are recomputed from the reference temb via the raw weights.
    let temb_ref = {
        let shape = dump.shape("time_guidance_embed_out0");
        Tensor::<B, 2>::from_data(
            TensorData::new(dump.floats("time_guidance_embed_out0"), [shape[0], shape[1]]),
            &device,
        )
    };
    let lin2 = |name: &str, din: usize, dout: usize| -> Tensor<B, 2> {
        let t: Tensor<B, 2> = src
            .open_tensor(name)
            .unwrap()
            .load_to_tensor(&device)
            .unwrap();
        assert_eq!(t.dims(), [dout, din], "{name}");
        t
    };
    let silu2 = burn::tensor::activation::silu(temb_ref.clone());
    let mods_img = split_mods(
        silu2.clone().matmul(lin2("double_stream_modulation_img.linear.weight", dim, dim * 6).transpose()),
        2,
        dim,
    );
    let mods_txt = split_mods(
        silu2.clone().matmul(lin2("double_stream_modulation_txt.linear.weight", dim, dim * 6).transpose()),
        2,
        dim,
    );
    let mods_single = split_mods(
        silu2.clone().matmul(lin2("single_stream_modulation.linear.weight", dim, dim * 3).transpose()),
        1,
        dim,
    );

    // Joint rope over text-first ids, exactly as forward() builds it.
    let mut ids = txt_ids.clone();
    ids.extend_from_slice(&img_ids);
    let (jcos, jsin) = rope_tables::<B>(&ids, &cfg.axes_dims_rope, cfg.rope_theta, &device);

    let mut img_h = dump.tensor3("x_embedder_out0");
    let mut txt_h = dump.tensor3("context_embedder_out0");
    for i in 0..cfg.num_layers {
        let block =
            Flux2DoubleBlock::<B>::load(src, &format!("transformer_blocks.{i}"), &cfg, &device)
                .expect("double block");
        let (t, im) = block.forward(img_h, txt_h, &mods_img, &mods_txt, (&jcos, &jsin), &cfg);
        check(&format!("double_{i}.txt"), &t, &dump.floats(&format!("double_{i}_out0")), 2e-4);
        check(&format!("double_{i}.img"), &im, &dump.floats(&format!("double_{i}_out1")), 2e-4);
        // Chain the REFERENCE outputs so drift never compounds.
        txt_h = dump.tensor3(&format!("double_{i}_out0"));
        img_h = dump.tensor3(&format!("double_{i}_out1"));
    }

    let mut h = Tensor::cat(vec![txt_h, img_h], 1);
    for i in 0..cfg.num_single_layers {
        let block = Flux2SingleBlock::<B>::load(
            src,
            &format!("single_transformer_blocks.{i}"),
            &cfg,
            &device,
        )
        .expect("single block");
        let out = block.forward(h, &mods_single[0], (&jcos, &jsin), &cfg);
        check(&format!("single_{i}"), &out, &dump.floats(&format!("single_{i}_out0")), 2e-4);
        h = dump.tensor3(&format!("single_{i}_out0"));
    }

    // Stage 4: end-to-end forward — everything at once, our own temb.
    let out = model.forward(input_img, input_txt, timestep, &img_ids, &txt_ids);
    check("end-to-end", &out, &dump.floats("output"), 5e-4);
}
