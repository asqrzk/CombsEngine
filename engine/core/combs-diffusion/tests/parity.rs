//! Component-parity harness vs torch-cpu reference activations.
//!
//! Gated on `COMBS_DIFFUSION_PARITY_DIR` (a dump produced by
//! `tools/goldens/gen_diffusion_reference.py`) and the local
//! stable-diffusion-v1-5 checkpoint. All inputs are READ from the dump —
//! no cross-language math — and each stage compares against the diffusers
//! reference on the NdArray backend. Stages run in dependency order; the
//! first failure localizes the defect.
//!
//! Run:
//! ```sh
//! COMBS_DIFFUSION_PARITY_DIR=/path/to/dump \
//!   cargo test --release -p combs-diffusion --test parity -- --ignored --nocapture
//! ```

use burn::backend::NdArray;
use burn::tensor::{Bool, Int, Tensor, TensorData};
use combs_diffusion::clip::{ClipAttention, ClipEncoderLayer, ClipTextEmbeddings};
use combs_diffusion::time::{timestep_embedding, TimeEmbedding};
use combs_diffusion::{ClipTextModel, UNet2DConditionModel, VAEDecoder};
use combs_formats::SafetensorsSource;

type B = NdArray<f32>;

struct Dump {
    dir: std::path::PathBuf,
    manifest: serde_json::Value,
}

impl Dump {
    fn open() -> Option<Dump> {
        let dir = std::path::PathBuf::from(std::env::var_os("COMBS_DIFFUSION_PARITY_DIR")?);
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).ok()?)
                .ok()?;
        Some(Dump { dir, manifest })
    }

    fn shape(&self, name: &str) -> Vec<usize> {
        self.manifest
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["name"] == name)
            .unwrap_or_else(|| panic!("{name} not in manifest"))["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect()
    }

    fn f32s(&self, name: &str) -> (Vec<f32>, Vec<usize>) {
        let bytes = std::fs::read(self.dir.join(format!("{name}.bin"))).expect(name);
        let values = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        (values, self.shape(name))
    }

    fn i32s(&self, name: &str) -> (Vec<i32>, Vec<usize>) {
        let bytes = std::fs::read(self.dir.join(format!("{name}.bin"))).expect(name);
        let values = bytes
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        (values, self.shape(name))
    }
}

fn model_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap();
    std::path::PathBuf::from(home).join(".cache/combs/models/stable-diffusion-v1-5")
}

/// Reports max/mean abs deviation and asserts the stage bound.
fn compare(stage: &str, ours: &[f32], reference: &[f32], max_abs_bound: f32) {
    assert_eq!(ours.len(), reference.len(), "{stage}: length mismatch");
    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    let mut worst = 0usize;
    for (i, (a, b)) in ours.iter().zip(reference).enumerate() {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
            worst = i;
        }
        sum_abs += d as f64;
    }
    let mean_abs = sum_abs / ours.len() as f64;
    println!(
        "[parity] {stage}: max_abs {max_abs:.6e} (idx {worst}: ours {} vs ref {}), mean_abs {mean_abs:.6e}",
        ours[worst], reference[worst]
    );
    assert!(
        max_abs <= max_abs_bound,
        "{stage} DIVERGES: max_abs {max_abs:.6e} > bound {max_abs_bound:.1e} — this is the defective stage"
    );
}

#[test]
#[ignore = "requires COMBS_DIFFUSION_PARITY_DIR + local stable-diffusion-v1-5"]
fn component_parity_vs_torch() {
    let Some(dump) = Dump::open() else {
        panic!("set COMBS_DIFFUSION_PARITY_DIR to the gen_diffusion_reference.py output");
    };
    let device = Default::default();
    let root = model_dir();

    // Stage 1: sinusoidal timestep projection at t=951.
    let t: Tensor<B, 1, Int> = Tensor::from_data(TensorData::from([951i64]), &device);
    let ours = timestep_embedding(&t, 320, &device);
    let (reference, _) = dump.f32s("time_proj_951");
    compare(
        "time_proj(951)",
        &ours.clone().into_data().to_vec::<f32>().unwrap(),
        &reference,
        1e-4,
    );

    // Stage 2: the time MLP on top of the projection.
    let unet_source = SafetensorsSource::load_weights_only(root.join("unet"), "sd-unet")
        .expect("unet source");
    let time_mlp =
        TimeEmbedding::<B>::load_from(&unet_source, "time_embedding", 320, 1280, &device)
            .expect("time embedding");
    let (reference, _) = dump.f32s("time_embedding_951");
    compare(
        "time_embedding(951)",
        &time_mlp.forward(ours).into_data().to_vec::<f32>().unwrap(),
        &reference,
        5e-4,
    );

    // Stage 3: CLIP last hidden state for the fixed prompt ids.
    let text_source =
        SafetensorsSource::load_weights_only(root.join("text_encoder"), "sd-clip")
            .expect("text source");
    let clip = ClipTextModel::<B>::load_from(&text_source, &device).expect("clip");
    let (ids, ids_shape) = dump.i32s("clip_ids");
    let ids: Tensor<B, 2, Int> = Tensor::from_data(
        TensorData::new(ids, [ids_shape[0], ids_shape[1]]),
        &device,
    );
    let ours_hidden = clip.forward(ids);
    let (reference, _) = dump.f32s("clip_hidden");
    compare(
        "clip.last_hidden",
        &ours_hidden.into_data().to_vec::<f32>().unwrap(),
        &reference,
        2e-3,
    );

    // Stage 4: full UNet noise prediction — conditioned on the REFERENCE
    // hidden state so this stage is independent of stage 3.
    let unet = UNet2DConditionModel::<B>::load_from(&unet_source, &device).expect("unet");
    let (latent, ls) = dump.f32s("latent_in");
    let latent: Tensor<B, 4> =
        Tensor::from_data(TensorData::new(latent, [ls[0], ls[1], ls[2], ls[3]]), &device);
    let (ctx, cs) = dump.f32s("clip_hidden");
    let ctx: Tensor<B, 3> =
        Tensor::from_data(TensorData::new(ctx, [cs[0], cs[1], cs[2]]), &device);
    let ours_noise = unet.forward(latent, 951.0, ctx);
    let (reference, _) = dump.f32s("unet_out_951");
    compare(
        "unet.noise_pred(951)",
        &ours_noise.into_data().to_vec::<f32>().unwrap(),
        &reference,
        5e-3,
    );

    // Stage 5: VAE decode of a fixed latent (incl. post_quant_conv).
    let vae_source = SafetensorsSource::load_weights_only(root.join("vae"), "sd-vae")
        .expect("vae source");
    let vae = VAEDecoder::<B>::load_from(&vae_source, &device).expect("vae");
    let (vlat, vs) = dump.f32s("vae_latent");
    let vlat: Tensor<B, 4> =
        Tensor::from_data(TensorData::new(vlat, [vs[0], vs[1], vs[2], vs[3]]), &device);
    let ours_img = vae.forward(vlat);
    let (reference, _) = dump.f32s("vae_out");
    compare(
        "vae.decode",
        &ours_img.into_data().to_vec::<f32>().unwrap(),
        &reference,
        5e-3,
    );

    println!("[parity] all stages within bounds");
}

/// Bisects INSIDE the CLIP text encoder when the top-level stage diverges:
/// embeddings → layer-0 sublayers (attention fed the REFERENCE ln1 output,
/// isolating it from the norm) → layer chain checkpoints at 0/5/11.
#[test]
#[ignore = "requires COMBS_DIFFUSION_PARITY_DIR + local stable-diffusion-v1-5"]
fn clip_bisect_vs_torch() {
    let Some(dump) = Dump::open() else {
        panic!("set COMBS_DIFFUSION_PARITY_DIR to the gen_diffusion_reference.py output");
    };
    let device = Default::default();
    let root = model_dir();
    let text_source =
        SafetensorsSource::load_weights_only(root.join("text_encoder"), "sd-clip")
            .expect("text source");

    let (ids, ids_shape) = dump.i32s("clip_ids");
    let seq = ids_shape[1];
    let ids: Tensor<B, 2, Int> =
        Tensor::from_data(TensorData::new(ids, [ids_shape[0], seq]), &device);

    // Stage b1: token + position embeddings.
    let embeddings = ClipTextEmbeddings::<B>::load_from(&text_source, &device).expect("emb");
    let emb_out = embeddings.forward(ids);
    let (reference, _) = dump.f32s("clip_emb_out");
    compare(
        "clip.embeddings",
        &emb_out.clone().into_data().to_vec::<f32>().unwrap(),
        &reference,
        1e-4,
    );

    // Same mask construction as clip.rs's causal_mask (burn tril_mask(0) =
    // TRUE strictly above the diagonal — the future to block).
    let tri: Tensor<B, 2, Bool> = Tensor::tril_mask([seq, seq], 0, &device);
    let mask = Tensor::<B, 4>::zeros([1, 1, seq, seq], &device)
        .mask_fill(tri.reshape([1, 1, seq, seq]), -1e9f32);

    // Stage b2: layer-0 attention on the REFERENCE ln1 output.
    let attn = ClipAttention::<B>::load_from(
        &text_source,
        "text_model.encoder.layers.0.self_attn",
        &device,
    )
    .expect("attn");
    let (ln1, s) = dump.f32s("clip_l0_ln1");
    let ln1: Tensor<B, 3> =
        Tensor::from_data(TensorData::new(ln1, [s[0], s[1], s[2]]), &device);
    // Manual attention from raw tensors with the simplest primitives
    // (narrow-per-head, no reshape/permute), comparing every intermediate —
    // pinpoints the defective step inside the module.
    {
        use combs_diffusion::weights::load_tensor;
        let p = "text_model.encoder.layers.0.self_attn";
        let proj = |name: &str| -> (Tensor<B, 3>, Tensor<B, 1>) {
            let w: Tensor<B, 2> =
                load_tensor(&text_source, &format!("{p}.{name}.weight"), &device).unwrap();
            let b: Tensor<B, 1> =
                load_tensor(&text_source, &format!("{p}.{name}.bias"), &device).unwrap();
            (w.transpose().unsqueeze::<3>(), b)
        };
        let apply = |x: &Tensor<B, 3>, name: &str| -> Tensor<B, 3> {
            let (w, b) = proj(name);
            x.clone().matmul(w) + b.reshape([1, 1, 768])
        };
        let q_m = apply(&ln1, "q_proj");
        let k_m = apply(&ln1, "k_proj");
        let v_m = apply(&ln1, "v_proj");
        for (name, ours) in [("clip_l0_q", &q_m), ("clip_l0_k", &k_m), ("clip_l0_v", &v_m)] {
            let (reference, _) = dump.f32s(name);
            compare(
                &format!("manual {name}"),
                &ours.clone().into_data().to_vec::<f32>().unwrap(),
                &reference,
                5e-5,
            );
        }
        // Per-head attention with narrow slices. burn triangle masks are
        // complements: tril_mask(0) is TRUE strictly above the diagonal.
        let mask2: Tensor<B, 2> = Tensor::<B, 2>::zeros([seq, seq], &device)
            .mask_fill(Tensor::tril_mask([seq, seq], 0, &device), -1e9f32);
        let mut ctx_heads = Vec::new();
        for h in 0..12 {
            let qh = q_m.clone().narrow(2, h * 64, 64);
            let kh = k_m.clone().narrow(2, h * 64, 64);
            let vh = v_m.clone().narrow(2, h * 64, 64);
            let scores = qh.matmul(kh.transpose()).mul_scalar(0.125)
                + mask2.clone().unsqueeze::<3>();
            let w = burn::tensor::activation::softmax(scores, 2);
            ctx_heads.push(w.matmul(vh));
        }
        let ctx_m = Tensor::cat(ctx_heads, 2);
        let (reference, _) = dump.f32s("clip_l0_ctx");
        compare(
            "manual ctx (pre out_proj)",
            &ctx_m.clone().into_data().to_vec::<f32>().unwrap(),
            &reference,
            5e-4,
        );
        let out_m = apply(&ctx_m, "out_proj");
        let (reference, _) = dump.f32s("clip_l0_attn");
        compare(
            "manual attn out",
            &out_m.into_data().to_vec::<f32>().unwrap(),
            &reference,
            5e-4,
        );
    }

    let ours_attn = attn.forward(ln1, &mask);
    let (reference, _) = dump.f32s("clip_l0_attn");
    compare(
        "clip.layer0.self_attn(ref ln1)",
        &ours_attn.into_data().to_vec::<f32>().unwrap(),
        &reference,
        5e-4,
    );

    // Stage b3: the full layer chain, checkpointed at 0, 5, 11.
    let mut hidden = emb_out;
    for i in 0..12 {
        let layer = ClipEncoderLayer::<B>::load_from(
            &text_source,
            &format!("text_model.encoder.layers.{i}"),
            &device,
        )
        .unwrap_or_else(|e| panic!("layer {i}: {e}"));
        hidden = layer.forward(hidden, &mask);
        if matches!(i, 0 | 5 | 11) {
            let (reference, _) = dump.f32s(&format!("clip_layer{i}_out"));
            compare(
                &format!("clip.layer{i}.out"),
                &hidden.clone().into_data().to_vec::<f32>().unwrap(),
                &reference,
                if i == 0 { 1e-3 } else { 5e-3 },
            );
        }
    }
    println!("[parity] clip bisect complete");
}

/// down_blocks[0] component probe: the first resnet on the REFERENCE
/// conv_in + t_emb, then the first spatial transformer on the REFERENCE
/// resnet output — isolates resnet math from attention math.
#[test]
#[ignore = "requires COMBS_DIFFUSION_PARITY_DIR + local stable-diffusion-v1-5"]
fn unet_down0_components_vs_torch() {
    use combs_diffusion::blocks::{ResnetBlock2D, SpatialTransformer};
    let Some(dump) = Dump::open() else {
        panic!("set COMBS_DIFFUSION_PARITY_DIR to the gen_diffusion_reference.py output");
    };
    let device = Default::default();
    let root = model_dir();
    let unet_source = SafetensorsSource::load_weights_only(root.join("unet"), "sd-unet")
        .expect("unet source");

    let t4 = |name: &str| -> Tensor<B, 4> {
        let (v, s) = dump.f32s(name);
        Tensor::from_data(TensorData::new(v, [s[0], s[1], s[2], s[3]]), &device)
    };
    let (temb, ts) = dump.f32s("time_embedding_951");
    let temb: Tensor<B, 2> = Tensor::from_data(TensorData::new(temb, [ts[0], ts[1]]), &device);

    let resnet0 = ResnetBlock2D::<B>::load_from(
        &unet_source,
        "down_blocks.0.resnets.0",
        320,
        320,
        Some(1280),
        1e-5,
        &device,
    )
    .expect("resnet0");
    let ours = resnet0.forward(t4("unet_conv_in"), Some(&temb));
    let (reference, _) = dump.f32s("unet_d0_res0");
    compare(
        "unet.down0.resnet0(ref inputs)",
        &ours.into_data().to_vec::<f32>().unwrap(),
        &reference,
        1e-3,
    );

    let attn0 = SpatialTransformer::<B>::load_from(
        &unet_source,
        "down_blocks.0.attentions.0",
        320,
        768,
        8,
        40,
        &device,
    )
    .expect("attn0");
    let (ctx, cs) = dump.f32s("clip_hidden");
    let ctx: Tensor<B, 3> =
        Tensor::from_data(TensorData::new(ctx, [cs[0], cs[1], cs[2]]), &device);
    let ours = attn0.forward(t4("unet_d0_res0"), &ctx);
    let (reference, _) = dump.f32s("unet_d0_attn0");
    compare(
        "unet.down0.attn0(ref inputs)",
        &ours.into_data().to_vec::<f32>().unwrap(),
        &reference,
        1e-3,
    );
}

/// Walks the UNet per-block taps (forward_traced) against the diffusers
/// hook dumps; the first diverging tap names the defective block.
#[test]
#[ignore = "requires COMBS_DIFFUSION_PARITY_DIR + local stable-diffusion-v1-5"]
fn unet_bisect_vs_torch() {
    let Some(dump) = Dump::open() else {
        panic!("set COMBS_DIFFUSION_PARITY_DIR to the gen_diffusion_reference.py output");
    };
    let device = Default::default();
    let root = model_dir();
    let unet_source = SafetensorsSource::load_weights_only(root.join("unet"), "sd-unet")
        .expect("unet source");
    let unet = UNet2DConditionModel::<B>::load_from(&unet_source, &device).expect("unet");

    let (latent, ls) = dump.f32s("latent_in");
    let latent: Tensor<B, 4> =
        Tensor::from_data(TensorData::new(latent, [ls[0], ls[1], ls[2], ls[3]]), &device);
    let (ctx, cs) = dump.f32s("clip_hidden");
    let ctx: Tensor<B, 3> =
        Tensor::from_data(TensorData::new(ctx, [cs[0], cs[1], cs[2]]), &device);

    let (out, taps) = unet.forward_traced(latent, 951.0, ctx);
    let mut first_bad: Option<String> = None;
    for (name, tap) in &taps {
        let ref_name = format!("unet_{name}");
        let (reference, _) = dump.f32s(&ref_name);
        let ours = tap.clone().into_data().to_vec::<f32>().unwrap();
        let max_abs = ours
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let status = if max_abs <= 5e-3 { "ok" } else { "DIVERGES" };
        println!("[parity] unet.{name}: max_abs {max_abs:.6e} {status}");
        if max_abs > 5e-3 && first_bad.is_none() {
            first_bad = Some(name.clone());
        }
    }
    let (reference, _) = dump.f32s("unet_out_951");
    let ours = out.into_data().to_vec::<f32>().unwrap();
    let max_abs = ours
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("[parity] unet.conv_out: max_abs {max_abs:.6e}");
    assert!(
        first_bad.is_none() && max_abs <= 5e-3,
        "first diverging UNet block: {}",
        first_bad.unwrap_or_else(|| "conv_out".into())
    );
}
