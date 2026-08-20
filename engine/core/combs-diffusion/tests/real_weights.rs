//! Real checkpoint test. Set `COMBS_DIFFUSION_SD15_DIR` to a local HF Diffusers
//! SD 1.5 checkpoint (with `unet/`, `vae/`, `text_encoder/` subdirs and a
//! `tokenizer.json`) to run.

use combs_diffusion::{load_diffusion_model, DiffusionArchitecture, DiffusionModel};

#[test]
fn sd15_real_checkpoint_generates() {
    let Ok(model_dir) = std::env::var("COMBS_DIFFUSION_SD15_DIR") else {
        return;
    };

    type B = burn::backend::NdArray;
    let device = Default::default();
    let mut pipeline =
        load_diffusion_model::<B>(DiffusionArchitecture::StableDiffusion1_5, &model_dir, &device)
            .expect("failed to load SD 1.5 checkpoint");

    let embed = pipeline
        .encode_prompt("a photo of an astronaut riding a horse", Some(""))
        .expect("failed to encode prompt");

    let (image, _seed) = pipeline
        .generate(embed, 64, 64, 5, 7.5, Some(42), combs_diffusion::SchedulerKind::default(), None)
        .expect("generation failed");

    assert_eq!(image.dims(), [1, 3, 64, 64]);
}
