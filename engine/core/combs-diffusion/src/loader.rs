//! Multi-component diffusion model loader.
//!
//! Real HuggingFace Diffusers checkpoints are split into sub-directories
//! (`unet/`, `vae/`, `text_encoder/`) and a shared tokenizer. This module
//! opens those components and builds the right pipeline for the requested
//! architecture.

use std::path::Path;

use combs_formats::{FormatError, Result, SafetensorsSource, TokenizerSpec};

use crate::StableDiffusionPipeline;
use burn::tensor::backend::Backend;

/// Supported diffusion model families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffusionArchitecture {
    /// Stable Diffusion 1.5 (and 2.x, which share the same component layout).
    StableDiffusion1_5,
}

impl DiffusionArchitecture {
    /// Detect the architecture from a Diffusers model directory.
    ///
    /// Tries `model_index.json` first, then falls back to the presence of the
    /// standard `unet/`, `vae/`, `text_encoder/` subdirectories.
    pub fn detect(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let index_path = model_dir.join("model_index.json");
        if index_path.exists() {
            let text = std::fs::read_to_string(&index_path)
                .map_err(|e| FormatError::Io(e))?;
            let json: serde_json::Value = serde_json::from_str(&text)
                .map_err(|e| FormatError::Json { context: index_path.display().to_string(), source: e })?;
            if let Some(class) = json.get("_class_name").and_then(|v| v.as_str()) {
                match class {
                    "StableDiffusionPipeline" | "StableDiffusionImg2ImgPipeline" | "StableDiffusionInpaintPipeline" => {
                        return Ok(Self::StableDiffusion1_5);
                    }
                    other => {
                        return Err(FormatError::Safetensors(format!(
                            "unsupported diffusion pipeline: {other}"
                        )));
                    }
                }
            }
        }

        // Fallback: verify the expected component layout.
        if model_dir.join("unet").is_dir()
            && model_dir.join("vae").is_dir()
            && model_dir.join("text_encoder").is_dir()
        {
            Ok(Self::StableDiffusion1_5)
        } else {
            Err(FormatError::MissingFile(
                "model_index.json or unet/vae/text_encoder subdirectories".to_string(),
            ))
        }
    }
}

/// Load a complete diffusion pipeline from a HuggingFace Diffusers checkpoint.
///
/// `model_dir` should contain `unet/`, `vae/`, `text_encoder/` subdirectories
/// and a `tokenizer.json` (or a `model_index.json` for architecture detection).
///
/// # Example
///
/// ```ignore
/// let pipeline = load_diffusion_model::<burn::backend::NdArray>(
///     DiffusionArchitecture::StableDiffusion1_5,
///     "/path/to/sd-1.5",
///     &Default::default(),
/// ).unwrap();
/// ```
pub fn load_diffusion_model<B: Backend>(
    architecture: DiffusionArchitecture,
    model_dir: impl AsRef<Path>,
    device: &B::Device,
) -> Result<StableDiffusionPipeline<B>> {
    load_diffusion_model_with_lora(architecture, model_dir, device, None)
}

/// Like [`load_diffusion_model`], with an optional LoRA fused into the
/// UNet (and text encoder, when the file carries `lora_te`/`text_encoder`
/// keys) at load time.
pub fn load_diffusion_model_with_lora<B: Backend>(
    architecture: DiffusionArchitecture,
    model_dir: impl AsRef<Path>,
    device: &B::Device,
    lora: Option<&combs_formats::LoraSpec>,
) -> Result<StableDiffusionPipeline<B>> {
    let model_dir = model_dir.as_ref();

    match architecture {
        DiffusionArchitecture::StableDiffusion1_5 => {
            load_stable_diffusion_1_5(model_dir, device, lora)
        }
    }
}

fn load_stable_diffusion_1_5<B: Backend>(
    model_dir: &Path,
    device: &B::Device,
    lora: Option<&combs_formats::LoraSpec>,
) -> Result<StableDiffusionPipeline<B>> {
    let unet_source = SafetensorsSource::load_weights_only(
        model_dir.join("unet"),
        "stable-diffusion-unet",
    )?;
    let vae_source = SafetensorsSource::load_weights_only(
        model_dir.join("vae"),
        "stable-diffusion-vae",
    )?;
    let text_source = SafetensorsSource::load_weights_only(
        model_dir.join("text_encoder"),
        "stable-diffusion-clip",
    )?;

    let tokenizer = load_tokenizer_spec(model_dir)?;

    if let Some(spec) = lora {
        let file = combs_formats::LoraFile::load(&spec.path)?;
        if !file.unrecognized.is_empty() {
            eprintln!(
                "lora: {} keys matched no known naming scheme (first: {})",
                file.unrecognized.len(),
                file.unrecognized[0]
            );
        }
        let unet_merged =
            combs_formats::merge_lora(unet_source, &file.unet, spec.scale, file.file_alpha)?;
        let text_merged =
            combs_formats::merge_lora(text_source, &file.text, spec.scale, file.file_alpha)?;
        for (label, applied, skipped) in [
            ("unet", unet_merged.applied, &unet_merged.skipped),
            ("text_encoder", text_merged.applied, &text_merged.skipped),
        ] {
            if applied > 0 || !skipped.is_empty() {
                eprintln!(
                    "lora {label}: fused {applied} tensors, skipped {}{}",
                    skipped.len(),
                    skipped.first().map(|s| format!(" (first: {s})")).unwrap_or_default()
                );
            }
        }
        if unet_merged.applied == 0 && text_merged.applied == 0 {
            return Err(FormatError::Safetensors(
                "lora file matched no tensors in this checkpoint".to_string(),
            ));
        }
        return StableDiffusionPipeline::load_from_dirs(
            &unet_merged,
            &vae_source,
            &text_merged,
            tokenizer,
            device,
        );
    }

    StableDiffusionPipeline::load_from_dirs(
        &unet_source,
        &vae_source,
        &text_source,
        tokenizer,
        device,
    )
}

/// Read `tokenizer.json` / `tokenizer_config.json` from a Diffusers root.
/// Accepts either a root `tokenizer.json` or `tokenizer/tokenizer.json`.
fn load_tokenizer_spec(model_dir: &Path) -> Result<TokenizerSpec> {
    let tokenizer_json = if model_dir.join("tokenizer.json").is_file() {
        model_dir.join("tokenizer.json")
    } else if model_dir.join("tokenizer/tokenizer.json").is_file() {
        model_dir.join("tokenizer/tokenizer.json")
    } else {
        return Err(FormatError::MissingFile(
            "tokenizer.json or tokenizer/tokenizer.json".to_string(),
        ));
    };

    let mut added_tokens = std::collections::HashMap::new();
    let mut chat_template = None;

    let config_path = model_dir.join("tokenizer_config.json");
    if config_path.exists() {
        let text = std::fs::read_to_string(&config_path).map_err(FormatError::Io)?;
        let config: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
            FormatError::Json {
                context: config_path.display().to_string(),
                source: e,
            }
        })?;
        if let Some(map) = config.get("added_tokens_decoder").and_then(|v| v.as_object()) {
            for (id, entry) in map {
                if let (Ok(id), Some(content)) = (
                    id.parse::<u32>(),
                    entry.get("content").and_then(|c| c.as_str()),
                ) {
                    added_tokens.insert(id, content.to_string());
                }
            }
        }
        chat_template = config
            .get("chat_template")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
    }

    Ok(TokenizerSpec {
        tokenizer: combs_formats::TokenizerSource::Path(tokenizer_json),
        added_tokens,
        chat_template,
        add_bos: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_from_component_layout() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir(root.join("unet")).unwrap();
        std::fs::create_dir(root.join("vae")).unwrap();
        std::fs::create_dir(root.join("text_encoder")).unwrap();

        assert_eq!(
            DiffusionArchitecture::detect(root).unwrap(),
            DiffusionArchitecture::StableDiffusion1_5,
        );
    }

    #[test]
    fn detect_from_model_index() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(
            root.join("model_index.json"),
            r#"{"_class_name": "StableDiffusionPipeline"}"#,
        )
        .unwrap();
        assert_eq!(
            DiffusionArchitecture::detect(root).unwrap(),
            DiffusionArchitecture::StableDiffusion1_5,
        );
    }
}
