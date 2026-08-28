//! `OnnxSource` — the ONNX weights container behind [`ModelSource`].
//!
//! ONNX LLM exports are DIRECTORY-shaped: the graph file (and its
//! optional external-data sidecar) sits beside — or one level below —
//! the ordinary HF siblings (config.json, tokenizer.json,
//! tokenizer_config.json / chat_template.jinja). This source mmaps
//! the graph and any sidecars, parses the initializer table once
//! ([`crate::onnx::OnnxModel`]), normalizes names to the HF canonical
//! forms the model loaders speak, and serves dense f32/f16/bf16
//! tensors zero-copy. Block-quantized `MatMulNBits` weights are
//! visible in the table but not yet served — that is the quant-bridge
//! step's job, and asking for one errs clearly instead of guessing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::onnx::{OnnxData, OnnxDtype, OnnxModel};
use crate::source::{ModelSource, TensorDtype, TensorReader};
use crate::tokenizer::{TokenizerSource, TokenizerSpec};
use crate::{FormatError, ModelMetadata, Result, SamplerConfig};

pub struct OnnxSource {
    metadata: ModelMetadata,
    tokenizer: TokenizerSpec,
    model: OnnxModel,
    image: Mmap,
    /// External-data sidecars by their manifest `location`.
    sidecars: HashMap<String, Mmap>,
    /// HF-canonical name → initializer name.
    names: HashMap<String, String>,
}

/// Normalize an ONNX initializer name to the HF-canonical form the
/// loaders speak. torch.onnx exports keep state-dict paths but may
/// prefix graph scoping (`/model/...`) or `onnx::` artifacts.
fn canonical_name(raw: &str) -> String {
    let mut s = raw.trim_start_matches('/').replace('/', ".");
    if let Some(rest) = s.strip_prefix("onnx..") {
        s = rest.to_string();
    }
    s
}

fn read_json(path: &Path) -> Result<Option<serde_json::Value>> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(Some(serde_json::from_str(&text).map_err(|e| {
            FormatError::Json { context: path.display().to_string(), source: e }
        })?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(FormatError::Io(e)),
    }
}

impl OnnxSource {
    /// Open a `.onnx` graph file. Siblings are searched in the file's
    /// directory first, then one level up (the `repo/onnx/model.onnx`
    /// layout HF exports use).
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)?;
        let image = unsafe { Mmap::map(&file)? };
        let model = OnnxModel::parse(&image)?;

        let dir = path.parent().unwrap_or(Path::new("."));
        let sibling_dirs: Vec<&Path> =
            std::iter::once(dir).chain(dir.parent()).collect();
        let find = |name: &str| -> Option<PathBuf> {
            sibling_dirs
                .iter()
                .map(|d| d.join(name))
                .find(|p| p.exists())
        };

        // --- metadata ----------------------------------------------------
        let config_path = find("config.json").ok_or_else(|| {
            FormatError::MissingFile(format!(
                "config.json beside {} (ONNX exports are directory-shaped)",
                path.display()
            ))
        })?;
        let config = read_json(&config_path)?.expect("existence checked");
        let generation_config = find("generation_config.json")
            .map(|p| read_json(&p))
            .transpose()?
            .flatten();
        let metadata = ModelMetadata::from_hf_config(&config, generation_config.as_ref())?;

        // --- tokenizer ---------------------------------------------------
        let tokenizer_json = find("tokenizer.json").ok_or_else(|| {
            FormatError::MissingFile(format!("tokenizer.json beside {}", path.display()))
        })?;
        let mut added_tokens = HashMap::new();
        let mut chat_template = None;
        let mut add_bos = None;
        if let Some(tc) = find("tokenizer_config.json").map(|p| read_json(&p)).transpose()?.flatten()
        {
            if let Some(map) = tc.get("added_tokens_decoder").and_then(|v| v.as_object()) {
                for (id, entry) in map {
                    if let (Ok(id), Some(content)) =
                        (id.parse::<u32>(), entry.get("content").and_then(|c| c.as_str()))
                    {
                        added_tokens.insert(id, content.to_string());
                    }
                }
            }
            chat_template = tc.get("chat_template").and_then(|v| v.as_str()).map(String::from);
            add_bos = tc.get("add_bos_token").and_then(|v| v.as_bool());
        }
        // Newer exports move the template into its own jinja file.
        if chat_template.is_none() {
            if let Some(jinja) = find("chat_template.jinja") {
                chat_template = Some(std::fs::read_to_string(jinja)?);
            }
        }
        let tokenizer = TokenizerSpec {
            tokenizer: TokenizerSource::Path(tokenizer_json),
            added_tokens,
            chat_template,
            add_bos,
        };

        // --- external sidecars ------------------------------------------
        let mut sidecars = HashMap::new();
        for info in model.tensors.values() {
            if let OnnxData::External { location, .. } = &info.data {
                if !sidecars.contains_key(location) {
                    let p = dir.join(location);
                    let f = std::fs::File::open(&p).map_err(|e| {
                        FormatError::MissingFile(format!(
                            "external data {} ({e})",
                            p.display()
                        ))
                    })?;
                    sidecars.insert(location.clone(), unsafe { Mmap::map(&f)? });
                }
            }
        }

        // --- name table --------------------------------------------------
        let mut names = HashMap::new();
        for raw in model.tensors.keys() {
            names.insert(canonical_name(raw), raw.clone());
        }

        Ok(Self { metadata, tokenizer, model, image, sidecars, names })
    }

    /// The parsed container table (quant-bridge and diagnostics).
    pub fn table(&self) -> &OnnxModel {
        &self.model
    }

    fn tensor_bytes(&self, info: &crate::onnx::OnnxTensorInfo) -> Result<&[u8]> {
        match &info.data {
            OnnxData::Inline { offset, len } => {
                self.image.get(*offset..offset + len).ok_or_else(|| {
                    FormatError::Safetensors(format!(
                        "onnx: tensor {} inline range out of bounds",
                        info.name
                    ))
                })
            }
            OnnxData::External { location, offset, length } => {
                let map = self.sidecars.get(location).ok_or_else(|| {
                    FormatError::MissingFile(location.clone())
                })?;
                let (s, e) = (*offset as usize, (*offset + *length) as usize);
                map.get(s..e).ok_or_else(|| {
                    FormatError::Safetensors(format!(
                        "onnx: tensor {} external range out of bounds ({location})",
                        info.name
                    ))
                })
            }
        }
    }
}

impl ModelSource for OnnxSource {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn tensor_names(&self) -> Vec<String> {
        self.names.keys().cloned().collect()
    }

    fn open_tensor(&self, name: &str) -> Result<TensorReader<'_>> {
        let raw = self
            .names
            .get(name)
            .ok_or_else(|| FormatError::TensorNotFound(name.to_string()))?;
        let info = &self.model.tensors[raw];
        let dtype = match info.dtype {
            OnnxDtype::F32 => TensorDtype::F32,
            OnnxDtype::F16 => TensorDtype::F16,
            OnnxDtype::BF16 => TensorDtype::BF16,
            OnnxDtype::I64 => TensorDtype::I64,
            OnnxDtype::I32 => TensorDtype::I32,
            other => {
                return Err(FormatError::UnsupportedDtype {
                    tensor: name.to_string(),
                    dtype: format!(
                        "{other:?} (block-quantized ONNX weights go through the quant bridge, \
                         not the dense path)"
                    ),
                })
            }
        };
        let shape: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
        Ok(TensorReader::new(
            name.to_string(),
            shape,
            dtype,
            self.tensor_bytes(info)?,
        ))
    }

    fn tokenizer(&self) -> Result<TokenizerSpec> {
        Ok(self.tokenizer.clone())
    }

    fn sampler_defaults(&self) -> Option<SamplerConfig> {
        None
    }
}
