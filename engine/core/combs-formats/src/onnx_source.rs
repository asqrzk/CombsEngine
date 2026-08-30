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
use crate::onnx_quant::{dequantize_matmul_nbits, repack_matmul_nbits_q4_0};
use crate::source::{ModelSource, QuantFormat, QuantTensor, TensorDtype, TensorReader};
use crate::tokenizer::{TokenizerSource, TokenizerSpec};
use crate::{FormatError, ModelMetadata, Result, SamplerConfig};

/// The graph's bytes, however they got here. Mapped from a file
/// natively; owned when a page handed them over, because a browser has
/// no filesystem to map and the alternative — a second code path for
/// reading initializers — is the one place a format adapter must not
/// have two of.
enum Image {
    #[cfg(not(target_family = "wasm"))]
    Mapped(Mmap),
    Owned(Vec<u8>),
}

impl std::ops::Deref for Image {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            #[cfg(not(target_family = "wasm"))]
            Image::Mapped(m) => m,
            Image::Owned(v) => v,
        }
    }
}

/// Everything an ONNX export keeps beside its graph. A directory holds
/// these as files; a page has to hand them over, so they are named once
/// here and both roads fill the same shape.
pub struct OnnxSiblings {
    /// `config.json` — required, it carries the architecture.
    pub config: String,
    /// `tokenizer.json` — required.
    pub tokenizer_json: Vec<u8>,
    pub tokenizer_config: Option<String>,
    pub generation_config: Option<String>,
    /// `chat_template.jinja`, for exports that moved it out of
    /// `tokenizer_config.json`.
    pub chat_template: Option<String>,
    /// External-data sidecars by their manifest `location`.
    pub sidecars: HashMap<String, Vec<u8>>,
}

pub struct OnnxSource {
    metadata: ModelMetadata,
    tokenizer: TokenizerSpec,
    model: OnnxModel,
    image: Image,
    /// External-data sidecars by their manifest `location`.
    sidecars: HashMap<String, Image>,
    /// HF-canonical name → (initializer name, needs transpose).
    names: HashMap<String, (String, bool)>,
    /// HF-canonical name → block-quantized MatMulNBits weight.
    quants: HashMap<String, QuantEntry>,
}

/// One MatMulNBits weight: the packed nibbles + scales initializers
/// and the geometry from the node attributes.
struct QuantEntry {
    packed: String,
    scales: String,
    k: usize,
    n: usize,
    block_size: usize,
    /// zero_points / g_idx present — Q4_0's fixed zero-point 8 no
    /// longer applies, so only the dequant fallback serves it.
    beyond_q4_0: bool,
}

/// Normalize an ONNX initializer name to the HF-canonical form the
/// loaders speak, and say whether its bytes need transposing.
///
/// The transformers.js / genai export dialect (verified on
/// onnx-community Qwen3): linear weights carry a fused-op suffix and
/// are stored `[in, out]` (ONNX MatMul computes X·W directly — the
/// transpose of HF's `[out, in]` linears); `self_attn` shortens to
/// `attn`; q/k norms gain a `.layernorm` segment; the final norm
/// masquerades as one-past-the-last layer
/// (`model.layers.{L}.final_norm_layernorm.weight`). Rope caches and
/// other non-weight initializers return None and stay hidden — the
/// tied head is a graph-level Transpose of the embedding, so
/// tied-by-absence loading works untouched.
fn canonical_name(raw: &str) -> Option<(String, bool)> {
    let mut s = raw.trim_start_matches('/').replace('/', ".");
    if let Some(rest) = s.strip_prefix("onnx..") {
        s = rest.to_string();
    }
    if !(s.starts_with("model.") || s.starts_with("lm_head.")) {
        return None; // cos_cache / sin_cache / graph scratch
    }
    if s.ends_with(".final_norm_layernorm.weight") && s.starts_with("model.layers.") {
        // The pseudo-layer index here is the layer COUNT, not a layer.
        return Some(("model.norm.weight".to_string(), false));
    }
    let transposed = s.contains(".MatMul.");
    let s = s
        .replace(".MatMul.weight", ".weight")
        .replace(".attn.", ".self_attn.")
        .replace(".layernorm.weight", ".weight");
    Some((s, transposed))
}

/// Transpose a row-major `[r, c]` element buffer to `[c, r]`.
fn transpose_bytes(data: &[u8], r: usize, c: usize, elem: usize) -> Vec<u8> {
    let mut out = vec![0u8; data.len()];
    for i in 0..r {
        for j in 0..c {
            let src = (i * c + j) * elem;
            let dst = (j * r + i) * elem;
            out[dst..dst + elem].copy_from_slice(&data[src..src + elem]);
        }
    }
    out
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
    ///
    /// Native only: a browser has no directory to search, and reaches
    /// the same source through [`OnnxSource::from_parts`].
    #[cfg(not(target_family = "wasm"))]
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)?;
        let image = Image::Mapped(unsafe { Mmap::map(&file)? });
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
                    sidecars.insert(location.clone(), Image::Mapped(unsafe { Mmap::map(&f)? }));
                }
            }
        }

        let (names, quants) = Self::index(&model);
        Ok(Self { metadata, tokenizer, model, image, sidecars, names, quants })
    }

    /// Open a graph a page handed over, with its siblings beside it.
    ///
    /// The directory road and this one build the SAME source: the
    /// quant table, the name table and the metadata are derived by the
    /// shared tail below, so a model cannot mean one thing when read
    /// from disk and another when read from a network.
    pub fn from_parts(graph: Vec<u8>, siblings: OnnxSiblings) -> Result<Self> {
        let image = Image::Owned(graph);
        let model = OnnxModel::parse(&image)?;

        let config: serde_json::Value = serde_json::from_slice(siblings.config.as_bytes())
            .map_err(|e| FormatError::Json { context: "config.json".into(), source: e })?;
        let generation_config = siblings
            .generation_config
            .as_deref()
            .map(|raw| {
                serde_json::from_str::<serde_json::Value>(raw).map_err(|e| FormatError::Json {
                    context: "generation_config.json".into(),
                    source: e,
                })
            })
            .transpose()?;
        let metadata = ModelMetadata::from_hf_config(&config, generation_config.as_ref())?;

        let mut added_tokens = HashMap::new();
        let mut chat_template = siblings.chat_template.clone();
        let mut add_bos = None;
        if let Some(raw) = siblings.tokenizer_config.as_deref() {
            let tc: serde_json::Value = serde_json::from_str(raw).map_err(|e| {
                FormatError::Json { context: "tokenizer_config.json".into(), source: e }
            })?;
            if let Some(map) = tc.get("added_tokens_decoder").and_then(|v| v.as_object()) {
                for (id, entry) in map {
                    if let (Ok(id), Some(content)) =
                        (id.parse::<u32>(), entry.get("content").and_then(|c| c.as_str()))
                    {
                        added_tokens.insert(id, content.to_string());
                    }
                }
            }
            if chat_template.is_none() {
                chat_template =
                    tc.get("chat_template").and_then(|v| v.as_str()).map(String::from);
            }
            add_bos = tc.get("add_bos_token").and_then(|v| v.as_bool());
        }
        let tokenizer = TokenizerSpec {
            tokenizer: TokenizerSource::Bytes(siblings.tokenizer_json),
            added_tokens,
            chat_template,
            add_bos,
        };

        let mut sidecars = HashMap::new();
        for info in model.tensors.values() {
            if let OnnxData::External { location, .. } = &info.data {
                if !sidecars.contains_key(location) {
                    let bytes = siblings.sidecars.get(location).ok_or_else(|| {
                        FormatError::MissingFile(format!(
                            "external data {location} was not delivered with the graph"
                        ))
                    })?;
                    sidecars.insert(location.clone(), Image::Owned(bytes.clone()));
                }
            }
        }

        let (names, quants) = Self::index(&model);
        Ok(Self { metadata, tokenizer, model, image, sidecars, names, quants })
    }

    /// The quant table and the name table, derived from the graph
    /// alone. Both constructors call this, which is what makes the
    /// claim that they build the same source true rather than
    /// intended.
    fn index(
        model: &OnnxModel,
    ) -> (HashMap<String, (String, bool)>, HashMap<String, QuantEntry>) {
        // --- quant table -------------------------------------------------
            // MatMulNBits weights come as `<name>.MatMul.weight_Q4` +
            // `..._scales` initializer PAIRS addressed by the node; they
            // surface under the canonical linear name, quantized — never
            // as raw dense tensors.
            let mut quants = HashMap::new();
            for node in &model.matmul_nbits {
                let Some(packed) = node.inputs.get(1) else { continue };
                let Some(scales) = node.inputs.get(2) else { continue };
                let base = packed.trim_end_matches("_Q4");
                let Some((canonical, _)) = canonical_name(base) else { continue };
                quants.insert(canonical, QuantEntry {
                    packed: packed.clone(),
                    scales: scales.clone(),
                    k: node.k as usize,
                    n: node.n as usize,
                    block_size: node.block_size as usize,
                    beyond_q4_0: node.inputs.len() > 3 || node.bits != 4,
                });
            }

            // --- name table --------------------------------------------------
            let quant_parts: std::collections::HashSet<&String> = quants
                .values()
                .flat_map(|q| [&q.packed, &q.scales])
                .collect();
            let mut names = HashMap::new();
            for raw in model.tensors.keys() {
                if quant_parts.contains(raw) {
                    continue;
                }
                if let Some((canonical, transposed)) = canonical_name(raw) {
                    names.insert(canonical, (raw.clone(), transposed));
                }
            }
            (names, quants)
    }

    /// A quant entry's scales as f32 (stored f16 or f32).
    fn scales_f32(&self, entry: &QuantEntry) -> Result<Vec<f32>> {
        let info = self.model.tensors.get(&entry.scales).ok_or_else(|| {
            FormatError::TensorNotFound(entry.scales.clone())
        })?;
        let bytes = self.tensor_bytes(info)?;
        Ok(match info.dtype {
            OnnxDtype::F32 => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            OnnxDtype::F16 => bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes(c.try_into().unwrap()).to_f32())
                .collect(),
            other => {
                return Err(FormatError::UnsupportedDtype {
                    tensor: entry.scales.clone(),
                    dtype: format!("{other:?} scales"),
                })
            }
        })
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
        self.names.keys().chain(self.quants.keys()).cloned().collect()
    }

    fn open_tensor(&self, name: &str) -> Result<TensorReader<'_>> {
        // Quantized weights: dequantize (the fallback for formats the
        // kernels don't take and for zero-point-carrying models).
        if let Some(entry) = self.quants.get(name) {
            let packed = self.tensor_bytes(
                self.model
                    .tensors
                    .get(&entry.packed)
                    .ok_or_else(|| FormatError::TensorNotFound(entry.packed.clone()))?,
            )?;
            let scales = self.scales_f32(entry)?;
            let values =
                dequantize_matmul_nbits(packed, &scales, entry.k, entry.n, entry.block_size)?;
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            return Ok(TensorReader::owned(
                name.to_string(),
                vec![entry.n, entry.k],
                bytes,
            ));
        }
        let (raw, transposed) = self
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
        let bytes = self.tensor_bytes(info)?;
        if *transposed {
            let [r, c] = shape[..] else {
                return Err(FormatError::Safetensors(format!(
                    "onnx: transposed tensor {name} is rank {}, expected 2",
                    shape.len()
                )));
            };
            return Ok(TensorReader::owned_with_dtype(
                name.to_string(),
                vec![c, r],
                dtype,
                transpose_bytes(bytes, r, c, dtype.size()),
            ));
        }
        Ok(TensorReader::new(name.to_string(), shape, dtype, bytes))
    }

    fn open_tensor_quant(&self, name: &str) -> Result<Option<QuantTensor<'_>>> {
        let Some(entry) = self.quants.get(name) else {
            return Ok(None);
        };
        if entry.beyond_q4_0 || entry.block_size != 32 || entry.k % 32 != 0 {
            return Ok(None); // dense fallback dequantizes instead
        }
        let packed = self.tensor_bytes(
            self.model
                .tensors
                .get(&entry.packed)
                .ok_or_else(|| FormatError::TensorNotFound(entry.packed.clone()))?,
        )?;
        let scales = self.scales_f32(entry)?;
        let data = repack_matmul_nbits_q4_0(packed, &scales, entry.k, entry.n, entry.block_size)?;
        Ok(Some(QuantTensor {
            format: QuantFormat::Q4_0,
            shape: vec![entry.n, entry.k],
            data: std::borrow::Cow::Owned(data),
        }))
    }

    fn tokenizer(&self) -> Result<TokenizerSpec> {
        Ok(self.tokenizer.clone())
    }

    fn sampler_defaults(&self) -> Option<SamplerConfig> {
        None
    }
}

#[cfg(test)]
mod parts_tests {
    use super::*;
    use crate::ModelSource;

    fn cache() -> Option<PathBuf> {
        let root = PathBuf::from(std::env::var("HOME").ok()?)
            .join(".cache/combs/models/qwen3-0.6b-onnx");
        root.join("onnx/model_q4f16.onnx").exists().then_some(root)
    }

    /// A graph handed over as bytes must build the same source as the
    /// same graph read from its directory.
    ///
    /// This is the whole claim behind putting ONNX in a browser: the
    /// page supplies what the filesystem supplied, and nothing else
    /// changes. Compared on what a model is actually made of — its
    /// metadata, every tensor name, and the bytes of the tensors
    /// themselves, quantized and dense.
    #[test]
    fn bytes_and_directory_build_the_same_source() {
        let Some(root) = cache() else {
            eprintln!("skip: qwen3-0.6b-onnx is not in the local cache");
            return;
        };
        let graph_path = root.join("onnx/model_q4f16.onnx");
        let from_dir = OnnxSource::load(&graph_path).expect("directory road");

        let read = |name: &str| std::fs::read(root.join(name)).ok();
        let read_str = |name: &str| std::fs::read_to_string(root.join(name)).ok();
        let from_bytes = OnnxSource::from_parts(
            std::fs::read(&graph_path).unwrap(),
            OnnxSiblings {
                config: read_str("config.json").expect("config.json"),
                tokenizer_json: read("tokenizer.json").expect("tokenizer.json"),
                tokenizer_config: read_str("tokenizer_config.json"),
                generation_config: read_str("generation_config.json"),
                chat_template: read_str("chat_template.jinja"),
                sidecars: HashMap::new(),
            },
        )
        .expect("bytes road");

        assert_eq!(
            from_dir.metadata().architecture,
            from_bytes.metadata().architecture
        );
        assert_eq!(from_dir.metadata().vocab_size, from_bytes.metadata().vocab_size);
        assert_eq!(
            from_dir.metadata().num_hidden_layers,
            from_bytes.metadata().num_hidden_layers
        );

        let mut a = from_dir.tensor_names();
        let mut b = from_bytes.tensor_names();
        a.sort();
        b.sort();
        assert_eq!(a, b, "the two roads disagree about which tensors exist");
        assert!(!a.is_empty(), "no tensors at all — the comparison would be empty");

        let mut quantized = 0;
        let mut dense = 0;
        for name in &a {
            match (
                from_dir.open_tensor_quant(name).unwrap(),
                from_bytes.open_tensor_quant(name).unwrap(),
            ) {
                (Some(x), Some(y)) => {
                    assert_eq!(x.format, y.format, "{name}: format");
                    assert_eq!(x.shape, y.shape, "{name}: shape");
                    assert_eq!(&*x.data, &*y.data, "{name}: packed bytes");
                    quantized += 1;
                }
                (None, None) => {
                    let x = from_dir.open_tensor(name).unwrap().load_data().unwrap();
                    let y = from_bytes.open_tensor(name).unwrap().load_data().unwrap();
                    assert_eq!(x.shape, y.shape, "{name}: dense shape");
                    assert_eq!(
                        x.to_vec::<f32>().unwrap(),
                        y.to_vec::<f32>().unwrap(),
                        "{name}: dense values"
                    );
                    dense += 1;
                }
                _ => panic!("{name}: the roads disagree on whether it is quantized"),
            }
        }
        eprintln!("{quantized} quantized and {dense} dense tensors identical across both roads");
    }
}
