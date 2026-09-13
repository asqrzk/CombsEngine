//! Qwen3.5 — the `qwen35` GGUF family, three Gated DeltaNet layers to every
//! gated full-attention layer — against the transformers reference, on the
//! host backend.
//!
//! The fixture is a 4-layer model with seeded random weights, converted to
//! GGUF the way llama.cpp's converter writes real files (tiled V heads,
//! A = -exp(A_log), norms +1 but the gated one). `fixtures/qwen35_tiny.py`
//! regenerates it and records the reference logits; the reference's own
//! chunked prefill and recurrent decode agree to 1.4e-4.

use std::path::PathBuf;

use burn::backend::NdArray;
use burn::tensor::{backend::Backend, Int, Tensor, TensorData};
use combs_formats::GgufSource;
use combs_models::{CacheConfig, GenerativeModel, ModelRegistry};

type B = NdArray<f32>;

/// Logit tolerance with f32 arithmetic on both sides (the host, and the
/// f32 production build): the logits span about ±16, the reference
/// disagrees with itself by 1.4e-4 across its chunked and recurrent roads,
/// and a wrong head order, gate or decay lands near 1e0.
const TOL: f32 = 1e-3;

/// On the f16 production build the embeddings, every projection output and
/// the residual stream are stored in f16, one rounding of up to 2^-11 of
/// each value per store; over four layers into a ±16 logit range that
/// bounds the drift near 2e-2. The delta rule, the norms and both gates run
/// in f32. A real mistake still lands near 1e0.
///
/// That argument counted the activations and missed the weights: the f16
/// build also holds every dense weight in f16, rounded once at load, and
/// this all-dense fixture feels it on every projection (a real packed file
/// keeps its weights packed). Weight rounding alone moves the reference by
/// 1.4e-1, so the f16 build is held to the reference computed with its
/// weights rounded the same way — the tolerance is unchanged — and its
/// drift from the f32-weight reference is printed beside it.
const TOL_F16_BUILD: f32 = 5e-2;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// Every number inside the array that follows `"key"` (nested arrays
/// flattened) — the fixture is machine-written, so a bracket scan is enough.
fn numbers(json: &str, key: &str) -> Vec<f32> {
    let at = json
        .find(&format!("\"{key}\""))
        .unwrap_or_else(|| panic!("fixture has no {key}"));
    let rest = &json[at..];
    let open = rest.find('[').expect("array");
    let mut depth = 0;
    let mut close = open;
    for (i, ch) in rest[open..].char_indices() {
        match ch {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    close = open + i;
                    break;
                }
            }
            _ => {}
        }
    }
    rest[open..=close]
        .split(|c: char| !(c.is_ascii_digit() || matches!(c, '-' | '.' | 'e' | 'E' | '+')))
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().expect("number"))
        .collect()
}

/// One reference: the prefill logits of every position, and the logits of
/// prefilling `prefix` tokens then decoding the rest one at a time.
struct Reference {
    logits: Vec<f32>,
    steps: Vec<f32>,
}

struct Fixture {
    ids: Vec<i64>,
    prefix: usize,
    vocab: usize,
    /// Weights as the file stores them (f32).
    f32_weights: Reference,
    /// The same model with its weights held in f16, as the f16 build
    /// holds them; arithmetic still f32.
    f16_weights: Reference,
}

/// The unsigned integer that follows `"key":`.
fn scalar(json: &str, key: &str) -> usize {
    let tag = format!("\"{key}\":");
    let at = json
        .find(&tag)
        .unwrap_or_else(|| panic!("fixture has no {key}"))
        + tag.len();
    json[at..]
        .trim_start()
        .split(|c: char| !c.is_ascii_digit())
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("fixture {key} is not an integer"))
}

fn fixture() -> Fixture {
    let json = std::fs::read_to_string(fixture_path("qwen35_tiny.json")).expect("fixture json");
    Fixture {
        ids: numbers(&json, "ids")
            .into_iter()
            .map(|v| v as i64)
            .collect(),
        prefix: scalar(&json, "prefix"),
        vocab: scalar(&json, "vocab"),
        f32_weights: Reference {
            logits: numbers(&json, "logits"),
            steps: numbers(&json, "step_logits"),
        },
        f16_weights: Reference {
            logits: numbers(&json, "logits_f16_weights"),
            steps: numbers(&json, "step_logits_f16_weights"),
        },
    }
}

fn load_on<Bk: Backend>(device: &Bk::Device) -> Box<dyn GenerativeModel<Bk>> {
    // From bytes, not the path: a path-backed source caches its synthesized
    // tokenizer beside the file, and a test must not write into the tree.
    let bytes = std::fs::read(fixture_path("qwen35_tiny.gguf")).expect("fixture gguf");
    let source = GgufSource::from_bytes(bytes).expect("fixture parses");
    ModelRegistry::<Bk>::new()
        .load(&source, device)
        .expect("the qwen35 family loads")
}

fn tokens_on<Bk: Backend>(ids: &[i64], device: &Bk::Device) -> Tensor<Bk, 2, Int> {
    Tensor::from_data(TensorData::new(ids.to_vec(), [1, ids.len()]), device)
}

fn max_abs(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "logit count");
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

/// The reference a build is held to, and the other one printed beside it.
fn references(fx: &Fixture, f16_weights: bool) -> (&Reference, &Reference) {
    if f16_weights {
        (&fx.f16_weights, &fx.f32_weights)
    } else {
        (&fx.f32_weights, &fx.f16_weights)
    }
}

fn prefill_parity_on<Bk: Backend>(device: &Bk::Device, tol: f32, f16_weights: bool) {
    let fx = fixture();
    let mut model = load_on::<Bk>(device);
    let mut cache = model.create_kv_cache(&CacheConfig::paged(64));
    let seq = fx.ids.len();
    let input = model.embed(tokens_on::<Bk>(&fx.ids, device));
    let logits = model
        .prefill_all_logits(input, cache.as_mut(), 0..seq as u32)
        .expect("all logits");
    assert_eq!(logits.dims(), [1, seq, fx.vocab]);
    let got: Vec<f32> = logits
        .into_data()
        .convert::<f32>()
        .to_vec()
        .expect("logits");
    assert!(
        got.iter().all(|v| v.is_finite()),
        "non-finite prefill logits"
    );
    let (want, other) = references(&fx, f16_weights);
    let d = max_abs(&got, &want.logits);
    println!(
        "prefill max |d| = {d:.2e} over {} logits (from the other reference: {:.2e})",
        got.len(),
        max_abs(&got, &other.logits)
    );
    assert!(
        d < tol,
        "prefill logits drift {d:.3e} from the reference (tol {tol:.0e})"
    );
}

fn decode_parity_on<Bk: Backend>(device: &Bk::Device, tol: f32, f16_weights: bool) {
    let fx = fixture();
    let mut model = load_on::<Bk>(device);
    let mut cache = model.create_kv_cache(&CacheConfig::paged(64));
    let input = model.embed(tokens_on::<Bk>(&fx.ids[..fx.prefix], device));
    let first = model.prefill(input, cache.as_mut(), 0..fx.prefix as u32);
    let mut got: Vec<f32> = first.into_data().convert::<f32>().to_vec().expect("logits");
    for t in fx.prefix..fx.ids.len() {
        let input = model.embed(tokens_on::<Bk>(&fx.ids[t..t + 1], device));
        let step = model.decode(input, cache.as_mut());
        got.extend(
            step.into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .expect("logits"),
        );
    }
    assert!(
        got.iter().all(|v| v.is_finite()),
        "non-finite decode logits"
    );
    let (want, other) = references(&fx, f16_weights);
    let d = max_abs(&got, &want.steps);
    println!(
        "prefill {} + decode {} max |d| = {d:.2e} (from the other reference: {:.2e})",
        fx.prefix,
        fx.ids.len() - fx.prefix,
        max_abs(&got, &other.steps)
    );
    assert!(
        d < tol,
        "decode logits drift {d:.3e} from the reference (tol {tol:.0e})"
    );
}

/// The production build's tolerance and reference: the f16 build is held to
/// the f16-weight reference, every other build to the f32 one.
fn production() -> (f32, bool) {
    if combs_core::BACKEND.dtype == "f16" {
        (TOL_F16_BUILD, true)
    } else {
        (TOL, false)
    }
}

#[test]
fn a_tiny_qwen35_prefill_matches_transformers() {
    prefill_parity_on::<B>(&Default::default(), TOL, false);
}

#[test]
fn decoding_one_token_at_a_time_carries_the_recurrent_state() {
    decode_parity_on::<B>(&Default::default(), TOL, false);
}

#[test]
#[ignore = "gpu"]
fn a_tiny_qwen35_prefill_matches_transformers_on_production_backend() {
    println!("backend: {}", combs_core::backend_line());
    let (tol, f16_weights) = production();
    prefill_parity_on::<combs_core::CombsBackend>(&Default::default(), tol, f16_weights);
}

#[test]
#[ignore = "gpu"]
fn decoding_carries_the_recurrent_state_on_production_backend() {
    println!("backend: {}", combs_core::backend_line());
    let (tol, f16_weights) = production();
    decode_parity_on::<combs_core::CombsBackend>(&Default::default(), tol, f16_weights);
}

#[test]
fn the_recurrent_cache_cannot_roll_back() {
    let fx = fixture();
    let device = Default::default();
    let mut model = load_on::<B>(&device);
    let mut cache = model.create_kv_cache(&CacheConfig::paged(64));
    let input = model.embed(tokens_on::<B>(&fx.ids[..fx.prefix], &device));
    let _ = model.prefill(input, cache.as_mut(), 0..fx.prefix as u32);
    // the recurrent state is one point in time: a partial rollback would
    // pair an old KV prefix with a state that has already seen the rest
    assert_eq!(cache.popn(2), 0, "a rollback was accepted");
    assert_eq!(cache.popn(0), 0);
    assert_eq!(cache.seq_len(), fx.prefix);
}
