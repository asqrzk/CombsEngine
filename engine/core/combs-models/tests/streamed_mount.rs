//! A model assembled from bytes arriving in arbitrary pieces must be
//! the same model as one read from the file.
//!
//! The claim streaming rests on is narrow and total: a mount that never
//! holds the file, that meets tensors in whatever order the network
//! hands them over and in whatever sized pieces, produces weights
//! indistinguishable from the whole-file load. Not close — the same. So
//! the test compares logits, which depend on every weight at once, and
//! it does it under chunk sizes chosen to be awkward: one byte at a
//! time, pieces smaller than a tensor, pieces spanning several.
//!
//! Runs against a locally cached model and skips, loudly, when it is
//! absent — the parity claim is about real checkpoints, and a synthetic
//! one would only prove the test agrees with itself.

use burn::tensor::{Int, Tensor, TensorData};
use combs_formats::{GgufSource, ModelSource, read_gguf_header};
use combs_models::staged::StagedWeights;
use combs_models::{CacheConfig, ModelRegistry};

type B = burn::backend::Wgpu<f32, i32, u32>;

fn cached(dir: &str) -> Option<std::path::PathBuf> {
    let path = std::path::PathBuf::from(std::env::var("HOME").ok()?)
        .join(".cache/combs/models")
        .join(dir)
        .join("model.gguf");
    path.exists().then_some(path)
}

/// Deterministic chunk sizes in `[lo, hi]`. Seeded so a failure is
/// reproducible: "it passed on my machine" is not a property a mount
/// should have.
struct Chunks {
    state: u64,
    lo: usize,
    hi: usize,
}

impl Chunks {
    fn new(seed: u64, lo: usize, hi: usize) -> Self {
        Chunks { state: seed | 1, lo, hi }
    }
    fn next(&mut self) -> usize {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.lo + (self.state as usize) % (self.hi - self.lo + 1)
    }
}

/// Mount a model the way a network would deliver it: header first from a
/// growing prefix, then payload in pieces, staging each tensor while its
/// bytes are briefly in the window and dropping them after.
fn mount_streamed(
    bytes: &[u8],
    device: &burn::backend::wgpu::WgpuDevice,
    mut chunks: Chunks,
) -> (Box<dyn combs_models::GenerativeModel<B>>, usize) {
    let total = bytes.len();

    // Phase one: feed until the header parses. The point of asking
    // repeatedly is that "not yet" has to be an answer — a mount cannot
    // know how big a header is before reading it.
    // The header phase pulls in real network-sized pieces even when the
    // payload phase is testing byte-at-a-time: re-parsing a megabyte of
    // header once per byte is quadratic and proves nothing the boundary
    // test below does not prove exactly.
    let mut have = 0usize;
    let mut asks = 0usize;
    let header = loop {
        have = (have + chunks.next().max(64 << 10)).min(total);
        asks += 1;
        match read_gguf_header(&bytes[..have], Some(total)).expect("header parses or fails") {
            Some(h) => break h,
            None => assert!(have < total, "header never parsed even with the whole file"),
        }
    };
    assert!(asks > 0);
    let header_bytes = bytes[..header.data_start].to_vec();
    assert!(
        ModelRegistry::<B>::supports_streaming(&header.architecture),
        "{} refused for streaming at the header",
        header.architecture
    );

    // Phase two: walk the payload. The window holds only what has
    // arrived and not yet been consumed; every tensor is staged the
    // moment it is complete, and the window then forgets it.
    let mut staged: Option<StagedWeights<B>> = None;
    let mut base = header.data_start;
    let mut window: Vec<u8> = Vec::new();
    let mut cursor = header.data_start;
    let mut next_tensor = 0usize;
    let mut peak_window = 0usize;

    while next_tensor < header.tensors.len() {
        let (name, start, size) = header.tensors[next_tensor].clone();
        // Pull until this tensor is entirely in the window.
        while cursor < start + size {
            let take = chunks.next().min(total - cursor);
            window.extend_from_slice(&bytes[cursor..cursor + take]);
            cursor += take;
            peak_window = peak_window.max(window.len());
            if take == 0 {
                break;
            }
        }
        let source = GgufSource::from_window(&header_bytes, window.clone(), base, total)
            .expect("windowed source");
        let weights = staged.get_or_insert_with(|| StagedWeights::new(source.metadata().clone()));
        for (hf, _range) in source.hf_names_for_ggml(&name) {
            weights.stage(&source, device, &hf).unwrap_or_else(|e| {
                panic!("staging {name} -> {hf} at {start}..{}: {e}", start + size)
            });
        }
        next_tensor += 1;
        // Forget everything up to the end of what was just consumed —
        // this is the line that makes the mount bounded rather than a
        // slow way of holding the file.
        let consumed = start + size - base;
        window.drain(..consumed);
        base = start + size;
    }

    let mut weights = staged.expect("at least one tensor");
    weights.seal();
    let model = ModelRegistry::<B>::new()
        .load_staged(&mut weights, device)
        .expect("staged model builds");
    (model, peak_window)
}

fn logits_of(
    model: &mut dyn combs_models::GenerativeModel<B>,
    device: &burn::backend::wgpu::WgpuDevice,
    ids: &[i32],
) -> Vec<f32> {
    let tokens = Tensor::<B, 2, Int>::from_data(
        TensorData::new(ids.to_vec(), [1, ids.len()]),
        device,
    );
    let embeds = model.embed(tokens);
    let mut cache = model.create_kv_cache(&CacheConfig::contiguous(64));
    model
        .prefill(embeds, cache.as_mut(), 0..ids.len() as u32)
        .into_data()
        .to_vec()
        .unwrap()
}

#[test]
fn a_streamed_mount_is_the_same_model_as_a_read_one() {
    let Some(path) = cached("smollm2-360m-instruct-gguf") else {
        eprintln!("skip: smollm2-360m-instruct-gguf is not in the local cache");
        return;
    };
    let device = burn::backend::wgpu::WgpuDevice::default();
    let ids: Vec<i32> = vec![1, 2278, 314, 1101, 460, 3021, 8, 15, 1000, 42];

    let classic = GgufSource::load(&path).expect("classic load");
    let mut reference = ModelRegistry::<B>::new()
        .load(&classic, &device)
        .expect("classic model");
    let expect = logits_of(reference.as_mut(), &device, &ids);
    assert!(
        expect.iter().all(|v| v.is_finite()),
        "the reference itself produced non-finite logits"
    );
    drop(reference);

    let bytes = std::fs::read(&path).expect("read file");
    // Byte-at-a-time proves nothing is assumed about alignment or about
    // a tensor arriving whole; the wide range is what a network does.
    for (seed, lo, hi) in [(0xC0FFEE_u64, 1, 1), (0x5EED, 1, 4096), (0xBEEF, 1 << 16, 8 << 20)] {
        let (mut streamed, peak) = mount_streamed(&bytes, &device, Chunks::new(seed, lo, hi));
        let got = logits_of(streamed.as_mut(), &device, &ids);
        assert_eq!(got.len(), expect.len(), "chunking {lo}..={hi}: logit count");
        assert_eq!(
            got, expect,
            "chunking {lo}..={hi}: logits differ from the whole-file load"
        );
        eprintln!(
            "chunks {lo}..={hi}: identical logits, window peaked at {:.1} MB of a {:.1} MB file",
            peak as f64 / 1e6,
            bytes.len() as f64 / 1e6
        );
        assert!(
            peak < bytes.len(),
            "the window held the whole file, which is not a stream"
        );
    }
}

/// The boundary itself: every prefix short of the header is "not yet",
/// and the first sufficient one parses. This is the property the mount
/// leans on when it decides whether to pull more or to start placing
/// weights, and it is cheaper to check directly than to crawl to it.
#[test]
fn a_short_header_says_not_yet_and_never_lies() {
    let Some(path) = cached("smollm2-360m-instruct-gguf") else {
        eprintln!("skip: smollm2-360m-instruct-gguf is not in the local cache");
        return;
    };
    let bytes = std::fs::read(&path).expect("read file");
    let total = bytes.len();
    let full = read_gguf_header(&bytes, Some(total))
        .expect("whole file parses")
        .expect("whole file is enough");
    let end = full.data_start;

    // `data_start` is the ALIGNMENT boundary after the info section, so
    // its last bytes are padding the parser never reads and a prefix a
    // few bytes short of it parses fine. The alignment is 32 by default
    // and the file may declare its own, so the last prefix checked here
    // stops well clear of it — asserting `end - 1` says not-yet would be
    // asserting something untrue about padding.
    for n in [0usize, 1, 4, 8, 24, end / 4, end / 2, end - 64] {
        let got = read_gguf_header(&bytes[..n], Some(total));
        assert!(
            matches!(got, Ok(None)),
            "a {n}-byte prefix of a {end}-byte header should be `not yet`, got {:?}",
            got.map(|o| o.map(|h| h.data_start))
        );
    }
    let at_end = read_gguf_header(&bytes[..end], Some(total))
        .expect("parses")
        .expect("the header is all there at data_start");
    assert_eq!(at_end.data_start, end);
    assert_eq!(at_end.tensors.len(), full.tensors.len());

    // Corruption is not shortness. A bad magic never becomes valid by
    // waiting for more bytes, and the mount must be told so rather than
    // asking forever.
    let mut wrong = bytes[..end].to_vec();
    wrong[0] ^= 0xFF;
    assert!(
        read_gguf_header(&wrong, Some(total)).is_err(),
        "a corrupt magic reported as `not yet` would hang a mount"
    );
}
