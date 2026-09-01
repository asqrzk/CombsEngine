//! The mount's two promises: it stays small, and it fails cleanly.
//!
//! Small, because the point of streaming is that a model larger than the
//! address space can still be mounted — a mount whose window grows with
//! the file is just a slower way of holding it. Cleanly, because a
//! stream that stops halfway is the ordinary case on a network, and a
//! half-built model that answers anyway is the worst possible outcome:
//! wrong weights are indistinguishable from right ones until the output
//! is wrong, and by then the cause is long gone.

use burn::tensor::{Int, Tensor, TensorData};
use combs_formats::{read_gguf_header, GgufSource, ModelSource};
use combs_models::ModelRegistry;
use combs_runtime::stream_mount::{MountError, StreamMount};

use combs_core::CombsBackend as B;

fn cached(dir: &str) -> Option<std::path::PathBuf> {
    let path = std::path::PathBuf::from(std::env::var("HOME").ok()?)
        .join(".cache/combs/models")
        .join(dir)
        .join("model.gguf");
    path.exists().then_some(path)
}

fn feed(mount: &mut StreamMount, bytes: &[u8], chunk: usize) -> Result<(), MountError> {
    let mut at = 0;
    while at < bytes.len() {
        let take = chunk.min(bytes.len() - at);
        mount.append(&bytes[at..at + take])?;
        at += take;
    }
    Ok(())
}

/// The residency claim, on a real checkpoint: the window never holds
/// more than the largest tensor plus a chunk in flight. That floor is
/// not incidental — a tensor has to be whole to be uploaded — and it is
/// what bounds a mount by the model's widest layer rather than by the
/// model.
#[test]
fn the_window_never_grows_past_the_largest_tensor() {
    let Some(path) = cached("smollm2-360m-instruct-gguf") else {
        eprintln!("skip: smollm2-360m-instruct-gguf is not in the local cache");
        return;
    };
    let bytes = std::fs::read(&path).unwrap();
    let device = combs_core::init_device();
    let header = read_gguf_header(&bytes, Some(bytes.len() as u64))
        .unwrap()
        .unwrap();
    let largest = header
        .tensors
        .iter()
        .map(|(_, _, size)| *size)
        .max()
        .unwrap();

    for chunk in [1 << 20, 8 << 20] {
        let mut mount = StreamMount::new(bytes.len() as u64, device.clone());
        feed(&mut mount, &bytes, chunk).expect("stream feeds");
        let (weights, _header) = mount.finish().expect("mount completes");
        drop(weights);
        // Re-measured from a fresh mount each time; `finish` consumes.
        let mut probe = StreamMount::new(bytes.len() as u64, device.clone());
        feed(&mut probe, &bytes, chunk).expect("stream feeds");
        let peak = probe.window_high_water();
        eprintln!(
            "chunk {:>5} KB: window peaked at {:.1} MB, largest tensor {:.1} MB, file {:.1} MB",
            chunk / 1024,
            peak as f64 / 1e6,
            largest as f64 / 1e6,
            bytes.len() as f64 / 1e6
        );
        assert!(
            peak as u64 <= largest + 2 * chunk as u64,
            "window peaked at {peak}, above the largest tensor {largest} plus two chunks"
        );
        assert!(
            (peak as u64) < bytes.len() as u64 / 2,
            "a window holding half the file is not a stream"
        );
    }
}

/// Five ways for a stream to stop early, each of which must produce an
/// error that says what was owed rather than a model that pretends.
#[test]
fn a_stream_that_stops_early_never_yields_a_model() {
    let Some(path) = cached("smollm2-360m-instruct-gguf") else {
        eprintln!("skip: smollm2-360m-instruct-gguf is not in the local cache");
        return;
    };
    let bytes = std::fs::read(&path).unwrap();
    let device = combs_core::init_device();
    let header = read_gguf_header(&bytes, Some(bytes.len() as u64))
        .unwrap()
        .unwrap();
    let data_start = header.data_start as usize;
    let (_, first_start, first_size) = header.tensors[0].clone();
    let (first_start, first_size) = (first_start as usize, first_size as usize);

    let cuts = [
        ("nothing at all", 0usize),
        ("mid-header", data_start / 2),
        ("header exactly, no payload", data_start),
        ("mid-first-tensor", first_start + first_size / 2),
        ("one byte short", bytes.len() - 1),
    ];
    for (what, cut) in cuts {
        let mut mount = StreamMount::new(bytes.len() as u64, device.clone());
        feed(&mut mount, &bytes[..cut], 4 << 20).expect("a short stream still feeds");
        let err = match mount.finish() {
            Ok(_) => panic!("{what}: a {cut}-byte stream produced a model"),
            Err(e) => e,
        };
        match err {
            MountError::Truncated {
                received, expected, ..
            } => {
                assert_eq!(
                    received, cut as u64,
                    "{what}: reported the wrong received count"
                );
                assert_eq!(
                    expected,
                    bytes.len() as u64,
                    "{what}: reported the wrong expectation"
                );
            }
            other => panic!("{what}: expected a truncation, got {other}"),
        }
        eprintln!("{what} at {cut}: {err}");
    }
}

/// More bytes than the file was said to hold is a different failure from
/// too few, and is caught before any of them are staged.
#[test]
fn a_stream_that_overruns_is_refused_at_the_append() {
    // Real header bytes, because a mount that has not recognized the
    // file yet would reject these for being the wrong shape long before
    // it got to counting them, and then the test would be measuring the
    // wrong refusal.
    let Some(path) = cached("smollm2-360m-instruct-gguf") else {
        eprintln!("skip: smollm2-360m-instruct-gguf is not in the local cache");
        return;
    };
    let bytes = std::fs::read(&path).unwrap();
    let device = combs_core::init_device();
    let mut mount = StreamMount::new(100_000, device);
    mount.append(&bytes[..60_000]).expect("under the limit");
    match mount.append(&bytes[60_000..120_000]) {
        Err(MountError::Overflow { expected, got }) => {
            assert_eq!(expected, 100_000);
            assert_eq!(got, 120_000);
        }
        other => panic!("expected an overflow, got {other:?}"),
    }
}

/// A file that is not a GGUF at all fails at the first bytes, not after
/// the whole thing has been pulled.
#[test]
fn a_malformed_header_fails_immediately() {
    let device = combs_core::init_device();
    let mut mount = StreamMount::new(1 << 20, device);
    match mount.append(b"not a gguf file at all, not even close") {
        Err(MountError::BadHeader(_)) => {}
        other => panic!("expected a bad header, got {other:?}"),
    }
}

/// Deterministic chunk sizes in `[lo, hi]`, seeded so a failure is
/// reproducible. The same generator the models crate feeds its own
/// helper with; here it drives the production mount.
struct Chunks {
    state: u64,
    lo: usize,
    hi: usize,
}

impl Chunks {
    fn new(seed: u64, lo: usize, hi: usize) -> Self {
        Chunks {
            state: seed | 1,
            lo,
            hi,
        }
    }
    fn next(&mut self) -> usize {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.lo + (self.state as usize) % (self.hi - self.lo + 1)
    }
}

fn feed_random(
    mount: &mut StreamMount,
    bytes: &[u8],
    mut chunks: Chunks,
) -> Result<(), MountError> {
    let mut at = 0;
    while at < bytes.len() {
        let take = chunks.next().min(bytes.len() - at);
        mount.append(&bytes[at..at + take])?;
        at += take;
    }
    Ok(())
}

/// Last-position logits of a fixed prompt, which depend on every weight
/// at once. Copied from the models crate's streamed-mount test rather
/// than shared: integration tests are not libraries, and a public
/// helper for one test would be API for nobody.
fn logits_of(
    model: &mut dyn combs_models::GenerativeModel<B>,
    device: &combs_core::CombsDevice,
    ids: &[i32],
) -> Vec<f32> {
    let tokens =
        Tensor::<B, 2, Int>::from_data(TensorData::new(ids.to_vec(), [1, ids.len()]), device);
    let embeds = model.embed(tokens);
    let mut cache = model.create_kv_cache(&combs_models::CacheConfig::contiguous(64));
    model
        .prefill(embeds, cache.as_mut(), 0..ids.len() as u32)
        .into_data()
        .convert::<f32>()
        .to_vec()
        .unwrap()
}

/// `data_start` is an alignment boundary, so the last bytes of the
/// header are padding the parser never reads (24 in this file). A chunk
/// boundary can land there, and when it does the next chunk's first byte
/// is file offset `len`, not `data_start`. Before the fix the mount filed
/// it at `data_start` and every tensor staged 24 bytes early — with every
/// byte accounted for and `finish` content. The check is the byte
/// ledger: after a second, sub-tensor-sized append the window must hold
/// exactly `received - data_start` bytes.
#[test]
fn a_chunk_boundary_inside_the_alignment_padding_does_not_shift_the_weights() {
    let Some(path) = cached("smollm2-360m-instruct-gguf") else {
        eprintln!("skip: smollm2-360m-instruct-gguf is not in the local cache");
        return;
    };
    let bytes = std::fs::read(&path).unwrap();
    let len = bytes.len() as u64;
    let header = read_gguf_header(&bytes, Some(len)).unwrap().unwrap();
    let data_start = header.data_start as usize;
    let cut = data_start - 1;

    // The parser's half of the contract.
    assert!(
        matches!(read_gguf_header(&bytes[..cut], Some(len)), Ok(None)),
        "a prefix one byte short of data_start parsed as a whole header"
    );
    // Smaller than the first tensor, so nothing can stage and the window
    // is the whole ledger. Asserted, because another file's first tensor
    // can be tiny (llama-3.2's rope table is 128 bytes) and a probe that
    // completed it would quietly measure something else.
    let probe = 1024usize;
    let (_, _, first_size) = &header.tensors[0];
    assert!(
        (probe as u64) < *first_size,
        "probe must not complete the first tensor"
    );

    let device = combs_core::init_device();
    let mut mount = StreamMount::new(len, device);
    let after_cut = mount
        .append(&bytes[..cut])
        .expect("header minus one byte feeds");
    assert_eq!(
        after_cut.tensors_total, 0,
        "the mount accepted a header short of its padding"
    );
    assert_eq!(after_cut.window, 0);
    let after_probe = mount
        .append(&bytes[cut..cut + probe])
        .expect("first payload bytes feed");
    assert_eq!(after_probe.tensors_total, header.tensors.len());
    assert_eq!(after_probe.received, (cut + probe) as u64);
    assert_eq!(
        after_probe.window as u64,
        after_probe.received - data_start as u64,
        "one window byte per byte past data_start; a mismatch means the payload is filed at the wrong offset"
    );
}

/// The same cut, carried through to logits: a mount whose header chunk
/// ended inside the padding must build the model the file describes.
/// Then the same oracle under seeded random chunking, which only the
/// models crate's helper had seen before — never the production mount.
#[test]
fn a_mount_cut_inside_the_padding_is_the_same_model_as_a_read_one() {
    let Some(path) = cached("smollm2-360m-instruct-gguf") else {
        eprintln!("skip: smollm2-360m-instruct-gguf is not in the local cache");
        return;
    };
    let device = combs_core::init_device();
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

    let bytes = std::fs::read(&path).unwrap();
    let len = bytes.len() as u64;
    let data_start = read_gguf_header(&bytes, Some(len))
        .unwrap()
        .unwrap()
        .data_start as usize;

    // The padding cut: `feed` walks its slice from index 0, so the first
    // chunk of the second call starts at file offset data_start - 1.
    let mut mount = StreamMount::new(len, device.clone());
    mount
        .append(&bytes[..data_start - 1])
        .expect("header minus one byte feeds");
    feed(&mut mount, &bytes[data_start - 1..], 4 << 20).expect("payload feeds");
    let (mut weights, header_only) = mount.finish().expect("mount completes");

    // What `finish` hands an engine besides weights.
    assert_eq!(
        header_only.metadata().architecture,
        classic.metadata().architecture
    );
    assert!(
        !header_only
            .tokenizer()
            .expect("tokenizer")
            .json_bytes()
            .expect("json")
            .is_empty(),
        "the header-only source must still carry the tokenizer"
    );

    let mut streamed = ModelRegistry::<B>::new()
        .load_staged(&mut weights, &device)
        .expect("staged model builds");
    let got = logits_of(streamed.as_mut(), &device, &ids);
    assert_eq!(
        got, expect,
        "padding cut: logits differ from the whole-file load"
    );
    drop(streamed);

    for (seed, lo, hi) in [(0x5EED_u64, 1usize, 4096usize), (0xBEEF, 1 << 16, 8 << 20)] {
        let mut mount = StreamMount::new(len, device.clone());
        feed_random(&mut mount, &bytes, Chunks::new(seed, lo, hi)).expect("random chunks feed");
        let peak = mount.window_high_water();
        let (mut weights, _) = mount.finish().expect("mount completes");
        let mut streamed = ModelRegistry::<B>::new()
            .load_staged(&mut weights, &device)
            .expect("staged model builds");
        let got = logits_of(streamed.as_mut(), &device, &ids);
        assert_eq!(
            got, expect,
            "chunks {lo}..={hi}: logits differ from the whole-file load"
        );
        eprintln!(
            "chunks {lo}..={hi}: identical logits, window peaked at {:.1} MB of a {:.1} MB file",
            peak as f64 / 1e6,
            len as f64 / 1e6
        );
    }
}

/// What a 7B costs to feed through the mount at a browser's chunk size —
/// a number, not a gate. The appends that stage nothing are the
/// per-append bookkeeping; the ones that stage are where a per-tensor
/// source build would show up.
///
/// COMBS_TEST_GGUF=$HOME/.cache/combs/models/qwen2.5-coder-7b-instruct-gguf/model.gguf \
///   cargo test --release -p combs-runtime --test stream_mount -- --ignored --nocapture
#[test]
#[ignore = "requires COMBS_TEST_GGUF (a multi-GB file) and a GPU"]
fn the_cost_of_feeding_a_7b_in_64_kib_chunks() {
    use std::io::Read;
    use std::time::{Duration, Instant};
    let path = std::env::var("COMBS_TEST_GGUF")
        .ok()
        .map(std::path::PathBuf::from)
        .or_else(|| cached("qwen2.5-coder-7b-instruct-gguf"));
    let Some(path) = path else {
        eprintln!("skip: set COMBS_TEST_GGUF");
        return;
    };
    let len = std::fs::metadata(&path).unwrap().len();
    let mut mount = StreamMount::new(len, combs_core::init_device());
    let mut file = std::fs::File::open(&path).unwrap();
    let mut chunk = vec![0u8; 64 << 10];
    let started = Instant::now();
    let (mut appends, mut idle, mut busy) = (0u64, 0u64, 0u64);
    let (mut idle_time, mut busy_time) = (Duration::ZERO, Duration::ZERO);
    let (mut header_phase, mut staged_before) = (None, 0usize);
    loop {
        let n = file.read(&mut chunk).unwrap();
        if n == 0 {
            break;
        }
        let t = Instant::now();
        let p = mount.append(&chunk[..n]).expect("feeds");
        let dt = t.elapsed();
        appends += 1;
        if header_phase.is_none() && p.tensors_total > 0 {
            header_phase = Some(started.elapsed());
        }
        if p.tensors_staged == staged_before {
            idle += 1;
            idle_time += dt;
        } else {
            busy += 1;
            busy_time += dt;
            staged_before = p.tensors_staged;
        }
    }
    let feed_time = started.elapsed();
    let peak = mount.window_high_water();
    let t = Instant::now();
    let (weights, _source) = mount.finish().expect("completes");
    let finish_time = t.elapsed();
    eprintln!(
        "[stream] {:.2} GB in {appends} appends: feed {:.1}s ({:.0} MB/s); header {:.2}s; \
         {idle} appends staged nothing in {:.2}s ({:.1} us each); {busy} staged in {:.1}s; \
         finish {:.2}s; window peaked at {:.0} MB",
        len as f64 / 1e9,
        feed_time.as_secs_f64(),
        len as f64 / 1e6 / feed_time.as_secs_f64(),
        header_phase.unwrap_or_default().as_secs_f64(),
        idle_time.as_secs_f64(),
        idle_time.as_micros() as f64 / idle.max(1) as f64,
        busy_time.as_secs_f64(),
        finish_time.as_secs_f64(),
        peak as f64 / 1e6
    );
    drop(weights);
}
