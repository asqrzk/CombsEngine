//! The mount's two promises: it stays small, and it fails cleanly.
//!
//! Small, because the point of streaming is that a model larger than the
//! address space can still be mounted — a mount whose window grows with
//! the file is just a slower way of holding it. Cleanly, because a
//! stream that stops halfway is the ordinary case on a network, and a
//! half-built model that answers anyway is the worst possible outcome:
//! wrong weights are indistinguishable from right ones until the output
//! is wrong, and by then the cause is long gone.

use combs_formats::read_gguf_header;
use combs_runtime::stream_mount::{MountError, StreamMount};

use combs_core::CombsBackend as B;

fn cached(dir: &str) -> Option<std::path::PathBuf> {
    let path = std::path::PathBuf::from(std::env::var("HOME").ok()?)
        .join(".cache/combs/models")
        .join(dir)
        .join("model.gguf");
    path.exists().then_some(path)
}

fn feed(
    mount: &mut StreamMount,
    bytes: &[u8],
    chunk: usize,
) -> Result<(), MountError> {
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
    let largest = header.tensors.iter().map(|(_, _, size)| *size).max().unwrap();

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
            MountError::Truncated { received, expected, .. } => {
                assert_eq!(received, cut as u64, "{what}: reported the wrong received count");
                assert_eq!(expected, bytes.len() as u64, "{what}: reported the wrong expectation");
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
