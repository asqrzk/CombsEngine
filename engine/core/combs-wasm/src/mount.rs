//! Chunk-appended model mounting — the buffering state machine.
//!
//! A browser tab cannot hand the engine a 2.5 GB model in one
//! ArrayBuffer: Chrome caps page-heap ArrayBuffers around 2.1 GB, so
//! the platform streams the cached file INTO wasm linear memory in
//! chunks instead. This module is the plain, natively-unit-tested
//! core; the `#[wasm_bindgen]` exports in lib.rs are thin wrappers
//! around it.
//!
//! The buffer is SEGMENTED, not one Vec: Rust caps every allocation at
//! `isize::MAX`, which on wasm32 is ~2.14 GB — a 2.5 GB image can
//! never be contiguous there, no matter the linear-memory ceiling.
//! Fixed 1 GiB segments feed [`combs_formats::open_model_source_segments`].
//!
//! Contract: `open` reserves every segment up front (while the heap is
//! still small, and with `try_reserve_exact` so an allocation failure
//! is an error, not an abort); `append` refuses overflow;
//! `finish_check` refuses short deliveries. Removal semantics (finish
//! consumes the state FIRST, abort is idempotent) live with the
//! caller's map, not here.

/// The open-gate ceiling. Beyond this, buffer-mode mounting cannot
/// leave room for load transients under wasm32's 4 GiB — those models
/// wait for the per-tensor streaming road, and the refusal must be
/// fast and name its numbers.
pub const MAX_BUFFER_MOUNT_BYTES: u64 = 3_200_000_000;

/// Segment size. Comfortably under the 32-bit allocation ceiling and
/// big enough that any GGUF header fits the first segment whole.
pub const SEGMENT_LEN: usize = 1 << 30;

/// One in-flight buffered mount.
#[derive(Debug)]
pub struct MountState {
    pub config_json: String,
    pub expected: usize,
    received: usize,
    seg_len: usize,
    segments: Vec<Vec<u8>>,
}

impl MountState {
    pub fn received(&self) -> usize {
        self.received
    }

    pub fn seg_len(&self) -> usize {
        self.seg_len
    }

    /// The completed image as its segments (finish-time handoff).
    pub fn into_segments(self) -> Vec<Vec<u8>> {
        self.segments
    }
}

/// Validate the open parameters and reserve the buffer.
pub fn open(config_json: String, expected_len: f64, mode: &str) -> Result<MountState, String> {
    open_with_segment_len(config_json, expected_len, mode, SEGMENT_LEN)
}

/// [`open`] with an explicit segment length — tests cross real seams
/// without gigabyte fixtures; production always uses [`SEGMENT_LEN`].
pub fn open_with_segment_len(
    config_json: String,
    expected_len: f64,
    mode: &str,
    seg_len: usize,
) -> Result<MountState, String> {
    match mode {
        "buffer" => {}
        "stream" => {
            return Err(
                "mount mode \"stream\" is not available yet — per-tensor streaming \
                 lands on its own road; use \"buffer\""
                    .to_string(),
            );
        }
        other => return Err(format!("unknown mount mode {other:?} (want \"buffer\")")),
    }
    if !expected_len.is_finite() || expected_len < 1.0 || expected_len.fract() != 0.0 {
        return Err(format!(
            "expected_len must be a positive whole byte count, got {expected_len}"
        ));
    }
    let expected = expected_len as u64;
    if expected > MAX_BUFFER_MOUNT_BYTES {
        return Err(format!(
            "model is {expected} bytes ({:.2} GB) — buffer mounting caps at \
             {MAX_BUFFER_MOUNT_BYTES} bytes ({:.1} GB): the image plus load \
             transients must fit wasm32's 4 GiB address space",
            expected as f64 / 1e9,
            MAX_BUFFER_MOUNT_BYTES as f64 / 1e9,
        ));
    }
    let expected = expected as usize;
    let mut segments = Vec::new();
    let mut remaining = expected;
    while remaining > 0 {
        let this = remaining.min(seg_len);
        let mut seg = Vec::new();
        seg.try_reserve_exact(this)
            .map_err(|e| format!("cannot reserve a {this}-byte model segment: {e}"))?;
        segments.push(seg);
        remaining -= this;
    }
    Ok(MountState { config_json, expected, received: 0, seg_len, segments })
}

/// Append one chunk, split across segment seams as needed; refuses to
/// grow past the declared length (checked BEFORE any byte lands, so a
/// refused append leaves the state untouched).
pub fn append(state: &mut MountState, chunk: &[u8]) -> Result<(), String> {
    if state.received + chunk.len() > state.expected {
        return Err(format!(
            "append overflows the mount: {} + {} > declared {}",
            state.received,
            chunk.len(),
            state.expected
        ));
    }
    let mut rest = chunk;
    while !rest.is_empty() {
        let seg = state.received / state.seg_len;
        let off = state.received - seg * state.seg_len;
        let take = rest.len().min(state.seg_len - off);
        state.segments[seg].extend_from_slice(&rest[..take]);
        state.received += take;
        rest = &rest[take..];
    }
    Ok(())
}

/// The finish precondition: every declared byte arrived.
pub fn finish_check(state: &MountState) -> Result<(), String> {
    if state.received != state.expected {
        return Err(format!(
            "mount finished short: received {} of {} declared bytes",
            state.received,
            state.expected
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i * 31 % 251) as u8).collect()
    }

    fn flat(st: MountState) -> Vec<u8> {
        st.into_segments().concat()
    }

    #[test]
    fn pathological_chunkings_reassemble_bitwise() {
        let data = payload(10_007); // prime-ish, misaligned everywhere
        for chunk_len in [1usize, 2, 3, 7, 4096, 10_006, 10_007] {
            let mut st = open("{}".into(), data.len() as f64, "buffer").unwrap();
            for chunk in data.chunks(chunk_len) {
                append(&mut st, chunk).unwrap();
            }
            finish_check(&st).unwrap();
            assert_eq!(flat(st), data, "chunk_len {chunk_len}");
        }
    }

    #[test]
    fn appends_split_across_segment_seams() {
        // Real seams at a test-sized segment length: chunks straddle
        // boundaries in every alignment, and each segment must hold
        // exactly the bytes its global offsets name.
        let data = payload(10_000);
        for chunk_len in [1usize, 3, 1000, 1024, 2047, 4096, 9_999] {
            let mut st =
                open_with_segment_len("{}".into(), data.len() as f64, "buffer", 1024).unwrap();
            for chunk in data.chunks(chunk_len) {
                append(&mut st, chunk).unwrap();
            }
            finish_check(&st).unwrap();
            let segs = st.into_segments();
            assert_eq!(segs.len(), 10, "chunk_len {chunk_len}");
            for (i, seg) in segs.iter().enumerate() {
                let want = if i == 9 { 784 } else { 1024 };
                assert_eq!(seg.len(), want, "segment {i} at chunk_len {chunk_len}");
                assert_eq!(seg[..], data[i * 1024..i * 1024 + want], "segment {i} bytes");
            }
        }
    }

    #[test]
    fn overflow_is_refused_and_state_unharmed() {
        let mut st = open("{}".into(), 8.0, "buffer").unwrap();
        append(&mut st, &[1, 2, 3, 4, 5, 6]).unwrap();
        let err = append(&mut st, &[7, 8, 9]).unwrap_err();
        assert!(err.contains("overflows"), "{err}");
        assert_eq!(st.received(), 6, "failed append must not partially write");
        append(&mut st, &[7, 8]).unwrap();
        finish_check(&st).unwrap();
    }

    #[test]
    fn short_finish_is_refused() {
        let mut st = open("{}".into(), 10.0, "buffer").unwrap();
        append(&mut st, &[0; 9]).unwrap();
        let err = finish_check(&st).unwrap_err();
        assert!(err.contains("9 of 10"), "{err}");
    }

    #[test]
    fn over_budget_open_names_its_numbers() {
        let err = open("{}".into(), 4.68e9, "buffer").unwrap_err();
        assert!(err.contains("4680000000"), "{err}");
        assert!(err.contains("3200000000"), "{err}");
    }

    #[test]
    fn bad_lengths_and_modes_are_refused() {
        assert!(open("{}".into(), 0.0, "buffer").is_err());
        assert!(open("{}".into(), -5.0, "buffer").is_err());
        assert!(open("{}".into(), f64::NAN, "buffer").is_err());
        assert!(open("{}".into(), 1024.5, "buffer").is_err());
        assert!(open("{}".into(), 1024.0, "stream").unwrap_err().contains("not available"));
        assert!(open("{}".into(), 1024.0, "mmap").unwrap_err().contains("unknown mount mode"));
    }
}
