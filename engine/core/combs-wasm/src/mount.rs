//! Chunk-appended model mounting — the buffering state machine.
//!
//! A browser tab cannot hand the engine a 2.5 GB model in one
//! ArrayBuffer: Chrome caps page-heap ArrayBuffers around 2.1 GB, so
//! the platform streams the cached file INTO wasm linear memory in
//! chunks instead. This module is the plain, natively-unit-tested
//! core; the `#[wasm_bindgen]` exports in lib.rs are thin wrappers
//! around it.
//!
//! Contract: `open` reserves the whole buffer up front (while the
//! heap is still small, and with `try_reserve_exact` so an allocation
//! failure is an error, not an abort); `append` refuses overflow;
//! `finish_check` refuses short deliveries. Removal semantics (finish
//! consumes the state FIRST, abort is idempotent) live with the
//! caller's map, not here.

/// The open-gate ceiling. Beyond this, buffer-mode mounting cannot
/// leave room for load transients under wasm32's 4 GiB — those models
/// wait for the per-tensor streaming road, and the refusal must be
/// fast and name its numbers.
pub const MAX_BUFFER_MOUNT_BYTES: u64 = 3_200_000_000;

/// One in-flight buffered mount.
#[derive(Debug)]
pub struct MountState {
    pub config_json: String,
    pub expected: usize,
    pub buf: Vec<u8>,
}

/// Validate the open parameters and reserve the buffer.
pub fn open(config_json: String, expected_len: f64, mode: &str) -> Result<MountState, String> {
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
    let mut buf = Vec::new();
    buf.try_reserve_exact(expected)
        .map_err(|e| format!("cannot reserve {expected} bytes for the model buffer: {e}"))?;
    Ok(MountState { config_json, expected, buf })
}

/// Append one chunk; refuses to grow past the declared length.
pub fn append(state: &mut MountState, chunk: &[u8]) -> Result<(), String> {
    let after = state.buf.len() + chunk.len();
    if after > state.expected {
        return Err(format!(
            "append overflows the mount: {} + {} > declared {}",
            state.buf.len(),
            chunk.len(),
            state.expected
        ));
    }
    state.buf.extend_from_slice(chunk);
    Ok(())
}

/// The finish precondition: every declared byte arrived.
pub fn finish_check(state: &MountState) -> Result<(), String> {
    if state.buf.len() != state.expected {
        return Err(format!(
            "mount finished short: received {} of {} declared bytes",
            state.buf.len(),
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

    #[test]
    fn pathological_chunkings_reassemble_bitwise() {
        let data = payload(10_007); // prime-ish, misaligned everywhere
        for chunk_len in [1usize, 2, 3, 7, 4096, 10_006, 10_007] {
            let mut st = open("{}".into(), data.len() as f64, "buffer").unwrap();
            for chunk in data.chunks(chunk_len) {
                append(&mut st, chunk).unwrap();
            }
            finish_check(&st).unwrap();
            assert_eq!(st.buf, data, "chunk_len {chunk_len}");
        }
    }

    #[test]
    fn overflow_is_refused_and_state_unharmed() {
        let mut st = open("{}".into(), 8.0, "buffer").unwrap();
        append(&mut st, &[1, 2, 3, 4, 5, 6]).unwrap();
        let err = append(&mut st, &[7, 8, 9]).unwrap_err();
        assert!(err.contains("overflows"), "{err}");
        assert_eq!(st.buf.len(), 6, "failed append must not partially write");
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
