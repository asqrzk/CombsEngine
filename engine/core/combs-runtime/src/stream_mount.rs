//! Mounting a model from a stream, without the file ever existing whole.
//!
//! The buffering mount holds the image and hands it over at the end. That
//! works to about three gigabytes and then stops: a browser tab has four,
//! and a 7B checkpoint plus its load transients does not fit under that
//! with a copy of the file underneath. So this mount never keeps the
//! file. Bytes arrive, tensors are uploaded the moment they are complete,
//! and the bytes behind them are dropped.
//!
//! What makes that safe is order. One `Response.body` delivers a file's
//! bytes strictly front to back, and a GGUF's tensor table says where
//! every payload starts. Sort the table by offset and the arriving stream
//! sweeps through the tensors in exactly that order — so a cursor over
//! the table, a window over the bytes, and the rule "stage when complete,
//! then forget" are enough. Nothing is ever needed twice, which is the
//! property the whole design rests on and the reason it can be bounded.
//!
//! The window's floor is therefore the largest single tensor: it must be
//! whole to be uploaded. Everything else is chunk-sized.

use combs_core::{BufferPool, CombsBackend, CombsDevice};
use combs_formats::{GgufHeaderInfo, GgufSource, ModelSource, read_gguf_header};
use combs_models::staged::StagedWeights;

/// Flush the device queue once this much has been staged since the last
/// one. Uploads are enqueued, not executed; without a periodic flush the
/// staging buffers behind them cannot be recycled and the queue grows
/// with the model rather than with the window.
const FLUSH_EVERY_BYTES: usize = 32 << 20;

/// What went wrong, in terms the caller can act on: pull more, give up,
/// or tell the user the file is not what it claimed.
#[derive(Debug)]
pub enum MountError {
    /// The header will never parse — not a truncation, a malformed file.
    BadHeader(String),
    /// The architecture cannot be mounted from a stream. Raised at the
    /// header, before any payload is pulled.
    Unsupported(String),
    /// More bytes arrived than the file was said to hold.
    Overflow { expected: usize, got: usize },
    /// The stream ended early. Names what was still owed, because "mount
    /// failed" sends someone to the wrong place.
    Truncated {
        received: usize,
        expected: usize,
        staged: usize,
        tensors: usize,
    },
    /// A tensor could not be turned into weights.
    Staging(String),
}

impl std::fmt::Display for MountError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MountError::BadHeader(why) => write!(f, "gguf header is malformed: {why}"),
            MountError::Unsupported(arch) => {
                write!(f, "{arch} cannot be mounted from a stream")
            }
            MountError::Overflow { expected, got } => write!(
                f,
                "stream delivered {got} bytes for a {expected}-byte model"
            ),
            MountError::Truncated { received, expected, staged, tensors } => write!(
                f,
                "stream ended at {received} of {expected} bytes with {staged} of \
                 {tensors} tensors staged"
            ),
            MountError::Staging(why) => write!(f, "staging failed: {why}"),
        }
    }
}

impl std::error::Error for MountError {}

/// How far along a mount is, for the caller to report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Progress {
    pub received: usize,
    pub expected: usize,
    pub tensors_staged: usize,
    pub tensors_total: usize,
    /// Bytes the window is holding right now.
    pub window: usize,
}

enum Phase {
    /// Still collecting the header. The bytes accumulate because a header
    /// cannot be parsed in pieces, only re-attempted with more.
    Header,
    /// Sweeping the tensor table in offset order.
    Body,
    Done,
}

/// A mount in flight.
pub struct StreamMount {
    phase: Phase,
    device: CombsDevice,
    pool: BufferPool,
    expected: usize,
    received: usize,

    header_bytes: Vec<u8>,
    header: Option<GgufHeaderInfo>,

    /// Bytes `[base, base + window.len())` — everything arrived and not
    /// yet consumed.
    window: Vec<u8>,
    base: usize,
    next_tensor: usize,
    since_flush: usize,

    staged: Option<StagedWeights<CombsBackend>>,
    window_high_water: usize,
}

impl StreamMount {
    /// Open a mount for a file of `expected` bytes.
    pub fn new(expected: usize, device: CombsDevice) -> Self {
        StreamMount {
            phase: Phase::Header,
            device,
            pool: BufferPool::new(),
            expected,
            received: 0,
            header_bytes: Vec::new(),
            header: None,
            window: Vec::new(),
            base: 0,
            next_tensor: 0,
            since_flush: 0,
            staged: None,
            window_high_water: 0,
        }
    }

    /// Feed the next piece of the file. Chunk boundaries carry no
    /// meaning — a tensor may span any number of them, and any number of
    /// tensors may land inside one.
    pub fn append(&mut self, chunk: &[u8]) -> Result<Progress, MountError> {
        if self.received + chunk.len() > self.expected {
            return Err(MountError::Overflow {
                expected: self.expected,
                got: self.received + chunk.len(),
            });
        }
        self.received += chunk.len();

        if matches!(self.phase, Phase::Header) {
            self.header_bytes.extend_from_slice(chunk);
            match read_gguf_header(&self.header_bytes, Some(self.expected)) {
                Err(e) => return Err(MountError::BadHeader(e.to_string())),
                Ok(None) => return Ok(self.progress()),
                Ok(Some(header)) => {
                    if !combs_models::ModelRegistry::<CombsBackend>::supports_streaming(
                        &header.architecture,
                    ) {
                        return Err(MountError::Unsupported(header.architecture));
                    }
                    // The header may have arrived with payload behind it
                    // in the same chunk; that payload opens the window.
                    self.base = header.data_start;
                    self.window = self.header_bytes[header.data_start.min(self.header_bytes.len())..]
                        .to_vec();
                    self.header_bytes.truncate(header.data_start);
                    self.header = Some(header);
                    self.phase = Phase::Body;
                }
            }
        } else {
            self.window.extend_from_slice(chunk);
        }

        self.window_high_water = self.window_high_water.max(self.window.len());
        self.drain_complete_tensors()?;
        Ok(self.progress())
    }

    /// Upload every tensor now wholly inside the window, then forget the
    /// bytes behind it. Runs after each append rather than at the end,
    /// which is what keeps the window from growing into the file.
    fn drain_complete_tensors(&mut self) -> Result<(), MountError> {
        let Some(header) = self.header.clone() else {
            return Ok(());
        };
        let end = self.base + self.window.len();
        while self.next_tensor < header.tensors.len() {
            let (name, start, size) = header.tensors[self.next_tensor].clone();
            if start + size > end {
                break;
            }
            // A source over the window as it stands. The header is kept
            // apart precisely so this can be rebuilt cheaply: the window
            // has no header in it any more.
            let source = GgufSource::from_window(
                &self.header_bytes,
                self.window.clone(),
                self.base,
                self.expected,
            )
            .map_err(|e| MountError::Staging(e.to_string()))?;
            let weights = self
                .staged
                .get_or_insert_with(|| StagedWeights::new(source.metadata().clone()));
            // Pinned the way the whole-file load pins: a staged weight
            // outlives the buffers that built it, and the pool facade
            // already knows where that is worth doing and where it is
            // still unproven (it declines on wasm, deliberately).
            for (hf, _range) in source.hf_names_for_ggml(&name) {
                self.pool
                    .pin_persistent(&self.device, || weights.stage(&source, &self.device, &hf))
                    .map_err(|e| MountError::Staging(format!("{name} -> {hf}: {e}")))?;
            }
            self.next_tensor += 1;

            let consumed = start + size - self.base;
            self.window.drain(..consumed);
            self.base = start + size;

            self.since_flush += size;
            if self.since_flush >= FLUSH_EVERY_BYTES {
                combs_models::flush_device::<CombsBackend>(&self.device);
                self.since_flush = 0;
            }
        }
        if self.next_tensor == header.tensors.len() {
            self.phase = Phase::Done;
        }
        Ok(())
    }

    pub fn progress(&self) -> Progress {
        Progress {
            received: self.received,
            expected: self.expected,
            tensors_staged: self.next_tensor,
            tensors_total: self.header.as_ref().map(|h| h.tensors.len()).unwrap_or(0),
            window: self.window.len(),
        }
    }

    /// The largest the window ever grew — the number a residency claim
    /// is made of.
    pub fn window_high_water(&self) -> usize {
        self.window_high_water
    }

    /// Close the mount and take the weights.
    ///
    /// A stream that stopped early fails here rather than producing a
    /// model with holes in it, and everything staged so far is dropped
    /// on the way out, which is what returns the device memory.
    pub fn finish(mut self) -> Result<StagedWeights<CombsBackend>, MountError> {
        let tensors = self.header.as_ref().map(|h| h.tensors.len()).unwrap_or(0);
        if self.received != self.expected || self.next_tensor != tensors {
            return Err(MountError::Truncated {
                received: self.received,
                expected: self.expected,
                staged: self.next_tensor,
                tensors,
            });
        }
        combs_models::flush_device::<CombsBackend>(&self.device);
        let mut weights = self.staged.take().ok_or(MountError::Truncated {
            received: self.received,
            expected: self.expected,
            staged: 0,
            tensors,
        })?;
        weights.seal();
        Ok(weights)
    }
}
