//! Pre-flight fit check: refuse a model whose largest single allocation
//! exceeds the adapter's per-binding cap BEFORE `Engine::load` enqueues
//! uploads. cubecl materializes buffers on its own compute threads where
//! an oversize allocation can only panic — never surface as a `Result` —
//! so the only honest moment to say "this model does not fit this
//! device" is before the first enqueue.

/// One tensor's contribution to the fit decision.
pub struct FitRow {
    pub name: String,
    /// Logical element count (dense materialization is `elements × 4`,
    /// the f32 widening every non-packed tensor goes through on load).
    pub elements: u64,
    /// Packed byte length + rank, set by the caller ONLY when the
    /// quant-linear path truly applies (rank-2 linear, block-aligned k,
    /// kernels enabled, not an embedding — embeddings always dequantize
    /// dense). Known limit: GGUF fused projections (phi3 attn_qkv) are
    /// measured per stored tensor, not per served slice.
    pub packed: Option<(u64, usize)>,
}

impl FitRow {
    /// Bytes this tensor materializes on the device. Unknown eligibility
    /// resolves to the LARGER dense size — over-refusing beats serving a
    /// model with holes in it.
    pub fn device_bytes(&self) -> u64 {
        match self.packed {
            Some((bytes, 2)) => bytes,
            _ => self.elements.saturating_mul(4),
        }
    }
}

pub struct FitReport {
    pub worst_name: String,
    pub worst_bytes: u64,
    pub tensors: usize,
}

/// Largest single device allocation across the model's tensors.
pub fn fit_report(rows: impl IntoIterator<Item = FitRow>) -> Option<FitReport> {
    let mut worst: Option<(String, u64)> = None;
    let mut tensors = 0usize;
    for row in rows {
        tensors += 1;
        let bytes = row.device_bytes();
        if worst.as_ref().map(|(_, b)| bytes > *b).unwrap_or(true) {
            worst = Some((row.name, bytes));
        }
    }
    worst.map(|(worst_name, worst_bytes)| FitReport {
        worst_name,
        worst_bytes,
        tensors,
    })
}

/// The refusal, worded to act on: device, cap, offending tensor, need.
pub fn check_fit(
    report: &FitReport,
    limit: u64,
    device_name: &str,
    device_type: &str,
) -> Result<(), String> {
    if report.worst_bytes <= limit {
        return Ok(());
    }
    Err(format!(
        "device {device_name} ({device_type}) caps allocations at {limit} bytes; \
         tensor {} needs {} — model does not fit this adapter \
         (checked {} tensors; a CPU-fallback adapter usually means the \
         container GPU is not reachable)",
        report.worst_name, report.worst_bytes, report.tensors
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(name: &str, elements: u64, packed: Option<(u64, usize)>) -> FitRow {
        FitRow { name: name.into(), elements, packed }
    }

    #[test]
    fn packed_rank2_uses_packed_bytes() {
        // Q8_0 math from the gguf fixtures: (64*8/32)*34 packed vs 64*8*4 dense.
        let r = row("w", 64 * 8, Some(((64 * 8 / 32) * 34, 2)));
        assert_eq!(r.device_bytes(), (64 * 8 / 32) * 34);
    }

    #[test]
    fn packed_non_rank2_falls_back_dense() {
        let r = row("w", 100, Some((10, 1)));
        assert_eq!(r.device_bytes(), 400);
    }

    #[test]
    fn dense_is_elements_times_four() {
        // A large vocab embed: 152064 × 3584 f32.
        let r = row("token_embd.weight", 152_064 * 3_584, None);
        assert_eq!(r.device_bytes(), 2_179_989_504);
    }

    #[test]
    fn report_finds_the_worst_and_refusal_names_it() {
        let report = fit_report(vec![
            row("small", 10, None),
            row("token_embd.weight", 152_064 * 3_584, None),
            row("mid", 1000, None),
        ])
        .unwrap();
        assert_eq!(report.worst_name, "token_embd.weight");
        assert_eq!(report.tensors, 3);
        let err = check_fit(&report, 134_217_728, "llvmpipe", "Cpu").unwrap_err();
        assert!(err.contains("llvmpipe (Cpu)"));
        assert!(err.contains("token_embd.weight"));
        assert!(err.contains("2179989504"));
        assert!(check_fit(&report, u64::MAX, "M3", "IntegratedGpu").is_ok());
    }

    #[test]
    fn packed_embed_counts_dense_and_refuses() {
        // A Q6_K token_embd is packed rank-2 but LOADS dense — the
        // walk passes packed: None for embeds, so the dense size
        // drives the refusal.
        let report = fit_report(vec![row("model.embed_tokens.weight", 152_064 * 3_584, None)])
            .unwrap();
        assert!(check_fit(&report, 2_147_483_647, "llvmpipe", "Cpu").is_err());
    }

    #[test]
    fn empty_model_yields_no_report() {
        assert!(fit_report(vec![]).is_none());
    }
}
