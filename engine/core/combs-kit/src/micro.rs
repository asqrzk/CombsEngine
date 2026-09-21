//! The box's fingerprint, independent of any model.
//!
//! Every cost figure this estate has ever held was entangled with a
//! model: a µs-per-dispatch inferred from a whole decode step, a
//! "launch-bound" read off a ratio. Three flat rungs in a row (§96,
//! §100, §101) said the wall on this desk is what each launch does,
//! not how many there are — and this is the instrument that would show
//! where. Four cells, each its own child process:
//!
//! - `dispatch`: the floor cost of one launch, measured alone.
//! - `bus`: host→device and device→host at three sizes.
//! - `gemv`: the Q4_K decode gemv at a real projection shape, as bytes
//!   read per second against the packed size — the roofline question.
//! - `submits`: the sweep §90 ran by hand over `CUBECL_WGPU_MAX_TASKS`,
//!   as `model` children with the env set, since the optimum is a
//!   property of the driver and every box must re-take it.

use crate::record::{Run, Value};

pub const CELLS: &[&str] = &["dispatch", "bus", "gemv", "gemv-cuts", "submits"];

/// Q4_K on the device, taken from the engine so the two cannot drift:
/// 148 B per 256 weights. The FILE packs the same superblock in 144 —
/// the repack widens it by 2.78% — so a GB/s divided by file bytes
/// reads 2.70% LOW, which is how every `*_gbs` figure back to §90 was
/// computed. The kernel reads these bytes; the rate is over these bytes.
const Q4K_DEVICE_SUPERBLOCK: usize = combs_models::Q4K_DEVICE_SUPERBLOCK_BYTES;

/// Rides every record that carries a corrected rate, so a reader
/// diffing this cell against §90-§131 sees the step and its size
/// instead of a 2.7% improvement that never happened.
const DENOMINATOR_NOTE: &str = "rates are over DEVICE bytes (q4_k 148 B/superblock) from this \
    build on; sittings up to §134 divided by the file's 144 B — 2.78% fewer bytes — and so read \
    2.70% low. The `_us` measurements are unchanged and stay comparable.";

/// One cell, inside a child. Never call from the parent.
pub fn run_cell(cell: &str) -> Run {
    let mut values = Vec::new();
    let mut notes = Vec::new();
    match cell {
        "dispatch" => {
            let n = 2000usize;
            let (took, delta) = combs_models::dispatch_latency(n);
            values.push(Value::measured("launches", n as f64, "dispatches"));
            values.push(Value::measured("counted", delta as f64, "dispatches"));
            values.push(Value::measured("wall_ms", took.as_secs_f64() * 1e3, "ms"));
            values.push(Value::derived(
                "us_per_launch",
                took.as_secs_f64() * 1e6 / n as f64,
                "us",
                "wall_ms * 1e3 / launches",
            ));
            if delta != n as u64 {
                notes.push(format!("the counter saw {delta} launches for {n} issued — the window was not ours alone"));
            }
        }
        "bus" => {
            for (label, mb) in [("1mb", 1usize), ("16mb", 16), ("256mb", 256)] {
                let bytes = mb * 1_048_576;
                let (h2d, d2h) = combs_models::bus_cell(bytes);
                let gbs = |d: std::time::Duration| bytes as f64 / d.as_secs_f64() / 1e9;
                values.push(Value::measured(format!("h2d_{label}_ms"), h2d.as_secs_f64() * 1e3, "ms"));
                values.push(Value::measured(format!("d2h_{label}_ms"), d2h.as_secs_f64() * 1e3, "ms"));
                values.push(Value::derived(format!("h2d_{label}_gbs"), gbs(h2d), "GB/s", "bytes / h2d_ms"));
                values.push(Value::derived(format!("d2h_{label}_gbs"), gbs(d2h), "GB/s", "bytes / d2h_ms"));
            }
            notes.push("device-to-device is not offered: the runtime has no plain copy kernel".to_string());
        }
        "gemv" => {
            // qwen3-0.6b's projection shapes: hidden 1024, intermediate 3072.
            for (label, n_out, k) in [("q_proj", 2048usize, 1024usize), ("up_proj", 3072, 1024), ("down_proj", 1024, 3072)] {
                let per = combs_models::gemv_cell(n_out, k, 200);
                let packed_bytes = (n_out * k / 256) * Q4K_DEVICE_SUPERBLOCK;
                values.push(Value::measured(format!("{label}_us"), per.as_secs_f64() * 1e6, "us"));
                values.push(Value::derived(
                    format!("{label}_gbs"),
                    packed_bytes as f64 / per.as_secs_f64() / 1e9,
                    "GB/s",
                    "q4_k DEVICE bytes (148 B/superblock) / time per rep",
                ));
            }
            notes.push(DENOMINATOR_NOTE.to_string());
        }
        "gemv-cuts" => {
            notes.push(DENOMINATOR_NOTE.to_string());
            // The experiment §102 asked for: does the narrow projection
            // want more workgroups or more work per lane? The two cuts
            // either side of the default answer opposite ways, so the
            // winner names the cause rather than only the cure.
            for (label, n_out, k) in [
                ("kv_proj_512", 512usize, 1024usize),
                ("q_proj_2048", 2048, 1024),
                ("up_proj_3072", 3072, 1024),
                ("down_proj_1024", 1024, 3072),
            ] {
                for (rows, lanes) in [(8u32, 32u32), (16, 16), (32, 8)] {
                    let (per, worst) = combs_models::gemv_cell_at(n_out, k, 200, rows, lanes);
                    let packed = (n_out * k / 256) * Q4K_DEVICE_SUPERBLOCK;
                    values.push(Value::measured(
                        format!("{label}_r{rows}l{lanes}_us"),
                        per.as_secs_f64() * 1e6,
                        "us",
                    ));
                    values.push(Value::derived(
                        format!("{label}_r{rows}l{lanes}_gbs"),
                        packed as f64 / per.as_secs_f64() / 1e9,
                        "GB/s",
                        "q4_k DEVICE bytes (148 B/superblock) / time per rep",
                    ));
                    // A cut that computes the wrong answer quickly is
                    // not a faster kernel; the error rides the record.
                    if worst > 1e-3 {
                        notes.push(format!("{label} at {rows}x{lanes}: rel error {worst:.1e} — WRONG, not faster"));
                    }
                }
            }
        }
        other => notes.push(format!("no measurement: unknown cell {other:?}")),
    }
    Run {
        id: cell.to_string(),
        kind: "micro".to_string(),
        subject: "this box".to_string(),
        config: serde_json::json!({ "cell": cell }),
        warm: true,
        values,
        notes,
    }
}

pub fn print_run(run: &Run) {
    if run.values.is_empty() {
        println!("  {:<10} {}", run.id, run.notes.join("; "));
        return;
    }
    let line: Vec<String> = run
        .values
        .iter()
        .map(|v| format!("{} {:.2} {}", v.name, v.value, v.unit))
        .collect();
    println!("  {:<10} {}", run.id, line.join(" | "));
    for n in &run.notes {
        println!("  {:<10} note: {n}", "");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The denominator a rate is divided by must be the bytes the kernel
    /// READS, which the repack widens past the file's. §90 through §134
    /// divided by 144 and read 2.78% low; the constant now comes from
    /// the engine's own `vram_bytes` arithmetic so the two cannot part
    /// company again, and the record says the step happened.
    #[test]
    fn a_rate_is_over_device_bytes_not_file_bytes() {
        const FILE_SUPERBLOCK: usize = 144;
        assert_eq!(Q4K_DEVICE_SUPERBLOCK, 148, "128 B quants + 12 B scales + 8 B d/dmin");
        assert!(
            Q4K_DEVICE_SUPERBLOCK > FILE_SUPERBLOCK,
            "the repack widens the superblock; a rate over file bytes under-reports"
        );
        let low = (FILE_SUPERBLOCK as f64 / Q4K_DEVICE_SUPERBLOCK as f64 - 1.0).abs() * 100.0;
        assert!((low - 2.70).abs() < 0.05, "the understatement is ~2.7%, got {low:.2}%");
        assert!(DENOMINATOR_NOTE.contains("148"), "the note names the layout it switched to");
    }

    /// Every cell either measures or says why — an empty run with no note
    /// is how a silent skip became a number once before.
    #[test]
    fn an_unknown_cell_refuses_by_name_instead_of_measuring_nothing() {
        let run = run_cell("not-a-cell");
        assert!(run.values.is_empty());
        assert!(
            run.notes.iter().any(|n| n.contains("unknown cell")),
            "{:?}",
            run.notes
        );
    }
}
