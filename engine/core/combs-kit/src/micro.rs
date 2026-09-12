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

pub const CELLS: &[&str] = &["dispatch", "bus", "gemv", "submits"];

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
                let packed_bytes = (n_out * k / 256) * 144;
                values.push(Value::measured(format!("{label}_us"), per.as_secs_f64() * 1e6, "us"));
                values.push(Value::derived(
                    format!("{label}_gbs"),
                    packed_bytes as f64 / per.as_secs_f64() / 1e9,
                    "GB/s",
                    "packed q4_k bytes / time per rep",
                ));
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
