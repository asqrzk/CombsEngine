//! Graphkit C ABI: a single `combsgraph_op_json` escape hatch (the
//! combs-mesh-ffi convention) so every algorithm rides one stable
//! symbol. Strings returned by `combsgraph_op_json` are owned by the
//! library and must be released with `combsgraph_string_free`.

use std::ffi::{c_char, CStr, CString};
use std::time::Instant;

use combs_graphkit::{activate_cpu, khop_cpu, ActivateParams, Graph};
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
struct OpRequest {
    op: String,
    #[serde(default)]
    nodes: usize,
    #[serde(default)]
    edges: Vec<(u32, u32)>,
    #[serde(default)]
    seeds: Vec<u32>,
    #[serde(default)]
    damping: Option<f32>,
    #[serde(default)]
    max_steps: Option<usize>,
    #[serde(default)]
    eps: Option<f32>,
    #[serde(default)]
    depth: Option<usize>,
    #[serde(default)]
    decay: Option<f32>,
    #[serde(default)]
    backend: Option<String>,
}

fn run_op(input: &str) -> Result<String, String> {
    let req: OpRequest = serde_json::from_str(input).map_err(|e| format!("bad request: {e}"))?;
    let graph = Graph::from_edges(req.nodes, &req.edges);
    let t0 = Instant::now();
    match req.op.as_str() {
        "activate" => {
            // Defaults come from the library's own Default — a divergent
            // literal here once shipped a cap the crate documents as
            // unable to converge.
            let defaults = ActivateParams::default();
            let params = ActivateParams {
                damping: req.damping.unwrap_or(defaults.damping),
                max_steps: req.max_steps.unwrap_or(defaults.max_steps),
                eps: req.eps.unwrap_or(defaults.eps),
            };
            let backend = req.backend.as_deref().unwrap_or("cpu");
            let (result, used) = match backend {
                #[cfg(feature = "gpu")]
                "gpu" => match combs_graphkit::gpu::activate_gpu(&graph, &req.seeds, &params) {
                    Ok(r) => (r, "gpu"),
                    // Over-cap or device failure: honest fallback, named.
                    Err(_) => (activate_cpu(&graph, &req.seeds, &params), "cpu-fallback"),
                },
                #[cfg(not(feature = "gpu"))]
                "gpu" => (activate_cpu(&graph, &req.seeds, &params), "cpu-fallback"),
                _ => (activate_cpu(&graph, &req.seeds, &params), "cpu"),
            };
            Ok(json!({
                "scores": result.scores,
                "steps_run": result.steps_run,
                "converged": result.converged,
                "backend": used,
                "ms": t0.elapsed().as_secs_f64() * 1000.0,
            })
            .to_string())
        }
        "khop" => {
            let scores = khop_cpu(
                &graph,
                &req.seeds,
                req.depth.unwrap_or(3),
                req.decay.unwrap_or(0.5),
            );
            Ok(json!({
                "scores": scores,
                "backend": "cpu",
                "ms": t0.elapsed().as_secs_f64() * 1000.0,
            })
            .to_string())
        }
        other => Err(format!("unknown op: {other}")),
    }
}

/// Runs one JSON op. Returns an owned JSON string (release with
/// `combsgraph_string_free`); errors come back as `{"error": "..."}`.
///
/// # Safety
/// `input` must be a valid NUL-terminated UTF-8 C string.
#[no_mangle]
pub unsafe extern "C" fn combsgraph_op_json(input: *const c_char) -> *mut c_char {
    let out = if input.is_null() {
        json!({"error": "null input"}).to_string()
    } else {
        match CStr::from_ptr(input).to_str() {
            Ok(s) => match run_op(s) {
                Ok(ok) => ok,
                Err(e) => json!({ "error": e }).to_string(),
            },
            Err(_) => json!({"error": "input is not utf-8"}).to_string(),
        }
    };
    CString::new(out)
        .unwrap_or_else(|_| CString::new("{\"error\":\"nul in output\"}").unwrap())
        .into_raw()
}

/// Frees a string previously returned by `combsgraph_op_json`.
///
/// # Safety
/// `s` must have been returned by this library and not yet freed.
#[no_mangle]
pub unsafe extern "C" fn combsgraph_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

#[cfg(test)]
mod tests {
    use super::run_op;

    #[test]
    fn activate_round_trips_json() {
        let out = run_op(
            r#"{"op":"activate","nodes":3,"edges":[[0,1],[1,2],[2,0]],"seeds":[0]}"#,
        )
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["scores"].as_array().unwrap().len(), 3);
        assert_eq!(v["backend"], "cpu");
    }

    #[test]
    fn unknown_op_is_an_error() {
        assert!(run_op(r#"{"op":"nope"}"#).is_err());
    }
}
