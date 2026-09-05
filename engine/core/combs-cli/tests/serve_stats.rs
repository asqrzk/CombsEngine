//! HTTP-surface test for `combs serve`'s /v1/stats — pinned here because
//! the route once carried TWO "build" keys (the manifest at the top, the
//! runtime env flags further down) and JSON parsers keep the last, so
//! the manifest was silently shadowed for every consumer. The manifest
//! now lives under "build_manifest"; both keys must exist with their
//! distinct shapes. Env-gated on the cached smollm2 gguf; skips loudly
//! otherwise so model-less CI stays green.

use std::io::Read;
use std::time::{Duration, Instant};

fn smollm2() -> Option<std::path::PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let f = std::path::PathBuf::from(home)
        .join(".cache/combs/models/smollm2-360m-instruct-gguf/model.gguf");
    f.is_file().then_some(f)
}

fn http_get(port: u16, path: &str) -> Option<(u16, Vec<u8>)> {
    use std::io::Write;
    let raw = format!("GET {path} HTTP/1.0\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream
        .set_read_timeout(Some(Duration::from_secs(120)))
        .ok()?;
    stream.write_all(raw.as_bytes()).ok()?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response).ok()?;
    let header_end = response.windows(4).position(|w| w == b"\r\n\r\n")? + 4;
    let head = String::from_utf8_lossy(&response[..header_end]);
    let status: u16 = head.split_whitespace().nth(1)?.parse().ok()?;
    Some((status, response[header_end..].to_vec()))
}

#[test]
fn stats_carries_the_manifest_and_the_env_flags_under_distinct_keys() {
    let Some(model) = smollm2() else {
        eprintln!("skipping: smollm2-360m gguf not cached");
        return;
    };

    let port = 18094_u16;
    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_combs"))
        .args([
            "serve",
            "--model",
            model.to_str().unwrap(),
            "--port",
            &port.to_string(),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn serve");

    let deadline = Instant::now() + Duration::from_secs(120);
    let healthy = loop {
        if let Some((200, _)) = http_get(port, "/health") {
            break true;
        }
        if Instant::now() > deadline {
            break false;
        }
        std::thread::sleep(Duration::from_millis(500));
    };

    let result = std::panic::catch_unwind(|| {
        assert!(healthy, "serve never became healthy");
        let (status, body) = http_get(port, "/v1/stats").expect("stats");
        assert_eq!(status, 200);
        let json: serde_json::Value = serde_json::from_slice(&body).expect("stats json");

        let manifest = &json["build_manifest"];
        assert!(
            manifest["git"]["commit"].is_string(),
            "build_manifest must carry the git identity: {manifest}"
        );
        assert!(manifest["built_at"].is_string(), "built_at missing");

        let flags = &json["build"];
        let dtype = flags["dtype"].as_str().unwrap_or_default();
        assert!(
            dtype == "f16" || dtype == "f32",
            "the env-flags key must keep its shape (platform reads .build.dtype): {flags}"
        );
        assert!(
            flags.get("git").is_none(),
            "the flags key must NOT be the manifest — the shadowing would be back"
        );
    });

    let _ = child.kill();
    let _ = child.wait();
    result.expect("stats assertions");
}
