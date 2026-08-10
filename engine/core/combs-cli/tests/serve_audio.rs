//! HTTP-surface test for the `serve-audio` worker — the first test against
//! any of the engine's HTTP servers. Env-gated: runs only when the Kokoro
//! checkpoint is cached (and espeak-ng is installed); skips silently
//! otherwise so CI without models stays green.

use std::io::Read;
use std::time::{Duration, Instant};

fn kokoro_dir() -> Option<std::path::PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let dir = std::path::PathBuf::from(home).join(".cache/combs/models/kokoro-82m");
    dir.join("voices").is_dir().then_some(dir)
}

// HTTP/1.0 keeps responses unchunked (close-delimited), so the naive
// read-to-end below sees the raw body bytes.
fn http_get(port: u16, path: &str) -> Option<(u16, Vec<u8>)> {
    http_request(port, &format!("GET {path} HTTP/1.0\r\nHost: localhost\r\nConnection: close\r\n\r\n"))
}

fn http_post_json(port: u16, path: &str, body: &str) -> Option<(u16, Vec<u8>)> {
    http_request(
        port,
        &format!(
            "POST {path} HTTP/1.0\r\nHost: localhost\r\nConnection: close\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        ),
    )
}

/// Tiny raw-TCP HTTP client (no reqwest dependency for one test): returns
/// (status, body).
fn http_request(port: u16, raw: &str) -> Option<(u16, Vec<u8>)> {
    use std::io::Write;
    let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(120))).ok()?;
    stream.write_all(raw.as_bytes()).ok()?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response).ok()?;
    let header_end = response.windows(4).position(|w| w == b"\r\n\r\n")? + 4;
    let head = String::from_utf8_lossy(&response[..header_end]);
    let status: u16 = head.split_whitespace().nth(1)?.parse().ok()?;
    Some((status, response[header_end..].to_vec()))
}

#[test]
fn serve_audio_http_surface() {
    let Some(model_dir) = kokoro_dir() else {
        eprintln!("skipping: kokoro-82m not cached");
        return;
    };

    let port = 18093_u16;
    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_combs"))
        .args([
            "serve-audio",
            "--model",
            model_dir.to_str().unwrap(),
            "--port",
            &port.to_string(),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn serve-audio");

    // Wait for /health (engine load is a few seconds).
    let deadline = Instant::now() + Duration::from_secs(60);
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
        assert!(healthy, "worker never became healthy");

        let (status, body) = http_get(port, "/v1/audio/voices").expect("voices");
        assert_eq!(status, 200);
        let json: serde_json::Value = serde_json::from_slice(&body).expect("voices json");
        let voices: Vec<&str> = json["voices"]
            .as_array()
            .expect("voices array")
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(voices.contains(&"af_heart"), "af_heart missing: {voices:?}");

        let (status, wav) = http_post_json(
            port,
            "/v1/audio/speech",
            r#"{"input":"Testing the audio worker.","voice":"af_heart"}"#,
        )
        .expect("speech");
        assert_eq!(status, 200, "speech failed: {}", String::from_utf8_lossy(&wav));
        assert!(wav.len() > 10_000, "wav too small: {} bytes", wav.len());
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        let rate = u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]);
        assert_eq!(rate, 24_000);

        // Bad requests fail with 400, not a hang or 500.
        let (status, _) = http_post_json(port, "/v1/audio/speech", r#"{"input":""}"#)
            .expect("empty input");
        assert_eq!(status, 400);
    });

    let _ = child.kill();
    let _ = child.wait();
    if let Err(panic) = result {
        std::panic::resume_unwind(panic);
    }
}
