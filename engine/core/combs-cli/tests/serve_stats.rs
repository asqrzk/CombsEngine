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

fn http_post(port: u16, path: &str, body: &str) -> Option<(u16, Vec<u8>)> {
    use std::io::Write;
    let raw = format!(
        "POST {path} HTTP/1.0\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
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

/// The anonymous-session mode must be READABLE, not merely configurable.
///
/// The platform can restart a worker carrying a new `--anonymous-sessions`
/// argv and call it "restarted" — but nothing reads the value back off a
/// LIVE engine, so a door change is "we asked for it", never "it took".
/// The readback it does have compares two fields, neither of them this
/// one. The engine already knows the answer and says it once, at startup,
/// into a log no door-verifier can read. This puts it where the platform
/// already looks: `.build`.
///
/// Two ports, and neither is the one above: a borrowed port is not yours
/// a moment later, and this suite has already paid for that once.
#[test]
fn the_stats_build_block_names_the_anonymous_session_mode() {
    let Some(model) = smollm2() else {
        eprintln!("skipping: smollm2-360m gguf not cached");
        return;
    };

    for (flag, want, port) in [(None, "keep", 18095_u16), (Some("off"), "off", 18096_u16)] {
        let mut args = vec![
            "serve".to_string(),
            "--model".to_string(),
            model.to_str().unwrap().to_string(),
            "--port".to_string(),
            port.to_string(),
        ];
        if let Some(mode) = flag {
            args.push("--anonymous-sessions".to_string());
            args.push(mode.to_string());
        }

        let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_combs"))
            .args(&args)
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

            let got = json["build"]["anonymous_sessions"].as_str();
            assert_eq!(
                got,
                Some(want),
                "a door that cannot be read back cannot be verified, only assumed: {}",
                json["build"]
            );
        });

        let _ = child.kill();
        let _ = child.wait();
        result.expect("anonymous-session mode on /v1/stats");
    }
}

/// The embedding road must be COUNTED, and counted APART.
///
/// Before this, `/v1/stats` had one `totals` block fed by generations
/// alone. A seat that only classified therefore read as a seat that had
/// done nothing at all, and a classify sitting could not say from the
/// route whether the classifier had run, how many questions it answered,
/// or which scorer answered them — §126's run could answer that only by
/// reading a console, which is why its record could not carry it.
///
/// So: `embeddings` and `classify` are siblings of `totals`, never folded
/// into it. `totals.requests` staying 0 after a classify call is the
/// assertion that matters; an embed decodes nothing and is not a
/// generation.
///
/// Its own port: a borrowed port is not yours a moment later, and this
/// suite has already paid for that once — this test was written on 18096
/// and quietly answered the anonymous-session test's probes instead.
/// Taken in this crate's tests: 18093 audio, 18094 and 18095/18096 here,
/// 18097 kv_events. A small arena too, because the default one refuses to
/// seat on a desk that has just built a release.
#[test]
fn the_stats_route_counts_the_embedding_road_apart_from_the_generation_totals() {
    let Some(model) = smollm2() else {
        eprintln!("skipping: smollm2-360m gguf not cached");
        return;
    };

    let port = 18098_u16;
    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_combs"))
        .args([
            "serve",
            "--model",
            model.to_str().unwrap(),
            "--port",
            &port.to_string(),
            "--context-size",
            "4096",
        ])
        // The log is read back at the end, so it is captured, and the
        // depth is turned up to where per-question detail is emitted.
        .env("COMBS_PROVENANCE", "all")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn serve");
    let mut log_pipe = child.stderr.take().expect("stderr piped");

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

        let before: serde_json::Value =
            serde_json::from_slice(&http_get(port, "/v1/stats").expect("stats").1).expect("json");
        assert_eq!(before["embeddings"]["requests"], 0, "nothing embedded yet: {}", before["embeddings"]);
        assert_eq!(before["classify"]["requests"], 0);
        assert_eq!(before["classify"]["backend"], "embed-cosine", "the route names the scorer");

        let body = serde_json::json!({
            "state": STATE,
            "questions": [{
                "name": "intent",
                "type": "choice",
                "instructions": "what is being asked for",
                "options": [
                    {"id": "cancel", "tether": {"means": "end an ongoing subscription"}},
                    {"id": "refund", "tether": {"means": "return money already paid"}},
                ],
            }],
        })
        .to_string();
        let (status, raw) = http_post(port, "/v1/classify", &body).expect("classify");
        let answer: serde_json::Value = serde_json::from_slice(&raw).expect("answer json");
        assert_eq!(status, 200, "classify: {answer}");

        let after: serde_json::Value =
            serde_json::from_slice(&http_get(port, "/v1/stats").expect("stats").1).expect("json");
        let emb = &after["embeddings"];
        let cls = &after["classify"];

        // The whole point of the sibling blocks.
        assert_eq!(
            after["totals"]["requests"], 0,
            "an embed is not a generation and must not be counted as one: {}",
            after["totals"]
        );
        assert_eq!(after["totals"]["generated_tokens"], 0, "nothing was decoded");

        assert_eq!(cls["requests"], 1, "the classify call was counted: {cls}");
        assert_eq!(cls["questions"], 1, "its questions were counted: {cls}");
        assert_eq!(
            cls["backend"], answer["backend"],
            "the route and the answer must name ONE scorer"
        );
        assert!(cls["cache"]["misses"].as_u64().unwrap_or(0) >= 1, "the first ask misses: {cls}");

        assert!(emb["requests"].as_u64().unwrap_or(0) >= 1, "the embed road ran: {emb}");
        // The state and both options' texts, at least.
        assert!(emb["texts"].as_u64().unwrap_or(0) >= 3, "every text was counted: {emb}");
        assert!(emb["prompt_tokens"].as_u64().unwrap_or(0) > 0, "tokens were counted: {emb}");
        assert!(emb["last_ms"].as_f64().unwrap_or(0.0) > 0.0, "the call was timed: {emb}");
        assert_eq!(emb["errors"], 0, "no embed failed: {emb}");

        // The answer must carry the score BEFORE the temperature, and it
        // must be the score the probabilities actually came from. Two
        // different score vectors can soften to the same distribution, so
        // a harness handed only `distribution` cannot check the softmax
        // it was told was used. This re-derives it.
        let a = &answer["answers"][0];
        let sims = a["similarities"].as_object().expect("similarities on the wire");
        let dist = a["distribution"].as_object().expect("distribution");
        assert_eq!(sims.len(), dist.len(), "one raw score per option: {a}");
        let t = a["temperature"].as_f64().expect("temperature") as f32;
        let ids: Vec<&String> = dist.keys().collect();
        let raw: Vec<f32> = ids.iter().map(|k| sims[*k].as_f64().unwrap() as f32).collect();
        let want = combs_classify::softmax_scaled(&raw, t);
        for (i, k) in ids.iter().enumerate() {
            let got = dist[*k].as_f64().unwrap() as f32;
            assert!(
                (got - want[i]).abs() < 1e-5,
                "softmax(similarities/T) must reproduce the distribution for {k}: {got} vs {}",
                want[i]
            );
        }
    });

    let _ = child.kill();
    let _ = child.wait();

    // The state is the user's text and never belongs in a log. Its
    // length and a hash are enough to tell two runs apart and to find
    // the call a figure came from.
    let mut log = String::new();
    let _ = log_pipe.read_to_string(&mut log);
    result.expect("embedding-road stats assertions");
    assert!(
        !log.contains(STATE),
        "the state reached the log verbatim:\n{log}"
    );
    assert!(log.contains("state_fnv="), "the turn must name the state by hash: {log}");
    assert!(log.contains("classify.question"), "per-question detail is missing at debug: {log}");
    assert!(log.contains("state_len="), "the turn must carry the state's length: {log}");
}

/// A state distinctive enough that finding it in a log is proof, not
/// coincidence.
const STATE: &str = "please cancel my subscription, reference zulu-7731-quokka";
