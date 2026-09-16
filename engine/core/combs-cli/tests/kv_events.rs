//! `POST /v1/tokenize` and `GET /v1/kv/events` on a live `combs serve`.
//!
//! The ids a router renders with must be the ids the chat path itself
//! prefills, and the page log must name exactly the pages the engine
//! holds: a multi-page prefill, a decode that crosses page boundaries,
//! a rollback, a session dropped while another still holds its prefix,
//! and a clear.
//!
//! The replica this test keeps is keyed by the engine's own page keys but
//! asserted in TOKENS, so it can never agree with the implementation by
//! sharing its arithmetic.
//!
//! Env-gated on a cached gguf (`COMBS_TEST_GGUF`, else a small one in the
//! model cache); skips loudly otherwise so a model-less box stays green.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde_json::Value;

const PORT: u16 = 18097;

fn model() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("COMBS_TEST_GGUF") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    let home = std::env::var("HOME").ok()?;
    let f = PathBuf::from(home).join(".cache/combs/models/qwen3-0.6b-gguf/model.gguf");
    f.is_file().then_some(f)
}

fn http_request(raw: &str) -> Option<(u16, Vec<u8>)> {
    use std::io::Write;
    let mut stream = std::net::TcpStream::connect(("127.0.0.1", PORT)).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(180))).ok()?;
    stream.write_all(raw.as_bytes()).ok()?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response).ok()?;
    let header_end = response.windows(4).position(|w| w == b"\r\n\r\n")? + 4;
    let head = String::from_utf8_lossy(&response[..header_end]);
    let status: u16 = head.split_whitespace().nth(1)?.parse().ok()?;
    Some((status, response[header_end..].to_vec()))
}

fn get(path: &str) -> Option<(u16, Value)> {
    let (status, body) = http_request(&format!(
        "GET {path} HTTP/1.0\r\nHost: localhost\r\nConnection: close\r\n\r\n"
    ))?;
    Some((status, serde_json::from_slice(&body).unwrap_or(Value::Null)))
}

fn post(path: &str, body: &str) -> Option<(u16, Value)> {
    let (status, body) = http_request(&format!(
        "POST {path} HTTP/1.0\r\nHost: localhost\r\nConnection: close\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    ))?;
    Some((status, serde_json::from_slice(&body).unwrap_or(Value::Null)))
}

fn delete(path: &str) -> Option<(u16, Value)> {
    let (status, body) = http_request(&format!(
        "DELETE {path} HTTP/1.0\r\nHost: localhost\r\nConnection: close\r\n\r\n"
    ))?;
    Some((status, serde_json::from_slice(&body).unwrap_or(Value::Null)))
}

/// What a reader believes the engine holds: page key -> the tokens that
/// page carried, in the order the log named them.
#[derive(Default)]
struct Replica {
    held: BTreeMap<String, Vec<u32>>,
    since: u64,
    page_size: usize,
    stored_runs: usize,
    stored_pages: usize,
    removed_pages: usize,
    cleared: usize,
}

impl Replica {
    /// Reads every event past `since` and applies it, refusing a run whose
    /// parent it has never seen (the ordering contract) — and returns the
    /// events applied.
    fn drain(&mut self) -> Vec<Value> {
        let (status, page) = get(&format!("/v1/kv/events?since={}", self.since)).expect("kv events");
        assert_eq!(status, 200, "kv events: {page}");
        self.page_size = page["page_size"].as_u64().expect("page_size") as usize;
        assert!(self.page_size > 0);
        assert_eq!(
            page["resync"].as_bool(),
            Some(false),
            "this test never falls behind the ring"
        );
        let events = page["events"].as_array().cloned().unwrap_or_default();
        self.stored_runs = 0;
        self.stored_pages = 0;
        self.removed_pages = 0;
        self.cleared = 0;
        for e in &events {
            match e["type"].as_str().expect("event type") {
                "stored" => {
                    let hashes: Vec<String> = e["hashes"]
                        .as_array()
                        .expect("hashes")
                        .iter()
                        .map(|h| h.as_str().expect("a key is a decimal string").to_string())
                        .collect();
                    let tokens: Vec<u32> = e["token_ids"]
                        .as_array()
                        .expect("token_ids")
                        .iter()
                        .map(|t| t.as_u64().expect("a token id") as u32)
                        .collect();
                    assert_eq!(
                        tokens.len(),
                        hashes.len() * self.page_size,
                        "a stored run carries whole pages"
                    );
                    if let Some(parent) = e["parent"].as_str() {
                        assert!(
                            self.held.contains_key(parent),
                            "a run hung off a page the reader has never seen: {e}"
                        );
                    }
                    for (i, h) in hashes.iter().enumerate() {
                        assert!(
                            !self.held.contains_key(h),
                            "the log named a page the reader already holds: {h}"
                        );
                        self.held.insert(
                            h.clone(),
                            tokens[i * self.page_size..(i + 1) * self.page_size].to_vec(),
                        );
                    }
                    self.stored_runs += 1;
                    self.stored_pages += hashes.len();
                }
                "removed" => {
                    for h in e["hashes"].as_array().expect("hashes") {
                        let h = h.as_str().expect("a key is a decimal string");
                        assert!(
                            self.held.remove(h).is_some(),
                            "the log removed a page the reader never held: {h}"
                        );
                        self.removed_pages += 1;
                    }
                }
                "cleared" => {
                    self.held.clear();
                    self.cleared += 1;
                }
                other => panic!("unknown event type {other}"),
            }
            self.since = e["seq"].as_u64().expect("seq");
        }
        events
    }

    fn holds_tokens(&self, tokens: &[u32]) -> bool {
        self.held.values().any(|t| t == tokens)
    }
}

/// The engine's own statement of how many tokens a session holds — the
/// count the page log must agree with, page for page.
fn session_history(id: &str) -> usize {
    let (status, v) = get("/v1/sessions").expect("sessions");
    assert_eq!(status, 200, "sessions: {v}");
    v["data"]
        .as_array()
        .expect("data")
        .iter()
        .find(|s| s["id"].as_str() == Some(id))
        .and_then(|s| s["history_tokens"].as_u64())
        .unwrap_or_else(|| panic!("session {id} is not live: {v}")) as usize
}

fn tokenize(body: &str) -> Vec<u32> {
    let (status, v) = post("/v1/tokenize", body).expect("tokenize");
    assert_eq!(status, 200, "tokenize: {v}");
    let ids: Vec<u32> = v["token_ids"]
        .as_array()
        .expect("token_ids")
        .iter()
        .map(|t| t.as_u64().expect("a token id") as u32)
        .collect();
    assert_eq!(v["count"].as_u64().expect("count") as usize, ids.len());
    ids
}

fn completion(prompt: &str, session: &str, max_tokens: usize) -> Value {
    let body = serde_json::json!({
        "prompt": prompt,
        "session_id": session,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    })
    .to_string();
    let (status, v) = post("/v1/completions", &body).expect("completion");
    assert_eq!(status, 200, "completion: {v}");
    v
}

fn words(n: usize) -> String {
    const BASE: [&str; 10] = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
    ];
    (0..n)
        .map(|i| BASE[i % BASE.len()])
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn tokenize_answers_the_chat_paths_own_ids_and_the_page_log_names_what_the_engine_holds() {
    let Some(model) = model() else {
        eprintln!("skipping: no cached gguf (set COMBS_TEST_GGUF)");
        return;
    };

    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_combs"))
        .args([
            "serve",
            "--model",
            model.to_str().unwrap(),
            "--port",
            &PORT.to_string(),
        ])
        .env("COMBS_KV_EVENTS", "1")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn serve");

    let deadline = Instant::now() + Duration::from_secs(180);
    let healthy = loop {
        if let Some((200, _)) = get("/health") {
            break true;
        }
        if Instant::now() > deadline {
            break false;
        }
        std::thread::sleep(Duration::from_millis(500));
    };

    let result = std::panic::catch_unwind(|| {
        assert!(healthy, "serve never became healthy");

        // --- the render contract: the ids the chat path prefills --------
        let chat_body = serde_json::json!({
            "messages": [{"role": "user", "content": "Name three colours."}],
        })
        .to_string();
        let ids = tokenize(&chat_body);
        assert!(ids.len() > 4, "a rendered chat turn is more than its words: {}", ids.len());

        let with_generation = serde_json::json!({
            "messages": [{"role": "user", "content": "Name three colours."}],
            "max_tokens": 1,
            "session_id": "render",
            "temperature": 0.0,
        })
        .to_string();
        let (status, answer) = post("/v1/chat/completions", &with_generation).expect("chat");
        assert_eq!(status, 200, "chat: {answer}");
        assert_eq!(
            answer["usage"]["prompt_tokens"].as_u64().expect("prompt_tokens") as usize,
            ids.len(),
            "the tokenize route must answer the ids this model's own chat path prefills"
        );

        // A raw prompt tokenizes as itself.
        let raw = tokenize(&serde_json::json!({"prompt": "alpha bravo"}).to_string());
        assert!(!raw.is_empty());
        let (status, refused) = post("/v1/tokenize", "{}").expect("tokenize refusal");
        assert_eq!(status, 400, "a body with neither messages nor prompt is refused: {refused}");

        // --- the page log ----------------------------------------------
        let mut replica = Replica::default();
        replica.drain();
        assert_eq!(delete("/v1/sessions").expect("clear").0, 200);
        replica.drain();
        assert!(replica.cleared >= 1, "clearing every session is one line, not a page each");
        assert!(replica.held.is_empty());

        // A prefill of several pages.
        let long = words(60);
        let long_ids = tokenize(&serde_json::json!({"prompt": long}).to_string());
        let ps = {
            let (_, page) = get("/v1/kv/events?since=0").expect("page size");
            page["page_size"].as_u64().expect("page_size") as usize
        };
        assert!(long_ids.len() >= 3 * ps, "the fixture prompt must fill several pages");
        completion(&long, "s1", 1);
        replica.drain();
        let full_pages = long_ids.len() / ps;
        assert_eq!(
            replica.held.len(),
            session_history("s1") / ps,
            "the log must name every full page the one live session holds"
        );
        assert_eq!(replica.removed_pages, 0, "a cold prefill removes nothing");
        for page in 0..full_pages {
            let tokens = &long_ids[page * ps..(page + 1) * ps];
            assert!(
                replica.holds_tokens(tokens),
                "page {page} of the prompt is not in the log"
            );
        }

        // A decode that crosses page boundaries publishes the pages it
        // filled. Which tokens a page came from cannot be read off the
        // tokens — a repeating prompt's greedy continuation repeats it —
        // so the bar is the engine's own count of what the session holds.
        let before = replica.held.len();
        completion(&long, "s1", 2 * ps + 3);
        replica.drain();
        let held_tokens = session_history("s1");
        assert!(
            held_tokens >= long_ids.len() + ps,
            "the fixture must decode past a page boundary: {} prompt tokens, {held_tokens} held",
            long_ids.len()
        );
        assert_eq!(
            replica.held.len(),
            held_tokens / ps,
            "the log must name every full page the session holds, the decode's own included"
        );
        assert!(
            replica.held.len() > before,
            "decoding past a page boundary must publish the pages it filled ({before} -> {})",
            replica.held.len()
        );

        // Two sessions on one prefix: the same pages, named once.
        let shared_before = replica.held.len();
        completion(&long, "s2", 1);
        replica.drain();
        assert!(
            replica.held.len() <= shared_before + 1,
            "a second session on the same prefix must not duplicate its pages \
             ({shared_before} -> {})",
            replica.held.len()
        );
        let sessions = get("/v1/sessions").expect("sessions").1;
        let live = sessions["data"].as_array().expect("sessions").len();
        assert!(live >= 2, "both sessions are live: {sessions}");

        // Dropping one holder keeps the pages the other still holds.
        assert_eq!(delete("/v1/sessions/s1").expect("drop s1").0, 200);
        replica.drain();
        assert!(
            replica.holds_tokens(&long_ids[..ps]),
            "the prefix session s2 still holds was removed with s1"
        );

        // A rollback below a page boundary drops the pages nobody holds.
        let other = words(60).replace("alpha", "zulu");
        let other_ids = tokenize(&serde_json::json!({"prompt": other}).to_string());
        assert!(other_ids.len() >= 2 * ps);
        completion(&other, "s2", 1);
        replica.drain();
        assert!(
            replica.removed_pages > 0,
            "a divergent prompt in the only session that held those pages removes them"
        );
        assert!(
            !replica.holds_tokens(&long_ids[ps..2 * ps]),
            "the rolled-back page is still in the log"
        );
        assert!(
            replica.holds_tokens(&other_ids[..ps]),
            "the new prompt's first page is not in the log"
        );

        // And a clear takes everything.
        assert_eq!(delete("/v1/sessions").expect("clear").0, 200);
        replica.drain();
        assert!(replica.cleared >= 1);
        assert!(replica.held.is_empty(), "a clear leaves the reader holding nothing");
    });

    let _ = child.kill();
    let _ = child.wait();
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}
