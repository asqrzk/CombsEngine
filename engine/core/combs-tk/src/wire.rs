//! The wire, as pure functions: an HTTP/1.1 request, a response head, a
//! chunk decoder that takes bytes as they arrive, and an SSE splitter
//! that turns text into the frames a client acts on.
//!
//! No socket lives here. Every rule was taken from bytes captured off a
//! live engine (`tests/fixtures/wire`), and three exist because the
//! captures showed the shape. The engine answers a stream with no length,
//! so tiny_http chunks it — and a plain JSON body past 32 KiB chunks
//! too, so the decoder is not optional for either road. It writes
//! `: keepalive` on an idle decode, which is a comment and never content.
//! And a failure mid-stream arrives as an in-band `data: {"error":…}`
//! frame at HTTP 200, so the status line is not the verdict.

use serde_json::Value;

/// One HTTP/1.1 request with a body, one per connection: `Connection:
/// close`, so the reply ends when the socket does and no length is
/// negotiated twice.
pub fn request(method: &str, path: &str, host: &str, body: &[u8]) -> Vec<u8> {
    let mut out = format!(
        "{method} {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .into_bytes();
    out.extend_from_slice(body);
    out
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// A response head: the status and the headers, names as sent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Head {
    pub status: u16,
    pub headers: Vec<(String, String)>,
}

impl Head {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    pub fn chunked(&self) -> bool {
        self.header("transfer-encoding")
            .is_some_and(|v| v.to_ascii_lowercase().contains("chunked"))
    }

    pub fn content_length(&self) -> Option<usize> {
        self.header("content-length").and_then(|v| v.trim().parse().ok())
    }
}

/// The head and the offset at which the body starts, `Ok(None)` while
/// the blank line has not arrived yet, or the reason the bytes are not
/// an HTTP head at all.
pub fn parse_head(bytes: &[u8]) -> Result<Option<(Head, usize)>, String> {
    let Some(end) = find(bytes, b"\r\n\r\n") else {
        return Ok(None);
    };
    let text = std::str::from_utf8(&bytes[..end]).map_err(|_| "a reply head that is not text".to_string())?;
    let mut lines = text.split("\r\n");
    let status_line = lines.next().unwrap_or("");
    let mut parts = status_line.splitn(3, ' ');
    let version = parts.next().unwrap_or("");
    if !version.starts_with("HTTP/1.") {
        return Err(format!("not an HTTP reply: {status_line:?}"));
    }
    let status: u16 = parts
        .next()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("a reply with no status: {status_line:?}"))?;
    let headers = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect();
    Ok(Some((Head { status, headers }, end + 4)))
}

/// A chunked-transfer decoder that is fed whatever slice the socket
/// hands over, however a chunk boundary falls across it. Payload is
/// handed on as soon as it is in hand rather than held until its chunk
/// ends: an answer arriving a token at a time must show a token at a
/// time.
#[derive(Debug)]
pub struct Dechunker {
    pending: Vec<u8>,
    state: Chunk,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Chunk {
    /// Waiting for a size line.
    Size,
    /// Inside a chunk: this many payload bytes still to come.
    Data(usize),
    /// A chunk's payload has ended and its CRLF has not arrived.
    Crlf,
    /// The zero-length chunk was seen; whatever follows is ignored.
    Done,
}

impl Default for Dechunker {
    fn default() -> Self {
        Self { pending: Vec::new(), state: Chunk::Size }
    }
}

impl Dechunker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed bytes; decoded payload is appended to `out`.
    pub fn feed(&mut self, bytes: &[u8], out: &mut Vec<u8>) -> Result<(), String> {
        if self.state == Chunk::Done {
            return Ok(());
        }
        self.pending.extend_from_slice(bytes);
        loop {
            match self.state {
                Chunk::Size => {
                    let Some(nl) = find(&self.pending, b"\r\n") else {
                        return Ok(());
                    };
                    let line = std::str::from_utf8(&self.pending[..nl])
                        .map_err(|_| "a chunk size that is not text".to_string())?;
                    let hex = line.split(';').next().unwrap_or("").trim();
                    let n = usize::from_str_radix(hex, 16)
                        .map_err(|_| format!("a chunk size that is not hex: {line:?}"))?;
                    self.pending.drain(..nl + 2);
                    if n == 0 {
                        self.state = Chunk::Done;
                        self.pending.clear();
                        return Ok(());
                    }
                    self.state = Chunk::Data(n);
                }
                Chunk::Data(remaining) => {
                    if self.pending.is_empty() {
                        return Ok(());
                    }
                    let take = remaining.min(self.pending.len());
                    out.extend_from_slice(&self.pending[..take]);
                    self.pending.drain(..take);
                    let left = remaining - take;
                    if left > 0 {
                        self.state = Chunk::Data(left);
                        return Ok(());
                    }
                    self.state = Chunk::Crlf;
                }
                Chunk::Crlf => {
                    if self.pending.len() < 2 {
                        return Ok(());
                    }
                    if &self.pending[..2] != b"\r\n" {
                        return Err("a chunk not followed by CRLF".into());
                    }
                    self.pending.drain(..2);
                    self.state = Chunk::Size;
                }
                Chunk::Done => return Ok(()),
            }
        }
    }

    /// True once the terminating zero-length chunk has been seen.
    pub fn done(&self) -> bool {
        self.state == Chunk::Done
    }
}

/// Splits SSE text into the payloads of its `data:` lines, dropping
/// comments, and holds an event that has not finished arriving.
#[derive(Debug, Default)]
pub struct SseSplitter {
    pending: String,
}

impl SseSplitter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed text; returns the data payloads of every event completed by
    /// it, in order.
    pub fn feed(&mut self, text: &str) -> Vec<String> {
        self.pending.push_str(text);
        let mut out = Vec::new();
        while let Some(i) = self.pending.find("\n\n") {
            let event: String = self.pending.drain(..i + 2).take(i).collect();
            let data: Vec<&str> = event
                .split('\n')
                .map(|l| l.trim_end_matches('\r'))
                .filter_map(|l| l.strip_prefix("data:"))
                .map(|rest| rest.strip_prefix(' ').unwrap_or(rest))
                .collect();
            // A line beginning ':' is a comment — the engine's keepalive —
            // and any other field is not ours to act on.
            if !data.is_empty() {
                out.push(data.join("\n"));
            }
        }
        out
    }
}

/// What one data payload means to a client.
#[derive(Debug, Clone, PartialEq)]
pub enum Frame {
    /// A piece of the answer.
    Content(String),
    /// The model called tools: the `tool_calls` array as sent.
    Calls(Value),
    /// The generation ended, and why: `stop`, `length` or `tool_calls`.
    Finish(String),
    /// The engine's accounting for the generation, on the final frame.
    Usage(Value),
    /// The generation failed after the status line said 200.
    Error(String),
    /// `[DONE]`: nothing more will come.
    Done,
}

/// The frames one payload carries. One payload may carry several — the
/// final one carries a finish reason and the usage together, and a
/// truncated tool call carries the call AND `length`.
pub fn frames(payload: &str) -> Result<Vec<Frame>, String> {
    if payload.trim() == "[DONE]" {
        return Ok(vec![Frame::Done]);
    }
    let v: Value =
        serde_json::from_str(payload).map_err(|e| format!("a frame that is not JSON: {e}"))?;
    if let Some(err) = v.get("error") {
        let msg = err
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("the engine reported an error without a message");
        return Ok(vec![Frame::Error(msg.to_string())]);
    }
    let mut out = Vec::new();
    if let Some(choice) = v.get("choices").and_then(|c| c.get(0)) {
        // A stream carries `delta`; a plain reply carries `message`. The
        // same reader serves both roads.
        if let Some(part) = choice.get("delta").or_else(|| choice.get("message")) {
            if let Some(s) = part.get("content").and_then(Value::as_str) {
                if !s.is_empty() {
                    out.push(Frame::Content(s.to_string()));
                }
            }
            if let Some(calls) = part.get("tool_calls") {
                if calls.as_array().is_some_and(|a| !a.is_empty()) {
                    out.push(Frame::Calls(calls.clone()));
                }
            }
        }
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            out.push(Frame::Finish(reason.to_string()));
        }
    }
    if let Some(u) = v.get("usage") {
        if !u.is_null() {
            out.push(Frame::Usage(u.clone()));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const STREAM: &[u8] = include_bytes!("../tests/fixtures/wire/stream.raw");
    const MULTI: &[u8] = include_bytes!("../tests/fixtures/wire/stream.multichunk.raw");
    const KEEPALIVE: &[u8] = include_bytes!("../tests/fixtures/wire/stream.keepalive.raw");
    const PLAIN: &[u8] = include_bytes!("../tests/fixtures/wire/plain.raw");
    const ERROR: &[u8] = include_bytes!("../tests/fixtures/wire/error.raw");

    /// Every stream fixture is one generation of this text, so every
    /// road through the decoder must arrive at exactly it.
    const CONTENT: &str = "<think>\n\n</think>\n\nCount from one to five:  \n1, 2, 3, 4, 5";

    fn split(raw: &[u8]) -> (Head, &[u8]) {
        let (head, at) = parse_head(raw).expect("a captured reply parses").expect("its head is complete");
        (head, &raw[at..])
    }

    /// Decode a raw capture end to end, feeding the body in slices of
    /// `step` bytes so no boundary is where it is convenient.
    fn decode(raw: &[u8], step: usize) -> (Head, Vec<Frame>) {
        let (head, body) = split(raw);
        let mut de = Dechunker::new();
        let mut text = Vec::new();
        for piece in body.chunks(step) {
            de.feed(piece, &mut text).expect("well-formed chunks decode");
        }
        assert!(de.done(), "the terminating chunk was not seen");
        let mut sse = SseSplitter::new();
        let mut out = Vec::new();
        for payload in sse.feed(&String::from_utf8(text).expect("utf-8")) {
            out.extend(frames(&payload).expect("a captured payload is a known frame"));
        }
        (head, out)
    }

    fn content_of(fr: &[Frame]) -> String {
        fr.iter()
            .filter_map(|f| if let Frame::Content(s) = f { Some(s.as_str()) } else { None })
            .collect()
    }

    #[test]
    fn a_request_is_http_1_1_with_a_length_and_a_close() {
        let r = request("POST", "/v1/chat/completions", "127.0.0.1:8080", b"{\"a\":1}");
        let text = String::from_utf8(r.clone()).unwrap();
        assert!(text.starts_with("POST /v1/chat/completions HTTP/1.1\r\n"), "{text}");
        assert!(text.contains("\r\nHost: 127.0.0.1:8080\r\n"), "{text}");
        assert!(text.contains("\r\nContent-Length: 7\r\n"), "{text}");
        assert!(text.contains("\r\nConnection: close\r\n"), "{text}");
        assert!(text.contains("\r\nContent-Type: application/json\r\n"), "{text}");
        assert!(r.ends_with(b"\r\n\r\n{\"a\":1}"), "the body follows the blank line");
    }

    #[test]
    fn a_head_says_its_status_and_how_the_body_is_framed() {
        let (h, body) = split(STREAM);
        assert_eq!(h.status, 200);
        assert!(h.chunked(), "a stream is chunked");
        assert_eq!(h.header("content-type"), Some("text/event-stream"));
        assert!(body.starts_with(b"14ac\r\n"), "the body starts at the first chunk size");
        let (p, pbody) = split(PLAIN);
        assert!(!p.chunked());
        assert_eq!(p.content_length(), Some(679));
        assert_eq!(pbody.len(), 679);
        let (e, _) = split(ERROR);
        assert_eq!(e.status, 400);
    }

    #[test]
    fn a_head_that_has_not_finished_arriving_is_not_an_error() {
        assert_eq!(parse_head(b"HTTP/1.1 200 OK\r\nServer: x\r\n").unwrap(), None);
        assert!(parse_head(b"not http at all\r\n\r\n").is_err());
    }

    /// The capture is one chunk. The multi-chunk twin is the same body
    /// re-cut so that boundaries fall inside a `data:` prefix, inside a
    /// JSON string and inside the blank line between events — and it is
    /// fed three bytes at a time on top of that.
    #[test]
    fn a_chunk_boundary_can_fall_anywhere() {
        let (_, whole) = decode(STREAM, usize::MAX);
        let (_, cut) = decode(MULTI, 3);
        assert_eq!(content_of(&whole), CONTENT);
        assert_eq!(content_of(&cut), CONTENT);
        assert_eq!(whole, cut, "re-chunking must not change a single frame");
    }

    /// The engine writes `: keepalive` every 100 ms of idle decode.
    /// It is an SSE comment: not content, not a frame, not an error.
    #[test]
    fn a_keepalive_is_never_content() {
        let (_, fr) = decode(KEEPALIVE, 5);
        assert_eq!(content_of(&fr), CONTENT);
        assert!(
            !fr.iter().any(|f| matches!(f, Frame::Content(s) if s.contains("keepalive"))),
            "a keepalive leaked into the answer"
        );
        assert!(!fr.iter().any(|f| matches!(f, Frame::Error(_))));
    }

    /// The last data frame carries the finish reason and the usage
    /// together, and `[DONE]` follows it. A client that stops at the
    /// first finish reason must not miss the accounting on that frame.
    #[test]
    fn the_final_frame_carries_the_finish_and_the_usage() {
        let (_, fr) = decode(STREAM, 64);
        let n = fr.len();
        assert_eq!(fr[n - 1], Frame::Done);
        assert!(fr[..n - 1].iter().any(|f| f == &Frame::Finish("length".into())), "{fr:?}");
        let usage = fr.iter().find_map(|f| if let Frame::Usage(u) = f { Some(u) } else { None });
        let usage = usage.expect("the final frame carries usage");
        assert_eq!(usage["prompt_tokens"], 18);
        assert_eq!(usage["completion_tokens"], 24);
        assert_eq!(usage["cache"]["cached_tokens"], 17);
    }

    /// An event split across two feeds is one event, not two halves.
    #[test]
    fn an_event_split_across_feeds_is_one_event() {
        let mut s = SseSplitter::new();
        assert!(s.feed("data: {\"a\"").is_empty(), "nothing has ended yet");
        let got = s.feed(":1}\n\ndata: [DONE]\n\n");
        assert_eq!(got, vec!["{\"a\":1}".to_string(), "[DONE]".to_string()]);
    }

    /// A failure mid-stream arrives at HTTP 200 as a data frame. The
    /// status line said success; the frame says otherwise, and the
    /// frame is right.
    #[test]
    fn an_error_frame_is_a_failure_at_200() {
        let fr = frames(r#"{"error":{"message":"the engine fell over","type":"engine_error"}}"#).unwrap();
        assert_eq!(fr, vec![Frame::Error("the engine fell over".into())]);
    }

    /// finish_reason precedence in the engine is length > tool_calls >
    /// stop, so a tool call cut off by max_tokens says `length`. The call
    /// is still there and is still a call.
    #[test]
    fn a_truncated_call_is_still_a_call() {
        let fr = frames(
            r#"{"choices":[{"delta":{"tool_calls":[{"id":"call_0","type":"function","function":{"name":"net.ping","arguments":"{\"host\":\"127.0.0.1\"}"}}]},"finish_reason":"length","index":0}]}"#,
        )
        .unwrap();
        assert!(fr.iter().any(|f| matches!(f, Frame::Calls(_))), "the call was dropped: {fr:?}");
        assert!(fr.contains(&Frame::Finish("length".into())));
    }

    #[test]
    fn done_ends_the_stream() {
        assert_eq!(frames("[DONE]").unwrap(), vec![Frame::Done]);
    }

    /// A payload that is not JSON and not `[DONE]` is a broken wire, not
    /// a piece of the answer.
    #[test]
    fn garbage_on_the_wire_is_refused_not_shown() {
        assert!(frames("this is not json").is_err());
    }
}
