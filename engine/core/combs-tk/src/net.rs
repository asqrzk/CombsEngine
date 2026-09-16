//! The socket: one blocking connection to the engine on loopback, one
//! request per connection, and a read timeout so a wedged engine cannot
//! hold the toolkit forever.
//!
//! Nothing here is clever, on purpose. The engine serves on 127.0.0.1
//! with no TLS and no auth, so a TLS stack would buy nothing and cost
//! the crate its size; and every byte that arrives goes straight to the
//! pure decoders in `wire`, which is where the rules live and where the
//! tests are.

use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpStream};
use std::time::Duration;

use crate::wire::{self, Dechunker, Frame, Head, SseSplitter};

/// A whole reply: its head and its body, dechunked if it was chunked.
#[derive(Debug)]
pub struct Reply {
    pub head: Head,
    pub body: Vec<u8>,
}

fn connect(port: u16, timeout: Duration) -> Result<TcpStream, String> {
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    let s = TcpStream::connect_timeout(&addr, Duration::from_secs(3))
        .map_err(|e| format!("no engine answers on 127.0.0.1:{port}: {e}"))?;
    s.set_read_timeout(Some(timeout))
        .map_err(|e| format!("could not arm the read timeout: {e}"))?;
    Ok(s)
}

/// Read until the head is complete; return it with whatever body bytes
/// arrived alongside it.
fn read_head(s: &mut TcpStream) -> Result<(Head, Vec<u8>), String> {
    let mut buf = Vec::with_capacity(4096);
    let mut piece = [0u8; 4096];
    loop {
        match wire::parse_head(&buf)? {
            Some((head, at)) => return Ok((head, buf[at..].to_vec())),
            None => {}
        }
        let n = s.read(&mut piece).map_err(|e| format!("reading the reply head: {e}"))?;
        if n == 0 {
            return Err("the engine closed the connection before a reply head".into());
        }
        buf.extend_from_slice(&piece[..n]);
        if buf.len() > 64 * 1024 {
            return Err("a reply head past 64 KiB is not a reply head".into());
        }
    }
}

/// Send one request and read the whole reply.
pub fn exchange(port: u16, method: &str, path: &str, body: &[u8], timeout: Duration) -> Result<Reply, String> {
    let mut s = connect(port, timeout)?;
    let req = wire::request(method, path, &format!("127.0.0.1:{port}"), body);
    s.write_all(&req).map_err(|e| format!("sending the request: {e}"))?;
    let (head, mut rest) = read_head(&mut s)?;
    let mut raw = Vec::new();
    raw.append(&mut rest);
    let mut piece = [0u8; 16 * 1024];
    loop {
        let n = s.read(&mut piece).map_err(|e| format!("reading the reply: {e}"))?;
        if n == 0 {
            break;
        }
        raw.extend_from_slice(&piece[..n]);
    }
    let body = if head.chunked() {
        let mut de = Dechunker::new();
        let mut out = Vec::new();
        de.feed(&raw, &mut out)?;
        out
    } else {
        raw
    };
    Ok(Reply { head, body })
}

/// The message inside an engine error body, or the raw text when it is
/// not the engine's shape.
pub fn error_message(body: &[u8]) -> String {
    let text = String::from_utf8_lossy(body);
    serde_json::from_str::<serde_json::Value>(&text)
        .ok()
        .and_then(|v| v.get("error")?.get("message")?.as_str().map(str::to_string))
        .unwrap_or_else(|| text.trim().to_string())
}

/// Send one request and hand every frame to `on_frame` as it arrives.
/// Returns the head once the stream has ended. A non-200 head is an
/// error carrying the engine's own message; an in-band error frame is
/// handed on like any other frame, because the status line has already
/// said 200 and only the caller can decide what that means.
pub fn stream(
    port: u16,
    path: &str,
    body: &[u8],
    timeout: Duration,
    mut on_frame: impl FnMut(&Frame),
) -> Result<Head, String> {
    let mut s = connect(port, timeout)?;
    let req = wire::request("POST", path, &format!("127.0.0.1:{port}"), body);
    s.write_all(&req).map_err(|e| format!("sending the request: {e}"))?;
    let (head, first) = read_head(&mut s)?;
    if head.status != 200 {
        let mut raw = first;
        let mut piece = [0u8; 4096];
        while let Ok(n) = s.read(&mut piece) {
            if n == 0 {
                break;
            }
            raw.extend_from_slice(&piece[..n]);
        }
        let body = if head.chunked() {
            let mut de = Dechunker::new();
            let mut out = Vec::new();
            let _ = de.feed(&raw, &mut out);
            out
        } else {
            raw
        };
        return Err(format!("the engine answered {}: {}", head.status, error_message(&body)));
    }
    let chunked = head.chunked();
    let mut de = Dechunker::new();
    let mut sse = SseSplitter::new();
    let mut text = Vec::new();
    let mut piece = [0u8; 16 * 1024];
    let mut pending = first;
    loop {
        if !pending.is_empty() {
            text.clear();
            if chunked {
                de.feed(&pending, &mut text)?;
            } else {
                text.extend_from_slice(&pending);
            }
            pending.clear();
            for payload in sse.feed(&String::from_utf8_lossy(&text)) {
                for f in wire::frames(&payload)? {
                    let done = f == Frame::Done;
                    on_frame(&f);
                    if done {
                        return Ok(head);
                    }
                }
            }
        }
        let n = s.read(&mut piece).map_err(|e| format!("reading the stream: {e}"))?;
        if n == 0 {
            return Ok(head);
        }
        pending.extend_from_slice(&piece[..n]);
    }
}
