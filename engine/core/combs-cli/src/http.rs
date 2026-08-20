//! Shared tiny_http helpers for the `serve*` commands (text, images, tts).

use std::io::Read;

use serde_json::{Value, json};

/// Response type used everywhere: a boxed reader so streaming and buffered
/// responses share one concrete type.
pub type HttpResponse = tiny_http::Response<Box<dyn Read + Send>>;

pub fn error_json(kind: &str, message: &str) -> Value {
    json!({"error": {"type": kind, "message": message}})
}

pub fn cors_header() -> tiny_http::Header {
    tiny_http::Header::from_bytes("Access-Control-Allow-Origin", "*").unwrap()
}

pub fn json_response(status: u16, body: Value) -> HttpResponse {
    let data = body.to_string().into_bytes();
    let len = data.len();
    tiny_http::Response::empty(status)
        .with_data(
            Box::new(std::io::Cursor::new(data)) as Box<dyn Read + Send>,
            Some(len),
        )
        .with_header(tiny_http::Header::from_bytes("Content-Type", "application/json").unwrap())
        .with_header(cors_header())
}

/// Non-JSON payload (e.g. a preview PNG) with an explicit content type.
pub fn bytes_response(status: u16, content_type: &str, data: Vec<u8>) -> HttpResponse {
    let len = data.len();
    tiny_http::Response::empty(status)
        .with_data(
            Box::new(std::io::Cursor::new(data)) as Box<dyn Read + Send>,
            Some(len),
        )
        .with_header(tiny_http::Header::from_bytes("Content-Type", content_type).unwrap())
        .with_header(cors_header())
}

/// Respond to a CORS preflight probe (browsers send OPTIONS before POSTs).
pub fn respond_preflight(request: tiny_http::Request) {
    let _ = request.respond(
        tiny_http::Response::empty(204)
            .with_header(cors_header())
            .with_header(
                tiny_http::Header::from_bytes(
                    "Access-Control-Allow-Methods",
                    "GET, POST, OPTIONS",
                )
                .unwrap(),
            )
            .with_header(
                tiny_http::Header::from_bytes(
                    "Access-Control-Allow-Headers",
                    "content-type, authorization",
                )
                .unwrap(),
            ),
    );
}
