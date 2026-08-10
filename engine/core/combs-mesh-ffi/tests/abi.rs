//! ABI tests: call the extern "C" exports directly from Rust (same pattern
//! as combs-ffi's tests — no C toolchain needed).

use std::ffi::{CStr, CString, c_char};
use std::ptr;
use std::sync::Mutex;

use combsmesh_ffi::*;

const KEY: &[u8] = b"abi test master key, 32 bytes!!!";

/// Serializes tests that mutate the process-wide keyring (tests in one
/// binary run on threads).
static KEYRING_LOCK: Mutex<()> = Mutex::new(());

fn last_error() -> Option<String> {
    let p = combsmesh_last_error();
    if p.is_null() {
        None
    } else {
        Some(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
    }
}

fn op_json(request: &str) -> Result<serde_json::Value, String> {
    let req = CString::new(request).unwrap();
    let raw = unsafe { combsmesh_op_json(req.as_ptr()) };
    if raw.is_null() {
        return Err(last_error().unwrap_or_else(|| "NULL, no error?".into()));
    }
    let s = unsafe { CStr::from_ptr(raw) }.to_string_lossy().into_owned();
    unsafe { combsmesh_string_free(raw) };
    Ok(serde_json::from_str(&s).expect("response is JSON"))
}

fn sample_binary() -> Vec<u8> {
    let emoji = combs_mesh::EmojiBuilder::new("abi-emoji")
        .description("built in an abi test")
        .add_todo("t1", "do the abi")
        .add_image_rgba(4, 4, (0..4 * 4 * 4).map(|i| i as u8).collect())
        .with_agent_lifecycle()
        .build();
    combs_mesh::EmojiExporter::to_binary(&emoji).expect("to_binary")
}

#[test]
fn init_encrypt_decrypt_shutdown_roundtrip() {
    let _guard = KEYRING_LOCK.lock().unwrap();
    assert_eq!(combsmesh_shutdown(), 0); // clean slate
    // Crypto before init → 1 + error set.
    let (mut out, mut out_len) = (ptr::null_mut(), 0usize);
    let rc = unsafe { combsmesh_encrypt_memory(b"x".as_ptr(), 1, &mut out, &mut out_len) };
    assert_eq!(rc, 1);
    assert!(last_error().expect("error").contains("not initialized"));

    assert_eq!(unsafe { combsmesh_init(KEY.as_ptr(), KEY.len()) }, 0);

    let data = b"hello mesh abi";
    let rc = unsafe { combsmesh_encrypt_memory(data.as_ptr(), data.len(), &mut out, &mut out_len) };
    assert_eq!(rc, 0, "encrypt: {:?}", last_error());
    assert!(!out.is_null() && out_len > data.len());
    let ct = unsafe { std::slice::from_raw_parts(out, out_len) }.to_vec();
    unsafe { combsmesh_bytes_free(out, out_len) };
    assert_ne!(ct, data);

    let (mut pt, mut pt_len) = (ptr::null_mut(), 0usize);
    let rc = unsafe { combsmesh_decrypt_memory(ct.as_ptr(), ct.len(), &mut pt, &mut pt_len) };
    assert_eq!(rc, 0, "decrypt: {:?}", last_error());
    let back = unsafe { std::slice::from_raw_parts(pt, pt_len) }.to_vec();
    unsafe { combsmesh_bytes_free(pt, pt_len) };
    assert_eq!(back, data);

    assert_eq!(combsmesh_shutdown(), 0);
    let rc = unsafe { combsmesh_decrypt_memory(ct.as_ptr(), ct.len(), &mut pt, &mut pt_len) };
    assert_eq!(rc, 1, "decrypt after shutdown must fail");
}

#[test]
fn init_with_null_key_generates_random_master() {
    let _guard = KEYRING_LOCK.lock().unwrap();
    assert_eq!(unsafe { combsmesh_init(ptr::null(), 0) }, 0);
    let (mut out, mut out_len) = (ptr::null_mut(), 0usize);
    let rc = unsafe { combsmesh_encrypt_memory(b"x".as_ptr(), 1, &mut out, &mut out_len) };
    assert_eq!(rc, 0, "encrypt with random master: {:?}", last_error());
    unsafe { combsmesh_bytes_free(out, out_len) };
    assert_eq!(combsmesh_shutdown(), 0);
}

#[test]
fn render_sprite_from_cmse_binary() {
    let binary = sample_binary();
    let (mut out, mut out_len) = (ptr::null_mut(), 0usize);
    let rc = unsafe {
        combsmesh_render_sprite(binary.as_ptr(), binary.len(), 0, &mut out, &mut out_len)
    };
    assert_eq!(rc, 0, "render: {:?}", last_error());
    assert_eq!(out_len, 4 * 4 * 4);
    let rgba = unsafe { std::slice::from_raw_parts(out, out_len) }.to_vec();
    unsafe { combsmesh_bytes_free(out, out_len) };
    // Frame 0 of add_image_rgba is the atlas verbatim.
    assert_eq!(rgba, (0..4 * 4 * 4).map(|i| i as u8).collect::<Vec<_>>());

    // Out-of-range frame → 1 + error.
    let rc = unsafe {
        combsmesh_render_sprite(binary.as_ptr(), binary.len(), 99, &mut out, &mut out_len)
    };
    assert_eq!(rc, 1);
    assert!(last_error().expect("error").contains("out of range"));
}

#[test]
fn op_json_build_from_binary_unicode_roundtrip() {
    let build = op_json(
        r#"{"op":"build","name":"json-emoji","description":"via op_json",
            "blocks":[{"type":"tdo","items":[{"key":"a","value":"task a","status":"Pending"}]}]}"#,
    )
    .expect("build");
    assert_eq!(build["emoji"]["name"], "json-emoji");
    let binary_b64 = build["binary_b64"].as_str().expect("binary_b64");
    assert!(!build["unicode"].as_str().expect("unicode").is_empty());

    // from_binary(build.binary) → same emoji.
    let req = format!(r#"{{"op":"from_binary","binary_b64":"{binary_b64}"}}"#);
    let parsed = op_json(&req).expect("from_binary");
    assert_eq!(parsed["emoji"], build["emoji"]);

    // unicode round trip through op_json.
    let unicode = build["unicode"].as_str().unwrap();
    let req = serde_json::json!({"op": "from_unicode", "unicode": unicode}).to_string();
    let parsed = op_json(&req).expect("from_unicode");
    assert_eq!(parsed["emoji"]["name"], "json-emoji");

    // to_unicode(emoji) → same string.
    let req = serde_json::json!({"op": "to_unicode", "emoji": build["emoji"]}).to_string();
    let encoded = op_json(&req).expect("to_unicode");
    assert_eq!(encoded["unicode"].as_str().unwrap(), unicode);
}

#[test]
fn op_json_render() {
    let binary = sample_binary();
    let b64 = base64_encode(&binary);
    let req = format!(r#"{{"op":"render","binary_b64":"{b64}","frame":0}}"#);
    let rendered = op_json(&req).expect("render");
    assert_eq!(rendered["width"], 4);
    assert_eq!(rendered["height"], 4);
    let rgba = base64_decode(rendered["rgba_b64"].as_str().unwrap());
    assert_eq!(rgba.len(), 4 * 4 * 4);
}

#[test]
fn op_json_registry_ops() {
    // Isolate the registry in a temp COMBS_HOME.
    let dir = std::env::temp_dir().join(format!("combsmesh-abi-{}", std::process::id()));
    std::env::set_var("COMBS_HOME", &dir);

    let binary = sample_binary();
    let b64 = base64_encode(&binary);
    let req = format!(r#"{{"op":"registry_register","binary_b64":"{b64}"}}"#);
    let registered = op_json(&req).expect("register");
    let hash = registered["hash"].as_str().expect("hash").to_string();
    assert_eq!(hash.len(), 64);

    let list = op_json(r#"{"op":"registry_list"}"#).expect("list");
    assert_eq!(list["entries"].as_array().expect("entries").len(), 1);
    assert_eq!(list["entries"][0]["name"], "abi-emoji");

    // Resolve by name and by hash.
    for key in ["abi-emoji", hash.as_str()] {
        let req = serde_json::json!({"op": "registry_resolve", "name_or_hash": key}).to_string();
        let resolved = op_json(&req).expect("resolve");
        assert_eq!(resolved["emoji"]["name"], "abi-emoji");
        assert_eq!(resolved["binary_b64"].as_str().unwrap(), b64);
    }

    let missing = op_json(r#"{"op":"registry_resolve","name_or_hash":"nope"}"#);
    assert!(missing.expect_err("unknown name").contains("no emoji"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn error_paths_set_last_error() {
    let _guard = KEYRING_LOCK.lock().unwrap();
    // NULL op request.
    assert!(unsafe { combsmesh_op_json(ptr::null()) }.is_null());
    assert!(last_error().expect("error").contains("request_json is NULL"));

    // Unknown op.
    let err = op_json(r#"{"op":"nonsense"}"#).expect_err("unknown op");
    assert!(err.contains("unknown op"));

    // Garbage binary to render_sprite.
    let garbage = [0xDEu8; 32];
    let (mut out, mut out_len) = (ptr::null_mut(), 0usize);
    let rc = unsafe {
        combsmesh_render_sprite(garbage.as_ptr(), garbage.len(), 0, &mut out, &mut out_len)
    };
    assert_eq!(rc, 1);
    assert!(last_error().is_some());

    // NULL data / out params.
    let rc = unsafe { combsmesh_render_sprite(ptr::null(), 8, 0, &mut out, &mut out_len) };
    assert_eq!(rc, 1);
    assert!(last_error().expect("error").contains("cmse is NULL"));

    assert_eq!(unsafe { combsmesh_init(KEY.as_ptr(), KEY.len()) }, 0);
    let rc = unsafe { combsmesh_encrypt_memory(b"x".as_ptr(), 1, ptr::null_mut(), &mut out_len) };
    assert_eq!(rc, 1);
    assert!(last_error().expect("error").contains("out"));
    assert_eq!(combsmesh_shutdown(), 0);

    // Garbage base64 in op.
    let err = op_json(r#"{"op":"from_binary","binary_b64":"!!!"}"#).expect_err("bad b64");
    assert!(err.contains("binary_b64"));
}

#[cfg(not(feature = "engine"))]
#[test]
fn infer_without_engine_feature_is_unsupported() {
    let prompt = CString::new("hello").unwrap();
    let mut out: *mut c_char = ptr::null_mut();
    let rc = unsafe { combsmesh_infer(prompt.as_ptr(), &mut out) };
    assert_eq!(rc, 1);
    let err = last_error().expect("error");
    assert!(err.contains("unsupported"), "error was: {err}");
    assert!(err.contains("engine"), "error was: {err}");

    // NULL prompt.
    let rc = unsafe { combsmesh_infer(ptr::null(), &mut out) };
    assert_eq!(rc, 1);
    assert!(last_error().expect("error").contains("prompt is NULL"));
}

fn base64_encode(bytes: &[u8]) -> String {
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    STANDARD.encode(bytes)
}

fn base64_decode(s: &str) -> Vec<u8> {
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    STANDARD.decode(s).expect("valid base64")
}
