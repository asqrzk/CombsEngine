//! # combs-mesh-ffi — CombsMesh C ABI
//!
//! Stable C ABI + JSON FFI for the CombsMesh emoji engine, mirroring the
//! combs-ffi conventions: panic-fenced `extern "C"` exports, a thread-local
//! error slot read via `combsmesh_last_error`, `int32_t` return codes
//! (0 = ok, 1 = error), and one `combsmesh_op_json` escape hatch for every
//! op that does not need a dedicated symbol (build/encode/registry/render).
//!
//! ## Ownership contract
//!
//! - Strings returned by `combsmesh_op_json` / `combsmesh_infer` are owned
//!   by the library and must be released with `combsmesh_string_free`.
//! - Byte buffers written to out-params (`combsmesh_encrypt_memory`,
//!   `combsmesh_decrypt_memory`, `combsmesh_render_sprite`) are leaked
//!   `Box<[u8]>` allocations; release them with `combsmesh_bytes_free`,
//!   which reconstructs the exact same boxed slice — the pair is symmetric.
//! - `combsmesh_last_error` returns a *borrowed* pointer: do not free it;
//!   invalidated by the next FFI call on the same thread.

mod ops;
#[cfg(feature = "engine")]
mod runtime_engine;
mod types;

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

use combs_mesh::crypto;
use combs_mesh::{CpuRenderer, Renderer};

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(msg: impl std::fmt::Display) {
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = Some(CString::new(msg.to_string().replace('\0', " ")).unwrap())
    });
}

/// Returns the last error message on this thread, or NULL. The pointer is
/// borrowed — do not free it; invalidated by the next FFI call on the same
/// thread.
#[no_mangle]
pub extern "C" fn combsmesh_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| match &*slot.borrow() {
        Some(s) => s.as_ptr(),
        None => ptr::null(),
    })
}

/// Frees a string previously returned by `combsmesh_op_json` or
/// `combsmesh_infer`.
///
/// # Safety
/// `s` must have been returned by this library and not yet freed.
#[no_mangle]
pub unsafe extern "C" fn combsmesh_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(unsafe { CString::from_raw(s) });
    }
}

/// Frees a byte buffer previously written to an out-param by
/// `combsmesh_encrypt_memory` / `combsmesh_decrypt_memory` /
/// `combsmesh_render_sprite`.
///
/// # Safety
/// `p`/`len` must come from this library, unchanged, and not yet freed.
/// Reconstructs the leaked `Box<[u8]>` (`slice_from_raw_parts_mut(p, len)`)
/// — exactly symmetric with the allocation in `leak_bytes`.
#[no_mangle]
pub unsafe extern "C" fn combsmesh_bytes_free(p: *mut u8, len: usize) {
    if !p.is_null() {
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(p, len)) });
    }
}

/// Leaks a byte buffer for the FFI boundary (see `combsmesh_bytes_free`).
fn leak_bytes(v: Vec<u8>) -> (*mut u8, usize) {
    let slice = Box::leak(v.into_boxed_slice());
    (slice.as_mut_ptr(), slice.len())
}

unsafe fn read_str<'a>(s: *const c_char, what: &str) -> Result<&'a str, String> {
    if s.is_null() {
        return Err(format!("{what} is NULL"));
    }
    unsafe { CStr::from_ptr(s) }
        .to_str()
        .map_err(|e| format!("{what} is not valid UTF-8: {e}"))
}

unsafe fn read_bytes<'a>(p: *const u8, len: usize, what: &str) -> Result<&'a [u8], String> {
    if p.is_null() && len > 0 {
        return Err(format!("{what} is NULL"));
    }
    Ok(unsafe { std::slice::from_raw_parts(p, len) })
}

/// Writes a byte buffer to the out-params; NULL out-params are an error.
unsafe fn write_out(out: *mut *mut u8, out_len: *mut usize, v: Vec<u8>) -> Result<(), String> {
    if out.is_null() || out_len.is_null() {
        return Err("out/out_len is NULL".into());
    }
    let (p, len) = leak_bytes(v);
    unsafe {
        *out = p;
        *out_len = len;
    }
    Ok(())
}

/// Initializes the process-wide keyring (`combs_mesh::crypto::init`).
/// `key == NULL` or `key_len == 0` generates a random master key.
/// Returns 0 on success, 1 on error.
///
/// # Safety
/// `key` must point to `key_len` readable bytes (or be NULL with len 0).
#[no_mangle]
pub unsafe extern "C" fn combsmesh_init(key: *const u8, key_len: usize) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        let key = if key.is_null() || key_len == 0 {
            None
        } else {
            Some(unsafe { read_bytes(key, key_len, "key") }?)
        };
        crypto::init(key).map_err(|e| e.to_string())
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            set_last_error(e);
            1
        }
        Err(_) => {
            set_last_error("panic during init");
            1
        }
    }
}

/// Shuts the library down, zeroizing the master key. Always returns 0.
#[no_mangle]
pub extern "C" fn combsmesh_shutdown() -> i32 {
    crypto::shutdown();
    0
}

/// Encrypts a memory blob (AES-256-GCM, nonce-prefixed). Requires a prior
/// `combsmesh_init`. Free `*out` with `combsmesh_bytes_free`.
///
/// # Safety
/// `data` must point to `len` readable bytes; `out`/`out_len` must be
/// valid for writing.
#[no_mangle]
pub unsafe extern "C" fn combsmesh_encrypt_memory(
    data: *const u8,
    len: usize,
    out: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    crypt_memory(data, len, out, out_len, true)
}

/// Decrypts a blob produced by `combsmesh_encrypt_memory`.
///
/// # Safety
/// Same contract as `combsmesh_encrypt_memory`.
#[no_mangle]
pub unsafe extern "C" fn combsmesh_decrypt_memory(
    data: *const u8,
    len: usize,
    out: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    crypt_memory(data, len, out, out_len, false)
}

fn crypt_memory(
    data: *const u8,
    len: usize,
    out: *mut *mut u8,
    out_len: *mut usize,
    encrypt: bool,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        let data = unsafe { read_bytes(data, len, "data") }?;
        let keyring = crypto::global().map_err(|e| e.to_string())?;
        let algorithm = combs_mesh::EncryptionAlgorithm::Aes256Gcm;
        let bytes = if encrypt {
            keyring.encrypt(data, algorithm)
        } else {
            keyring.decrypt(data, algorithm)
        }
        .map_err(|e| e.to_string())?;
        unsafe { write_out(out, out_len, bytes) }
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            set_last_error(e);
            1
        }
        Err(_) => {
            set_last_error("panic during memory crypto");
            1
        }
    }
}

/// Renders frame `frame` of the emoji in `cmse` (a `.cmse` binary) to a
/// tightly packed `frame_width * frame_height * 4` RGBA8 buffer. Free
/// `*out_rgba` with `combsmesh_bytes_free`. Encrypted containers are
/// accepted when the process keyring holds the right key.
///
/// # Safety
/// `cmse` must point to `len` readable bytes; out-params must be valid.
#[no_mangle]
pub unsafe extern "C" fn combsmesh_render_sprite(
    cmse: *const u8,
    len: usize,
    frame: u32,
    out_rgba: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        let bytes = unsafe { read_bytes(cmse, len, "cmse") }?;
        let emoji = crate::ops::decode_emoji(bytes)?;
        let image = emoji
            .get_image()
            .ok_or_else(|| "emoji has no image block".to_string())?;
        let rgba = CpuRenderer::new()
            .render_frame(&image.atlas, frame)
            .map_err(|e| e.to_string())?;
        unsafe { write_out(out_rgba, out_len, rgba) }
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            set_last_error(e);
            1
        }
        Err(_) => {
            set_last_error("panic during sprite render");
            1
        }
    }
}

/// Runs inference on `prompt`; the returned string must be freed with
/// `combsmesh_string_free`. Requires the `engine` feature and a model
/// loaded via `combsmesh_op_json` (`{"op":"engine_load", ...}`); otherwise
/// returns 1 with an "unsupported" error.
///
/// # Safety
/// `prompt` must be a valid NUL-terminated UTF-8 string; `out` must be
/// valid for writing.
#[no_mangle]
pub unsafe extern "C" fn combsmesh_infer(prompt: *const c_char, out: *mut *mut c_char) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        if out.is_null() {
            return Err("out is NULL".into());
        }
        let prompt = unsafe { read_str(prompt, "prompt") }?;
        let text = infer_impl(prompt)?;
        let cstr = CString::new(text.replace('\0', " ")).map_err(|e| e.to_string())?;
        unsafe { *out = CString::into_raw(cstr) };
        Ok(())
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            set_last_error(e);
            1
        }
        Err(_) => {
            set_last_error("panic during inference");
            1
        }
    }
}

#[cfg(feature = "engine")]
fn infer_impl(prompt: &str) -> Result<String, String> {
    runtime_engine::RuntimeEngine::global()
        .infer(prompt)
        .map_err(|e| e.to_string())
}

#[cfg(not(feature = "engine"))]
fn infer_impl(prompt: &str) -> Result<String, String> {
    use combs_mesh::ffi_trait::{CombsEngineCore, DefaultEngine};
    DefaultEngine::new()
        .infer(prompt)
        .map_err(|e| e.to_string())
}

/// JSON-FFI escape hatch: one stable symbol for every op that does not
/// need a dedicated C signature. Request: `{"op": "...", ...}`; response:
/// an op-specific JSON object. Free the returned string with
/// `combsmesh_string_free`. Returns NULL on error (see
/// `combsmesh_last_error`). Supported ops: `build`, `from_binary`,
/// `to_unicode`, `from_unicode`, `registry_register`, `registry_resolve`,
/// `registry_list`, `render`, `engine_load` (engine feature).
///
/// # Safety
/// `request_json` must be a valid NUL-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn combsmesh_op_json(request_json: *const c_char) -> *mut c_char {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<CString, String> {
        let json = unsafe { read_str(request_json, "request_json") }?;
        let response = ops::dispatch(json)?;
        CString::new(response).map_err(|e| e.to_string())
    }));
    match result {
        Ok(Ok(s)) => CString::into_raw(s),
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error("panic during op dispatch");
            ptr::null_mut()
        }
    }
}
