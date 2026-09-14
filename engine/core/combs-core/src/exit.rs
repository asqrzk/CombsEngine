//! Ending a process that has touched a GPU.
//!
//! cubecl-wgpu gives every device stream a poll thread that owns a
//! `wgpu::Device` and waits on submissions (vendor/cubecl-wgpu
//! compute/poll.rs). Its stream lives in a per-device client that lives as
//! long as the process, so the thread outlives `main`. When a process then
//! ends through a normal return or `std::process::exit`, libc runs atexit
//! handlers and C++ static destructors — among them a GPU driver's —
//! while that thread may still be inside the driver. When it
//! is, a process that did all its work dies at teardown ("double free or
//! corruption", exit 134); upstream this is gfx-rs/wgpu#8365.
//!
//! [`exit_process`] ends the process without running those destructors:
//! it flushes what the process wrote, then calls `_exit`. Nothing a combs
//! command needs happens in a destructor — records, reports and logs are
//! written and flushed before this is called.

use std::io::Write;

/// Flush stdout and stderr, then end the process with `code` without
/// running atexit handlers or C++ static destructors (unix: `_exit`).
/// Elsewhere this is `std::process::exit`.
pub fn exit_process(code: i32) -> ! {
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    #[cfg(unix)]
    {
        // SAFETY: `_exit` takes an int and never returns. Buffers held by
        // Rust's stdio were flushed above; nothing else needs unwinding.
        unsafe { libc::_exit(code) }
    }
    #[cfg(not(unix))]
    {
        std::process::exit(code)
    }
}
