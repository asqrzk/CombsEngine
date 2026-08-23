//! Monotonic time that exists on every target the engine runs on.
//!
//! `std::time::Instant::now()` does not merely return something useless on
//! `wasm32-unknown-unknown` — it panics, because there is no clock behind
//! it. `web-time` presents the same API over `performance.now()` in a
//! browser and re-exports `std::time` everywhere else, so the engine
//! measures itself identically in both places and `Duration` stays the std
//! type `GenerationStats` already exposes to callers.

//! Both clocks are here, and every clock the engine reads goes through
//! this module. `SystemTime` is the one that bit us: it is not read for
//! timing but for *entropy* — the sampler seeds its RNG from the wall
//! clock when a request supplies no seed — so a build that never touched
//! it in a greedy test still panicked on the first sampled turn in a
//! browser. Nothing here may be imported from `std::time` directly; the
//! test at the bottom of this file enforces that.

#[cfg(not(target_family = "wasm"))]
pub(crate) use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[cfg(target_family = "wasm")]
pub(crate) use web_time::{Instant, SystemTime, UNIX_EPOCH};

/// No source file outside this one may name `std::time::Instant` or
/// `std::time::SystemTime`.
///
/// A clock that panics only in a browser is invisible to every native
/// test, so the guard cannot be behavioural — it has to be a rule about
/// the source. Reading the crate's own files is a blunt instrument, and
/// it is the instrument that would have caught this before it shipped.
#[cfg(test)]
mod tests {
    #[test]
    fn no_crate_file_reads_std_time_directly() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut offenders = Vec::new();
        for entry in std::fs::read_dir(&dir).expect("read src") {
            let path = entry.expect("entry").path();
            if path.extension().is_none_or(|e| e != "rs") {
                continue;
            }
            if path.file_name().is_some_and(|f| f == "time.rs") {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("read file");
            for (n, line) in text.lines().enumerate() {
                if line.trim_start().starts_with("//") {
                    continue;
                }
                if line.contains("std::time::Instant") || line.contains("std::time::SystemTime") {
                    offenders.push(format!(
                        "{}:{}: {}",
                        path.file_name().unwrap().to_string_lossy(),
                        n + 1,
                        line.trim()
                    ));
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "these read a std clock directly, which panics in a browser — \
             use crate::time instead:\n  {}",
            offenders.join("\n  ")
        );
    }
}
