//! Is this binary newer than the code it claims to measure?
//!
//! The house already refuses a stale binary in one place: a battery
//! probe dies when any crate source is newer than the app binary,
//! because a chain that trusts a stale binary lies. The kit judges
//! every other instrument, so it takes the same law — and it has
//! needed it twice in one day (an engine six days old serving every
//! road; an f16 kit half an hour behind its own source with a proof
//! run queued against it).
//!
//! Only a developer tree can be checked: a shipped kit has no sources
//! beside it, and their absence is not staleness. That case reports
//! `Unknown` and says so, rather than passing silently.

pub enum Freshness {
    /// The binary is newer than every source found.
    Fresh,
    /// Sources are newer. The names are the evidence.
    Stale(Vec<String>),
    /// No source tree beside the binary — a shipped kit. Not a failure.
    Unknown,
}

/// When this binary was built — read from the BINARY ITSELF, not from a
/// compile-time stamp.
///
/// The first version of this used a build script. It was wrong in the
/// way that matters: cargo does not re-run a build script when an
/// unrelated crate changes, so after a real rebuild the stamp was stale
/// and the guard refused a binary that was in fact fresh. Caught by
/// running it — refused, rebuilt, refused again. The file's own
/// modification time cannot drift from the file.
pub fn built_at() -> u64 {
    std::env::current_exe()
        .and_then(std::fs::metadata)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The workspace this binary was built from, if it is still on disk.
fn workspace() -> Option<std::path::PathBuf> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent()?;
    dir.is_dir().then(|| dir.to_path_buf())
}

pub fn check() -> Freshness {
    let Some(root) = workspace() else {
        return Freshness::Unknown;
    };
    let built = built_at();
    if built == 0 {
        return Freshness::Unknown;
    }
    let mut newer = Vec::new();
    // The crates whose code this binary's numbers depend on. Not the
    // whole tree: a change to the diffusion crate does not make a
    // decode measurement stale, and crying about it would teach people
    // to pass the override.
    for crate_dir in [
        "combs-kit",
        "combs-core",
        "combs-models",
        "combs-runtime",
        "combs-formats",
    ] {
        walk(&root.join(crate_dir).join("src"), built, &mut newer);
        if newer.len() > 6 {
            break;
        }
    }
    if newer.is_empty() {
        Freshness::Fresh
    } else {
        Freshness::Stale(newer)
    }
}

fn walk(dir: &std::path::Path, built: u64, found: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, built, found);
        } else if p.extension().is_some_and(|x| x == "rs" || x == "wgsl") {
            let newer = e
                .metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .is_some_and(|d| d.as_secs() > built);
            if newer {
                found.push(p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
            }
        }
        if found.len() > 6 {
            return;
        }
    }
}

/// The words a stale binary earns, and the one way past them.
pub fn refusal(names: &[String]) -> String {
    format!(
        "this kit was built before {} — the numbers would be yesterday's \
         wearing today's date. Rebuild it, or pass --allow-stale if you \
         mean to measure the old binary on purpose.",
        names.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stamp must be real, or the check is theatre — and it must
    /// come from the binary, so a rebuild always moves it.
    #[test]
    fn the_build_stamp_is_the_binarys_own_time() {
        let t = built_at();
        // After 2020 and before 2100: a zero here means the binary could
        // not read itself and the guard would pass everything.
        assert!(t > 1_577_836_800, "the build stamp is missing or absurd: {t}");
        assert!(t < 4_102_444_800, "the build stamp is in the far future: {t}");
        // And it is THIS file's time, not a compile-time constant: the
        // test binary was written after the sources it was built from.
        let exe = std::env::current_exe().expect("a test binary has a path");
        let m = std::fs::metadata(&exe).expect("and a modification time");
        let from_file = m
            .modified()
            .unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        assert_eq!(t, from_file);
    }

    /// The refusal names what made it stale and the way past it.
    #[test]
    fn the_refusal_names_the_evidence_and_the_override() {
        let words = refusal(&["qlinear.rs".into(), "kv.rs".into()]);
        assert!(words.contains("qlinear.rs"), "{words}");
        assert!(words.contains("--allow-stale"), "{words}");
    }
}
