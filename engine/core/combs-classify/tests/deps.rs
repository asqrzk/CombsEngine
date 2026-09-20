//! The gate takes serde and nothing else.
//!
//! An ALLOW list, not a deny list, for the reason the toolkit's own
//! guard gives: a deny list only stops what somebody already thought of.
//! This crate is linked by the serve route, by the kit's suite and by
//! anything that routes on a decision; a tensor crate added here would
//! make every one of them pay minutes of build for arithmetic that
//! costs microseconds.

const ALLOWED: &[&str] = &["serde", "serde_json"];

/// The dependency names a manifest declares, minus the allowed ones.
/// Pure, so the guard can be shown to catch something before it is
/// trusted about the real manifest.
fn foreign_deps(manifest: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut in_deps = false;
    for line in manifest.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_deps = line.contains("dependencies");
            continue;
        }
        if !in_deps || line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((lhs, _)) = line.split_once('=') else {
            continue;
        };
        let name = lhs.trim().split('.').next().unwrap_or("").trim();
        if name.is_empty() || ALLOWED.contains(&name) || found.iter().any(|f| f == name) {
            continue;
        }
        found.push(name.to_string());
    }
    found
}

/// A GUARD, not a red: it passes on its first run because the manifest
/// is already clean. What earns it its place is the negative control —
/// a check that cannot fail proves nothing about the one it passed.
#[test]
fn the_gate_takes_only_the_dependencies_it_is_allowed() {
    let smuggled = "[dependencies]\nserde = \"1\"\nburn = { version = \"0.21\" }\n";
    assert_eq!(
        foreign_deps(smuggled),
        vec!["burn".to_string()],
        "the guard did not catch a dependency nobody allowed"
    );

    let manifest = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"))
        .expect("read the gate's manifest");
    let foreign = foreign_deps(&manifest);
    assert!(
        foreign.is_empty(),
        "combs-classify has taken dependencies nobody allowed: {foreign:?}"
    );
}
