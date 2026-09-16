//! The toolkit names the code behind it the way the CLI and the kit do:
//! git where the source is a checkout, else the commit a packed archive
//! was made from (COMBS_SOURCE_COMMIT / COMBS_SOURCE_REF, the pod road's
//! stamp). A build from `git archive` has no .git to ask; a binary that
//! stamps only from git names such a build "unknown", and a toolkit that
//! cannot name its own build has no business reporting on anyone else's.

fn build_script() -> String {
    std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/build.rs"))
        .expect("read the toolkit's build.rs")
}

#[test]
fn the_toolkit_resolves_its_source_through_the_shared_stamp() {
    let build = build_script();
    assert!(
        build.contains("include!(\"build/stamp.rs\")"),
        "the toolkit's build.rs does not include the shared source resolver (build/stamp.rs)"
    );
    assert!(
        build.contains("source_identity("),
        "the toolkit's build.rs never calls source_identity"
    );
}

#[test]
fn the_toolkit_restamps_when_the_packed_commit_changes() {
    let build = build_script();
    for var in ["COMBS_SOURCE_COMMIT", "COMBS_SOURCE_REF"] {
        assert!(
            build.contains(&format!("rerun-if-env-changed={var}")),
            "a change to {var} would not re-stamp the build"
        );
    }
}

/// The stamp resolver is a byte copy shared with combs-cli. Kept in step
/// by xtask's drift guard; asserted here too so a copy that drifts fails
/// the toolkit's own tier rather than only the xtask one.
#[test]
fn the_shared_resolver_is_a_byte_copy_of_the_cli_s() {
    let mine = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/build/stamp.rs"))
        .expect("read the toolkit's stamp resolver");
    let theirs = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../combs-cli/build/stamp.rs"
    ))
    .expect("read the cli's stamp resolver");
    assert_eq!(
        mine, theirs,
        "combs-tk/build/stamp.rs has drifted from combs-cli/build/stamp.rs"
    );
}

/// Every crate this toolkit is allowed to take, and nothing else.
///
/// An ALLOW list, not a deny list, because a deny list only stops the
/// dependencies somebody already thought of: the guard below it names
/// six engine and GPU crates, so `ureq`, `tokio` or `sha2` could be
/// added tomorrow and pass in silence. The toolkit talks to 127.0.0.1
/// against a server with no TLS and no auth, drives binaries that
/// already answer every hard question, and is worth keeping near a
/// megabyte beside an engine of a hundred. Adding a name here is a
/// decision, and it should read like one.
const ALLOWED: &[&str] = &["clap", "serde", "serde_json", "console", "libc"];

/// The dependency names a manifest declares, minus the allowed ones.
/// Pure, so the guard can be shown to catch something before it is
/// trusted to catch everything.
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
        // `serde.workspace = true` names the crate before the dot.
        let name = lhs.trim().split('.').next().unwrap_or("").trim();
        if name.is_empty() || ALLOWED.contains(&name) || found.iter().any(|f| f == name) {
            continue;
        }
        found.push(name.to_string());
    }
    found
}

/// The guard must be shown to catch a smuggled dependency before it is
/// believed when it says the real manifest is clean — a check that
/// cannot fail proves nothing about the one it passed.
#[test]
fn the_toolkit_takes_only_the_dependencies_it_is_allowed() {
    let smuggled = "[dependencies]\nclap = \"4\"\nureq = { version = \"3\" }\n";
    assert_eq!(
        foreign_deps(smuggled),
        vec!["ureq".to_string()],
        "the guard did not catch a dependency nobody allowed"
    );

    let manifest = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"))
        .expect("read the toolkit's manifest");
    let foreign = foreign_deps(&manifest);
    assert!(
        foreign.is_empty(),
        "combs-tk has taken dependencies nobody allowed: {foreign:?}"
    );
}

/// The toolkit drives binaries; it must not grow a dependency on the
/// engine crates. Linking combs-core would pull burn, wgpu and cubecl in
/// for facts the kit already answers, and a toolkit that takes a minute
/// to build is one people stop rebuilding.
#[test]
fn the_toolkit_links_no_engine_crate() {
    let manifest = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"))
        .expect("read the toolkit's manifest");
    let deps = manifest
        .split("[dependencies]")
        .nth(1)
        .expect("the manifest has a dependencies section");
    for heavy in ["combs-core", "combs-runtime", "combs-models", "combs-formats", "burn", "cubecl"] {
        assert!(
            !deps.contains(heavy),
            "combs-tk depends on {heavy} — it drives the tools, it does not link them"
        );
    }
}
