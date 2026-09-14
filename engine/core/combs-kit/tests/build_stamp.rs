//! The kit's build stamp names the code behind a record the way the CLI's
//! does: git where the source is a checkout, else the commit a packed
//! archive was made from (COMBS_SOURCE_COMMIT / COMBS_SOURCE_REF, the pod
//! road's stamp). A build from `git archive` has no .git to ask; a kit that
//! stamps only from git names such a build "unknown", and a record that
//! cannot name its code cannot be checked by a stranger.

fn build_script() -> String {
    std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/build.rs")).expect("read the kit's build.rs")
}

#[test]
fn the_kit_resolves_its_source_through_the_shared_stamp() {
    let build = build_script();
    assert!(
        build.contains("include!(\"build/stamp.rs\")"),
        "the kit's build.rs does not include the shared source resolver (build/stamp.rs)"
    );
    assert!(
        build.contains("source_identity("),
        "the kit's build.rs never calls source_identity"
    );
}

#[test]
fn the_kit_restamps_when_the_packed_commit_changes() {
    let build = build_script();
    for var in ["COMBS_SOURCE_COMMIT", "COMBS_SOURCE_REF"] {
        assert!(
            build.contains(&format!("rerun-if-env-changed={var}")),
            "the kit's stamp is not re-run when {var} changes"
        );
    }
}
