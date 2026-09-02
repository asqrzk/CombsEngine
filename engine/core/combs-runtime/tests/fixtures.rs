//! H3d — fixtures stay small enough to review and to clone. The
//! ceilings live in docs/TOLERANCES.md §4 (local): 256 KiB per file,
//! 2 MiB per tests/data directory. Harmony's generator scripts write
//! fixtures here too; anything larger belongs in the model cache, not
//! the repository.

#[test]
fn fixtures_stay_under_the_ceiling() {
    const FILE_CEILING: u64 = 256 * 1024;
    const DIR_CEILING: u64 = 2 * 1024 * 1024;
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let mut total = 0u64;
    for entry in std::fs::read_dir(&dir).expect("tests/data exists") {
        let entry = entry.unwrap();
        let meta = entry.metadata().unwrap();
        if !meta.is_file() {
            continue;
        }
        let len = meta.len();
        assert!(
            len <= FILE_CEILING,
            "{} is {} bytes (ceiling {})",
            entry.path().display(),
            len,
            FILE_CEILING
        );
        total += len;
    }
    assert!(
        total <= DIR_CEILING,
        "tests/data totals {total} bytes (ceiling {DIR_CEILING})"
    );
    assert!(
        total > 0,
        "tests/data unexpectedly empty — the ceiling test lost its subject"
    );
}
