//! Embeds the Svelte UI template (`engine/ui/template`) into the `combs`
//! binary so `combs chew` works no matter how the CLI was installed
//! (cargo / npm / pip / prebuilt binary — no repo checkout required).
//!
//! Template resolution order at build time:
//!   1. `$CARGO_MANIFEST_DIR/vendor/ui-template` (staged by `cargo xtask
//!      package` before `cargo publish`, since published crates cannot
//!      reference files outside their manifest dir)
//!   2. `../../ui/template` (normal in-repo build)
//!
//! Emits `$OUT_DIR/template_manifest.rs` with
//! `pub static TEMPLATE_FILES: &[(&str, &[u8])]`.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const SKIP_DIRS: &[&str] = &["node_modules", "dist", ".svelte-kit", ".vite", "data"];
/// Runtime secrets/state the proxy creates — never embed these in a binary.
const SKIP_FILES: &[&str] = &["master.key", "permissions.json", "manifest.json", "authn.json", "package-lock.json"];

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let vendored = manifest.join("vendor/ui-template");
    let repo = manifest.join("../../ui/template");
    let root = if vendored.is_dir() { vendored } else { repo };
    assert!(
        root.is_dir(),
        "UI template not found at {}",
        root.display()
    );

    let mut entries: Vec<(String, PathBuf)> = Vec::new();
    collect(&root, &root, &mut entries);
    entries.sort();

    let mut code = String::from(
        "/// (relative path, contents) for every file in the embedded UI template.\n\
         pub static TEMPLATE_FILES: &[(&str, &[u8])] = &[\n",
    );
    for (rel, abs) in &entries {
        println!("cargo:rerun-if-changed={}", abs.display());
        // Forward slashes work in include_bytes! on every platform.
        let abs_fwd = abs.display().to_string().replace('\\', "/");
        code.push_str(&format!(
            "    ({rel:?}, include_bytes!(\"{abs_fwd}\")),\n"
        ));
    }
    code.push_str("];\n");

    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("template_manifest.rs");
    fs::write(out, code).unwrap();
}

fn collect(root: &Path, dir: &Path, out: &mut Vec<(String, PathBuf)>) {
    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if entry.file_type().unwrap().is_dir() {
            if !SKIP_DIRS.contains(&name.as_str()) {
                collect(root, &path, out);
            }
        } else if !SKIP_FILES.contains(&name.as_str()) {
            let rel = path
                .strip_prefix(root)
                .unwrap()
                .display()
                .to_string()
                .replace('\\', "/");
            out.push((rel, path));
        }
    }
}
