//! A GPU process ends through `combs_core::exit::exit_process`: flush, then
//! `_exit`, so a GPU driver's static destructors never race cubecl-wgpu's
//! poll thread (gfx-rs/wgpu#8365: a finished process aborting at teardown).
//! `_exit` skips Rust's own stdio teardown too, so these tests hold the fix to
//! what a caller relies on: the whole output arrives, and the exit code and
//! error shape are the ones a normal return would have produced.

use std::process::Command;

fn combs() -> Command {
    Command::new(env!("CARGO_BIN_EXE_combs"))
}

#[test]
fn build_info_json_arrives_whole_through_the_exit() {
    let out = combs().args(["build-info", "--json"]).output().expect("spawn combs build-info");
    assert!(out.status.success(), "build-info exited {:?}", out.status.code());
    let text = String::from_utf8(out.stdout).expect("utf8");
    let trimmed = text.trim_end();
    assert!(trimmed.starts_with('{') && trimmed.ends_with('}'), "the JSON was cut: {trimmed}");
    assert_eq!(trimmed.matches('{').count(), trimmed.matches('}').count(), "unbalanced JSON: {trimmed}");
    for key in ["\"git\"", "\"commit\"", "\"serving_dtype\"", "\"version\""] {
        assert!(trimmed.contains(key), "{key} missing from: {trimmed}");
    }
}

#[test]
fn a_failing_command_keeps_its_exit_code_and_error_shape() {
    let out = combs()
        .args(["inspect", "--model", "/nonexistent/combs-exit-test/model.gguf"])
        .output()
        .expect("spawn combs inspect");
    assert_eq!(out.status.code(), Some(1), "a failed command must exit 1");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.lines().any(|l| l.starts_with("Error: ")), "std's `Error: …` line is gone: {err}");
}

#[test]
fn the_doctor_ends_cleanly_with_its_whole_report() {
    if std::env::var_os("COMBS_TEST_NO_GPU").is_some() {
        eprintln!("skipping: COMBS_TEST_NO_GPU");
        return;
    }
    let out = combs().args(["doctor", "--json", "--skip-e2e"]).output().expect("spawn combs doctor");
    let text = String::from_utf8_lossy(&out.stdout);
    let trimmed = text.trim_end();
    assert!(trimmed.ends_with('}'), "the doctor's report was cut: {trimmed}");
    let verdict_pass = trimmed.contains("\"verdict\": \"pass\"");
    let code = out.status.code();
    if verdict_pass {
        assert_eq!(code, Some(0), "a passing doctor must exit 0, not {code:?} (a crash at teardown)");
    } else {
        assert_eq!(code, Some(1), "a refusing doctor must exit 1, not {code:?}");
    }
}
