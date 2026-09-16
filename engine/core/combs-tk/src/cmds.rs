//! The commands. Each one finds a tool, drives it, and renders what
//! came back — and every one of them says which build answered.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use crate::render;
use crate::state::{self, Liveness};
use crate::tools::{self, Tool};
use crate::ui::{self, Status};

/// The engine's own headroom when it prices a load: memory it refuses to
/// count as available, because a machine driven to its last byte swaps
/// and a swapping machine is a crashed machine. The twin of
/// `DEFAULT_HEADROOM_BYTES` in combs-cli's fit.
const HEADROOM_BYTES: u64 = 2 * 1024 * 1024 * 1024;

fn fail(msg: &str) -> i32 {
    eprintln!("{} {}", ui::bad("combs-tk:"), msg);
    1
}

/// Run a tool and hand back its stdout, or say why not.
fn capture(bin: &PathBuf, args: &[&str]) -> Result<String, String> {
    let out = Command::new(bin)
        .args(args)
        .output()
        .map_err(|e| format!("{} could not be run: {e}", bin.display()))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let tail = err.lines().rev().take(3).collect::<Vec<_>>().join(" / ");
        return Err(format!(
            "{} {} exited {} — {tail}",
            bin.display(),
            args.first().copied().unwrap_or(""),
            out.status.code().unwrap_or(-1)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// The machine block the kit reports. One implementation of these facts
/// exists and it is the kit's — this asks it rather than reading the
/// machine a second, differently-wrong way.
fn machine() -> Result<serde_json::Value, String> {
    let kit = tools::find(Tool::Kit)?;
    let text = capture(&kit, &["system", "--json"])?;
    let v: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("the kit's system record is not JSON: {e}"))?;
    v.get("machine")
        .cloned()
        .ok_or_else(|| "the kit's record carries no machine block".to_string())
}

pub fn system(json: bool) -> i32 {
    let m = match machine() {
        Ok(m) => m,
        Err(e) => return fail(&e),
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&m).unwrap_or_default());
        return 0;
    }
    let s = |k: &str| m.get(k).and_then(|v| v.as_str()).unwrap_or("?").to_string();
    let gpu = m.get("gpu").cloned().unwrap_or(serde_json::Value::Null);
    let g = |k: &str| gpu.get(k).and_then(|v| v.as_str()).unwrap_or("?").to_string();

    println!("{}", ui::head("machine"));
    println!("  {:<12} {} {}", "os", s("os"), s("arch"));
    println!("  {:<12} {}", "cpu", s("cpu"));
    match m.get("available_bytes").and_then(|v| v.as_u64()) {
        Some(b) => println!("  {:<12} {} free", "memory", ui::gib(b)),
        None => println!(
            "  {:<12} {}",
            "memory",
            ui::warn("NO PROBE — every host-memory gate stands down here")
        ),
    }
    let power = s("power");
    let power_line = if power == "battery" {
        ui::warn("battery — timings taken here are ratios, not absolutes")
    } else {
        power.clone()
    };
    println!("  {:<12} {}", "power", power_line);
    println!("  {:<12} {} · {} · {}", "gpu", g("name"), g("backend"), g("device_type"));
    0
}

pub fn doctor(model: Option<&str>, json: bool) -> i32 {
    let bin = match tools::find(Tool::Engine) {
        Ok(b) => b,
        Err(e) => return fail(&e),
    };
    let build = tools::build_of(&bin);
    match &build {
        Ok(b) => println!("{} {}", ui::dim("engine"), b.line()),
        Err(e) => return fail(e),
    }
    let mut args: Vec<&str> = vec!["doctor"];
    if json {
        args.push("--json");
    }
    if let Some(m) = model {
        args.push("--model");
        args.push(m);
    }
    // The doctor owns its process and its exit code IS the verdict, so
    // its output is passed through rather than reinterpreted.
    // The doctor prints its own checks and its own verdict line; echoing
    // a second one would be two sources of truth for the same question.
    // What is added here is only the exit code, passed through unchanged.
    let status = Command::new(&bin).args(&args).status();
    match status {
        Ok(s) if s.success() => 0,
        Ok(s) => s.code().unwrap_or(1),
        Err(e) => fail(&format!("could not run the doctor: {e}")),
    }
}

fn cache_root() -> Result<PathBuf, String> {
    Ok(state::home()?.join("models"))
}

/// Bytes on disk under a directory, one level of nesting deep — enough
/// for a model directory, and it never walks a whole home.
fn dir_bytes(dir: &PathBuf) -> u64 {
    let mut total = 0;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_file() {
                total += e.metadata().map(|m| m.len()).unwrap_or(0);
            } else if p.is_dir() {
                if let Ok(inner) = std::fs::read_dir(&p) {
                    for i in inner.flatten() {
                        total += i.metadata().map(|m| m.len()).unwrap_or(0);
                    }
                }
            }
        }
    }
    total
}

pub fn models() -> i32 {
    let root = match cache_root() {
        Ok(r) => r,
        Err(e) => return fail(&e),
    };
    let Ok(entries) = std::fs::read_dir(&root) else {
        println!("{}", ui::dim(&format!("no model cache at {}", root.display())));
        println!("  pull one:  combs pull qwen3-0.6b-gguf");
        return 0;
    };
    let mut rows: Vec<(String, u64)> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .map(|p| {
            let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            (name, dir_bytes(&p))
        })
        .collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1));
    if rows.is_empty() {
        println!("{}", ui::dim("the cache is empty — combs pull qwen3-0.6b-gguf"));
        return 0;
    }
    println!("{}", ui::head(&format!("models in {}", root.display())));
    let total: u64 = rows.iter().map(|r| r.1).sum();
    for (name, bytes) in &rows {
        println!("  {:<40} {:>10}", name, ui::gib(*bytes));
    }
    println!("  {:<40} {:>10}", ui::dim("total"), ui::dim(&ui::gib(total)));
    0
}

pub fn fit(model: &str, context: Option<u64>, kv_per_token: Option<u64>) -> i32 {
    let engine = match tools::find(Tool::Engine) {
        Ok(b) => b,
        Err(e) => return fail(&e),
    };
    let text = match capture(&engine, &["inspect", "--model", model, "--json"]) {
        Ok(t) => t,
        Err(e) => return fail(&e),
    };
    let v: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => return fail(&format!("inspect did not answer JSON: {e}")),
    };
    let w = v.get("weights").cloned().unwrap_or(serde_json::Value::Null);
    let packed = w.get("packed_bytes_total").and_then(|x| x.as_u64()).unwrap_or(0);
    let dense = w.get("dense_bytes_total").and_then(|x| x.as_u64()).unwrap_or(0);
    let available = match machine() {
        Ok(m) => m.get("available_bytes").and_then(|x| x.as_u64()),
        Err(e) => return fail(&e),
    };

    println!("{}", ui::head(&render::seat_header("fit", model)));
    println!(
        "  {:<22} {} ({} packed, {} widened dense at load)",
        "weights",
        ui::gib(packed + dense),
        ui::gib(packed),
        ui::mib(dense)
    );

    // The arena is the other half of the working set and a header cannot
    // tell you its size: KV bytes per token comes from the model's heads
    // and the storage width, which is a property of the BUILD that will
    // serve it. Rather than guess, this asks for the number or says
    // plainly that it is missing.
    let kv_bytes = match (context, kv_per_token) {
        (Some(ctx), Some(per)) => {
            let kv = ctx * per;
            println!(
                "  {:<22} {} ({} tokens × {} B/token)",
                "kv arena", ui::gib(kv), ctx, per
            );
            Some(kv)
        }
        _ => {
            println!(
                "  {:<22} {}",
                "kv arena",
                ui::dim("not priced — pass --context and --kv-bytes-per-token (combs-tk measure reports it)")
            );
            None
        }
    };

    let Some(available) = available else {
        println!(
            "  {:<22} {}",
            "verdict",
            ui::warn("host memory is unreadable on this platform — the fit gate stands down")
        );
        return 0;
    };
    println!("  {:<22} {} free · {} headroom", "machine", ui::gib(available), ui::gib(HEADROOM_BYTES));

    // The same arithmetic combs-cli's check_llm_fit applies before it
    // serves: weights + arena + headroom against what is free.
    let need = packed + dense + kv_bytes.unwrap_or(0);
    let total = need.saturating_add(HEADROOM_BYTES);
    if total <= available {
        println!(
            "  {:<22} {} — needs {}, {} free",
            "verdict",
            ui::good("fits"),
            ui::gib(total),
            ui::gib(available)
        );
        if kv_bytes.is_none() {
            println!("  {}", ui::dim("(weights only — the arena is not in this sum)"));
        }
        0
    } else {
        println!(
            "  {:<22} {} — needs {}, only {} free",
            "verdict",
            ui::bad("does not fit"),
            ui::gib(total),
            ui::gib(available)
        );
        println!("  {}", ui::dim("ways out: a smaller --context-size, a smaller quant, or close other applications"));
        1
    }
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn up(
    model: &str,
    role: &str,
    port: Option<u16>,
    context: Option<usize>,
    wait_secs: u64,
) -> i32 {
    let bin = match tools::find(Tool::Engine) {
        Ok(b) => b,
        Err(e) => return fail(&e),
    };
    let build = match tools::build_of(&bin) {
        Ok(b) => b,
        Err(e) => return fail(&e),
    };
    let dir = match state::root() {
        Ok(d) => d,
        Err(e) => return fail(&e),
    };
    // One engine per role: a second would take a second copy of the
    // weights, and on a laptop that is the difference between working
    // and swapping.
    if let Some(r) = state::load_all(&dir).into_iter().find(|r| r.role == role) {
        if state::liveness(state::pid_alive(r.pid), state::port_answers(r.port)) != Liveness::Stale {
            return fail(&format!(
                "a {role} engine is already up on port {} (pid {}) — stop it first: combs-tk down {role}",
                r.port, r.pid
            ));
        }
        state::forget(&dir, role);
    }
    let port = match port.map(Ok).unwrap_or_else(state::free_port) {
        Ok(p) => p,
        Err(e) => return fail(&format!("no port to listen on: {e}")),
    };
    let log_path = dir.join(format!("{role}.log"));
    if let Err(e) = std::fs::create_dir_all(&dir) {
        return fail(&format!("creating {}: {e}", dir.display()));
    }
    let log = match std::fs::File::create(&log_path) {
        Ok(f) => f,
        Err(e) => return fail(&format!("creating {}: {e}", log_path.display())),
    };
    let errlog = match log.try_clone() {
        Ok(f) => f,
        Err(e) => return fail(&format!("cloning the log handle: {e}")),
    };

    // The seat's name in the pane as well as in the record: a status
    // line carrying someone's home directory cannot be pasted anywhere,
    // and it blows out every column beside it.
    let name = state::identity(model);
    let mut status = Status::new();
    status.paint(&[
        format!("{} {}", ui::head("starting"), name),
        format!("  {:<10} {}", "engine", build.line()),
        format!("  {:<10} {}", "port", port),
        format!("  {:<10} {}", "state", ui::mark("starting")),
    ]);

    // A context size is passed only when one is asked for: an explicit
    // size is the operator's word, and the engine's own sizing is better
    // informed than a guess made out here.
    let mut args: Vec<String> = vec![
        "serve".into(),
        "--model".into(),
        model.into(),
        "--port".into(),
        port.to_string(),
    ];
    if let Some(tokens) = context {
        args.push("--context-size".into());
        args.push(tokens.to_string());
    }

    let child = Command::new(&bin)
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(errlog))
        .spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(e) => return fail(&format!("could not start the engine: {e}")),
    };
    let pid = child.id();

    let run = state::Run {
        role: role.to_string(),
        pid,
        port,
        // The seat's name, never the caller's path: a record that carries
        // a home directory cannot be shared, and `ps` has one column for it.
        model: state::identity(model),
        commit: build.commit.clone(),
        dtype: build.dtype.clone(),
        log: log_path.clone(),
        started: now_unix(),
    };
    if let Err(e) = state::save(&dir, &run) {
        return fail(&e);
    }

    // The engine binds only after the model is loaded, so "not answering
    // yet" is the normal state for as long as the weights take.
    let began = std::time::Instant::now();
    let outcome = state::wait_for_start(
        &mut child,
        port,
        std::time::Duration::from_secs(wait_secs),
        |elapsed| {
            status.progress(&[
                format!("{} {}", ui::head("starting"), name),
                format!("  {:<10} {}", "engine", build.line()),
                format!("  {:<10} http://127.0.0.1:{}", "address", port),
                format!(
                    "  {:<10} {} loading weights, {}",
                    "state",
                    ui::mark("starting"),
                    ui::ms(elapsed.as_millis() as f64)
                ),
            ]);
        },
    );

    match outcome {
        state::Start::Ready => {
            status.paint(&[
                format!("{} {}", ui::head("up"), name),
                format!("  {:<10} {}", "engine", build.line()),
                format!("  {:<10} http://127.0.0.1:{}", "address", port),
                format!("  {:<10} {}", "state", ui::mark("up")),
                format!("  {:<10} {} · pid {}", "took", ui::ms(began.elapsed().as_millis() as f64), pid),
            ]);
            status.settle();
            println!("{}", ui::dim(&format!("  logs: combs-tk logs {role} -f    ·    stop: combs-tk down {role}")));
            0
        }
        state::Start::Died(code) => {
            status.settle();
            let said = code.map(|c| format!(" (exit {c})")).unwrap_or_default();
            println!("{} the engine exited while loading{said} — its own words:", ui::bad("failed"));
            if let Ok(text) = std::fs::read_to_string(&log_path) {
                for line in text.lines().rev().take(8).collect::<Vec<_>>().into_iter().rev() {
                    println!("  {line}");
                }
            }
            state::forget(&dir, role);
            1
        }
        // Not killed, and the record is kept: a large seat may still be
        // loading. What is reported is only that this wait could not
        // confirm it, which a caller must never read as "up".
        state::Start::Waiting => {
            status.settle();
            println!(
                "{} it did not answer within {wait_secs}s and is still loading (pid {pid})",
                ui::warn("waiting")
            );
            println!("{}", ui::dim(&format!("  watch: combs-tk logs {role} -f    ·    check: combs-tk ps    ·    stop: combs-tk down {role}")));
            1
        }
    }
}

pub fn ps() -> i32 {
    let dir = match state::root() {
        Ok(d) => d,
        Err(e) => return fail(&e),
    };
    let runs = state::load_all(&dir);
    if runs.is_empty() {
        println!("{}", ui::dim("nothing running — combs-tk up <model>"));
        return 0;
    }
    println!(
        "{}",
        ui::head(&format!("  {:<8} {:<7} {:<26} {:<12} {}", "ROLE", "PORT", "MODEL", "BUILD", "STATE"))
    );
    for r in runs {
        let live = state::liveness(state::pid_alive(r.pid), state::port_answers(r.port));
        let word = match live {
            Liveness::Up => "up",
            Liveness::Starting => "starting",
            Liveness::Stale => "stale",
        };
        println!(
            "  {:<8} {:<7} {:<26} {:<12} {}",
            r.role,
            r.port,
            r.model,
            format!("{} {}", r.dtype, r.commit),
            ui::mark(word)
        );
    }
    0
}

pub fn down(role: Option<&str>, all: bool) -> i32 {
    let dir = match state::root() {
        Ok(d) => d,
        Err(e) => return fail(&e),
    };
    let runs = state::load_all(&dir);
    let targets: Vec<_> = if all {
        runs
    } else if let Some(r) = role {
        runs.into_iter().filter(|x| x.role == r).collect()
    } else {
        return fail("name a role or pass --all");
    };
    if targets.is_empty() {
        println!("{}", ui::dim("nothing to stop"));
        return 0;
    }
    for r in targets {
        // Only a pid this toolkit recorded, and only while it is alive.
        // Nothing here searches for a process by name.
        if !state::pid_alive(r.pid) {
            println!("  {:<8} {}", r.role, ui::dim("already gone — record cleared"));
            state::forget(&dir, &r.role);
            continue;
        }
        #[cfg(unix)]
        // SAFETY: SIGTERM to a pid this process recorded when it spawned
        // it. The engine drains its queue on TERM before exiting.
        unsafe {
            libc::kill(r.pid as libc::pid_t, libc::SIGTERM);
        }
        let mut gone = false;
        for _ in 0..40 {
            if !state::pid_alive(r.pid) {
                gone = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        if !gone {
            #[cfg(unix)]
            // SAFETY: same pid; it ignored TERM for four seconds.
            unsafe {
                libc::kill(r.pid as libc::pid_t, libc::SIGKILL);
            }
            println!("  {:<8} {}", r.role, ui::warn("ignored SIGTERM — killed"));
        } else {
            println!("  {:<8} {}", r.role, ui::mark("stopped"));
        }
        state::forget(&dir, &r.role);
    }
    0
}

pub fn logs(role: &str, follow: bool) -> i32 {
    let dir = match state::root() {
        Ok(d) => d,
        Err(e) => return fail(&e),
    };
    let path = dir.join(format!("{role}.log"));
    if !path.is_file() {
        return fail(&format!("no log for {role} — is it running? combs-tk ps"));
    }
    if !follow {
        match std::fs::read_to_string(&path) {
            Ok(t) => {
                print!("{t}");
                return 0;
            }
            Err(e) => return fail(&format!("reading {}: {e}", path.display())),
        }
    }
    // Follow: print what is there, then keep reading from where we
    // stopped. `tail -f` in twelve lines, and portable to both platforms.
    let mut pos = 0u64;
    loop {
        let Ok(mut f) = std::fs::File::open(&path) else {
            return fail("the log went away");
        };
        use std::io::{Read, Seek, SeekFrom};
        let len = f.metadata().map(|m| m.len()).unwrap_or(0);
        if len < pos {
            pos = 0; // truncated: start over rather than print garbage
        }
        if len > pos {
            let _ = f.seek(SeekFrom::Start(pos));
            let mut buf = String::new();
            if f.read_to_string(&mut buf).is_ok() {
                print!("{buf}");
                pos = len;
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(300));
    }
}

pub fn measure(model: &str, depths: &str, repeat: usize, arena: Option<usize>) -> i32 {
    let kit = match tools::find(Tool::Kit) {
        Ok(b) => b,
        Err(e) => return fail(&e),
    };
    // A measurement is only as good as the machine's state, so the two
    // readings that have poisoned comparisons here are said first.
    if let Ok(m) = machine() {
        if m.get("power").and_then(|v| v.as_str()) == Some("battery") {
            println!(
                "{}",
                ui::warn("on battery — ratios from this run are usable, absolutes are not")
            );
        }
    }
    let out = std::env::temp_dir().join(format!("combs-tk-measure-{}.json", std::process::id()));
    let repeat_s = repeat.to_string();
    let mut args: Vec<String> = vec![
        "model".into(),
        model.into(),
        "--depths".into(),
        depths.into(),
        "--repeat".into(),
        repeat_s,
        "--out".into(),
        out.to_string_lossy().into_owned(),
    ];
    if let Some(a) = arena {
        args.push("--arena".into());
        args.push(a.to_string());
    }

    let child = Command::new(&kit)
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(e) => return fail(&format!("could not start the kit: {e}")),
    };

    let name = state::identity(model);
    let mut status = Status::new();
    let began = std::time::Instant::now();
    status.paint(&[format!("{} {}", ui::head("measuring"), name)]);
    if let Some(stdout) = child.stdout.take() {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            status.log("kit", &line);
            // Motion, not content: in a terminal this keeps a running
            // clock above the scrolling log; piped to a file it prints
            // nothing, where a redraw per line would bury the very log
            // it was meant to frame.
            status.progress(&[format!(
                "{} {} · {}",
                ui::head("measuring"),
                name,
                ui::ms(began.elapsed().as_millis() as f64)
            )]);
        }
    }
    let code = child.wait().map(|s| s.code().unwrap_or(1)).unwrap_or(1);
    status.settle();
    if code != 0 {
        if let Some(mut e) = child.stderr.take() {
            use std::io::Read;
            let mut buf = String::new();
            let _ = e.read_to_string(&mut buf);
            for l in buf.lines().rev().take(5).collect::<Vec<_>>().into_iter().rev() {
                eprintln!("  {l}");
            }
        }
        return fail("the measurement did not finish");
    }
    match std::fs::read_to_string(&out).ok().and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok()) {
        Some(rec) => {
            print_record(&rec);
            let _ = std::fs::remove_file(&out);
            0
        }
        None => fail("the kit wrote no record to read"),
    }
}

/// Render the kit's own record. Every figure carries how it was obtained
/// and how far it moved across readings — a number without its spread is
/// a story, and this estate has been told several.
fn print_record(rec: &serde_json::Value) {
    let empty = Vec::new();
    let runs = rec.get("runs").and_then(|r| r.as_array()).unwrap_or(&empty);
    if runs.is_empty() {
        println!("{}", ui::dim("the record holds no runs"));
        return;
    }
    let build = rec.get("build").cloned().unwrap_or(serde_json::Value::Null);
    println!(
        "\n{} {} · {} · {}",
        ui::head("measured with"),
        build.get("commit").and_then(|v| v.as_str()).unwrap_or("?"),
        build.get("dtype").and_then(|v| v.as_str()).unwrap_or("?"),
        if build.get("dirty").and_then(|v| v.as_bool()).unwrap_or(false) {
            ui::warn("dirty tree — not reproducible from the commit")
        } else {
            ui::dim("clean")
        }
    );
    println!(
        "  {:<7} {:>9} {:>12} {:>10} {:>12} {:>9}",
        ui::dim("DEPTH"), ui::dim("PROMPT"), ui::dim("PREFILL"), ui::dim("TTFT"), ui::dim("DECODE"), ui::dim("SPREAD")
    );
    for r in runs {
        let get = |name: &str| -> Option<&serde_json::Value> {
            r.get("values")?
                .as_array()?
                .iter()
                .find(|v| v.get("name").and_then(|n| n.as_str()) == Some(name))
        };
        let num = |name: &str| get(name).and_then(|v| v.get("value")).and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
        let decode_spread = get("decode_tok_s")
            .and_then(|v| v.get("spread_pct"))
            .and_then(|v| v.as_f64());
        // The kit's own band: past it a figure is a range, not a number.
        let spread = match decode_spread {
            Some(s) if s > 10.0 => ui::warn(&format!("±{s:.0}% RANGE")),
            Some(s) => ui::dim(&format!("±{s:.1}%")),
            None => ui::warn("unmeasured"),
        };
        println!(
            "  {:<7} {:>9.0} {:>9.1} t/s {:>10} {:>9.2} t/s {:>9}",
            r.get("id").and_then(|v| v.as_str()).unwrap_or("?"),
            num("prompt_tokens"),
            num("prefill_tok_s"),
            ui::ms(num("ttft_ms")),
            num("decode_tok_s"),
            spread
        );
    }
    if let Some(kv) = runs
        .first()
        .and_then(|r| r.get("values"))
        .and_then(|v| v.as_array())
        .and_then(|vs| vs.iter().find(|v| v.get("name").and_then(|n| n.as_str()) == Some("kv_bytes_per_token")))
        .and_then(|v| v.get("value"))
        .and_then(|v| v.as_f64())
    {
        println!(
            "\n  {} {:.0} B/token — pass it to `combs-tk fit --kv-bytes-per-token` to price an arena",
            ui::dim("kv"),
            kv
        );
    }
}
