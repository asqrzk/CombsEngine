//! # combs-kit — measure a box, a model and a build, and write a record
//! you can compare.
//!
//! ## What this is not
//!
//! It is not the operational toolkit. It never starts a server, never
//! kills a process, never takes a port, never gates a commit and never
//! installs anything. Those belong to the hive console, the gate scripts
//! and the deploy toolkit, and they keep every inch of that ground. The
//! kit measures, compares, explains, and writes records.
//!
//! ## Why it exists
//!
//! Every number this estate stands behind lives either in a private
//! markdown ledger or inside a display string in a catalog. So an
//! outside audit picked up a figure from before the kernels existed,
//! judged the engine two times worse than it is, and nothing in the tree
//! could contradict it. A measurement that cannot be reproduced by a
//! stranger on their own machine is a story.
//!
//! ## The rules it will not break
//!
//! - Never report the first run. A reported timing is the warm second of
//!   a pair, and the record says so.
//! - Never report a number from a box that failed the doctor without
//!   stamping it. A software rasterizer serves plausible garbage at a
//!   crawl and calls itself healthy.
//! - Record the power source: a laptop on battery is a different machine.
//! - One model per process. The pool does not return memory until a
//!   process exits, and a suite in one process has taken this machine
//!   down before.
//! - Never download anything, and never install a foreign runtime. The
//!   kit names the command; the person runs it.

mod child;
mod diff;
mod doors;
mod model;
mod ops;
mod report;
mod machine;
mod micro;
mod record;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "combs-kit",
    version,
    about = "Measure a box, a model and a build — and write a record you can compare"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// What this box is: cpu, memory, adapter, power, and which probes
    /// this platform actually has.
    System {
        /// Write the record as JSON instead of a human table.
        #[arg(long)]
        json: bool,
        /// Write the record to a file.
        #[arg(long)]
        out: Option<std::path::PathBuf>,
    },
    /// Every environment door the engine reads, what it does, and
    /// whether its effect has ever been measured.
    Doors {
        #[arg(long)]
        json: bool,
    },
    /// What this engine can do, where, and WHY NOT — each row with its
    /// reason and a pointer to the evidence.
    Ops {
        /// Only rows whose platform or backend contains this word.
        #[arg(long)]
        r#where: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// What THIS model costs on THIS box: load, prefill, time to first
    /// token, decode, dispatches per token and pool, at each depth.
    /// Every depth is its own process and every figure is the warm
    /// second run of a pair.
    Model {
        /// The model directory or file.
        path: String,
        /// Prompt depths to measure, in tokens.
        #[arg(long, default_value = "4,2000", value_delimiter = ',')]
        depths: Vec<usize>,
        /// Tokens to generate per depth.
        #[arg(long, default_value_t = 16)]
        tokens: usize,
        #[arg(long)]
        json: bool,
        #[arg(long)]
        out: Option<std::path::PathBuf>,
    },
    /// The box's fingerprint without a model: launch latency, the bus,
    /// the gemv against its packed bytes, and the submit sweep. Each
    /// cell its own process.
    Micro {
        /// Cells to run; default all four.
        #[arg(long, value_delimiter = ',')]
        cells: Option<Vec<String>>,
        /// The seat the submit sweep decodes (packed; the 0.6b by default
        /// when present).
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        json: bool,
        #[arg(long)]
        out: Option<std::path::PathBuf>,
    },
    /// One micro cell. Spawned by `micro`; not for hand use.
    #[command(hide = true)]
    RunMicro {
        #[arg(long)]
        cell: String,
    },
    /// Measure ONE subject. Spawned by `model`; not for hand use — the
    /// one-subject-per-process rule is the reason this exists.
    #[command(hide = true)]
    RunOne {
        #[arg(long)]
        subject: String,
    },
    /// Render a record as the ledger's own section shape, so a
    /// measurement and its write-up cannot disagree.
    Report {
        record: std::path::PathBuf,
        /// The section number this will be filed as.
        #[arg(long, default_value = "N")]
        section: String,
        /// The one-sentence claim the section stands on.
        #[arg(long, default_value = "what this run measured")]
        claim: String,
        #[arg(long, default_value = "today")]
        date: String,
        #[arg(long, default_value = "")]
        branch: String,
    },
    /// Did it get worse? Compares two records figure by figure, knowing
    /// which direction is better for each. Exits non-zero when anything
    /// regressed past the band.
    Diff {
        before: std::path::PathBuf,
        after: std::path::PathBuf,
        /// How far a figure may move before it counts, in percent. Wide
        /// by default: this estate has produced a 2.4x outlier from a
        /// single reading, and a tool that cries regression at noise is
        /// one people learn to ignore.
        #[arg(long, default_value_t = 5.0)]
        tolerance: f64,
    },
    /// Answer one operation. Exit 0 when it works, 1 when it does not,
    /// 2 when the name is unknown — a question we did not understand is
    /// never answered "no".
    Can {
        /// The operation id, as `combs-kit ops` lists them.
        op: String,
        #[arg(long)]
        r#where: Option<String>,
    },
}

fn main() {
    // A tool people will pipe into `head` must not die of it. Rust masks
    // SIGPIPE at startup, so a closed reader turns a println into a panic
    // and the exit code says "crash" where the truth is "you stopped
    // reading". Restore the default and let the process end quietly.
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }

    let cli = Cli::parse();
    let code = match cli.command {
        Command::System { json, out } => cmd_system(json, out.as_deref()),
        Command::Doors { json } => cmd_doors(json),
        Command::Ops { r#where, json } => {
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&ops::as_json()).unwrap_or_default()
                );
            } else {
                ops::print_all(r#where.as_deref());
            }
            0
        }
        Command::Can { op, r#where } => ops::answer(&op, r#where.as_deref()),
        Command::Model { path, depths, tokens, json, out } => {
            cmd_model(&path, &depths, tokens, json, out.as_deref())
        }
        Command::RunOne { subject } => cmd_run_one(&subject),
        Command::Micro { cells, model, json, out } => cmd_micro(cells, model, json, out.as_deref()),
        Command::RunMicro { cell } => {
            let run = micro::run_cell(&cell);
            match serde_json::to_string(&run) {
                Ok(t) => {
                    println!("{t}");
                    0
                }
                Err(e) => {
                    eprintln!("combs-kit: {e}");
                    1
                }
            }
        }
        Command::Report { record, section, claim, date, branch } => {
            match load(&record) {
                Ok(rec) => {
                    println!("{}", report::markdown(&rec, &section, &claim, &date, &branch));
                    0
                }
                Err(e) => {
                    eprintln!("combs-kit: {e}");
                    1
                }
            }
        }
        Command::Diff { before, after, tolerance } => match (load(&before), load(&after)) {
            (Ok(a), Ok(b)) => {
                let v = diff::compare(&a, &b, tolerance);
                if let Some(why) = v.refused {
                    eprintln!("combs-kit: {why}");
                    std::process::exit(2);
                }
                for line in &v.lines {
                    println!("{line}");
                }
                if v.regressions > 0 {
                    println!("\n{} figure(s) regressed past {}%.", v.regressions, tolerance);
                }
                i32::from(v.regressions > 0)
            }
            (Err(e), _) | (_, Err(e)) => {
                eprintln!("combs-kit: {e}");
                1
            }
        },
    };
    std::process::exit(code);
}

fn cmd_system(json: bool, out: Option<&std::path::Path>) -> i32 {
    let m = machine::read();
    if !json && out.is_none() {
        machine::print_human(&m);
        return 0;
    }
    let machine_json = match serde_json::to_value(&m) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("combs-kit: could not render the machine: {e}");
            return 1;
        }
    };
    // Trust is stamped, never assumed. Until the kit runs the doctor
    // itself, it says plainly that it did not.
    let trust = record::Trust {
        doctor: "not run".to_string(),
        detail: "combs-kit system reads the machine; it does not judge it. Run `combs doctor`."
            .to_string(),
    };
    let rec = record::Record::new(machine_json, build_stamp(), trust);
    match rec.write(out) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("combs-kit: {e}");
            1
        }
    }
}

fn load(path: &std::path::Path) -> Result<record::Record, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let rec: record::Record =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    if rec.schema != record::SCHEMA {
        return Err(format!(
            "{} is schema {}, and this kit reads {}",
            path.display(),
            rec.schema,
            record::SCHEMA
        ));
    }
    Ok(rec)
}

fn cmd_model(
    path: &str,
    depths: &[usize],
    tokens: usize,
    json: bool,
    out: Option<&std::path::Path>,
) -> i32 {
    if child::is_child() {
        eprintln!("combs-kit: a child may not run a sweep of its own");
        return 2;
    }
    let m = machine::read();
    if !json && out.is_none() {
        machine::print_human(&m);
        // The build, on screen, every time. Half of every number here is
        // which binary produced it, and this estate has already measured
        // the wrong backend once because a feature flag did not forward
        // and nothing on screen said so.
        let b = build_stamp();
        println!(
            "  build       {} · fusion {} · {}",
            b["dtype"].as_str().unwrap_or("?"),
            b["fusion"].as_bool().unwrap_or(false),
            b["profile"].as_str().unwrap_or("?"),
        );
        println!("\nmodel {path}");
    }
    let mut runs = Vec::new();
    for depth in depths {
        let subject = child::Subject {
            id: format!("d{depth}"),
            model: path.to_string(),
            prompt_tokens: *depth,
            max_tokens: tokens,
        };
        let run = child::measure(&subject);
        if !json && out.is_none() {
            model::print_run(&run);
        }
        runs.push(run);
    }
    let measured = runs.iter().filter(|r| !r.values.is_empty()).count();
    if !json && out.is_none() {
        if measured < runs.len() {
            println!(
                "\n{} of {} depths produced no measurement — see the notes above.",
                runs.len() - measured,
                runs.len()
            );
        }
        return i32::from(measured == 0);
    }
    let machine_json = serde_json::to_value(&m).unwrap_or(serde_json::Value::Null);
    let trust = record::Trust {
        doctor: "not run".to_string(),
        detail: "these are timings, not a verdict on the machine. Run `combs doctor`.".to_string(),
    };
    let mut rec = record::Record::new(machine_json, build_stamp(), trust);
    rec.runs = runs;
    match rec.write(out) {
        Ok(()) => i32::from(measured == 0),
        Err(e) => {
            eprintln!("combs-kit: {e}");
            1
        }
    }
}

fn cmd_micro(
    cells: Option<Vec<String>>,
    model: Option<String>,
    json: bool,
    out: Option<&std::path::Path>,
) -> i32 {
    if child::is_child() {
        eprintln!("combs-kit: a child may not run a sweep of its own");
        return 2;
    }
    let m = machine::read();
    let human = !json && out.is_none();
    if human {
        machine::print_human(&m);
        println!("\nmicro");
    }
    let wanted: Vec<String> = cells.unwrap_or_else(|| micro::CELLS.iter().map(|c| c.to_string()).collect());
    let mut runs = Vec::new();
    for cell in &wanted {
        if cell == "submits" {
            let seat = model.clone().or_else(|| {
                let home = std::env::var("HOME").ok()?;
                let p = std::path::PathBuf::from(home).join(".cache/combs/models/qwen3-0.6b-gguf");
                p.is_dir().then(|| p.to_string_lossy().into_owned())
            });
            let Some(seat) = seat else {
                runs.push(record::Run {
                    id: "submits".into(),
                    kind: "micro".into(),
                    subject: "this box".into(),
                    config: serde_json::Value::Null,
                    warm: false,
                    values: vec![],
                    notes: vec!["no measurement: no packed seat to decode; pass --model".into()],
                });
                continue;
            };
            for tasks in [8usize, 16, 32, 64, 128] {
                let subject = child::Subject {
                    id: format!("tasks{tasks}"),
                    model: seat.clone(),
                    prompt_tokens: 4,
                    max_tokens: 16,
                };
                let run = child::measure_with_env(&subject, &[("CUBECL_WGPU_MAX_TASKS".into(), tasks.to_string())]);
                if human {
                    model::print_run(&run);
                }
                runs.push(run);
            }
            continue;
        }
        let run = child::measure_cell(cell, &[]);
        if human {
            micro::print_run(&run);
        }
        runs.push(run);
    }
    if human {
        return 0;
    }
    let trust = record::Trust {
        doctor: "not run".into(),
        detail: "timings of the box itself; run `combs doctor` for its verdict".into(),
    };
    let mut rec = record::Record::new(
        serde_json::to_value(&m).unwrap_or(serde_json::Value::Null),
        build_stamp(),
        trust,
    );
    rec.runs = runs;
    match rec.write(out) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("combs-kit: {e}");
            1
        }
    }
}

fn cmd_run_one(subject: &str) -> i32 {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(subject) else {
        eprintln!("combs-kit: the subject did not parse");
        return 2;
    };
    let Some(subject) = child::Subject::from_json(&v) else {
        eprintln!("combs-kit: the subject is missing a field");
        return 2;
    };
    let run = model::run_one(&subject);
    match serde_json::to_string(&run) {
        Ok(text) => {
            println!("{text}");
            0
        }
        Err(e) => {
            eprintln!("combs-kit: could not render the run: {e}");
            1
        }
    }
}

fn cmd_doors(json: bool) -> i32 {
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&doors::as_json()).unwrap_or_default()
        );
    } else {
        doors::print_table();
    }
    0
}

/// What built this kit. A record without it cannot be compared to
/// anything, because dtype and features are half of every number here.
fn build_stamp() -> serde_json::Value {
    serde_json::json!({
        "kit_version": env!("CARGO_PKG_VERSION"),
        "dtype": if cfg!(feature = "f16") { "f16" } else { "f32" },
        "fusion": !cfg!(feature = "f16") || cfg!(feature = "f16-fusion"),
        "cpu_backend": cfg!(feature = "cpu"),
        "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
        "target": format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The build stamp is not decoration: an f16 record and an f32
    /// record of the same seat are different measurements, and §92 is
    /// the proof (the same command measured the wrong backend once
    /// because a feature did not forward).
    #[test]
    fn the_build_stamp_names_the_dtype_and_the_fusion() {
        let s = build_stamp();
        assert!(s.get("dtype").is_some());
        assert!(s.get("fusion").is_some());
        let dtype = s["dtype"].as_str().unwrap();
        assert!(dtype == "f16" || dtype == "f32");
    }

    #[test]
    fn the_cli_parses_its_own_surface() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }
}
