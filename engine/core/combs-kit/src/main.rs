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

mod doors;
mod machine;
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
