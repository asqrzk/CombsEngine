//! # combs-tk — know the box, run the engine, watch it work.
//!
//! ## What this is
//!
//! The operational half. `combs` serves models and `combs-kit` measures
//! them; this drives both and shows you what is happening while it does.
//! It is the thing you run on a laptop in the morning and on a rented
//! Linux box in the afternoon, and it behaves the same on each.
//!
//! ## What it will not do
//!
//! It does not compute. It loads no model, takes no GPU, and links
//! neither the engine nor the kit — it finds their binaries and drives
//! them. That is why it builds in seconds and starts instantly, and it
//! is also forced by the tools: a doctor run must own its process
//! because a GPU teardown is not re-entrant, and the kit measures one
//! subject per process because the pool does not return memory until a
//! process exits.
//!
//! It also never kills a process it did not start. Runs are remembered
//! in a file with their pid and port, and a process counts as ours only
//! when the record, the pid and the port all agree. Sweeping by command
//! line has executed the sweeper in this estate before.
//!
//! ## What it says
//!
//! Every status names the build behind it — version, dtype, commit —
//! because "the engine is up" is not a fact until you know which engine.

mod api;
mod cmds;
mod net;
mod render;
mod state;
mod tools;
mod ui;
mod wire;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "combs-tk",
    version,
    about = "Know the box, run the engine, watch it work"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// What this machine is: cpu, memory, gpu, power, and which probes
    /// it actually has.
    System {
        #[arg(long)]
        json: bool,
    },
    /// Whether this machine can be trusted with real work. Exits non-zero
    /// when any check refuses, so it works as a gate in a script.
    Doctor {
        /// Prove it end to end on a model as well as the canaries.
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// The models in the local cache, with their sizes.
    Models,
    /// Will this model fit on this machine? Prices the weights against
    /// what is free, and names the part it cannot know without loading.
    Fit {
        /// A preset id or a path.
        model: String,
        /// KV arena size in tokens, when you want the arena priced too.
        #[arg(long)]
        context: Option<u64>,
        /// Bytes of KV per token, from a previous measurement — the one
        /// number a header cannot tell you.
        #[arg(long)]
        kv_bytes_per_token: Option<u64>,
    },
    /// Start an engine and watch it come up.
    Up {
        /// A preset id or a path.
        model: String,
        /// What it serves. One engine per role.
        #[arg(long, default_value = "text")]
        role: String,
        /// Port to listen on. Default: one the OS lends us.
        #[arg(long)]
        port: Option<u16>,
        /// KV arena size in tokens. Default: the engine's own sizing.
        #[arg(long)]
        context: Option<usize>,
        /// Seconds to wait for it to answer. Past this the wait gives up
        /// and says so; a slow engine is never killed for being slow.
        #[arg(long, default_value_t = 180)]
        wait: u64,
    },
    /// What is running, and whether it answers.
    Ps,
    /// Stop an engine this toolkit started.
    Down {
        /// Which role to stop. Omit with --all.
        role: Option<String>,
        #[arg(long)]
        all: bool,
    },
    /// Follow an engine's output.
    Logs {
        #[arg(default_value = "text")]
        role: String,
        /// Keep printing as it is written.
        #[arg(long, short)]
        follow: bool,
    },
    /// How fast is this model on this machine? Drives the measurement
    /// kit, streams its work, and prints the table it recorded.
    Measure {
        /// A preset id or a path.
        model: String,
        /// Prompt depths, in tokens.
        #[arg(long, default_value = "4,2000")]
        depths: String,
        /// Readings per depth. One reading cannot show its own spread.
        #[arg(long, default_value = "3")]
        repeat: usize,
        /// The KV arena in tokens.
        #[arg(long)]
        arena: Option<usize>,
    },
}

fn main() {
    // A tool people pipe into `head` must not die of it: Rust masks
    // SIGPIPE at startup, so a closed reader turns a println into a panic
    // and the exit code says "crash" where the truth is "you stopped
    // reading". The kit takes this for the same reason.
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }

    let cli = Cli::parse();
    let code = match cli.command {
        Command::System { json } => cmds::system(json),
        Command::Doctor { model, json } => cmds::doctor(model.as_deref(), json),
        Command::Models => cmds::models(),
        Command::Fit { model, context, kv_bytes_per_token } => {
            cmds::fit(&model, context, kv_bytes_per_token)
        }
        Command::Up { model, role, port, context, wait } => {
            cmds::up(&model, &role, port, context, wait)
        }
        Command::Ps => cmds::ps(),
        Command::Down { role, all } => cmds::down(role.as_deref(), all),
        Command::Logs { role, follow } => cmds::logs(&role, follow),
        Command::Measure { model, depths, repeat, arena } => {
            cmds::measure(&model, &depths, repeat, arena)
        }
    };
    std::process::exit(code);
}
