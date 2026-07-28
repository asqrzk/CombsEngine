//! `combs chew` — interactive UI scaffolder (Firebase-CLI style).
//!
//! `combs chew chat-ui` / `combs chew debate-ui` walks the user through a
//! feature selection (SPACE to toggle, arrows to move, ENTER to confirm)
//! and scaffolds a configured Svelte 5 app from `Engine/Ui/template` into
//! the target directory. Every prompt has a flag equivalent
//! (`--reasoning=true`, `--authentication=false`, ...) so the command is
//! scriptable end-to-end.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Args;
use console::style;
use dialoguer::{Confirm, Input, MultiSelect, Select, theme::ColorfulTheme};

/// Features the scaffolded UI can include.
const FEATURES: &[&str] = &["reasoning", "vision", "audio", "save-chats"];
const THEMES: &[&str] = &["system", "dark", "light"];

#[derive(Args, Clone, Default)]
pub struct ChewArgs {
    /// Target directory for the scaffolded app (default: ./<mode>).
    #[arg(long)]
    pub dir: Option<PathBuf>,
    /// Include the reasoning panel (multi-step answers).
    #[arg(long)]
    pub reasoning: Option<bool>,
    /// Include vision input (image attachments).
    #[arg(long)]
    pub vision: Option<bool>,
    /// Include audio input (microphone capture).
    #[arg(long)]
    pub audio: Option<bool>,
    /// Persist chat sessions locally.
    #[arg(long)]
    pub save_chats: Option<bool>,
    /// UI theme: system | dark | light.
    #[arg(long)]
    pub theme: Option<String>,
    /// Model preset id to use. REQUIRED — every app runs against an
    /// explicitly chosen model (with `--yes` this flag is mandatory).
    #[arg(long)]
    pub model: Option<String>,
    /// `combs serve` URL the UI talks to.
    #[arg(long, default_value = "http://localhost:8080")]
    pub server: String,
    /// Debate: agent names, comma-separated (default: "Pro,Con").
    #[arg(long)]
    pub agents: Option<String>,
    /// Debate topic.
    #[arg(long)]
    pub topic: Option<String>,
    /// Debate turns (total speaker turns).
    #[arg(long)]
    pub turns: Option<u32>,
    /// Skip all prompts and use defaults for anything not flagged.
    #[arg(long)]
    pub yes: bool,
    /// Overwrite the target directory if it already exists.
    #[arg(long)]
    pub force: bool,
    /// Do not run `npm install` after scaffolding.
    #[arg(long)]
    pub no_install: bool,
    /// Do not start the dev server (`npm run dev`) after scaffolding.
    #[arg(long)]
    pub no_start: bool,
}

/// Resolved scaffold configuration (written to combs.ui.json).
#[derive(Debug, serde::Serialize, Clone)]
struct UiConfig {
    mode: String,
    features: Features,
    authentication: bool,
    theme: String,
    model: String,
    server: String,
    debate: Option<DebateConfig>,
}

#[derive(Debug, serde::Serialize, Clone)]
struct Features {
    reasoning: bool,
    vision: bool,
    audio: bool,
    save_chats: bool,
}

#[derive(Debug, serde::Serialize, Clone)]
struct DebateConfig {
    agents: Vec<DebateAgent>,
    topic: String,
    turns: u32,
}

#[derive(Debug, serde::Serialize, Clone)]
struct DebateAgent {
    name: String,
    stance: String,
    behavior: String,
}

/// Entry point for `combs chew chat-ui|debate-ui`.
pub fn chew(mode: &str, args: ChewArgs) -> Result<()> {
    let theme = ColorfulTheme::default();
    println!(
        "{}",
        style(format!("combs chew — scaffold a {mode} app")).bold().cyan()
    );

    // --- feature selection -------------------------------------------------
    let all_flagged = args.reasoning.is_some()
        && args.vision.is_some()
        && args.audio.is_some()
        && args.save_chats.is_some();
    let features = if all_flagged || args.yes {
        Features {
            reasoning: args.reasoning.unwrap_or(false),
            vision: args.vision.unwrap_or(false),
            audio: args.audio.unwrap_or(false),
            save_chats: args.save_chats.unwrap_or(true),
        }
    } else {
        let flags = [
            args.reasoning.unwrap_or(false),
            args.vision.unwrap_or(false),
            args.audio.unwrap_or(false),
            args.save_chats.unwrap_or(true),
        ];
        let chosen = MultiSelect::with_theme(&theme)
            .with_prompt("Select features (SPACE to toggle, ENTER to continue)")
            .items(FEATURES)
            .defaults(&flags)
            .interact()?;
        Features {
            reasoning: chosen.contains(&0),
            vision: chosen.contains(&1),
            audio: chosen.contains(&2),
            save_chats: chosen.contains(&3),
        }
    };

    // --- auth ---------------------------------------------------------------
    // Always on. First-run keypair auth is part of every scaffolded app;
    // no configuration escapes it.
    let authentication = true;

    // --- theme ---------------------------------------------------------------
    let theme_name = match args.theme.clone() {
        Some(t) => t,
        None if args.yes => "system".into(),
        None => THEMES[Select::with_theme(&theme)
            .with_prompt("Theme")
            .items(THEMES)
            .default(0)
            .interact()?]
            .to_string(),
    };

    // --- model (mandatory) ---------------------------------------------------
    let model = match args.model.clone() {
        Some(m) if !m.trim().is_empty() => m,
        Some(_) => bail!("--model must not be empty"),
        None if args.yes => bail!(
            "--model is required with --yes — every app must explicitly choose a model \
             (e.g. --model smollm2-135m)"
        ),
        None => Input::with_theme(&theme)
            .with_prompt("Model preset (e.g. smollm2-135m)")
            .validate_with(|v: &String| {
                if v.trim().is_empty() {
                    Err("a model is required")
                } else {
                    Ok(())
                }
            })
            .interact_text()?,
    };

    // --- debate specifics ----------------------------------------------------
    let debate = if mode == "debate-ui" {
        Some(prompt_debate(&args, &theme)?)
    } else {
        None
    };

    let config = UiConfig {
        mode: mode.to_string(),
        features,
        authentication,
        theme: theme_name,
        model,
        server: args.server.clone(),
        debate,
    };

    // --- scaffold -------------------------------------------------------------
    let dir = args
        .dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("./{mode}")));
    scaffold(&dir, &config, args.force || args.yes)?;

    println!();
    println!("{}", style("scaffold complete!").bold().green());
    println!("  dir:   {}", dir.display());

    // --- install + launch ---------------------------------------------------
    if args.no_install {
        print_manual_steps(&dir, &args.server);
        return Ok(());
    }
    match npm_install(&dir) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("{} {e:#}", style("npm install failed:").red());
            print_manual_steps(&dir, &args.server);
            return Ok(());
        }
    }
    if args.no_start {
        println!();
        println!("start it later with:  cd {} && npm run dev", dir.display());
        println!("(UI talks to {})", args.server);
        return Ok(());
    }
    // --- model server (combs serve) -----------------------------------------
    let mut serve_child = maybe_start_serve(&args, &dir, &config.model);

    println!();
    println!(
        "{}",
        style("starting dev server (Ctrl-C to stop)...").bold().cyan()
    );
    println!("  ui server will talk to combs at: {}", args.server);
    let result = npm_run_dev(&dir);
    if let Some(mut child) = serve_child.take() {
        let _ = child.kill();
        let _ = child.wait();
    }
    result
}

/// Resolves the `--model` value to a local model path: either a direct
/// path, or a preset id under the combs model cache.
fn resolve_model_path(model: &str) -> Option<PathBuf> {
    let direct = PathBuf::from(model);
    if direct.exists() {
        return Some(direct);
    }
    crate::pull::cached_dir(model)
}

fn server_port(server: &str) -> u16 {
    server
        .rsplit(':')
        .next()
        .and_then(|p| p.trim_end_matches('/').parse().ok())
        .unwrap_or(8080)
}

/// Starts `combs serve` for the UI's upstream unless one is already
/// answering at the --server URL. Returns the child so it can be stopped
/// when the dev server exits.
fn maybe_start_serve(args: &ChewArgs, dir: &Path, model: &str) -> Option<std::process::Child> {
    let port = server_port(&args.server);
    if std::net::TcpStream::connect(format!("127.0.0.1:{port}")).is_ok() {
        println!(
            "{}",
            style(format!("combs server already running at {} — reusing it", args.server)).dim()
        );
        return None;
    }
    let mut resolved = resolve_model_path(model);
    if resolved.is_none() {
        // Not cached — download it (auto with --yes; ask otherwise).
        let get = args.yes
            || dialoguer::Confirm::new()
                .with_prompt(format!("model '{model}' is not cached — download it now?"))
                .default(true)
                .interact()
                .unwrap_or(false);
        if get {
            match crate::pull::pull(model) {
                Ok(dir) => resolved = Some(dir),
                Err(e) => eprintln!("{} {e:#}", style("model download failed:").red()),
            }
        }
    }
    let Some(model_path) = resolved else {
        println!(
            "{}",
            style(format!(
                "model '{model}' not available — start the server yourself later:\n  \
                 combs pull {model} && combs serve --model {model} --port {port}"
            ))
            .yellow()
        );
        return None;
    };
    let exe = std::env::current_exe().ok()?;
    let log = std::fs::File::create(dir.join("combs-serve.log")).ok()?;
    let log_err = log.try_clone().ok()?;
    println!(
        "{}",
        style(format!("starting combs serve ({}) on :{port} — log: combs-serve.log", model_path.display()))
            .bold()
    );
    let child = std::process::Command::new(exe)
        .args(["serve", "--model"])
        .arg(&model_path)
        .args(["--port", &port.to_string()])
        .stdout(std::process::Stdio::from(log))
        .stderr(std::process::Stdio::from(log_err))
        .spawn()
        .ok()?;
    // Wait for the engine to bind the port (model load can take a while).
    for _ in 0..240 {
        if std::net::TcpStream::connect(format!("127.0.0.1:{port}")).is_ok() {
            println!("{}", style("combs serve is up").green());
            return Some(child);
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    eprintln!("{}", style("combs serve did not come up in 120s — see combs-serve.log").red());
    Some(child)
}

fn print_manual_steps(dir: &Path, server: &str) {
    println!();
    println!("next steps:");
    println!("  cd {}", dir.display());
    println!("  npm install");
    println!("  npm run dev");
    println!();
    println!("then point your browser at the printed URL (UI talks to {server}).");
}

/// `npm` is `npm.cmd` on Windows.
fn npm_program() -> &'static str {
    if cfg!(windows) {
        "npm.cmd"
    } else {
        "npm"
    }
}

fn npm_install(dir: &Path) -> Result<()> {
    let npm = npm_program();
    println!();
    println!("{}", style("installing dependencies (npm install)...").bold());
    let status = std::process::Command::new(npm)
        .arg("install")
        .current_dir(dir)
        .status()
        .context("npm not found on PATH — install Node.js (https://nodejs.org)")?;
    if !status.success() {
        bail!("npm install exited with {status}");
    }
    Ok(())
}

fn npm_run_dev(dir: &Path) -> Result<()> {
    let mut cmd = std::process::Command::new(npm_program());
    cmd.args(["run", "dev", "--", "--open"]).current_dir(dir);
    // Hand the proxy the absolute path to THIS combs binary so it can spawn
    // second engines (`/api/engine/spawn`) even when `combs` is not on PATH
    // (e.g. invoked as ./combs from target/release). COMBS_BIN wins in
    // engine.mjs; an explicit env var still overrides.
    if std::env::var_os("COMBS_BIN").is_none() {
        if let Ok(exe) = std::env::current_exe() {
            cmd.env("COMBS_BIN", exe);
        }
    }
    let status = cmd.status().context("failed to start dev server")?;
    if !status.success() {
        bail!("dev server exited with {status}");
    }
    Ok(())
}

fn prompt_debate(args: &ChewArgs, theme: &ColorfulTheme) -> Result<DebateConfig> {
    let agents_raw = match args.agents.clone() {
        Some(a) => a,
        None if args.yes => "Pro,Con".into(),
        None => Input::with_theme(theme)
            .with_prompt("Agent names (comma-separated)")
            .default("Pro,Con".into())
            .interact_text()?,
    };
    let names: Vec<String> = agents_raw
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if names.len() < 2 {
        bail!("debate-ui needs at least two agents");
    }

    let topic = match args.topic.clone() {
        Some(t) => t,
        None if args.yes => "Is local-first AI better than cloud AI?".into(),
        None => Input::with_theme(theme)
            .with_prompt("Debate topic")
            .default("Is local-first AI better than cloud AI?".into())
            .interact_text()?,
    };

    let turns = match args.turns {
        Some(t) => t,
        None if args.yes => 8,
        None => Input::with_theme(theme)
            .with_prompt("Total turns")
            .default(8)
            .interact_text()?,
    };

    let mut agents = Vec::new();
    for (i, name) in names.iter().enumerate() {
        let stance = if i % 2 == 0 { "pro" } else { "against" }.to_string();
        let behavior = if args.yes {
            format!("Argue {stance} the topic with sharp, concise points.")
        } else {
            Input::with_theme(theme)
                .with_prompt(format!("Behavior/persona for {name} ({stance})"))
                .default(format!("Argue {stance} the topic with sharp, concise points."))
                .interact_text()?
        };
        agents.push(DebateAgent {
            name: name.clone(),
            stance,
            behavior,
        });
    }

    Ok(DebateConfig { agents, topic, turns })
}

// Generated by build.rs — the UI template, embedded in the binary.
mod embedded {
    include!(concat!(env!("OUT_DIR"), "/template_manifest.rs"));
}

/// The UI template source directory.
///
/// Resolution order:
///   1. `COMBS_UI_TEMPLATE` env var (custom template override)
///   2. The template embedded in the binary, extracted once to
///      `$COMBS_HOME/ui-template/<version>` (default `$COMBS_HOME` =
///      `~/.cache/combs`) so the source also lives on the user's system
///      for inspection/customization.
fn template_root() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("COMBS_UI_TEMPLATE") {
        let p = PathBuf::from(p);
        if p.is_dir() {
            return Ok(p);
        }
        bail!("COMBS_UI_TEMPLATE={} is not a directory", p.display());
    }
    extract_embedded_template()
}

/// Writes the embedded template to the user cache (once per CLI version)
/// and returns the extracted path.
fn extract_embedded_template() -> Result<PathBuf> {
    let home = std::env::var("COMBS_HOME")
        .map(PathBuf::from)
        .or_else(|_| {
            std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .map(|h| PathBuf::from(h).join(".cache/combs"))
        })
        .context("cannot locate a home directory (set COMBS_HOME)")?;
    let version = env!("CARGO_PKG_VERSION");
    let root = home.join("ui-template").join(version);
    let stamp = root.join(".combs-version");
    let expected = template_stamp();
    if stamp.is_file() && fs::read_to_string(&stamp).ok().as_deref() == Some(expected.as_str()) {
        return Ok(root); // already extracted, contents unchanged
    }
    if root.exists() {
        fs::remove_dir_all(&root).ok();
    }
    for (rel, bytes) in embedded::TEMPLATE_FILES {
        let target = root.join(rel);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&target, bytes)?;
    }
    fs::write(&stamp, expected)?;
    Ok(root)
}

/// FNV-1a over the embedded template contents — the extraction stamp, so a
/// rebuilt binary with a changed template always re-extracts even when the
/// CLI version is unchanged.
fn template_stamp() -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for (rel, bytes) in embedded::TEMPLATE_FILES {
        for b in rel.as_bytes().iter().chain(bytes.iter()) {
            hash ^= u64::from(*b);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    format!("{}-{hash:016x}", env!("CARGO_PKG_VERSION"))
}

/// Copies the template into `dir` and writes combs.ui.json.
fn scaffold(dir: &Path, config: &UiConfig, overwrite: bool) -> Result<()> {
    if dir.exists() && fs::read_dir(dir)?.next().is_some() {
        let rewrite = overwrite
            || Confirm::new()
                .with_prompt(format!(
                    "directory {} is not empty — overwrite it?",
                    dir.display()
                ))
                .default(true)
                .interact()?;
        if !rewrite {
            bail!("target directory {} is not empty", dir.display());
        }
        fs::remove_dir_all(dir)?;
    }
    let template = template_root()?;
    copy_dir(&template, dir).with_context(|| format!("copying {}", template.display()))?;
    let config_json = serde_json::to_string_pretty(config)?;
    fs::write(dir.join("combs.ui.json"), config_json)?;
    Ok(())
}

fn copy_dir(src: &Path, dst: &Path) -> Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let path = entry.path();
        let target = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            let name = entry.file_name();
            if name == "node_modules"
                || name == ".svelte-kit"
                || name == "dist"
                || name == "data"
            {
                continue;
            }
            copy_dir(&path, &target)?;
        } else {
            fs::copy(&path, &target)?;
        }
    }
    Ok(())
}
