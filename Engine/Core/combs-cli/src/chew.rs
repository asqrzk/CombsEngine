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
    /// First-run auth: keypair generation + backup prompt. `--authentication=false`
    /// disables auth AND local persistence (incognito mode).
    #[arg(long)]
    pub authentication: Option<bool>,
    /// UI theme: system | dark | light.
    #[arg(long)]
    pub theme: Option<String>,
    /// Model preset id to use by default.
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
    let authentication = match args.authentication {
        Some(a) => a,
        None if args.yes => true,
        None => Confirm::with_theme(&theme)
            .with_prompt("Enable first-run authentication (keypair + backup prompt)?")
            .default(true)
            .interact()?,
    };
    if !authentication {
        println!(
            "{}",
            style("incognito mode: no auth, no local persistence").yellow()
        );
    }

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

    // --- model ---------------------------------------------------------------
    let model = match args.model.clone() {
        Some(m) => m,
        None if args.yes => "smollm2-135m".into(),
        None => Input::with_theme(&theme)
            .with_prompt("Default model preset")
            .default("smollm2-135m".into())
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
    scaffold(&dir, &config)?;

    println!();
    println!("{}", style("scaffold complete!").bold().green());
    println!("  dir:   {}", dir.display());
    println!();
    println!("next steps:");
    println!("  cd {}", dir.display());
    println!("  npm install");
    println!("  npm run dev");
    println!();
    println!("then point your browser at the printed URL (UI talks to {}).", args.server);
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

/// The UI template source directory (Engine/Ui/template).
fn template_root() -> Result<PathBuf> {
    // combs-cli lives at Engine/Core/combs-cli; the template is Engine/Ui/template.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("Ui/template"));
    match root {
        Some(r) if r.is_dir() => Ok(r),
        _ => bail!(
            "UI template not found at Engine/Ui/template — are you running from the CombsEngine repo?"
        ),
    }
}

/// Copies the template into `dir` and writes combs.ui.json.
fn scaffold(dir: &Path, config: &UiConfig) -> Result<()> {
    if dir.exists() && fs::read_dir(dir)?.next().is_some() {
        bail!("target directory {} is not empty", dir.display());
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
            if entry.file_name() == "node_modules" || entry.file_name() == ".svelte-kit" {
                continue;
            }
            copy_dir(&path, &target)?;
        } else {
            fs::copy(&path, &target)?;
        }
    }
    Ok(())
}
