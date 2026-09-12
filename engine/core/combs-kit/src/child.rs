//! One subject, one process. The rule the bench is built around.
//!
//! This is not a style preference. The GPU pool does not return memory
//! until a process exits, and autotune's working set scales
//! super-linearly with vocabulary: one qwen3-0.6b test measured a 20.5
//! GB peak against 3.2 GB for a smaller model, and a suite run in one
//! process has taken this machine down. The parity sweep already runs
//! one test per process for exactly this reason.
//!
//! A second reason arrived later and is just as binding: the dispatch
//! counter is process-wide and monotone, so two subjects measured in
//! one process cannot tell each other's launches apart. Two independent
//! arguments, one rule.
//!
//! So the parent NEVER loads a model. It spawns itself with a hidden
//! subcommand, reads one JSON object from the child's stdout, and reaps
//! what it spawned — nothing else. It does not scan for processes, read
//! ports, or kill by name. That ground belongs to the operational
//! toolkit and the kit does not touch it.

use std::io::Read;
use std::process::{Command, Stdio};

use crate::record::Run;

/// How long one subject may take before the parent stops waiting. A
/// model that will not load, or a machine gone to swap, must produce a
/// recorded timeout rather than a harness that hangs — a skip is named,
/// never mistaken for a pass.
const DEADLINE_SECS: u64 = 900;

pub struct Subject {
    pub id: String,
    pub model: String,
    pub prompt_tokens: usize,
    pub max_tokens: usize,
}

impl Subject {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "max_tokens": self.max_tokens,
        })
    }

    /// The shape that goes into a RECORD: the model by name, never by
    /// path. A committed baseline must not carry someone's home
    /// directory, and a path makes a record impossible to compare
    /// against a run on another machine.
    pub fn to_json_for_record(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id,
            "model": crate::model::identity(&self.model),
            "prompt_tokens": self.prompt_tokens,
            "max_tokens": self.max_tokens,
        })
    }

    pub fn from_json(v: &serde_json::Value) -> Option<Self> {
        Some(Self {
            id: v["id"].as_str()?.to_string(),
            model: v["model"].as_str()?.to_string(),
            prompt_tokens: v["prompt_tokens"].as_u64()? as usize,
            max_tokens: v["max_tokens"].as_u64()? as usize,
        })
    }
}

/// Run one subject in a child of this same binary and return its Run.
///
/// A failure is a recorded outcome, not a panic: the caller gets a Run
/// carrying the reason so the record shows what was attempted and why it
/// produced nothing. Erasing a failed subject would make a partial
/// sweep look complete.
pub fn measure(subject: &Subject) -> Run {
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => return failed(subject, &format!("cannot find my own binary: {e}")),
    };
    let mut child = match Command::new(exe)
        .arg("run-one")
        .arg("--subject")
        .arg(subject.to_json().to_string())
        .env("COMBS_KIT_CHILD", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => return failed(subject, &format!("cannot spawn a child: {e}")),
    };

    let started = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut out = String::new();
                if let Some(mut pipe) = child.stdout.take() {
                    let _ = pipe.read_to_string(&mut out);
                }
                if !status.success() {
                    return failed(
                        subject,
                        &format!("the child exited {status}; its stderr is above"),
                    );
                }
                return match serde_json::from_str::<Run>(out.trim()) {
                    Ok(run) => run,
                    Err(e) => failed(subject, &format!("the child's answer did not parse: {e}")),
                };
            }
            Ok(None) => {
                if started.elapsed().as_secs() > DEADLINE_SECS {
                    let _ = child.kill();
                    let _ = child.wait();
                    return failed(
                        subject,
                        &format!("no answer within {DEADLINE_SECS}s — killed"),
                    );
                }
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
            Err(e) => return failed(subject, &format!("cannot wait on the child: {e}")),
        }
    }
}

fn failed(subject: &Subject, why: &str) -> Run {
    Run {
        id: subject.id.clone(),
        kind: "model".to_string(),
        subject: crate::model::identity(&subject.model),
        config: subject.to_json_for_record(),
        warm: false,
        values: Vec::new(),
        notes: vec![format!("no measurement: {why}")],
    }
}

/// True inside a child, so the child never spawns children of its own.
pub fn is_child() -> bool {
    std::env::var_os("COMBS_KIT_CHILD").is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_subject_round_trips_through_its_json() {
        let s = Subject {
            id: "d2000".into(),
            model: "/models/qwen3-0.6b-gguf".into(),
            prompt_tokens: 2000,
            max_tokens: 16,
        };
        let back = Subject::from_json(&s.to_json()).expect("round trip");
        assert_eq!(back.id, s.id);
        assert_eq!(back.prompt_tokens, 2000);
    }

    /// A subject that produced nothing must still appear in the record,
    /// carrying why. A sweep that silently drops its failures reads as
    /// complete when it is not. It is named rather than pathed, like
    /// every other subject: a committed baseline must not publish
    /// someone's home directory, and the hygiene gate refuses one that
    /// does.
    #[test]
    fn a_failed_subject_is_recorded_not_erased() {
        let s = Subject {
            id: "d4".into(),
            model: "/nowhere".into(),
            prompt_tokens: 4,
            max_tokens: 16,
        };
        let run = failed(&s, "the model does not exist");
        assert!(run.values.is_empty());
        assert!(!run.warm, "a failure is never reported as a warm reading");
        assert_eq!(run.notes.len(), 1);
        assert!(run.notes[0].contains("the model does not exist"));
        // Named, not pathed — even a failure must not carry a home
        // directory into a record that may be committed.
        assert_eq!(run.subject, "nowhere");
    }
}
