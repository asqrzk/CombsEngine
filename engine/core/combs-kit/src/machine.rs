//! What this box IS, recorded before anything is measured on it.
//!
//! Two readings here are easy to skip and expensive to lack. The POWER
//! SOURCE, because a laptop on battery throttles and an unlabelled
//! battery run has poisoned more comparisons in this estate than any
//! bug — §92 holds two figures for one seat that differ 2.2x with
//! matching dispatch counts, which is desk state and nothing else. And
//! WHICH PROBES EXIST, because a platform with no host-memory probe
//! silently disarms the fit gates, and a reader deserves to know that
//! the gate stood down rather than passed.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Power {
    Ac,
    Battery,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Probes {
    /// `combs_core::host::available_memory_bytes` answers here.
    pub available_memory: bool,
    /// `combs_core::host::process_footprint_bytes` answers here.
    pub process_footprint: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Machine {
    pub os: String,
    pub arch: String,
    pub cpu: String,
    pub ram_bytes: Option<u64>,
    pub available_bytes: Option<u64>,
    pub power: Power,
    pub probes: Probes,
    pub gpu: serde_json::Value,
    /// Whether this adapter admits f16 in a shader. wgpu 29 hides the
    /// typed flag, so this reads the raw feature dump — which is why
    /// `DeviceCaps` carries that string at all.
    pub shader_f16: Option<bool>,
    /// The machine's shape: cores, quota, memory limit, accelerator memory, unified or not.
    pub profile: crate::profile::Profile,
}

fn shell(cmd: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(cmd).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn cpu_name() -> String {
    #[cfg(target_os = "macos")]
    {
        if let Some(v) = shell("sysctl", &["-n", "machdep.cpu.brand_string"]) {
            return v;
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in text.lines() {
                if let Some(v) = line.strip_prefix("model name") {
                    if let Some(v) = v.split(':').nth(1) {
                        return v.trim().to_string();
                    }
                }
            }
        }
    }
    "unknown".to_string()
}

fn ram_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        return shell("sysctl", &["-n", "hw.memsize"])?.parse().ok();
    }
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in text.lines() {
            if let Some(v) = line.strip_prefix("MemTotal:") {
                let kb: u64 = v.trim().trim_end_matches(" kB").trim().parse().ok()?;
                return Some(kb * 1024);
            }
        }
        return None;
    }
    #[allow(unreachable_code)]
    None
}

/// AC or battery. Unknown is an honest answer and is recorded as one —
/// the alternative is a number that silently means something else.
fn power() -> Power {
    #[cfg(target_os = "macos")]
    {
        if let Some(out) = shell("pmset", &["-g", "batt"]) {
            if out.contains("'AC Power'") {
                return Power::Ac;
            }
            if out.contains("'Battery Power'") {
                return Power::Battery;
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(v) = std::fs::read_to_string("/sys/class/power_supply/AC/online") {
            return match v.trim() {
                "1" => Power::Ac,
                "0" => Power::Battery,
                _ => Power::Unknown,
            };
        }
    }
    Power::Unknown
}

pub fn read() -> Machine {
    let (gpu, shader_f16) = if combs_core::gpu_available() {
        let device = combs_core::init_device();
        let caps = combs_core::device_caps(&device);
        let f16 = caps.features.to_ascii_uppercase().contains("SHADER_F16");
        (serde_json::to_value(&caps).unwrap_or(serde_json::Value::Null), Some(f16))
    } else {
        (
            serde_json::json!({"name": "none", "backend": "none",
                "note": "no wgpu adapter is visible on this machine"}),
            None,
        )
    };

    Machine {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        cpu: cpu_name(),
        ram_bytes: ram_bytes(),
        available_bytes: combs_core::host::available_memory_bytes(),
        power: power(),
        probes: Probes {
            available_memory: combs_core::host::available_memory_bytes().is_some(),
            process_footprint: combs_core::host::process_footprint_bytes().is_some(),
        },
        profile: crate::profile::read(gpu.get("device_type").and_then(|v| v.as_str()).unwrap_or("")),
        gpu,
        shader_f16,
    }
}

pub fn print_human(m: &Machine) {
    println!("machine");
    println!("  os          {} {}", m.os, m.arch);
    println!("  cpu         {}", m.cpu);
    match m.ram_bytes {
        Some(b) => println!("  ram         {:.1} GiB", b as f64 / 1073741824.0),
        None => println!("  ram         unreadable on this platform"),
    }
    match m.available_bytes {
        Some(b) => println!("  available   {:.1} GiB", b as f64 / 1073741824.0),
        None => println!("  available   NO PROBE — every host-memory gate stands down here"),
    }
    println!("  power       {:?}", m.power);
    let name = m.gpu.get("name").and_then(|v| v.as_str()).unwrap_or("?");
    let backend = m.gpu.get("backend").and_then(|v| v.as_str()).unwrap_or("?");
    let kind = m.gpu.get("device_type").and_then(|v| v.as_str()).unwrap_or("?");
    println!("  gpu         {name} · {backend} · {kind}");
    match m.shader_f16 {
        Some(true) => println!("  shader f16  yes — f16 storage kernels are reachable on this adapter"),
        Some(false) => println!("  shader f16  no — f16 storage kernels stay doored off here"),
        None => println!("  shader f16  unknown — no adapter to ask"),
    }
    let p = &m.profile;
    let opt = |v: Option<u32>| v.map(|n| n.to_string()).unwrap_or_else(|| "unread".into());
    println!(
        "  cores       logical {} · physical {} · packages {} · quota {}",
        opt(p.cpu.logical),
        opt(p.cpu.physical),
        opt(p.cpu.sockets),
        p.limits.cpu_quota_cores.map(|c| format!("{c} cores")).unwrap_or_else(|| "none set".into())
    );
    if !p.cpu.features.is_empty() {
        println!("  isa         {}", p.cpu.features.join(" "));
    }
    if let Some(b) = p.limits.memory_limit_bytes {
        println!("  mem limit   {:.1} GiB (cgroup)", b as f64 / 1073741824.0);
    }
    match (p.accel.bytes, p.accel.source.as_str()) {
        (Some(b), src) => println!(
            "  accel mem   {:.1} GiB ({src}){}",
            b as f64 / 1073741824.0,
            p.accel.link.as_ref().map(|l| format!(" · link {l}")).unwrap_or_default()
        ),
        (None, "unified") => println!("  accel mem   unified — the adapter draws on host memory"),
        (None, src) => println!("  accel mem   {src}"),
    }
    println!(
        "  probes      available_memory {} · process_footprint {}",
        yes_no(m.probes.available_memory),
        yes_no(m.probes.process_footprint)
    );
}

fn yes_no(b: bool) -> &'static str {
    if b { "yes" } else { "NO" }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unknown must be reachable and must serialize as a word, not as a
    /// silent default that reads like a measurement.
    #[test]
    fn power_serializes_as_a_word() {
        let text = serde_json::to_string(&Power::Unknown).unwrap();
        assert_eq!(text, "\"unknown\"");
        assert_eq!(serde_json::to_string(&Power::Battery).unwrap(), "\"battery\"");
    }

    /// A missing probe is recorded as missing, never as zero. A zero
    /// available-memory reading is a refusal; None is a stand-down, and
    /// the two must not be confusable downstream.
    #[test]
    fn a_missing_probe_is_none_not_zero() {
        let m = Machine {
            os: "windows".into(),
            arch: "x86_64".into(),
            cpu: "test".into(),
            ram_bytes: None,
            available_bytes: None,
            power: Power::Unknown,
            probes: Probes { available_memory: false, process_footprint: false },
            gpu: serde_json::Value::Null,
            shader_f16: None,
            profile: crate::profile::Profile::default(),
        };
        let text = serde_json::to_string(&m).unwrap();
        assert!(text.contains("\"available_bytes\":null"));
        assert!(!text.contains("\"available_bytes\":0"));
    }
}
