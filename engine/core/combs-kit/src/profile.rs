//! What kind of machine this is — the facts that decide how an engine should run on it, read before any cell is
//! measured, so two records from two machines can be told apart by their shape and not only by their names.
//!
//! A rented container sees the host's CPU list and the host's memory in `/proc`, but runs under a quota and a limit it
//! cannot exceed; the quota and the limit are what a build and an engine actually get. An accelerator's own memory
//! is not visible through wgpu, so it is read from the vendor's tool where one is installed and is otherwise said to
//! be unreadable — or, on a unified-memory machine, to be the host's memory. Every field is an Option: a fact that
//! could not be read is recorded as unread, never as zero.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CpuShape {
    /// Logical processors the kernel lists.
    pub logical: Option<u32>,
    /// Distinct (package, core) pairs.
    pub physical: Option<u32>,
    /// Distinct packages.
    pub sockets: Option<u32>,
    /// Instruction sets that matter to an inference engine, as the kernel names them.
    pub features: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Limits {
    /// CPU time this process group may use, in cores (a cgroup quota), when one is set.
    pub cpu_quota_cores: Option<f64>,
    /// Memory this process group may use, in bytes (a cgroup limit), when one is set.
    pub memory_limit_bytes: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AccelMemory {
    pub bytes: Option<u64>,
    /// Where `bytes` came from: "vendor-tool", "unified", or "unread".
    pub source: String,
    /// The host link as the vendor tool reports it, e.g. "gen4 x16".
    pub link: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Profile {
    pub cpu: CpuShape,
    pub limits: Limits,
    pub accel: AccelMemory,
    /// True when the accelerator draws on host memory, false when it has its own, None when unknown.
    pub unified: Option<bool>,
}

/// The instruction sets worth recording, in the kernel's spelling.
pub const FEATURES_OF_INTEREST: &[&str] = &[
    "avx2", "avx512f", "avx512_bf16", "avx512_vnni", "amx_tile", "amx_int8", "amx_bf16", "asimd", "sve", "sve2",
];

/// Logical, physical and package counts, and the features of interest, from `/proc/cpuinfo` text.
pub fn parse_cpuinfo(text: &str) -> CpuShape {
    let mut logical = 0u32;
    let mut packages = std::collections::BTreeSet::new();
    let mut cores = std::collections::BTreeSet::new();
    let mut package: Option<String> = None;
    let mut topology = true;
    let mut flags: Option<Vec<String>> = None;
    for block in text.split("\n\n") {
        let mut core_id: Option<String> = None;
        let mut seen = false;
        package = None;
        for line in block.lines() {
            let Some((k, v)) = line.split_once(':') else { continue };
            let (k, v) = (k.trim(), v.trim());
            match k {
                "processor" => seen = true,
                "physical id" => package = Some(v.to_string()),
                "core id" => core_id = Some(v.to_string()),
                "flags" | "Features" if flags.is_none() => {
                    flags = Some(v.split_whitespace().map(str::to_string).collect())
                }
                _ => {}
            }
        }
        if !seen {
            continue;
        }
        logical += 1;
        match (&package, &core_id) {
            (Some(p), Some(c)) => {
                packages.insert(p.clone());
                cores.insert((p.clone(), c.clone()));
            }
            _ => topology = false,
        }
    }
    let flags = flags.unwrap_or_default();
    CpuShape {
        logical: (logical > 0).then_some(logical),
        physical: (topology && !cores.is_empty()).then_some(cores.len() as u32),
        sockets: (topology && !packages.is_empty()).then_some(packages.len() as u32),
        features: FEATURES_OF_INTEREST
            .iter()
            .filter(|f| flags.iter().any(|x| x == *f))
            .map(|f| f.to_string())
            .collect(),
    }
}

/// A cgroup v2 `cpu.max` line ("<quota> <period>" or "max <period>") in cores.
pub fn parse_cpu_max(text: &str) -> Option<f64> {
    let mut it = text.split_whitespace();
    let (q, p) = (it.next()?, it.next()?);
    parse_cfs(q, p)
}

/// A cgroup v1 quota and period pair (quota -1 = unlimited) in cores.
pub fn parse_cfs(quota: &str, period: &str) -> Option<f64> {
    let q: i64 = quota.trim().parse().ok()?;
    let p: i64 = period.trim().parse().ok()?;
    if q <= 0 || p <= 0 {
        return None;
    }
    Some((q as f64 / p as f64 * 100.0).round() / 100.0)
}

/// A cgroup memory limit ("max", a huge sentinel, or bytes).
pub fn parse_memory_limit(text: &str) -> Option<u64> {
    let v: u64 = text.trim().parse().ok()?;
    // cgroup v1 reports "unlimited" as a page-rounded i64::MAX; anything past 2^62 bytes is that sentinel.
    (v > 0 && v < (1u64 << 62)).then_some(v)
}

/// One line of `nvidia-smi --query-gpu=memory.total,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader,nounits`.
pub fn parse_vendor_memory(line: &str) -> (Option<u64>, Option<String>) {
    let parts: Vec<&str> = line.split(',').map(str::trim).collect();
    let mib = parts.first().and_then(|v| v.parse::<u64>().ok());
    let gen = parts.get(1).and_then(|v| v.parse::<u32>().ok());
    let width = parts.get(2).and_then(|v| v.parse::<u32>().ok());
    let link = match (gen, width) {
        (Some(g), Some(w)) => Some(format!("gen{g} x{w}")),
        _ => None,
    };
    (mib.map(|m| m * 1024 * 1024), link)
}

fn shell(cmd: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(cmd).args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

#[cfg(target_os = "linux")]
fn read_os(device_type: &str) -> Profile {
    let file = |p: &str| std::fs::read_to_string(p).ok();
    let cpu = file("/proc/cpuinfo").map(|t| parse_cpuinfo(&t)).unwrap_or_default();
    let cpu_quota_cores = file("/sys/fs/cgroup/cpu.max")
        .and_then(|t| parse_cpu_max(&t))
        .or_else(|| parse_cfs(&file("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")?, &file("/sys/fs/cgroup/cpu/cpu.cfs_period_us")?));
    let memory_limit_bytes = file("/sys/fs/cgroup/memory.max")
        .and_then(|t| parse_memory_limit(&t))
        .or_else(|| parse_memory_limit(&file("/sys/fs/cgroup/memory/memory.limit_in_bytes")?));
    let unified = match device_type {
        "DiscreteGpu" => Some(false),
        "IntegratedGpu" => Some(true),
        _ => None,
    };
    let vendor = shell(
        "nvidia-smi",
        &["--query-gpu=memory.total,pcie.link.gen.max,pcie.link.width.max", "--format=csv,noheader,nounits"],
    )
    .and_then(|t| t.lines().next().map(parse_vendor_memory));
    let accel = match (vendor, unified) {
        (Some((Some(b), link)), _) => AccelMemory { bytes: Some(b), source: "vendor-tool".into(), link },
        (_, Some(true)) => AccelMemory { bytes: None, source: "unified".into(), link: None },
        _ => AccelMemory { bytes: None, source: "unread".into(), link: None },
    };
    Profile { cpu, limits: Limits { cpu_quota_cores, memory_limit_bytes }, accel, unified }
}

#[cfg(target_os = "macos")]
fn read_os(device_type: &str) -> Profile {
    let num = |k: &str| shell("sysctl", &["-n", k]).and_then(|v| v.parse::<u32>().ok());
    let mut features = Vec::new();
    // Apple silicon names its instruction sets through hw.optional; ASIMD is always present on arm64.
    if std::env::consts::ARCH == "aarch64" {
        features.push("asimd".to_string());
    }
    let cpu = CpuShape {
        logical: num("hw.logicalcpu"),
        physical: num("hw.physicalcpu"),
        sockets: num("hw.packages"),
        features,
    };
    let unified = match device_type {
        "IntegratedGpu" => Some(true),
        "DiscreteGpu" => Some(false),
        _ => None,
    };
    let accel = if unified == Some(true) {
        AccelMemory { bytes: None, source: "unified".into(), link: None }
    } else {
        AccelMemory { bytes: None, source: "unread".into(), link: None }
    };
    Profile { cpu, limits: Limits::default(), accel, unified }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn read_os(_device_type: &str) -> Profile {
    Profile { accel: AccelMemory { bytes: None, source: "unread".into(), link: None }, ..Profile::default() }
}

pub fn read(device_type: &str) -> Profile {
    read_os(device_type)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CPUINFO_X86: &str = "processor\t: 0\nphysical id\t: 0\ncore id\t\t: 0\nflags\t\t: fpu sse avx2 avx512f amx_tile amx_int8 other\n\n\
processor\t: 1\nphysical id\t: 0\ncore id\t\t: 0\nflags\t\t: fpu sse avx2 avx512f amx_tile amx_int8 other\n\n\
processor\t: 2\nphysical id\t: 0\ncore id\t\t: 1\nflags\t\t: fpu sse avx2 avx512f amx_tile amx_int8 other\n\n\
processor\t: 3\nphysical id\t: 1\ncore id\t\t: 0\nflags\t\t: fpu sse avx2 avx512f amx_tile amx_int8 other\n";

    const CPUINFO_ARM: &str = "processor\t: 0\nFeatures\t: fp asimd sve crc32\n\nprocessor\t: 1\nFeatures\t: fp asimd sve crc32\n";

    #[test]
    fn cpuinfo_counts_logical_physical_and_packages() {
        let s = parse_cpuinfo(CPUINFO_X86);
        assert_eq!(s.logical, Some(4), "four processor lines");
        assert_eq!(s.physical, Some(3), "(0,0) (0,1) (1,0)");
        assert_eq!(s.sockets, Some(2), "packages 0 and 1");
        assert_eq!(s.features, vec!["avx2", "avx512f", "amx_tile", "amx_int8"], "only the features of interest, in their listed order");
    }

    #[test]
    fn cpuinfo_without_topology_says_logical_only() {
        let s = parse_cpuinfo(CPUINFO_ARM);
        assert_eq!(s.logical, Some(2));
        assert_eq!(s.physical, None, "no core ids: physical cores are unread, not guessed");
        assert_eq!(s.sockets, None);
        assert_eq!(s.features, vec!["asimd", "sve"]);
    }

    #[test]
    fn a_cgroup_quota_reads_in_cores_and_max_means_none() {
        assert_eq!(parse_cpu_max("272000 100000\n"), Some(2.72));
        assert_eq!(parse_cpu_max("max 100000\n"), None);
        assert_eq!(parse_cpu_max("garbage"), None);
        assert_eq!(parse_cfs("2720000", "100000"), Some(27.2));
        assert_eq!(parse_cfs("-1", "100000"), None);
    }

    #[test]
    fn a_memory_limit_ignores_the_unlimited_sentinels() {
        assert_eq!(parse_memory_limit("max\n"), None);
        assert_eq!(parse_memory_limit("9223372036854771712\n"), None, "cgroup v1's unlimited");
        assert_eq!(parse_memory_limit("124999999488\n"), Some(124_999_999_488));
        assert_eq!(parse_memory_limit(""), None);
    }

    #[test]
    fn a_vendor_memory_line_gives_bytes_and_the_link() {
        assert_eq!(parse_vendor_memory("32607, 5, 16"), (Some(32607 * 1024 * 1024), Some("gen5 x16".to_string())));
        assert_eq!(parse_vendor_memory("24564, [N/A], [N/A]"), (Some(24564 * 1024 * 1024), None));
        assert_eq!(parse_vendor_memory("[N/A], 4, 8"), (None, Some("gen4 x8".to_string())));
        assert_eq!(parse_vendor_memory(""), (None, None));
    }

    #[test]
    fn an_unread_fact_serializes_as_null_never_zero() {
        let text = serde_json::to_string(&Profile::default()).unwrap();
        assert!(text.contains("\"cpu_quota_cores\":null"));
        assert!(text.contains("\"bytes\":null"));
        assert!(!text.contains(":0,") && !text.contains(":0}"));
    }
}
