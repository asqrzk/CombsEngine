//! What the HOST machine will admit about its own memory.
//!
//! Two probes, and the `None` arms matter as much as the readings: a
//! platform with no trusted probe must skip a check rather than guess,
//! because a fabricated number here becomes either a refusal that lies
//! or a gate that never fires. The CLI's fit gates stand down BY NAME
//! when these answer `None`, and the measurement kit records which
//! probes a box actually has, so a reader knows a gate stood down
//! rather than passed.
//!
//! Moved here from the CLI so there is one implementation: the engine,
//! the CLI and the kit all ask the same question of the same code.

const MB: u64 = 1024 * 1024;
const GB: u64 = 1024 * MB;

/// Memory the machine can hand out, or `None` where we have no probe we
/// trust — an unknown platform skips the check rather than guessing.
#[cfg(target_os = "macos")]
pub fn available_memory_bytes() -> Option<u64> {
    // free + speculative + inactive: macOS keeps "free" deliberately
    // small, and inactive pages are what a large allocation actually
    // draws on. Optimistic by design — DEFAULT_HEADROOM_BYTES is the
    // counterweight.
    let mut stats: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
    let mut count = libc::HOST_VM_INFO64_COUNT;
    // libc marks mach_host_self deprecated in favour of the mach2
    // crate; one call does not justify a new dependency, and the port
    // name it returns is stable.
    #[allow(deprecated)]
    let ok = unsafe {
        libc::host_statistics64(
            libc::mach_host_self(),
            libc::HOST_VM_INFO64,
            &mut stats as *mut _ as *mut libc::integer_t,
            &mut count,
        )
    };
    if ok != 0 {
        return None;
    }
    let page = unsafe { libc::vm_page_size } as u64;
    Some(
        (stats.free_count as u64 + stats.speculative_count as u64 + stats.inactive_count as u64)
            * page,
    )
}

#[cfg(target_os = "linux")]
pub fn available_memory_bytes() -> Option<u64> {
    parse_kb_field(
        &std::fs::read_to_string("/proc/meminfo").ok()?,
        "MemAvailable:",
    )
}

/// Pull a `kB`-suffixed field out of a /proc table. Kept off the
/// platform gate so the parsing is tested everywhere, not only where it
/// runs ("MemAvailable:" is the kernel's own estimate of what a fresh
/// allocation can get — exactly this question).
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse_kb_field(table: &str, field: &str) -> Option<u64> {
    table
        .lines()
        .find_map(|l| l.strip_prefix(field))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|kb| kb.parse::<u64>().ok())
        .map(|kb| kb * 1024)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn available_memory_bytes() -> Option<u64> {
    None
}

/// This process's physical footprint — the number Activity Monitor
/// shows, which unlike RSS counts GPU allocations on unified memory.
#[cfg(target_os = "macos")]
pub fn process_footprint_bytes() -> Option<u64> {
    let mut info: libc::rusage_info_v2 = unsafe { std::mem::zeroed() };
    let ok = unsafe {
        libc::proc_pid_rusage(
            std::process::id() as libc::c_int,
            libc::RUSAGE_INFO_V2,
            &mut info as *mut _ as *mut libc::rusage_info_t,
        )
    };
    (ok == 0).then_some(info.ri_phys_footprint)
}

/// No probe off macOS yet: Linux's VmRSS omits GPU allocations (GTT
/// pages an integrated driver maps are not charged to the process), and
/// the whole model is denominated in a footprint that INCLUDES them.
/// Reporting VmRSS here would make every run after the first look cold
/// and refuse it. Skipping is the honest answer until someone measures
/// the platform properly.
#[cfg(not(target_os = "macos"))]
pub fn process_footprint_bytes() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proc_tables_parse_kb_fields() {
        let meminfo =
            "MemTotal:       32000000 kB\nMemFree:  100 kB\nMemAvailable:   12345678 kB\n";
        assert_eq!(
            parse_kb_field(meminfo, "MemAvailable:"),
            Some(12_345_678 * 1024)
        );
        assert_eq!(
            parse_kb_field("VmRSS:\t  4096 kB\n", "VmRSS:"),
            Some(4 * MB)
        );
        assert_eq!(parse_kb_field(meminfo, "Nope:"), None);
        assert_eq!(
            parse_kb_field("MemAvailable:   what kB\n", "MemAvailable:"),
            None
        );
    }

    /// The live probes must agree with reality on this machine: a
    /// footprint in a plausible range, and free memory under total.
    #[test]
    fn live_probes_return_plausible_numbers() {
        // The probes' own contract is "None where we have no probe we
        // trust" — the footprint probe has no Linux arm at all, so an
        // expect() here failed every ubuntu CI run by construction
        // (born on macOS, where proc_pid_rusage answers). Each probe
        // is judged only where it exists.
        match process_footprint_bytes() {
            Some(footprint) => {
                assert!(footprint > 1024 * 1024, "implausibly small: {footprint}");
                assert!(footprint < 64 * GB, "implausibly large: {footprint}");
            }
            None => eprintln!("skip: no trusted footprint probe on this platform"),
        }
        match available_memory_bytes() {
            Some(available) => {
                assert!(available > 0);
                assert!(available < 1024 * GB);
            }
            None => eprintln!("skip: no trusted available-memory probe on this platform"),
        }
    }
}
