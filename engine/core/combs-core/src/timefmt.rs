// Timestamps and durations in the interchange forms, so a record can
// be read by a person, sorted by a machine, and parsed by anything
// that speaks the standards: RFC 3339 for instants (UTC, always with
// the `Z`), ISO 8601 for durations.
//
// Hand-rolled rather than pulled from a crate because `build.rs`
// includes this same source — the build's own timestamp and the
// running binary's log lines must be produced by one implementation,
// or they can disagree about what time it is.

/// `2026-08-29T12:34:56Z` from seconds since the Unix epoch (UTC).
pub fn rfc3339(epoch_secs: u64) -> String {
    let days = (epoch_secs / 86_400) as i64;
    let secs_of_day = epoch_secs % 86_400;
    let (y, m, d) = civil_from_days(days);
    let (h, min, s) = (secs_of_day / 3600, (secs_of_day % 3600) / 60, secs_of_day % 60);
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{min:02}:{s:02}Z")
}

/// Howard Hinnant's civil-from-days, shifted to a March-based year so
/// the leap day lands at the end and needs no special case.
fn civil_from_days(days_since_epoch: i64) -> (i64, u32, u32) {
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `PT38.5S` from milliseconds — the ISO 8601 duration form, kept
/// alongside the raw millisecond count so neither a human nor a parser
/// has to do arithmetic to read a record.
pub fn iso_duration(ms: u128) -> String {
    let secs = ms / 1000;
    let (h, m, s, frac) = (secs / 3600, (secs % 3600) / 60, secs % 60, ms % 1000);
    let mut out = String::from("PT");
    if h > 0 {
        out.push_str(&format!("{h}H"));
    }
    if m > 0 {
        out.push_str(&format!("{m}M"));
    }
    if frac > 0 {
        out.push_str(&format!("{s}.{frac:03}S"));
    } else {
        out.push_str(&format!("{s}S"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc3339_matches_known_instants() {
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339(1_000_000_000), "2001-09-09T01:46:40Z");
        // A leap day, the case the March-based shift exists to handle.
        assert_eq!(rfc3339(1_709_164_800), "2024-02-29T00:00:00Z");
        assert_eq!(rfc3339(1_787_966_364), "2026-08-29T01:19:24Z");
    }

    #[test]
    fn durations_are_iso8601() {
        assert_eq!(iso_duration(0), "PT0S");
        assert_eq!(iso_duration(500), "PT0.500S");
        assert_eq!(iso_duration(38_500), "PT38.500S");
        assert_eq!(iso_duration(90_000), "PT1M30S");
        assert_eq!(iso_duration(3_723_000), "PT1H2M3S");
    }
}
