//! The one budget authority. Every layer that asks "what can this
//! device hold?" asks HERE — the engine's load preflight, a host's
//! doctor, a fit gate before a download. The zero-means-absence
//! interpretation happens at exactly this choke point, and the type
//! carries the meaning: `None` IS "no such constraint exists", which
//! no consumer can misread as a cap of nothing. (The misread cost the
//! cpu floor every model it was built to serve.)

use combs_core::CombsDevice;

#[derive(Debug, Clone, Copy)]
pub struct DeviceBudget {
    /// Largest single storage binding, when the backend has bindings
    /// at all. `None` on backends with no device buffers (cpu) — the
    /// absence of a limit, never a limit of zero.
    pub binding_limit: Option<u64>,
    /// Largest single buffer, same contract.
    pub buffer_limit: Option<u64>,
}

fn present(cap: u64) -> Option<u64> {
    (cap > 0).then_some(cap)
}

pub fn device_budget(device: &CombsDevice) -> DeviceBudget {
    let caps = combs_core::device_caps(device);
    DeviceBudget {
        binding_limit: present(caps.max_storage_buffer_binding_size),
        buffer_limit: present(caps.max_buffer_size),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_caps_mean_absence_not_a_cap() {
        assert_eq!(present(0), None);
        assert_eq!(present(1), Some(1));
        assert_eq!(present(u64::MAX), Some(u64::MAX));
    }

    /// On a cpu build the whole budget is constraint-free by
    /// construction — the property the fit check stands on.
    #[cfg(feature = "cpu")]
    #[test]
    fn cpu_budget_carries_no_device_constraints() {
        let device = combs_core::init_device();
        let b = device_budget(&device);
        assert_eq!(b.binding_limit, None);
        assert_eq!(b.buffer_limit, None);
    }
}
