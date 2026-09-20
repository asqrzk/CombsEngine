//! The cascade threshold, swept rather than guessed.
//!
//! A cheap classifier answers, and below some confidence the caller
//! falls back to a dearer system. Where that line sits is an expected
//! value question with four terms — the two systems' costs, their
//! latencies, and what an error is worth — and the answer moves with the
//! deployment. So it is a swept number carried by a door, never a
//! constant in code.

/// One labelled item: what the cheap system said, and whether each
/// system was right about it.
#[derive(Debug, Clone, Copy)]
pub struct Row {
    pub sys1_confidence: f32,
    pub sys1_correct: bool,
    pub sys2_correct: bool,
}

/// The weights of the trade, units named at every field.
#[derive(Debug, Clone, Copy)]
pub struct Costs {
    /// Cost of one cheap answer, in the caller's own unit.
    pub c_sys1: f32,
    /// Cost of one fallback answer, same unit.
    pub c_sys2: f32,
    /// Milliseconds a cheap answer takes.
    pub l_sys1_ms: f32,
    /// Milliseconds a fallback answer takes.
    pub l_sys2_ms: f32,
    /// What one millisecond of wait is worth, in the cost unit.
    pub w_lat_per_ms: f32,
    /// What one wrong answer is worth, in the cost unit.
    pub w_err: f32,
}

/// The sweep's value at one threshold.
#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub t: f32,
    pub p_fallback: f32,
    /// The cheap system's error rate among the items it KEEPS at this
    /// threshold. Zero when it keeps nothing — there is no error rate
    /// over an empty set, and the record says so rather than inventing
    /// one.
    pub e_sys1: f32,
    pub cost: f32,
    pub latency: f32,
    pub error: f32,
    pub ev: f32,
}

/// Sweep t over 0.00..=1.00 in hundredths and return the argmin and the
/// whole curve.
///
/// Ties take the LOWEST t: when the expected value is flat, prefer the
/// cheap system — the dear one has to earn the switch.
pub fn sweep_thresholds(rows: &[Row], costs: &Costs) -> (f32, Vec<Point>) {
    if rows.is_empty() {
        return (0.0, Vec::new());
    }
    let n = rows.len() as f32;
    let e2 = 1.0 - rows.iter().filter(|r| r.sys2_correct).count() as f32 / n;

    let mut curve = Vec::with_capacity(101);
    let mut best_t = 0.0f32;
    let mut best_ev = f32::MAX;
    for i in 0..=100 {
        let t = i as f32 / 100.0;
        let kept: Vec<&Row> = rows.iter().filter(|r| r.sys1_confidence >= t).collect();
        let p_fallback = (n - kept.len() as f32) / n;
        let e_sys1 = if kept.is_empty() {
            0.0
        } else {
            1.0 - kept.iter().filter(|r| r.sys1_correct).count() as f32 / kept.len() as f32
        };
        let cost = costs.c_sys1 + p_fallback * costs.c_sys2;
        let latency = costs.l_sys1_ms + p_fallback * costs.l_sys2_ms;
        let error = (1.0 - p_fallback) * e_sys1 + p_fallback * e2;
        let ev = cost + costs.w_lat_per_ms * latency + costs.w_err * error;
        curve.push(Point { t, p_fallback, e_sys1, cost, latency, error, ev });
        if ev < best_ev - 1e-9 {
            best_ev = ev;
            best_t = t;
        }
    }
    (best_t, curve)
}
