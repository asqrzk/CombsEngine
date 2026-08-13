//! Logits processors: composable, CPU-side transforms applied to a logits row
//! before sampling.
//!
//! A processor receives the mutable logits slice plus the full token history
//! (prompt + generated so far) and adjusts logits in place. Processors are
//! chained in the canonical order: penalties → temperature → top-k → top-p.

/// A transform applied to a logits row before sampling.
pub trait LogitsProcessor: Send + Sync {
    /// Adjusts `logits` in place. `history` is prompt + generated tokens.
    fn process(&self, logits: &mut [f32], history: &[u32]);
}

/// An ordered chain of processors, applied front to back.
#[derive(Default)]
pub struct LogitsProcessorChain {
    processors: Vec<Box<dyn LogitsProcessor>>,
}

impl LogitsProcessorChain {
    /// Creates an empty chain (a no-op).
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a processor to the end of the chain.
    pub fn push(&mut self, processor: impl LogitsProcessor + 'static) {
        self.processors.push(Box::new(processor));
    }

    /// Applies all processors in order.
    pub fn process(&self, logits: &mut [f32], history: &[u32]) {
        for p in &self.processors {
            p.process(logits, history);
        }
    }
}

/// Divides all logits by the temperature. Only constructed with `t > 0`;
/// the `t == 0` (greedy) case is handled by sampler selection.
pub struct TemperatureScaler {
    temperature: f32,
}

impl TemperatureScaler {
    /// Creates a scaler. `temperature <= 0` is treated as a no-op.
    pub fn new(temperature: f32) -> Self {
        TemperatureScaler { temperature }
    }
}

impl LogitsProcessor for TemperatureScaler {
    fn process(&self, logits: &mut [f32], _history: &[u32]) {
        if self.temperature <= 0.0 || self.temperature == 1.0 {
            return;
        }
        for v in logits.iter_mut() {
            *v /= self.temperature;
        }
    }
}

/// HuggingFace-style repetition penalty: for every token already present in
/// the history, positive logits are divided by the penalty and negative
/// logits are multiplied by it (both reduce the token's probability).
pub struct RepetitionPenalty {
    penalty: f32,
}

impl RepetitionPenalty {
    /// Creates the processor. `penalty == 1.0` is a no-op.
    pub fn new(penalty: f32) -> Self {
        RepetitionPenalty { penalty }
    }
}

impl LogitsProcessor for RepetitionPenalty {
    fn process(&self, logits: &mut [f32], history: &[u32]) {
        if self.penalty == 1.0 {
            return;
        }
        // HF reference semantics: a token is penalized at most once no
        // matter how often it appears (same dedup as PresencePenalty).
        // Count-based compounding (penalty^n) pushes generations past
        // their natural stop — measured on qwen2.5-7b: rope 1.1 ran to
        // the length cap with spaceless run-ons; once-per-unique stays
        // clean and terminates.
        let mut seen = std::collections::HashSet::new();
        for &id in history {
            if !seen.insert(id) {
                continue;
            }
            let Some(v) = logits.get_mut(id as usize) else {
                continue;
            };
            if *v > 0.0 {
                *v /= self.penalty;
            } else {
                *v *= self.penalty;
            }
        }
    }
}

/// OpenAI-style frequency penalty: subtracts `penalty * count` from the logit
/// of every token in the history, where `count` is how often the token
/// appears.
pub struct FrequencyPenalty {
    penalty: f32,
}

impl FrequencyPenalty {
    /// Creates the processor. `penalty == 0.0` is a no-op.
    pub fn new(penalty: f32) -> Self {
        FrequencyPenalty { penalty }
    }
}

impl LogitsProcessor for FrequencyPenalty {
    fn process(&self, logits: &mut [f32], history: &[u32]) {
        if self.penalty == 0.0 {
            return;
        }
        for &id in history {
            if let Some(v) = logits.get_mut(id as usize) {
                *v -= self.penalty;
            }
        }
    }
}

/// OpenAI-style presence penalty: subtracts `penalty` once from the logit of
/// every token that appears at least once in the history.
pub struct PresencePenalty {
    penalty: f32,
}

impl PresencePenalty {
    /// Creates the processor. `penalty == 0.0` is a no-op.
    pub fn new(penalty: f32) -> Self {
        PresencePenalty { penalty }
    }
}

impl LogitsProcessor for PresencePenalty {
    fn process(&self, logits: &mut [f32], history: &[u32]) {
        if self.penalty == 0.0 {
            return;
        }
        let mut seen = std::collections::HashSet::new();
        for &id in history {
            if seen.insert(id) {
                if let Some(v) = logits.get_mut(id as usize) {
                    *v -= self.penalty;
                }
            }
        }
    }
}

/// Per-token logit offsets (OpenAI `logit_bias`), added before any other
/// processing. Lives in the shared penalty chain so BOTH samplers honor
/// it — a bias changes the argmax, so greedy must see it too.
pub struct LogitBias {
    bias: std::collections::HashMap<u32, f32>,
}

impl LogitBias {
    /// Creates the processor. An empty map is a no-op.
    pub fn new(bias: std::collections::HashMap<u32, f32>) -> Self {
        LogitBias { bias }
    }
}

impl LogitsProcessor for LogitBias {
    fn process(&self, logits: &mut [f32], _history: &[u32]) {
        for (&id, &b) in &self.bias {
            if let Some(v) = logits.get_mut(id as usize) {
                *v += b;
            }
        }
    }
}

/// Min-p filtering: keeps tokens whose probability is at least `p` times
/// the top token's probability. Works in logit space — the condition
/// `prob(i)/prob(max) >= p` is exactly `logit(i) >= logit(max) + ln(p)` —
/// so no softmax is needed. Multinomial-chain only: the filter can never
/// change an argmax.
pub struct MinP {
    p: f32,
}

impl MinP {
    /// Creates the processor. `p <= 0` disables the filter.
    pub fn new(p: f32) -> Self {
        MinP { p }
    }
}

impl LogitsProcessor for MinP {
    fn process(&self, logits: &mut [f32], _history: &[u32]) {
        if self.p <= 0.0 || logits.is_empty() {
            return;
        }
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if !max.is_finite() {
            return;
        }
        let threshold = max + self.p.ln();
        for v in logits.iter_mut() {
            if *v < threshold {
                *v = f32::NEG_INFINITY;
            }
        }
    }
}

/// Keeps the `k` largest logits and sets the rest to `-inf`.
pub struct TopK {
    k: usize,
}

impl TopK {
    /// Creates the processor. `k == 0` disables the filter.
    pub fn new(k: usize) -> Self {
        TopK { k }
    }
}

impl LogitsProcessor for TopK {
    fn process(&self, logits: &mut [f32], _history: &[u32]) {
        if self.k == 0 || self.k >= logits.len() {
            return;
        }
        // Threshold = k-th largest logit.
        let mut sorted: Vec<f32> = logits.to_vec();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let threshold = sorted[self.k - 1];
        // Strict `<` keeps ties at the threshold, so slightly more than k
        // logits may survive when logits are exactly equal — harmless.
        for v in logits.iter_mut() {
            if *v < threshold {
                *v = f32::NEG_INFINITY;
            }
        }
    }
}

/// Nucleus (top-p) filtering: sorts the logits descending, walks the
/// cumulative softmax mass and cuts the tail once it exceeds `p`. The
/// highest-probability token is always kept.
///
/// Costs O(vocab log vocab) per step due to the sort — acceptable for the
/// CPU-side sampler (one readback per token); a GPU-side kernel is a later
/// phase if this shows up in profiles.
pub struct TopP {
    p: f32,
}

impl TopP {
    /// Creates the processor. `p >= 1.0` disables the filter.
    pub fn new(p: f32) -> Self {
        TopP { p }
    }
}

impl LogitsProcessor for TopP {
    fn process(&self, logits: &mut [f32], _history: &[u32]) {
        if self.p >= 1.0 || logits.is_empty() {
            return;
        }
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));

        // Cumulative softmax mass in descending order.
        let max = logits[idx[0]];
        let mut total = 0.0f32;
        for &i in &idx {
            total += (logits[i] - max).exp();
        }
        let mut cumulative = 0.0f32;
        let mut keep = idx.len();
        for (rank, &i) in idx.iter().enumerate() {
            cumulative += (logits[i] - max).exp();
            if cumulative / total > self.p {
                // Keep this token (HF keeps the first that crosses the
                // threshold) and cut everything after it.
                keep = rank + 1;
                break;
            }
        }
        for &i in &idx[keep..] {
            logits[i] = f32::NEG_INFINITY;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repetition_penalty_reduces_probability_both_signs() {
        // Positive logit: divided by penalty -> smaller.
        // Negative logit: multiplied by penalty -> more negative.
        let p = RepetitionPenalty::new(2.0);
        let mut logits = vec![4.0f32, -4.0, 1.0];
        p.process(&mut logits, &[0, 1]);
        assert_eq!(logits[0], 2.0);
        assert_eq!(logits[1], -8.0);
        assert_eq!(logits[2], 1.0, "untouched token");
    }

    #[test]
    fn repetition_penalty_one_is_noop() {
        let p = RepetitionPenalty::new(1.0);
        let mut logits = vec![4.0f32, -4.0];
        p.process(&mut logits, &[0, 1]);
        assert_eq!(logits, vec![4.0, -4.0]);
    }

    #[test]
    fn repetition_penalty_applies_once_per_unique_token() {
        // A token repeated in history is penalized exactly once (HF
        // reference), never penalty^count.
        let p = RepetitionPenalty::new(2.0);
        let mut repeated = vec![4.0f32, -4.0];
        p.process(&mut repeated, &[0, 0, 0, 1, 1]);
        let mut once = vec![4.0f32, -4.0];
        p.process(&mut once, &[0, 1]);
        assert_eq!(repeated, once);
        assert_eq!(repeated, vec![2.0, -8.0]);
    }

    #[test]
    fn frequency_penalty_scales_with_count() {
        let p = FrequencyPenalty::new(0.5);
        let mut logits = vec![10.0f32, 10.0, 10.0];
        p.process(&mut logits, &[0, 0, 1]);
        assert_eq!(logits[0], 9.0, "count 2 -> -1.0");
        assert_eq!(logits[1], 9.5, "count 1 -> -0.5");
        assert_eq!(logits[2], 10.0);
    }

    #[test]
    fn presence_penalty_applies_once() {
        let p = PresencePenalty::new(0.5);
        let mut logits = vec![10.0f32, 10.0, 10.0];
        p.process(&mut logits, &[0, 0, 1]);
        assert_eq!(logits[0], 9.5);
        assert_eq!(logits[1], 9.5);
        assert_eq!(logits[2], 10.0);
    }

    #[test]
    fn top_k_keeps_exactly_k() {
        let p = TopK::new(2);
        let mut logits = vec![1.0f32, 5.0, 3.0, 2.0];
        p.process(&mut logits, &[]);
        let finite: Vec<_> = logits.iter().filter(|v| v.is_finite()).collect();
        assert_eq!(finite.len(), 2);
        assert!(logits[1].is_finite() && logits[2].is_finite());
    }

    #[test]
    fn top_p_cuts_tail() {
        // logits [2, 1, 0, -10] -> softmax mass ≈ [0.64, 0.24, 0.09, ~0].
        // p = 0.8 keeps the first two (0.64 <= 0.8, 0.88 > 0.8 at rank 1).
        let p = TopP::new(0.8);
        let mut logits = vec![2.0f32, 1.0, 0.0, -10.0];
        p.process(&mut logits, &[]);
        assert!(logits[0].is_finite());
        assert!(logits[1].is_finite());
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    #[test]
    fn top_p_always_keeps_argmax() {
        let p = TopP::new(0.01);
        let mut logits = vec![0.0f32, 10.0, 5.0];
        p.process(&mut logits, &[]);
        assert!(logits[1].is_finite());
    }

    #[test]
    fn temperature_scales() {
        let p = TemperatureScaler::new(2.0);
        let mut logits = vec![4.0f32, -2.0];
        p.process(&mut logits, &[]);
        assert_eq!(logits, vec![2.0, -1.0]);
    }

    #[test]
    fn logit_bias_offsets_named_tokens() {
        let bias = std::collections::HashMap::from([(0u32, -100.0f32), (2, 3.0)]);
        let p = LogitBias::new(bias);
        let mut logits = vec![5.0f32, 1.0, 1.0];
        p.process(&mut logits, &[]);
        assert_eq!(logits, vec![-95.0, 1.0, 4.0]);
    }

    #[test]
    fn min_p_keeps_relative_mass() {
        // p = 0.5 with max logit 2.0 → threshold 2.0 + ln(0.5) ≈ 1.307.
        let p = MinP::new(0.5);
        let mut logits = vec![2.0f32, 1.5, 1.0, -3.0];
        p.process(&mut logits, &[]);
        assert!(logits[0].is_finite());
        assert!(logits[1].is_finite());
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    #[test]
    fn min_p_zero_is_noop() {
        let p = MinP::new(0.0);
        let mut logits = vec![2.0f32, -10.0];
        p.process(&mut logits, &[]);
        assert_eq!(logits, vec![2.0, -10.0]);
    }

    #[test]
    fn chain_applies_in_order() {
        let mut chain = LogitsProcessorChain::new();
        chain.push(RepetitionPenalty::new(2.0));
        chain.push(TemperatureScaler::new(2.0));
        let mut logits = vec![4.0f32, 8.0];
        chain.process(&mut logits, &[0]);
        // token 0: 4/2 (penalty) then /2 (temperature) = 1; token 1: 8/2 = 4.
        assert_eq!(logits, vec![1.0, 4.0]);
    }
}
