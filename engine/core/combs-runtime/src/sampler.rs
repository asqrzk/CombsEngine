//! Pluggable CPU-side samplers.
//!
//! The engine reads the logits row back from the device once per decode step
//! and hands the mutable slice to a [`Sampler`], which applies its
//! [`LogitsProcessorChain`] (penalties → temperature → top-k → top-p) and
//! picks the next token. GPU-side sampling kernels are a later phase.

use crate::logits::{
    FrequencyPenalty, LogitsProcessorChain, PresencePenalty, RepetitionPenalty,
    TemperatureScaler, TopK, TopP,
};

/// Picks the next token id from a mutable logits row (vocab-sized).
///
/// `history` is the full context: prompt tokens + tokens generated so far.
pub trait Sampler: Send {
    /// Samples one token id, mutating `logits` in place.
    fn sample(&mut self, logits: &mut [f32], history: &[u32]) -> u32;
}

/// Sampling parameters for one generation call.
///
/// `Default` is greedy-ish: temperature 0 (argmax), no filters, no penalties.
#[derive(Debug, Clone, Default)]
pub struct SamplingParams {
    /// Sampling temperature; `0.0` = greedy argmax.
    pub temperature: f32,
    /// Top-k cutoff (`None`/`Some(0)` = disabled).
    pub top_k: Option<usize>,
    /// Nucleus threshold (`None`/`Some(>= 1.0)` = disabled).
    pub top_p: Option<f32>,
    /// HF-style repetition penalty (`None`/`Some(1.0)` = disabled).
    pub repetition_penalty: Option<f32>,
    /// OpenAI-style frequency penalty (`None`/`Some(0.0)` = disabled).
    pub frequency_penalty: Option<f32>,
    /// OpenAI-style presence penalty (`None`/`Some(0.0)` = disabled).
    pub presence_penalty: Option<f32>,
    /// RNG seed for reproducible sampling (`None` = seed from system time).
    pub seed: Option<u64>,
}

/// Builds the penalty-only chain shared by both samplers.
fn penalty_chain(params: &SamplingParams) -> LogitsProcessorChain {
    let mut chain = LogitsProcessorChain::new();
    if let Some(p) = params.repetition_penalty {
        chain.push(RepetitionPenalty::new(p));
    }
    if let Some(p) = params.frequency_penalty {
        chain.push(FrequencyPenalty::new(p));
    }
    if let Some(p) = params.presence_penalty {
        chain.push(PresencePenalty::new(p));
    }
    chain
}

/// Picks a sampler for `params` (greedy when `temperature <= 0`).
pub fn sampler_from_params(params: &SamplingParams) -> Box<dyn Sampler> {
    if params.temperature <= 0.0 {
        Box::new(GreedySampler::with_params(params))
    } else {
        Box::new(MultinomialSampler::new(params))
    }
}

/// Argmax sampler — deterministic. Still applies the penalty processors
/// (they can change the argmax); temperature/top-k/top-p are irrelevant to
/// argmax and are skipped.
pub struct GreedySampler {
    chain: LogitsProcessorChain,
}

impl GreedySampler {
    /// Plain argmax with no processors.
    pub fn new() -> Self {
        GreedySampler {
            chain: LogitsProcessorChain::new(),
        }
    }

    /// Argmax with the penalty processors from `params`.
    pub fn with_params(params: &SamplingParams) -> Self {
        GreedySampler {
            chain: penalty_chain(params),
        }
    }
}

impl Default for GreedySampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for GreedySampler {
    fn sample(&mut self, logits: &mut [f32], history: &[u32]) -> u32 {
        self.chain.process(logits, history);
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }
}

/// Softmax multinomial sampler with a seedable xorshift RNG (no external
/// `rand` dependency).
pub struct MultinomialSampler {
    chain: LogitsProcessorChain,
    rng_state: u64,
}

impl MultinomialSampler {
    /// Creates a sampler with the full processor chain from `params`
    /// (penalties → temperature → top-k → top-p).
    pub fn new(params: &SamplingParams) -> Self {
        let mut chain = penalty_chain(params);
        chain.push(TemperatureScaler::new(params.temperature));
        if let Some(k) = params.top_k {
            chain.push(TopK::new(k));
        }
        if let Some(p) = params.top_p {
            chain.push(TopP::new(p));
        }
        MultinomialSampler {
            chain,
            rng_state: seed_or_time(params.seed),
        }
    }

    fn next_f32(&mut self) -> f32 {
        // xorshift64*
        let mut x = self.rng_state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng_state = x;
        let v = x.wrapping_mul(0x2545F4914F6CDD1D);
        // map to (0, 1]
        1.0 - (v >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// splitmix64 finalize, used to scramble a user seed into the RNG state.
fn seed_or_time(seed: Option<u64>) -> u64 {
    let raw = seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E3779B97F4A7C15)
    });
    let mut z = raw.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    (z ^ (z >> 31)) | 1
}

impl Sampler for MultinomialSampler {
    fn sample(&mut self, logits: &mut [f32], history: &[u32]) -> u32 {
        self.chain.process(logits, history);
        // Numerically stable softmax + inverse-CDF draw.
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if !max.is_finite() {
            return 0;
        }
        let mut sum = 0.0f32;
        for v in logits.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        let mut threshold = self.next_f32() * sum;
        for (i, v) in logits.iter().enumerate() {
            threshold -= *v;
            if threshold <= 0.0 {
                return i as u32;
            }
        }
        (logits.len() - 1) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let mut s = GreedySampler::new();
        let mut logits = vec![0.1f32, 0.2, 5.0, -1.0];
        assert_eq!(s.sample(&mut logits, &[]), 2);
    }

    #[test]
    fn greedy_applies_repetition_penalty() {
        // Token 2 has the raw argmax, but it is in the history; a strong
        // penalty should push token 1 ahead.
        let params = SamplingParams {
            repetition_penalty: Some(4.0),
            ..Default::default()
        };
        let mut s = GreedySampler::with_params(&params);
        let mut logits = vec![0.1f32, 2.0, 5.0, -1.0];
        assert_eq!(s.sample(&mut logits, &[2]), 1);
    }

    #[test]
    fn seeded_multinomial_is_reproducible() {
        let params = SamplingParams {
            temperature: 1.0,
            seed: Some(42),
            ..Default::default()
        };
        let mut a = MultinomialSampler::new(&params);
        let mut b = MultinomialSampler::new(&params);
        for _ in 0..16 {
            let mut la = vec![0.1f32, 0.2, 5.0, -1.0, 3.0];
            let mut lb = la.clone();
            assert_eq!(a.sample(&mut la, &[]), b.sample(&mut lb, &[]));
        }
    }

    #[test]
    fn low_temperature_is_nearly_greedy() {
        let params = SamplingParams {
            temperature: 1e-4,
            seed: Some(7),
            ..Default::default()
        };
        let mut s = MultinomialSampler::new(&params);
        for _ in 0..8 {
            let mut logits = vec![0.1f32, 0.2, 5.0, -1.0];
            assert_eq!(s.sample(&mut logits, &[]), 2);
        }
    }

    #[test]
    fn top_k_restricts_support() {
        let params = SamplingParams {
            temperature: 1.0,
            top_k: Some(1),
            seed: Some(1),
            ..Default::default()
        };
        let mut s = MultinomialSampler::new(&params);
        for _ in 0..8 {
            let mut logits = vec![0.1f32, 0.2, 5.0, -1.0];
            assert_eq!(s.sample(&mut logits, &[]), 2);
        }
    }
}
