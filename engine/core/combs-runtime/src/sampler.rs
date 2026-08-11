//! Pluggable CPU-side samplers.
//!
//! The engine reads the logits row back from the device once per decode step
//! and hands the mutable slice to a [`Sampler`], which applies its
//! [`LogitsProcessorChain`] (penalties → temperature → top-k → top-p) and
//! picks the next token. GPU-side sampling kernels are a later phase.

use crate::logits::{
    FrequencyPenalty, LogitBias, LogitsProcessorChain, MinP, PresencePenalty,
    RepetitionPenalty, TemperatureScaler, TopK, TopP,
};

/// Picks the next token id from a mutable logits row (vocab-sized).
///
/// `history` is the full context: prompt tokens + tokens generated so far.
pub trait Sampler: Send {
    /// Samples one token, mutating `logits` in place.
    fn sample(&mut self, logits: &mut [f32], history: &[u32]) -> SampleOutcome;
}

/// One sampled token, with log-probability capture when requested.
#[derive(Debug, Clone)]
pub struct SampleOutcome {
    /// The sampled token id.
    pub token: u32,
    /// Log-probability capture; `None` unless the request asked for it.
    pub logprobs: Option<TokenLogprobs>,
}

impl SampleOutcome {
    fn plain(token: u32) -> Self {
        SampleOutcome {
            token,
            logprobs: None,
        }
    }
}

/// Log-probability capture for one sampled token: ln P under the
/// distribution the sampler actually drew from (after its full processor
/// chain — penalties/bias for greedy, plus temperature and filters for
/// multinomial).
#[derive(Debug, Clone)]
pub struct TokenLogprobs {
    /// ln P(sampled token).
    pub logprob: f32,
    /// Top-n `(token id, ln P)` alternatives, descending; empty when the
    /// request asked for the sampled token's logprob only.
    pub top: Vec<(u32, f32)>,
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
    /// Min-p probability floor relative to the top token
    /// (`None`/`Some(<= 0.0)` = disabled; multinomial only — the filter
    /// can never change an argmax).
    pub min_p: Option<f32>,
    /// Per-token logit offsets (OpenAI `logit_bias`), applied before all
    /// other processing so both samplers honor them.
    pub logit_bias: Option<std::collections::HashMap<u32, f32>>,
    /// Capture per-token log-probabilities: `Some(n)` returns the sampled
    /// token's logprob plus the top-n alternatives (`n` may be 0).
    pub logprobs: Option<usize>,
}

/// Builds the penalty-only chain shared by both samplers.
fn penalty_chain(params: &SamplingParams) -> LogitsProcessorChain {
    let mut chain = LogitsProcessorChain::new();
    if let Some(bias) = &params.logit_bias {
        if !bias.is_empty() {
            chain.push(LogitBias::new(bias.clone()));
        }
    }
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
    logprobs: Option<usize>,
}

impl GreedySampler {
    /// Plain argmax with no processors.
    pub fn new() -> Self {
        GreedySampler {
            chain: LogitsProcessorChain::new(),
            logprobs: None,
        }
    }

    /// Argmax with the penalty processors from `params`.
    pub fn with_params(params: &SamplingParams) -> Self {
        GreedySampler {
            chain: penalty_chain(params),
            logprobs: params.logprobs,
        }
    }
}

impl Default for GreedySampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for GreedySampler {
    fn sample(&mut self, logits: &mut [f32], history: &[u32]) -> SampleOutcome {
        self.chain.process(logits, history);
        let token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        match self.logprobs {
            // Log-softmax only on request — the greedy path stays free.
            Some(n) => SampleOutcome {
                token,
                logprobs: Some(capture_from_logits(logits, token, n)),
            },
            None => SampleOutcome::plain(token),
        }
    }
}

/// Log-softmax capture over a processed logits row.
fn capture_from_logits(logits: &[f32], token: u32, top_n: usize) -> TokenLogprobs {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&v| (v - max).exp()).sum();
    let lse = max + sum.ln();
    TokenLogprobs {
        logprob: logits.get(token as usize).copied().unwrap_or(f32::NEG_INFINITY) - lse,
        top: top_n_ids(logits, top_n)
            .into_iter()
            .map(|i| (i as u32, logits[i] - lse))
            .collect(),
    }
}

/// Indices of the `n` largest finite values, descending (small-n heap
/// select: O(vocab · log n)).
fn top_n_ids(values: &[f32], n: usize) -> Vec<usize> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    /// `(value, index)` ordered by `total_cmp` on the value; index ties
    /// prefer the smaller index (evict larger ones from the min-heap).
    struct Item(f32, usize);
    impl PartialEq for Item {
        fn eq(&self, other: &Self) -> bool {
            self.cmp(other) == std::cmp::Ordering::Equal
        }
    }
    impl Eq for Item {}
    impl PartialOrd for Item {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Item {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.0.total_cmp(&other.0).then(other.1.cmp(&self.1))
        }
    }

    if n == 0 {
        return Vec::new();
    }
    let mut heap: BinaryHeap<Reverse<Item>> = BinaryHeap::with_capacity(n + 1);
    for (i, &v) in values.iter().enumerate() {
        if !v.is_finite() {
            continue;
        }
        heap.push(Reverse(Item(v, i)));
        if heap.len() > n {
            heap.pop();
        }
    }
    let mut out: Vec<Item> = heap.into_iter().map(|Reverse(item)| item).collect();
    out.sort_by(|a, b| b.cmp(a));
    out.into_iter().map(|item| item.1).collect()
}

/// Softmax multinomial sampler with a seedable xorshift RNG (no external
/// `rand` dependency).
pub struct MultinomialSampler {
    chain: LogitsProcessorChain,
    rng_state: u64,
    logprobs: Option<usize>,
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
        if let Some(p) = params.min_p {
            chain.push(MinP::new(p));
        }
        MultinomialSampler {
            chain,
            rng_state: seed_or_time(params.seed),
            logprobs: params.logprobs,
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
    fn sample(&mut self, logits: &mut [f32], history: &[u32]) -> SampleOutcome {
        self.chain.process(logits, history);
        // Numerically stable softmax + inverse-CDF draw.
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if !max.is_finite() {
            return SampleOutcome::plain(0);
        }
        let mut sum = 0.0f32;
        for v in logits.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        let mut threshold = self.next_f32() * sum;
        let mut token = (logits.len() - 1) as u32;
        for (i, v) in logits.iter().enumerate() {
            threshold -= *v;
            if threshold <= 0.0 {
                token = i as u32;
                break;
            }
        }
        match self.logprobs {
            // The row now holds exp(logit - max): ln P(i) = ln v[i] - ln sum,
            // so capture is nearly free after the draw.
            Some(n) => {
                let ln_sum = sum.ln();
                let lp = |v: f32| {
                    if v > 0.0 {
                        v.ln() - ln_sum
                    } else {
                        f32::NEG_INFINITY
                    }
                };
                let top = top_n_ids(logits, n)
                    .into_iter()
                    .filter(|&i| logits[i] > 0.0)
                    .map(|i| (i as u32, lp(logits[i])))
                    .collect();
                SampleOutcome {
                    token,
                    logprobs: Some(TokenLogprobs {
                        logprob: lp(logits[token as usize]),
                        top,
                    }),
                }
            }
            None => SampleOutcome::plain(token),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let mut s = GreedySampler::new();
        let mut logits = vec![0.1f32, 0.2, 5.0, -1.0];
        assert_eq!(s.sample(&mut logits, &[]).token, 2);
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
        assert_eq!(s.sample(&mut logits, &[2]).token, 1);
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
            assert_eq!(a.sample(&mut la, &[]).token, b.sample(&mut lb, &[]).token);
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
            assert_eq!(s.sample(&mut logits, &[]).token, 2);
        }
    }

    #[test]
    fn logit_bias_flips_greedy_argmax() {
        let params = SamplingParams {
            logit_bias: Some(std::collections::HashMap::from([(2u32, -100.0f32)])),
            ..Default::default()
        };
        let mut s = GreedySampler::with_params(&params);
        let mut logits = vec![0.1f32, 0.2, 5.0, -1.0];
        assert_eq!(s.sample(&mut logits, &[]).token, 1);
    }

    #[test]
    fn greedy_logprobs_are_log_softmax() {
        let params = SamplingParams {
            logprobs: Some(2),
            ..Default::default()
        };
        let mut s = GreedySampler::with_params(&params);
        let mut logits = vec![1.0f32, 2.0, 3.0];
        let out = s.sample(&mut logits, &[]);
        assert_eq!(out.token, 2);
        let lp = out.logprobs.expect("captured");
        // log-softmax of [1,2,3]: lse = 3 + ln(1 + e^-1 + e^-2).
        let lse = 3.0 + (1.0f32 + (-1.0f32).exp() + (-2.0f32).exp()).ln();
        assert!((lp.logprob - (3.0 - lse)).abs() < 1e-5);
        assert_eq!(lp.top.len(), 2);
        assert_eq!(lp.top[0].0, 2);
        assert_eq!(lp.top[1].0, 1);
        assert!((lp.top[0].1 - (3.0 - lse)).abs() < 1e-5);
    }

    #[test]
    fn min_p_restricts_multinomial_support() {
        let params = SamplingParams {
            temperature: 1.0,
            min_p: Some(0.9),
            seed: Some(3),
            ..Default::default()
        };
        let mut s = MultinomialSampler::new(&params);
        for _ in 0..16 {
            let mut logits = vec![5.0f32, 0.0, -1.0, 4.9];
            // p(top)=~0.5; only ids 0 and 3 clear a 0.9 relative floor.
            let t = s.sample(&mut logits, &[]).token;
            assert!(t == 0 || t == 3, "sampled {t}");
        }
    }

    #[test]
    fn multinomial_logprobs_cover_survivors() {
        let params = SamplingParams {
            temperature: 1.0,
            top_k: Some(2),
            logprobs: Some(4),
            seed: Some(9),
            ..Default::default()
        };
        let mut s = MultinomialSampler::new(&params);
        let mut logits = vec![0.1f32, 4.0, 3.0, -2.0];
        let out = s.sample(&mut logits, &[]);
        let lp = out.logprobs.expect("captured");
        // top-k=2 leaves ids 1 and 2; alternatives exclude masked tokens.
        assert_eq!(lp.top.len(), 2);
        assert_eq!(lp.top[0].0, 1);
        assert_eq!(lp.top[1].0, 2);
        let p: f32 = lp.top.iter().map(|(_, l)| l.exp()).sum();
        assert!((p - 1.0).abs() < 1e-5, "filtered distribution sums to 1");
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
            assert_eq!(s.sample(&mut logits, &[]).token, 2);
        }
    }
}
