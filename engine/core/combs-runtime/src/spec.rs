//! Prompt-lookup speculative drafting: find the longest suffix n-gram of
//! the generation history that recurred earlier, and propose the tokens
//! that followed that earlier occurrence. No second model — repetition in
//! real text (code, quotes, lists, structured output) is the signal.

/// Longest recurring suffix n-gram (most recent occurrence wins ties),
/// proposing up to `k` follow-up tokens. Returns `None` when no suffix of
/// length ≥ `min_ngram` recurs — the caller decodes normally instead of
/// guessing. Short minimums trigger on the incidental bigrams every text
/// contains and drown the wins in failed verification rounds; production
/// callers should pass 3+.
pub fn propose(
    history: &[u32],
    k: usize,
    min_ngram: usize,
    max_ngram: usize,
) -> Option<Vec<u32>> {
    let n = history.len();
    let min_len = min_ngram.max(2);
    if n < min_len + 1 || k == 0 {
        return None;
    }
    // Cheap early-out: if the final token never appears earlier, no suffix
    // n-gram can either. Keeps the no-repetition steady state ~O(n).
    let last = history[n - 1];
    if !history[..n - 1].contains(&last) {
        return None;
    }
    let max_len = max_ngram.min(n - 1);
    if max_len < min_len {
        return None;
    }
    for len in (min_len..=max_len).rev() {
        let suffix = &history[n - len..];
        for start in (0..n - len).rev() {
            if &history[start..start + len] == suffix {
                let follow = start + len;
                let take = k.min(n - follow);
                if take == 0 {
                    continue;
                }
                return Some(history[follow..follow + take].to_vec());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proposes_continuation_of_repeated_ngram() {
        // "1 2 3 4 ... 1 2" — suffix [1,2] recurred at 0; 3,4 followed.
        let h = [1, 2, 3, 4, 9, 1, 2];
        assert_eq!(propose(&h, 2, 2, 8), Some(vec![3, 4]));
        assert_eq!(propose(&h, 1, 2, 8), Some(vec![3]));
    }

    #[test]
    fn prefers_longest_match_then_most_recent() {
        // Suffix [7,8,9] matches at 0 (followed by 5); the shorter [8,9]
        // also matches later — longest must win.
        let h = [7, 8, 9, 5, 0, 8, 9, 6, 7, 8, 9];
        assert_eq!(propose(&h, 1, 2, 8), Some(vec![5]));
        // Two occurrences of the bigram suffix [3,4]: most recent (followed
        // by 6) wins over the older one (followed by 5).
        let h2 = [3, 4, 5, 3, 4, 6, 0, 3, 4];
        assert_eq!(propose(&h2, 1, 2, 2), Some(vec![6]));
    }

    #[test]
    fn declines_without_recurrence() {
        assert_eq!(propose(&[1, 2, 3, 4, 5], 4, 2, 8), None);
        assert_eq!(propose(&[1, 2], 4, 2, 8), None);
        assert_eq!(propose(&[], 4, 2, 8), None);
    }

    #[test]
    fn min_ngram_floor_rejects_short_matches() {
        // Only a bigram recurs; a trigram minimum must decline it while
        // the bigram minimum accepts it.
        let h = [1, 2, 9, 8, 1, 2];
        assert_eq!(propose(&h, 2, 2, 8), Some(vec![9, 8]));
        assert_eq!(propose(&h, 2, 3, 8), None);
    }

    #[test]
    fn clamps_to_available_history() {
        // k=8 requested but only 3 tokens follow the match — and the
        // proposal may overlap the suffix itself (self-continuation).
        let h = [5, 6, 7, 5, 6];
        assert_eq!(propose(&h, 8, 2, 8), Some(vec![7, 5, 6]));
        assert_eq!(propose(&h, 2, 2, 8), Some(vec![7, 5]));
    }
}
