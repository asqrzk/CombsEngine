//! Stop-condition detection: stop token ids + stop strings.
//!
//! Stop strings are matched against a rolling window of the incrementally
//! detokenized text, so matches spanning token/chunk boundaries (e.g. "END"
//! arriving as "E", "ND") are caught.

/// Combined stop detector for one generation call.
pub struct StopDetector {
    token_ids: Vec<u32>,
    matcher: StopStringMatcher,
}

impl StopDetector {
    /// Creates a detector from stop token ids (model eos + config extras)
    /// and stop strings.
    pub fn new(token_ids: Vec<u32>, stop_strings: Vec<String>) -> Self {
        StopDetector {
            token_ids,
            matcher: StopStringMatcher::new(stop_strings),
        }
    }

    /// Whether `id` is a stop token.
    pub fn is_stop_token(&self, id: u32) -> bool {
        self.token_ids.contains(&id)
    }

    /// Whether any stop strings are configured.
    pub fn watches_strings(&self) -> bool {
        self.matcher.is_active()
    }

    /// Feeds a newly detokenized text piece. See
    /// [`StopStringMatcher::push`].
    pub fn push_text(&mut self, piece: &str) -> Option<usize> {
        self.matcher.push(piece)
    }
}

/// Rolling-window matcher for stop strings over a text stream.
///
/// Keeps only the last `max(stop.len()) - 1` chars of already-clean text:
/// any stop string that starts earlier than that can no longer end at the
/// stream tail, and since detection runs on every push, earlier occurrences
/// would already have stopped generation.
pub struct StopStringMatcher {
    stops: Vec<String>,
    tail: String,
    /// Maximum chars of clean tail needed to catch boundary-spanning matches.
    keep: usize,
}

impl StopStringMatcher {
    /// Creates a matcher. Empty strings in `stops` are ignored.
    pub fn new(stops: Vec<String>) -> Self {
        let stops: Vec<String> = stops.into_iter().filter(|s| !s.is_empty()).collect();
        let max_len = stops.iter().map(|s| s.chars().count()).max().unwrap_or(0);
        StopStringMatcher {
            stops,
            tail: String::new(),
            keep: max_len.saturating_sub(1),
        }
    }

    /// Whether any stop strings are configured.
    pub fn is_active(&self) -> bool {
        !self.stops.is_empty()
    }

    /// Appends the newly detokenized `piece` and returns `Some(byte_idx)` —
    /// the offset **within `piece`** where the earliest stop string begins —
    /// if a stop string is now present in the rolling tail; `None` otherwise.
    /// Callers should emit `piece[..byte_idx]` and stop.
    pub fn push(&mut self, piece: &str) -> Option<usize> {
        if !self.is_active() {
            return None;
        }
        let base = self.tail.len();
        self.tail.push_str(piece);

        // Earliest (leftmost) match of any stop string in the tail.
        let hit = self
            .stops
            .iter()
            .filter_map(|s| self.tail.find(s))
            .min();

        // Trim the tail, keeping enough context for boundary-spanning
        // matches on the next push.
        if self.tail.chars().count() > self.keep {
            let split = self
                .tail
                .char_indices()
                .nth(self.tail.chars().count() - self.keep)
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.tail.drain(..split);
        }

        hit.map(|idx| idx.saturating_sub(base).min(piece.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_within_one_piece() {
        let mut m = StopStringMatcher::new(vec!["END".to_string()]);
        assert_eq!(m.push("stop here END now"), Some(10));
    }

    #[test]
    fn matches_across_chunk_boundaries() {
        let mut m = StopStringMatcher::new(vec!["END".to_string()]);
        assert_eq!(m.push("E"), None);
        assert_eq!(m.push("ND"), Some(0));
    }

    #[test]
    fn match_starting_inside_piece() {
        let mut m = StopStringMatcher::new(vec!["END".to_string()]);
        assert_eq!(m.push("abc E"), None);
        // "ND" completes the match; the part of the piece before the stop
        // string is empty (the 'E' was already emitted), so idx = 0.
        assert_eq!(m.push("ND rest"), Some(0));
    }

    #[test]
    fn emits_piece_prefix_before_stop() {
        let mut m = StopStringMatcher::new(vec!["END".to_string()]);
        assert_eq!(m.push("hello END"), Some(6));
    }

    #[test]
    fn no_match_returns_none() {
        let mut m = StopStringMatcher::new(vec!["END".to_string()]);
        assert_eq!(m.push("hello "), None);
        assert_eq!(m.push("world"), None);
        assert_eq!(m.push("EN"), None);
        // "EN" + "D" spans three pushes.
        assert_eq!(m.push("D"), Some(0));
    }

    #[test]
    fn inactive_when_no_stops() {
        let mut m = StopStringMatcher::new(vec![]);
        assert!(!m.is_active());
        assert_eq!(m.push("END"), None);
    }

    #[test]
    fn detector_checks_token_ids() {
        let d = StopDetector::new(vec![1, 2], vec![]);
        assert!(d.is_stop_token(1));
        assert!(!d.is_stop_token(3));
        assert!(!d.watches_strings());
    }
}
