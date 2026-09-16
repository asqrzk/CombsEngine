//! The display: a status block that redraws in place, and logs that
//! stream underneath it.
//!
//! Two rules keep this honest on every machine it will meet.
//!
//! A terminal is not guaranteed. Piped into a file, run from a CI step,
//! or opened through the pod gateway's tty, stdout may not accept cursor
//! control — so every redraw asks `is_term()` first and falls back to
//! plain appended lines. The information is identical either way; only
//! the motion is lost.
//!
//! And colour is decoration, never meaning. Every state says its name in
//! words as well: a reader on a monochrome terminal, or one who cannot
//! distinguish the colours, must lose nothing.

use console::{style, Term};

/// A block of lines that can be rewritten in place.
///
/// `drawn` is how many lines the last paint left on screen, which is the
/// only state a redraw needs: clear that many, write the new ones. When
/// stdout is not a terminal the count stays zero and nothing is ever
/// cleared, so a log file reads as an append-only transcript.
pub struct Status {
    term: Term,
    drawn: usize,
    live: bool,
}

impl Status {
    pub fn new() -> Self {
        let term = Term::stdout();
        let live = term.is_term();
        Self { term, drawn: 0, live }
    }

    /// Replace the block with these lines.
    pub fn paint(&mut self, lines: &[String]) {
        if self.live && self.drawn > 0 {
            let _ = self.term.clear_last_lines(self.drawn);
        }
        for l in lines {
            println!("{l}");
        }
        self.drawn = if self.live { lines.len() } else { 0 };
    }

    /// A frame of motion: the same block, redrawn to show time passing.
    ///
    /// Where there is a terminal this replaces the block in place. Where
    /// there is not it prints NOTHING — a transcript gains nothing from
    /// twenty copies of one header differing only in how many
    /// milliseconds have passed, and that is precisely what buried the
    /// kit's log lines the first time a measurement ran piped to a file.
    pub fn progress(&mut self, lines: &[String]) {
        if should_draw(self.live, true) {
            self.paint(lines);
        }
    }

    /// Leave the current block on screen and stop managing it, so
    /// anything printed afterwards appends below rather than overwriting.
    pub fn settle(&mut self) {
        self.drawn = 0;
    }

    /// A log line from a child, printed under the status block. The block
    /// is cleared first so the line lands above it, then repainted by the
    /// next `paint` — which is what makes logs scroll while the status
    /// stays put.
    pub fn log(&mut self, prefix: &str, line: &str) {
        if self.live && self.drawn > 0 {
            let _ = self.term.clear_last_lines(self.drawn);
            self.drawn = 0;
        }
        println!("{} {}", style(format!("{prefix:>8}")).dim(), line);
    }
}

impl Default for Status {
    fn default() -> Self {
        Self::new()
    }
}

/// Content is drawn wherever a command runs; motion is drawn only where
/// there is a terminal that can move. These were one call once, and a
/// piped run printed its header again for every log line beneath it.
pub fn should_draw(live: bool, motion: bool) -> bool {
    live || !motion
}

/// A state's colour AND its word. The word is the contract; the colour
/// is help for the eye that can use it.
pub fn mark(state: &str) -> String {
    match state {
        "up" | "ok" | "pass" | "ready" => style(format!("● {state}")).green().to_string(),
        "down" | "stopped" => style(format!("○ {state}")).dim().to_string(),
        "starting" | "loading" => style(format!("◐ {state}")).yellow().to_string(),
        "fail" | "refused" | "dead" => style(format!("✗ {state}")).red().to_string(),
        other => format!("· {other}"),
    }
}

pub fn head(text: &str) -> String {
    style(text).bold().to_string()
}

pub fn dim(text: &str) -> String {
    style(text).dim().to_string()
}

pub fn warn(text: &str) -> String {
    style(text).yellow().to_string()
}

pub fn bad(text: &str) -> String {
    style(text).red().to_string()
}

pub fn good(text: &str) -> String {
    style(text).green().to_string()
}

/// Bytes as the size a person reads. Binary units, because every memory
/// figure the engine reports is binary and mixing the two is how a
/// "16 GB" machine appears to hold a 16 GiB model.
pub fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / 1_073_741_824.0)
}

pub fn mib(bytes: u64) -> String {
    format!("{:.0} MiB", bytes as f64 / 1_048_576.0)
}

/// A duration a person reads, from milliseconds.
pub fn ms(v: f64) -> String {
    if v >= 1000.0 {
        format!("{:.1} s", v / 1000.0)
    } else {
        format!("{v:.0} ms")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Colour is decoration; the word is the contract. A reader who
    /// cannot see colour must still learn the state, so every mark
    /// carries its name in text.
    #[test]
    fn every_mark_says_its_state_in_words() {
        for s in ["up", "down", "starting", "fail", "whatever"] {
            let m = mark(s);
            assert!(m.contains(s), "the mark for {s:?} does not name it: {m:?}");
        }
    }

    /// Binary units throughout: the engine's memory figures are binary,
    /// and mixing them is how a machine appears to hold a model it cannot.
    #[test]
    fn sizes_are_binary_units() {
        assert_eq!(gib(1_073_741_824), "1.0 GiB");
        assert_eq!(mib(1_048_576), "1 MiB");
    }

    /// Motion belongs to terminals; content belongs everywhere. A run
    /// piped to a file must still say what it did — and must not say it
    /// once per line of somebody else's log.
    #[test]
    fn motion_is_drawn_only_where_it_can_move() {
        assert!(should_draw(true, true), "a terminal shows motion");
        assert!(!should_draw(false, true), "a file is spared motion");
        assert!(should_draw(true, false), "content is always drawn");
        assert!(should_draw(false, false), "content is always drawn");
    }

    #[test]
    fn durations_turn_over_at_a_second() {
        assert_eq!(ms(149.0), "149 ms");
        assert_eq!(ms(4725.0), "4.7 s");
    }
}
