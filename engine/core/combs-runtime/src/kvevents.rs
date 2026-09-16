//! KV events: the pages this engine's sessions hold, published as a log
//! a cache-aware router's indexer can apply.
//!
//! A session owns its KV arena, so two sessions sharing a prefix hold two
//! copies of the same tokens. What a router needs to know is not how many
//! copies exist but whether this engine can serve a prefix from cache at
//! all — so pages are named by CONTENT: every full page of a session's
//! history gets a key chained from the page before it, and each key is
//! reported once however many sessions hold it. `Stored` when the first
//! holder fills a page, `Removed` when the last one lets it go (a
//! rollback below the page, an LRU eviction, a released session), and
//! `Cleared` when the whole table goes at once.
//!
//! The keys are opaque to the reader: an indexer computes its own request
//! keys from the token ids a `Stored` event carries and keeps a map from
//! ours to its own, so `Removed` needs nothing but the keys. Order is the
//! contract that lets it: removals first, then each run of new pages
//! after the page it hangs off — a child never arrives before its parent.
//!
//! Readers page the log by sequence number. One that falls behind the
//! ring is not handed a gap: it gets a resync — `Cleared` followed by one
//! `Stored` per page still held, parents first.
//!
//! What this does NOT claim: that a page is reusable by any caller. Reuse
//! is per named session (the engine rolls a session's own cache back to
//! its longest common prefix); the log says which token pages are on this
//! device, which is what a router scores a pod by.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Condvar, Mutex};
use std::time::Duration;

use crate::time::Instant;

/// `COMBS_KV_EVENTS`, parsed: shut unless told `1`. A page log costs
/// memory proportional to what the sessions hold and publishes token ids
/// to whoever can read the route, so an engine nobody routes to pays
/// nothing and exposes nothing.
pub(crate) fn kv_events_door(env: Option<&str>) -> bool {
    env == Some("1")
}

/// `COMBS_KV_EVENTS`, read once per process.
pub(crate) fn kv_events_enabled() -> bool {
    static DOOR: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DOOR.get_or_init(|| kv_events_door(std::env::var("COMBS_KV_EVENTS").ok().as_deref()))
}

/// The content key of one KV page: FNV-1a 64 over the parent's key and
/// the page's token ids, so a key names the whole prefix behind it.
/// Never zero — zero is the reader's "no parent".
pub fn kv_page_hash(parent: u64, tokens: &[u32]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET;
    for b in parent.to_le_bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(PRIME);
    }
    for t in tokens {
        for b in t.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(PRIME);
        }
    }
    if h == 0 { 1 } else { h }
}

/// The keys of every FULL page of a history, chained from the first. A
/// trailing partial page has no key: nothing can reuse it by content
/// until it fills.
pub fn page_hashes(history: &[u32], page_size: usize) -> Vec<u64> {
    if page_size == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(history.len() / page_size);
    let mut parent = 0u64;
    for page in history.chunks_exact(page_size) {
        let h = kv_page_hash(parent, page);
        out.push(h);
        parent = h;
    }
    out
}

/// One change to what this engine holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KvEvent {
    /// A run of consecutive pages nobody held before: `parent` is the key
    /// of the page in front of the run (`None` when it starts a history),
    /// `token_ids` exactly these pages' tokens in order.
    Stored {
        hashes: Vec<u64>,
        parent: Option<u64>,
        token_ids: Vec<u32>,
    },
    /// Pages no session holds any more.
    Removed { hashes: Vec<u64> },
    /// Every page is gone.
    Cleared,
}

/// An event with its place in the log.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeqEvent {
    pub seq: u64,
    pub event: KvEvent,
}

/// What a reader gets for a sequence number: the events after it, or —
/// when the ring dropped events it never saw — a resync, every line
/// stamped with the newest sequence number.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvEventsPage {
    pub events: Vec<SeqEvent>,
    pub resync: bool,
}

impl KvEventsPage {
    /// The sequence number a reader passes next; `since` itself when
    /// nothing came.
    pub fn newest(&self, since: u64) -> u64 {
        self.events.last().map_or(since, |e| e.seq)
    }
}

/// What the sessions held at the last look, diffed against what they
/// hold now.
pub struct KvIndex {
    page_size: usize,
    held: HashSet<u64>,
}

impl KvIndex {
    pub fn new(page_size: usize) -> Self {
        KvIndex {
            page_size,
            held: HashSet::new(),
        }
    }

    /// Tokens per page — the block size a reader must chunk by.
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Diffs the sessions' histories against the last look. Removals come
    /// first; then, session by session, each run of pages nobody held,
    /// hung off the page in front of it — which is either held already or
    /// was stored earlier in this same batch, so a chained reader never
    /// sees a child before its parent.
    pub fn reconcile<'a>(&mut self, histories: impl IntoIterator<Item = &'a [u32]>) -> Vec<KvEvent> {
        let ps = self.page_size;
        let histories: Vec<&[u32]> = histories.into_iter().collect();
        let lists: Vec<Vec<u64>> = histories.iter().map(|h| page_hashes(h, ps)).collect();
        let now: HashSet<u64> = lists.iter().flatten().copied().collect();

        let mut events = Vec::new();
        let mut removed: Vec<u64> = self.held.difference(&now).copied().collect();
        if !removed.is_empty() {
            removed.sort_unstable();
            events.push(KvEvent::Removed { hashes: removed });
        }

        let mut fresh: HashSet<u64> = HashSet::new();
        for (hashes, history) in lists.iter().zip(histories.iter()) {
            let mut i = 0;
            while i < hashes.len() {
                if self.held.contains(&hashes[i]) || fresh.contains(&hashes[i]) {
                    i += 1;
                    continue;
                }
                let start = i;
                while i < hashes.len()
                    && !self.held.contains(&hashes[i])
                    && !fresh.contains(&hashes[i])
                {
                    fresh.insert(hashes[i]);
                    i += 1;
                }
                events.push(KvEvent::Stored {
                    hashes: hashes[start..i].to_vec(),
                    parent: (start > 0).then(|| hashes[start - 1]),
                    token_ids: history[start * ps..i * ps].to_vec(),
                });
            }
        }
        self.held = now;
        events
    }

    /// Forgets every page at once.
    pub fn clear(&mut self) -> Vec<KvEvent> {
        self.held.clear();
        vec![KvEvent::Cleared]
    }
}

/// A page a reader can be resynced to: its parent, its tokens, and its
/// depth in the chain so a resync lists parents first.
#[derive(Clone, Debug)]
struct HeldPage {
    parent: Option<u64>,
    tokens: Vec<u32>,
    depth: usize,
}

struct LogInner {
    ring: VecDeque<SeqEvent>,
    /// Tokens carried by the `Stored` events still in the ring.
    ring_tokens: usize,
    next_seq: u64,
    /// The newest sequence number the ring has dropped; zero while it has
    /// dropped nothing.
    dropped_through: u64,
    held: HashMap<u64, HeldPage>,
}

/// The reader side: a bounded ring of sequenced events plus the pages
/// currently held, so a reader behind the ring can be resynced rather
/// than silently given a gap.
pub struct KvEventLog {
    inner: Mutex<LogInner>,
    landed: Condvar,
    cap_events: usize,
    cap_tokens: usize,
}

impl KvEventLog {
    /// A log keeping at most `cap_events` events and `cap_tokens` tokens
    /// of stored-page content; the newest event always stays.
    pub fn new(cap_events: usize, cap_tokens: usize) -> Self {
        KvEventLog {
            inner: Mutex::new(LogInner {
                ring: VecDeque::new(),
                ring_tokens: 0,
                next_seq: 1,
                dropped_through: 0,
                held: HashMap::new(),
            }),
            landed: Condvar::new(),
            cap_events: cap_events.max(1),
            cap_tokens,
        }
    }

    /// Appends events in order, stamping each with the next sequence
    /// number, and wakes every waiting reader.
    pub fn record(&self, events: Vec<KvEvent>) {
        if events.is_empty() {
            return;
        }
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        for event in events {
            match &event {
                KvEvent::Stored {
                    hashes,
                    parent,
                    token_ids,
                } => {
                    let ps = if hashes.is_empty() {
                        0
                    } else {
                        token_ids.len() / hashes.len()
                    };
                    let mut parent = *parent;
                    let mut depth = parent
                        .and_then(|p| g.held.get(&p))
                        .map_or(0, |held| held.depth + 1);
                    for (i, h) in hashes.iter().enumerate() {
                        g.held.insert(
                            *h,
                            HeldPage {
                                parent,
                                tokens: token_ids[i * ps..(i + 1) * ps].to_vec(),
                                depth,
                            },
                        );
                        parent = Some(*h);
                        depth += 1;
                    }
                    g.ring_tokens += token_ids.len();
                }
                KvEvent::Removed { hashes } => {
                    for h in hashes {
                        g.held.remove(h);
                    }
                }
                KvEvent::Cleared => g.held.clear(),
            }
            let seq = g.next_seq;
            g.next_seq += 1;
            g.ring.push_back(SeqEvent { seq, event });
            while g.ring.len() > 1
                && (g.ring.len() > self.cap_events || g.ring_tokens > self.cap_tokens)
            {
                if let Some(old) = g.ring.pop_front() {
                    if let KvEvent::Stored { token_ids, .. } = &old.event {
                        g.ring_tokens -= token_ids.len();
                    }
                    g.dropped_through = old.seq;
                }
            }
        }
        self.landed.notify_all();
    }

    /// The events after `since`, waiting up to `wait` for one to land
    /// when none has. A reader whose `since` lies behind what the ring
    /// still holds gets a resync instead.
    pub fn since(&self, since: u64, wait: Duration) -> KvEventsPage {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let deadline = Instant::now() + wait;
        while g.next_seq - 1 <= since {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            let (guard, _) = self
                .landed
                .wait_timeout(g, deadline - now)
                .unwrap_or_else(|e| e.into_inner());
            g = guard;
        }
        let newest = g.next_seq - 1;
        if since < g.dropped_through {
            let mut pages: Vec<(&u64, &HeldPage)> = g.held.iter().collect();
            pages.sort_by_key(|(h, p)| (p.depth, **h));
            let mut events = vec![SeqEvent {
                seq: newest,
                event: KvEvent::Cleared,
            }];
            events.extend(pages.into_iter().map(|(h, p)| SeqEvent {
                seq: newest,
                event: KvEvent::Stored {
                    hashes: vec![*h],
                    parent: p.parent,
                    token_ids: p.tokens.clone(),
                },
            }));
            return KvEventsPage {
                events,
                resync: true,
            };
        }
        KvEventsPage {
            events: g.ring.iter().filter(|e| e.seq > since).cloned().collect(),
            resync: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stored(hashes: &[u64], parent: Option<u64>, token_ids: &[u32]) -> KvEvent {
        KvEvent::Stored {
            hashes: hashes.to_vec(),
            parent,
            token_ids: token_ids.to_vec(),
        }
    }

    #[test]
    fn the_events_door_is_shut_unless_told_one() {
        assert!(!kv_events_door(None), "a router nobody asked for costs nothing");
        assert!(!kv_events_door(Some("0")));
        assert!(!kv_events_door(Some("")));
        assert!(kv_events_door(Some("1")));
    }

    #[test]
    fn a_page_key_chains_from_its_parent_and_is_never_zero() {
        let page: Vec<u32> = (0..16).collect();
        let first = kv_page_hash(0, &page);
        assert_ne!(first, 0, "zero is the reader's 'no parent'");
        assert_ne!(
            first,
            kv_page_hash(first, &page),
            "the same tokens after a different parent are a different page"
        );
        assert_eq!(first, kv_page_hash(0, &page), "a key is a function of its content");

        let history: Vec<u32> = (0..50).collect();
        let keys = page_hashes(&history, 16);
        assert_eq!(keys.len(), 3, "50 tokens hold three full pages of 16");
        assert_eq!(keys[0], kv_page_hash(0, &history[..16]));
        assert_eq!(keys[1], kv_page_hash(keys[0], &history[16..32]));
        assert_eq!(keys[2], kv_page_hash(keys[1], &history[32..48]));
        assert!(page_hashes(&history, 0).is_empty());
    }

    #[test]
    fn reconcile_names_each_page_once_parent_first_and_drops_only_what_no_session_holds() {
        let ps = 4;
        let short: Vec<u32> = (0..9).collect(); // two full pages, one partial
        let long: Vec<u32> = (0..13).collect(); // three full pages, sharing both
        let keys = page_hashes(&long, ps);
        let mut index = KvIndex::new(ps);

        let events = index.reconcile([long.as_slice(), short.as_slice()]);
        assert_eq!(
            events,
            vec![stored(&keys[..3], None, &long[..12])],
            "one run from the first holder; the second holder adds nothing"
        );

        let events = index.reconcile([short.as_slice(), long.as_slice()]);
        assert!(events.is_empty(), "nothing changed hands: {events:?}");

        // The short session grows past the long one — a decode crossing
        // page boundaries looks exactly like this.
        let grown: Vec<u32> = (0..20).collect();
        let grown_keys = page_hashes(&grown, ps);
        let events = index.reconcile([long.as_slice(), grown.as_slice()]);
        assert_eq!(
            events,
            vec![stored(&grown_keys[3..5], Some(keys[2]), &grown[12..20])],
            "the new run hangs off the page before it"
        );

        // The long session leaves; every page it held is still held.
        let events = index.reconcile([grown.as_slice()]);
        assert!(events.is_empty(), "a page with another holder is not removed: {events:?}");

        // A rollback below the second page: what nobody holds goes, and
        // the branch after it is one run parented on the page that stayed.
        let branch: Vec<u32> = (0..4).chain(100..108).collect();
        let branch_keys = page_hashes(&branch, ps);
        let events = index.reconcile([branch.as_slice()]);
        let mut gone = grown_keys[1..5].to_vec();
        gone.sort_unstable();
        assert_eq!(
            events,
            vec![
                KvEvent::Removed { hashes: gone },
                stored(&branch_keys[1..3], Some(keys[0]), &branch[4..12]),
            ]
        );

        assert_eq!(index.clear(), vec![KvEvent::Cleared]);
        let events = index.reconcile([branch.as_slice()]);
        assert_eq!(
            events,
            vec![stored(&branch_keys, None, &branch)],
            "after a clear every page is new again"
        );
    }

    #[test]
    fn a_reader_behind_the_ring_is_resynced_to_what_is_held_parents_first() {
        let ps = 2;
        let log = KvEventLog::new(3, usize::MAX);
        let mut index = KvIndex::new(ps);
        let a: Vec<u32> = (0..6).collect();
        let b: Vec<u32> = (0..4).chain(50..52).collect();
        log.record(index.reconcile([a.as_slice()]));
        log.record(index.reconcile([a.as_slice(), b.as_slice()]));
        log.record(index.reconcile([b.as_slice()]));
        log.record(index.reconcile([a.as_slice(), b.as_slice()]));
        log.record(index.reconcile([]));
        log.record(index.reconcile([b.as_slice()]));

        let page = log.since(0, Duration::ZERO);
        assert!(page.resync, "the ring holds three of six events; a reader at zero cannot be paged");
        let newest = page.newest(0);
        assert_eq!(newest, 6);
        assert!(page.events.iter().all(|e| e.seq == newest), "a resync is stamped with the newest sequence");
        assert_eq!(page.events[0].event, KvEvent::Cleared);
        let keys = page_hashes(&b, ps);
        let listed: Vec<(Vec<u64>, Option<u64>, Vec<u32>)> = page.events[1..]
            .iter()
            .map(|e| match &e.event {
                KvEvent::Stored { hashes, parent, token_ids } => {
                    (hashes.clone(), *parent, token_ids.clone())
                }
                other => panic!("a resync lists stored pages only, got {other:?}"),
            })
            .collect();
        assert_eq!(
            listed,
            vec![
                (vec![keys[0]], None, b[..2].to_vec()),
                (vec![keys[1]], Some(keys[0]), b[2..4].to_vec()),
                (vec![keys[2]], Some(keys[1]), b[4..6].to_vec()),
            ],
            "the pages the last session holds, parents before children"
        );

        let caught_up = log.since(newest, Duration::ZERO);
        assert!(!caught_up.resync);
        assert!(caught_up.events.is_empty());
        let recent = log.since(4, Duration::ZERO);
        assert!(!recent.resync, "sequence 4 is still in the ring");
        assert_eq!(recent.events.iter().map(|e| e.seq).collect::<Vec<_>>(), vec![5, 6]);
    }

    #[test]
    fn a_waiting_reader_wakes_on_the_event_and_not_only_on_the_timeout() {
        let log = std::sync::Arc::new(KvEventLog::new(8, usize::MAX));
        let writer = log.clone();
        let handle = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            writer.record(vec![KvEvent::Cleared]);
        });
        let started = Instant::now();
        let page = log.since(0, Duration::from_secs(5));
        handle.join().unwrap();
        assert_eq!(page.events.len(), 1, "the reader saw the event that landed while it waited");
        assert!(
            started.elapsed() < Duration::from_secs(4),
            "the reader woke on the event, not the timeout"
        );

        let idle = Instant::now();
        let none = log.since(1, Duration::from_millis(60));
        assert!(none.events.is_empty());
        assert!(
            idle.elapsed() >= Duration::from_millis(55),
            "with nothing to read the wait runs its course"
        );
    }
}
