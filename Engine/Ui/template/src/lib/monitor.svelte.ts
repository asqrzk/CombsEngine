/**
 * Realtime network & storage monitor (top bar).
 *
 * All counters come from the backend proxy (GET /api/monitor) — it sees
 * every byte that enters or leaves the app, so it is the single source of
 * truth. "Storage" is the proxy's data directory (chats, cache).
 */

export interface MonitorState {
  downBytes: number;
  upBytes: number;
  storageUsed: number;
}

function format(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

class Monitor {
  state = $state<MonitorState>({ downBytes: 0, upBytes: 0, storageUsed: 0 });
  /** False while the proxy is unreachable (counters stay stale-but-visible). */
  reachable = $state(true);

  private timer: ReturnType<typeof setTimeout> | null = null;
  private failures = 0;

  start(): void {
    if (this.timer) return;
    const poll = async () => {
      try {
        const res = await fetch("/api/monitor");
        if (res.ok) {
          const m = await res.json();
          this.state.downBytes = m.downBytes ?? 0;
          this.state.upBytes = m.upBytes ?? 0;
          this.state.storageUsed = m.dataBytes ?? 0;
          this.failures = 0;
          this.reachable = true;
        } else {
          this.failures += 1;
          this.reachable = false;
        }
      } catch {
        // proxy not reachable — keep last known counters, back off
        this.failures += 1;
        this.reachable = false;
      }
      // Self-schedule: the next poll is only queued after this one settles,
      // so polls never pile up while the proxy is busy (engine spawn, two
      // SSE relay streams). Back off on repeated failures: 2s → 10s cap.
      const delay = Math.min(2000 * 2 ** this.failures, 10_000);
      this.timer = setTimeout(poll, delay);
    };
    this.timer = setTimeout(poll, 0);
  }

  get downLabel(): string {
    return format(this.state.downBytes);
  }
  get upLabel(): string {
    return format(this.state.upBytes);
  }
  get storageLabel(): string {
    return format(this.state.storageUsed);
  }
}

export const monitor = new Monitor();
