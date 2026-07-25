/**
 * Realtime network & storage monitor (top bar).
 *
 * Network counters are fed by the API layer's byte accounting; storage
 * comes from `navigator.storage.estimate()` polled periodically.
 */

export interface MonitorState {
  downBytes: number;
  upBytes: number;
  storageUsed: number;
  storageQuota: number;
}

function format(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

class Monitor {
  state = $state<MonitorState>({
    downBytes: 0,
    upBytes: 0,
    storageUsed: 0,
    storageQuota: 0,
  });

  private timer: ReturnType<typeof setInterval> | null = null;

  start(): void {
    if (this.timer) return;
    const poll = async () => {
      try {
        const est = await navigator.storage.estimate();
        this.state.storageUsed = est.usage ?? 0;
        this.state.storageQuota = est.quota ?? 0;
      } catch {
        // not supported
      }
    };
    poll();
    this.timer = setInterval(poll, 5000);
  }

  netDown(bytes: number): void {
    this.state.downBytes += bytes;
  }
  netUp(bytes: number): void {
    this.state.upBytes += bytes;
  }

  get downLabel(): string {
    return format(this.state.downBytes);
  }
  get upLabel(): string {
    return format(this.state.upBytes);
  }
  get storageLabel(): string {
    if (!this.state.storageQuota) return "—";
    const pct = Math.round((this.state.storageUsed / this.state.storageQuota) * 100);
    return `${format(this.state.storageUsed)} (${pct}%)`;
  }
}

export const monitor = new Monitor();
