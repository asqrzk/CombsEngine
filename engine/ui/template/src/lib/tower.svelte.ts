/**
 * Control Tower — the browser's observe store.
 *
 * Subscribes to the proxy's realtime bus (WS /api/observe/ws) and keeps a
 * reactive, filterable event list plus derived rollups (per-source state,
 * runs grouped by trace, network + permission activity). The tower UI reads
 * only this store — it never touches the network directly.
 */

export interface ObsEvent {
  id: string;
  ts: number;
  source: string;
  kind: "span.start" | "span.end" | "event" | "metric" | "log" | "context";
  name: string;
  traceId?: string;
  spanId?: string;
  parentSpanId?: string;
  attrs?: Record<string, string | number | boolean | undefined>;
  input?: unknown;
  output?: unknown;
  context?: unknown;
  status?: "ok" | "error";
  error?: string;
  severity?: string;
}

export interface SourceInfo {
  id: string;
  lastSeen: number;
  kinds: Set<string>;
  /** Latest meaningful state (port, model, status). */
  state: Record<string, unknown>;
}

const CAPACITY = 1000;

class TowerStore {
  events = $state<ObsEvent[]>([]);
  connected = $state(false);
  /** Source id -> rollup. */
  sources = $state<Record<string, SourceInfo>>({});
  /** Currently selected event for the detail inspector. */
  selected = $state<ObsEvent | null>(null);
  /** Filters. */
  sourceFilter = $state<string | null>(null);
  kindFilter = $state<string | null>(null);
  textFilter = $state("");

  private ws: WebSocket | null = null;
  private retry = 0;

  start(): void {
    if (this.ws) return;
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/api/observe/ws`);
    this.ws = ws;
    ws.onopen = () => {
      this.connected = true;
      this.retry = 0;
    };
    ws.onmessage = (msg) => {
      try {
        this.ingest(JSON.parse(String(msg.data)) as ObsEvent);
      } catch {
        // ignore malformed frame
      }
    };
    ws.onclose = () => {
      this.connected = false;
      this.ws = null;
      // reconnect with light backoff
      const delay = Math.min(1000 * 2 ** this.retry++, 8000);
      setTimeout(() => this.start(), delay);
    };
    ws.onerror = () => ws.close();
  }

  private ingest(e: ObsEvent): void {
    this.events.push(e);
    if (this.events.length > CAPACITY) this.events.splice(0, this.events.length - CAPACITY);
    const s = this.sources[e.source] ?? { id: e.source, lastSeen: 0, kinds: new Set(), state: {} };
    s.lastSeen = e.ts;
    s.kinds.add(e.kind);
    if (e.name === "proxy.engine.spawn" || e.name === "proxy.engine.healthy") {
      s.state = { ...s.state, port: e.attrs?.port, model: e.attrs?.model, url: e.attrs?.url };
    }
    this.sources[e.source] = s;
  }

  /** Events after filters (newest last). */
  get filtered(): ObsEvent[] {
    const q = this.textFilter.trim().toLowerCase();
    return this.events.filter((e) => {
      if (this.sourceFilter && e.source !== this.sourceFilter) return false;
      if (this.kindFilter && e.kind !== this.kindFilter) return false;
      if (q && !`${e.name} ${e.source} ${JSON.stringify(e.attrs ?? {})}`.toLowerCase().includes(q)) return false;
      return true;
    });
  }

  /** Runs: traceId -> ordered events (span.start first). */
  get runs(): { traceId: string; name: string; source: string; status: string; durationMs?: number; events: ObsEvent[] }[] {
    const byTrace = new Map<string, ObsEvent[]>();
    for (const e of this.filtered) {
      if (!e.traceId) continue;
      const arr = byTrace.get(e.traceId) ?? [];
      arr.push(e);
      byTrace.set(e.traceId, arr);
    }
    const runs = [];
    for (const [traceId, events] of byTrace) {
      events.sort((a, b) => a.ts - b.ts);
      const start = events.find((e) => e.kind === "span.start");
      const end = events.find((e) => e.kind === "span.end");
      runs.push({
        traceId,
        name: start?.name ?? events[0].name,
        source: start?.source ?? events[0].source,
        status: end?.status ?? "running",
        durationMs: end?.attrs?.durationMs as number | undefined,
        events,
      });
    }
    return runs.sort((a, b) => (b.events[0]?.ts ?? 0) - (a.events[0]?.ts ?? 0));
  }

  sourceList(): SourceInfo[] {
    return Object.values(this.sources).sort((a, b) => b.lastSeen - a.lastSeen);
  }

  clear(): void {
    this.events = [];
    this.selected = null;
  }

  select(e: ObsEvent | null): void {
    this.selected = e;
  }
}

export const tower = new TowerStore();
