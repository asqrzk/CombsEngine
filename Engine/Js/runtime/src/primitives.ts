/**
 * Atomic runtime primitives: ports, tokens, locks, queues, sessions.
 *
 * These are the building blocks the flow-level APIs compose:
 * find a free port → mint an auth token → start an agent server →
 * register it with the orchestrator → delegate over WebSocket →
 * persist sessions in SQLite.
 */

import { DatabaseSync } from "node:sqlite";

// --- ports ----------------------------------------------------------------

/** Finds a free TCP port by binding :0 and reading back the assigned port. */
export function findFreePort(preferred?: number): number {
  if (preferred) {
    try {
      const l = Deno.listen({ port: preferred });
      l.close();
      return preferred;
    } catch {
      // fall through to :0
    }
  }
  const listener = Deno.listen({ port: 0 });
  const { port } = listener.addr as Deno.NetAddr;
  listener.close();
  return port;
}

// --- auth -----------------------------------------------------------------

/** Cryptographically secure bearer token for inter-process auth. */
export function generateToken(bytes = 24): string {
  const buf = new Uint8Array(bytes);
  crypto.getRandomValues(buf);
  return btoa(String.fromCharCode(...buf))
    .replaceAll("+", "-")
    .replaceAll("/", "_")
    .replaceAll("=", "");
}

/** Checks a request's `Authorization: Bearer <token>` header. */
export function checkBearer(request: Request, token: string): boolean {
  const header = request.headers.get("authorization");
  return header === `Bearer ${token}`;
}

// --- locks ----------------------------------------------------------------

/** Per-key async mutex: serializes async work per key (agent, thread, ...). */
export class KeyedMutex {
  private tails = new Map<string, Promise<void>>();

  /** Runs `fn` holding the lock for `key`. */
  async lock<T>(key: string, fn: () => Promise<T>): Promise<T> {
    const previous = this.tails.get(key) ?? Promise.resolve();
    let release!: () => void;
    const current = new Promise<void>((resolve) => (release = resolve));
    this.tails.set(key, previous.then(() => current));
    await previous;
    try {
      return await fn();
    } finally {
      release();
      if (this.tails.get(key) === current) this.tails.delete(key);
    }
  }
}

/** Counting semaphore for concurrency limits (worker pools, LLM calls). */
export class Semaphore {
  private waiting: (() => void)[] = [];
  constructor(private permits: number) {}

  async acquire(): Promise<void> {
    if (this.permits > 0) {
      this.permits--;
      return;
    }
    await new Promise<void>((resolve) => this.waiting.push(resolve));
  }

  release(): void {
    const next = this.waiting.shift();
    if (next) next();
    else this.permits++;
  }

  async run<T>(fn: () => Promise<T>): Promise<T> {
    await this.acquire();
    try {
      return await fn();
    } finally {
      this.release();
    }
  }
}

// --- queues ---------------------------------------------------------------

/** Durable task queue over Deno KV queues (atomic, resumable). */
export class KvTaskQueue<T = unknown> {
  private constructor(private kv: Deno.Kv) {}

  static async open(path?: string): Promise<KvTaskQueue> {
    return new KvTaskQueue(await Deno.openKv(path));
  }

  /** Enqueues a task (optionally delayed / backed off). */
  async enqueue(task: T, opts: { delay?: number; backoffSchedule?: number[] } = {}): Promise<void> {
    await this.kv.enqueue(task, {
      delay: opts.delay,
      backoffSchedule: opts.backoffSchedule,
    });
  }

  /** Starts consuming; handler errors trigger KV's backoff schedule. */
  listen(handler: (task: T) => Promise<void>): void {
    this.kv.listenQueue(async (value) => {
      await handler(value as T);
    });
  }

  close(): void {
    this.kv.close();
  }
}

// --- sessions -------------------------------------------------------------

export interface Session {
  id: string;
  agent: string;
  threadId: string;
  /** Serialized session payload (graph state, metadata). */
  data: Record<string, unknown>;
  createdAt: number;
  updatedAt: number;
}

/** SQLite session store: save/resume/list agent sessions. */
export class SessionStore {
  private db: DatabaseSync;

  constructor(path = "combs-sessions.sqlite") {
    this.db = new DatabaseSync(path);
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        agent TEXT NOT NULL,
        thread_id TEXT NOT NULL,
        data TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_sessions_agent ON sessions(agent, updated_at DESC);
    `);
  }

  /** Creates or updates a session. */
  save(session: Omit<Session, "createdAt" | "updatedAt"> & { createdAt?: number }): Session {
    const now = Date.now();
    const existing = this.get(session.id);
    const full: Session = {
      ...session,
      createdAt: existing?.createdAt ?? session.createdAt ?? now,
      updatedAt: now,
    };
    this.db
      .prepare(
        "INSERT OR REPLACE INTO sessions (id, agent, thread_id, data, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
      )
      .run(full.id, full.agent, full.threadId, JSON.stringify(full.data), full.createdAt, full.updatedAt);
    return full;
  }

  get(id: string): Session | undefined {
    const row = this.db
      .prepare("SELECT id, agent, thread_id, data, created_at, updated_at FROM sessions WHERE id = ?")
      .get(id) as
      | { id: string; agent: string; thread_id: string; data: string; created_at: number; updated_at: number }
      | undefined;
    if (!row) return undefined;
    return {
      id: row.id,
      agent: row.agent,
      threadId: row.thread_id,
      data: JSON.parse(row.data),
      createdAt: row.created_at,
      updatedAt: row.updated_at,
    };
  }

  list(agent?: string, limit = 50): Session[] {
    const rows = (agent
      ? this.db
        .prepare("SELECT id, agent, thread_id, data, created_at, updated_at FROM sessions WHERE agent = ? ORDER BY updated_at DESC LIMIT ?")
        .all(agent, limit)
      : this.db
        .prepare("SELECT id, agent, thread_id, data, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?")
        .all(limit)) as {
      id: string;
      agent: string;
      thread_id: string;
      data: string;
      created_at: number;
      updated_at: number;
    }[];
    return rows.map((row) => ({
      id: row.id,
      agent: row.agent,
      threadId: row.thread_id,
      data: JSON.parse(row.data),
      createdAt: row.created_at,
      updatedAt: row.updated_at,
    }));
  }

  delete(id: string): void {
    this.db.prepare("DELETE FROM sessions WHERE id = ?").run(id);
  }

  close(): void {
    this.db.close();
  }
}
