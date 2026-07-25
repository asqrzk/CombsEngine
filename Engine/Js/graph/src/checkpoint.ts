/**
 * Checkpointers: durable graph state per thread.
 *
 * A checkpoint captures channel values, channel versions and per-node
 * `versionsSeen` at a superstep boundary, plus pending writes (used to
 * persist interrupts and resume values). Restoring a checkpoint is what
 * enables resume, time travel (`getStateHistory`) and human state editing
 * (`updateState`).
 *
 * Three implementations:
 * - `MemoryCheckpointer` — tests, ephemeral runs.
 * - `KvCheckpointer` — Deno KV (works on Deno Deploy, atomic put).
 * - `SqliteCheckpointer` — node:sqlite, durable single-file store.
 */

import { DatabaseSync } from "node:sqlite";

export interface Checkpoint {
  /** Superstep number (monotonic per thread). */
  step: number;
  /** channel name -> checkpointed value. */
  channelValues: Record<string, unknown>;
  /** channel name -> version counter. */
  versions: Record<string, number>;
  /** node name -> channel name -> last version the node saw. */
  versionsSeen: Record<string, Record<string, number>>;
  /** Resume values queued for interrupted nodes (FIFO per thread). */
  resumeQueue: unknown[];
  /** The interrupt payloads that paused the run, if any. */
  pendingInterrupts?: { node: string; value: unknown }[];
  /** Nodes that completed before the checkpoint (join barriers, resume). */
  completedNodes?: string[];
  /** Frontier to continue with after an interruptAfter resume. */
  resumeFrontier?: { node: string; sendArgs?: unknown }[];
}

export interface Checkpointer {
  get(threadId: string): Promise<Checkpoint | undefined>;
  put(threadId: string, checkpoint: Checkpoint): Promise<void>;
  list(threadId: string, limit?: number): Promise<Checkpoint[]>;
  delete(threadId: string): Promise<void>;
}

/** In-memory checkpointer (default when none is provided). */
export class MemoryCheckpointer implements Checkpointer {
  private store = new Map<string, Checkpoint[]>();

  get(threadId: string): Promise<Checkpoint | undefined> {
    const history = this.store.get(threadId);
    return Promise.resolve(history?.[history.length - 1]);
  }
  put(threadId: string, checkpoint: Checkpoint): Promise<void> {
    const history = this.store.get(threadId) ?? [];
    history.push(structuredClone(checkpoint));
    this.store.set(threadId, history);
    return Promise.resolve();
  }
  list(threadId: string, limit = 100): Promise<Checkpoint[]> {
    return Promise.resolve([...(this.store.get(threadId) ?? [])].reverse().slice(0, limit));
  }
  delete(threadId: string): Promise<void> {
    this.store.delete(threadId);
    return Promise.resolve();
  }
}

/** Deno KV-backed checkpointer (atomic, Deploy-compatible). */
export class KvCheckpointer implements Checkpointer {
  private constructor(
    private kv: Deno.Kv,
    private prefix: string[] = ["combs", "graph"],
  ) {}

  static async open(path?: string, prefix?: string[]): Promise<KvCheckpointer> {
    return new KvCheckpointer(await Deno.openKv(path), prefix);
  }

  async get(threadId: string): Promise<Checkpoint | undefined> {
    const entries = this.kv.list<Checkpoint>({
      prefix: [...this.prefix, "threads", threadId],
    }, { limit: 1, reverse: true });
    for await (const entry of entries) return entry.value;
    return undefined;
  }

  async put(threadId: string, checkpoint: Checkpoint): Promise<void> {
    // Key includes the step so history is preserved; `get` reads the latest.
    const key = [
      ...this.prefix,
      "threads",
      threadId,
      String(checkpoint.step).padStart(10, "0"),
    ];
    await this.kv.atomic().set(key, checkpoint).commit();
  }

  async list(threadId: string, limit = 100): Promise<Checkpoint[]> {
    const out: Checkpoint[] = [];
    const entries = this.kv.list<Checkpoint>({
      prefix: [...this.prefix, "threads", threadId],
    }, { limit, reverse: true });
    for await (const entry of entries) out.push(entry.value);
    return out;
  }

  async delete(threadId: string): Promise<void> {
    const entries = this.kv.list({ prefix: [...this.prefix, "threads", threadId] });
    for await (const entry of entries) await this.kv.delete(entry.key);
  }
}

/** SQLite-backed checkpointer (node:sqlite, single durable file). */
export class SqliteCheckpointer implements Checkpointer {
  private db: DatabaseSync;

  constructor(path = "combs-checkpoints.sqlite") {
    this.db = new DatabaseSync(path);
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS checkpoints (
        thread_id TEXT NOT NULL,
        step INTEGER NOT NULL,
        payload TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        PRIMARY KEY (thread_id, step)
      );
      CREATE INDEX IF NOT EXISTS idx_checkpoints_thread ON checkpoints(thread_id, step DESC);
    `);
  }

  get(threadId: string): Promise<Checkpoint | undefined> {
    const row = this.db
      .prepare("SELECT payload FROM checkpoints WHERE thread_id = ? ORDER BY step DESC LIMIT 1")
      .get(threadId) as { payload: string } | undefined;
    return Promise.resolve(row ? JSON.parse(row.payload) : undefined);
  }

  put(threadId: string, checkpoint: Checkpoint): Promise<void> {
    this.db
      .prepare(
        "INSERT OR REPLACE INTO checkpoints (thread_id, step, payload, created_at) VALUES (?, ?, ?, ?)",
      )
      .run(threadId, checkpoint.step, JSON.stringify(checkpoint), Date.now());
    return Promise.resolve();
  }

  list(threadId: string, limit = 100): Promise<Checkpoint[]> {
    const rows = this.db
      .prepare("SELECT payload FROM checkpoints WHERE thread_id = ? ORDER BY step DESC LIMIT ?")
      .all(threadId, limit) as { payload: string }[];
    return Promise.resolve(rows.map((r) => JSON.parse(r.payload)));
  }

  delete(threadId: string): Promise<void> {
    this.db.prepare("DELETE FROM checkpoints WHERE thread_id = ?").run(threadId);
    return Promise.resolve();
  }

  close(): void {
    this.db.close();
  }
}
