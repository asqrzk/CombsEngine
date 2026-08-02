/**
 * Memory: long-term agent memory stores.
 *
 * Short-term (conversation) memory lives in the graph state / checkpointer;
 * this is the long-term store agents can `remember`/`recall` across threads.
 * Two backends matching the checkpointer backends: Deno KV and SQLite.
 *
 * Recall is recency + tag/user filtering (no embeddings yet — the interface
 * leaves room for a vector index later without call-site changes).
 */

import { DatabaseSync } from "node:sqlite";

export interface MemoryEntry {
  id: string;
  /** The remembered content (fact, preference, summary, ...). */
  content: string;
  /** Scoping tags (e.g. user id, agent name, topic). */
  tags: Record<string, string>;
  createdAt: number;
}

export interface MemoryStore {
  remember(content: string, tags?: Record<string, string>): Promise<MemoryEntry>;
  /** Latest-first; optionally filtered by exact tag matches. */
  recall(limit?: number, tags?: Record<string, string>): Promise<MemoryEntry[]>;
  forget(id: string): Promise<void>;
  clear(): Promise<void>;
}

/** Zero-padded timestamp for lexicographically sortable KV keys. */
function tsKey(createdAt: number): string {
  return String(createdAt).padStart(16, "0");
}

/** Deno KV-backed memory. */
export class KvMemoryStore implements MemoryStore {
  private lastTs = 0;

  private constructor(
    private kv: Deno.Kv,
    private prefix: string[],
  ) {}

  static async open(path?: string, scope = "memory"): Promise<KvMemoryStore> {
    return new KvMemoryStore(await Deno.openKv(path), ["combs", scope]);
  }

  async remember(content: string, tags: Record<string, string> = {}): Promise<MemoryEntry> {
    // Strictly monotonic timestamps: rapid successive writes can land in the
    // same Date.now() millisecond (CI runners), which would scramble the
    // "latest first" recall order.
    const createdAt = Math.max(Date.now(), this.lastTs + 1);
    this.lastTs = createdAt;
    const entry: MemoryEntry = {
      id: crypto.randomUUID(),
      content,
      tags,
      createdAt,
    };
    // Timestamp-prefixed key so `reverse: true` lists latest first.
    await this.kv.set([...this.prefix, tsKey(entry.createdAt), entry.id], entry);
    return entry;
  }

  async recall(limit = 20, tags: Record<string, string> = {}): Promise<MemoryEntry[]> {
    const out: MemoryEntry[] = [];
    const entries = this.kv.list<MemoryEntry>({ prefix: this.prefix }, { reverse: true });
    for await (const { value } of entries) {
      if (out.length >= limit) break;
      if (Object.entries(tags).every(([k, v]) => value.tags[k] === v)) out.push(value);
    }
    return out;
  }

  async forget(id: string): Promise<void> {
    const entries = this.kv.list<MemoryEntry>({ prefix: this.prefix });
    for await (const entry of entries) {
      if (entry.value.id === id) {
        await this.kv.delete(entry.key);
        return;
      }
    }
  }

  async clear(): Promise<void> {
    const entries = this.kv.list({ prefix: this.prefix });
    for await (const entry of entries) await this.kv.delete(entry.key);
  }
}

/** SQLite-backed memory (node:sqlite). */
export class SqliteMemoryStore implements MemoryStore {
  private db: DatabaseSync;

  constructor(path = "combs-memory.sqlite") {
    this.db = new DatabaseSync(path);
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        tags TEXT NOT NULL DEFAULT '{}',
        created_at INTEGER NOT NULL
      );
    `);
  }

  remember(content: string, tags: Record<string, string> = {}): Promise<MemoryEntry> {
    const entry: MemoryEntry = {
      id: crypto.randomUUID(),
      content,
      tags,
      createdAt: Date.now(),
    };
    this.db
      .prepare("INSERT INTO memories (id, content, tags, created_at) VALUES (?, ?, ?, ?)")
      .run(entry.id, content, JSON.stringify(tags), entry.createdAt);
    return Promise.resolve(entry);
  }

  recall(limit = 20, tags: Record<string, string> = {}): Promise<MemoryEntry[]> {
    const rows = this.db
      .prepare("SELECT id, content, tags, created_at FROM memories ORDER BY created_at DESC LIMIT ?")
      .all(limit * (Object.keys(tags).length ? 4 : 1)) as {
      id: string;
      content: string;
      tags: string;
      created_at: number;
    }[];
    const out: MemoryEntry[] = [];
    for (const row of rows) {
      const parsed: MemoryEntry = {
        id: row.id,
        content: row.content,
        tags: JSON.parse(row.tags),
        createdAt: row.created_at,
      };
      if (Object.entries(tags).every(([k, v]) => parsed.tags[k] === v)) out.push(parsed);
      if (out.length >= limit) break;
    }
    return Promise.resolve(out);
  }

  forget(id: string): Promise<void> {
    this.db.prepare("DELETE FROM memories WHERE id = ?").run(id);
    return Promise.resolve();
  }

  clear(): Promise<void> {
    this.db.exec("DELETE FROM memories");
    return Promise.resolve();
  }

  close(): void {
    this.db.close();
  }
}
