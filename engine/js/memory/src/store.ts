/**
 * GraphStore — SQLite-backed knowledge graph (entities, observations,
 * relations, embeddings) with the caste lifecycle on entities.
 *
 * The db file is shared by more than one process (an MCP stdio bin and
 * a platform server), so WAL + busy_timeout are part of the contract,
 * not tuning. All public methods are async so the surface is uniform
 * whether the at-rest crypto door is open (WebCrypto is async) or not.
 */

import { DatabaseSync } from "node:sqlite";
import type {
  BlobCrypto,
  Caste,
  CasteDoors,
  Entity,
  EntityHit,
  EntitySpec,
  GraphStats,
  ManifestEntry,
  NeighborsResult,
  Observation,
  Relation,
  RelationSpec,
  SearchQuery,
} from "./types.ts";
import { DEFAULT_CASTE_DOORS } from "./types.ts";

const CASTES: Caste[] = ["youngling", "worker", "queen", "drone"];

function defaultDbPath(): string {
  const home = Deno.env.get("HOME") ?? ".";
  return `${home}/.combs/memory/graph.db`;
}

interface EntityRow {
  name: string;
  type: string;
  project: string;
  caste: string;
  usage_count: number;
  last_used_at: number;
  created_at: number;
  updated_at: number;
}

function rowToEntity(r: EntityRow): Entity {
  return {
    name: r.name,
    type: r.type,
    project: r.project,
    caste: (CASTES.includes(r.caste as Caste) ? r.caste : "youngling") as Caste,
    usageCount: r.usage_count,
    lastUsedAt: r.last_used_at,
    createdAt: r.created_at,
    updatedAt: r.updated_at,
  };
}

const ENTITY_COLS =
  "name, type, project, caste, usage_count, last_used_at, created_at, updated_at";

export interface GraphStoreOptions {
  path?: string;
  crypto?: BlobCrypto;
  casteDoors?: Partial<CasteDoors>;
}

export class GraphStore {
  private db: DatabaseSync;
  private crypto: BlobCrypto | null;
  readonly casteDoors: CasteDoors;

  constructor(opts: GraphStoreOptions = {}) {
    const path = opts.path ?? defaultDbPath();
    if (path !== ":memory:") {
      const dir = path.substring(0, path.lastIndexOf("/"));
      if (dir) Deno.mkdirSync(dir, { recursive: true });
    }
    this.db = new DatabaseSync(path);
    this.crypto = opts.crypto ?? null;
    this.casteDoors = { ...DEFAULT_CASTE_DOORS, ...opts.casteDoors };
    this.db.exec(`
      PRAGMA journal_mode=WAL;
      PRAGMA busy_timeout=5000;
      PRAGMA foreign_keys=ON;
      CREATE TABLE IF NOT EXISTS entities (
        name TEXT PRIMARY KEY,
        type TEXT NOT NULL,
        project TEXT NOT NULL DEFAULT '',
        caste TEXT NOT NULL DEFAULT 'youngling',
        usage_count INTEGER NOT NULL DEFAULT 0,
        last_used_at INTEGER NOT NULL DEFAULT 0,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_entities_project ON entities(project, updated_at DESC);
      CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type, project);
      CREATE TABLE IF NOT EXISTS observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entity TEXT NOT NULL REFERENCES entities(name) ON DELETE CASCADE ON UPDATE CASCADE,
        content TEXT NOT NULL,
        enc TEXT,
        created_at INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_obs_entity ON observations(entity, created_at DESC);
      CREATE TABLE IF NOT EXISTS relations (
        from_entity TEXT NOT NULL REFERENCES entities(name) ON DELETE CASCADE ON UPDATE CASCADE,
        to_entity TEXT NOT NULL REFERENCES entities(name) ON DELETE CASCADE ON UPDATE CASCADE,
        rel_type TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        PRIMARY KEY (from_entity, to_entity, rel_type)
      );
      CREATE INDEX IF NOT EXISTS idx_rel_to ON relations(to_entity, rel_type);
      CREATE TABLE IF NOT EXISTS embeddings (
        entity TEXT PRIMARY KEY REFERENCES entities(name) ON DELETE CASCADE ON UPDATE CASCADE,
        dim INTEGER NOT NULL,
        vec BLOB NOT NULL
      );
    `);
  }

  /**
   * Sync transaction wrapper (BEGIN IMMEDIATE). The fn must not await:
   * an await would yield the shared connection mid-transaction. Async
   * work (crypto seals) is done BEFORE entering.
   */
  atomically<T>(fn: () => T): T {
    this.db.exec("BEGIN IMMEDIATE");
    try {
      const out = fn();
      this.db.exec("COMMIT");
      return out;
    } catch (e) {
      try {
        this.db.exec("ROLLBACK");
      } catch { /* already rolled back */ }
      throw e;
    }
  }

  // ── entities ──────────────────────────────────────────────────────

  async upsertEntities(specs: EntitySpec[]): Promise<Entity[]> {
    const now = Date.now();
    // Crypto seals happen BEFORE the transaction (no awaits inside it).
    const sealed = new Map<number, { content: string; enc: string | null }[]>();
    for (let i = 0; i < specs.length; i++) {
      if (specs[i].observations?.length) {
        sealed.set(i, await this.sealContents(specs[i].observations!));
      }
    }
    // Single-statement UPSERT: no SELECT-then-INSERT window against the
    // other process sharing this db. COALESCE keeps caste/project when
    // the spec is silent (graphify re-runs must not demote promotions).
    const upsert = this.db.prepare(
      `INSERT INTO entities (name, type, project, caste, usage_count, last_used_at, created_at, updated_at)
       VALUES (?, ?, COALESCE(?, ''), COALESCE(?, 'youngling'), 0, 0, ?, ?)
       ON CONFLICT(name) DO UPDATE SET
         type = excluded.type,
         project = COALESCE(?, entities.project),
         caste = COALESCE(?, entities.caste),
         updated_at = excluded.updated_at`,
    );
    return this.atomically(() => {
      const out: Entity[] = [];
      for (let i = 0; i < specs.length; i++) {
        const spec = specs[i];
        upsert.run(
          spec.name,
          spec.type,
          spec.project ?? null,
          spec.caste ?? null,
          now,
          now,
          spec.project ?? null,
          spec.caste ?? null,
        );
        const rows = sealed.get(i);
        if (rows) {
          if (spec.replaceObservations) {
            this.db.prepare("DELETE FROM observations WHERE entity = ?").run(spec.name);
          }
          this.insertObservationRows(spec.name, rows, now);
        }
        out.push(
          rowToEntity(
            this.db.prepare(`SELECT ${ENTITY_COLS} FROM entities WHERE name = ?`)
              .get(spec.name) as unknown as EntityRow,
          ),
        );
      }
      return out;
    });
  }

  async getEntity(name: string): Promise<
    | { entity: Entity; observations: Observation[]; out: Relation[]; inn: Relation[] }
    | undefined
  > {
    const row = this.db
      .prepare(`SELECT ${ENTITY_COLS} FROM entities WHERE name = ?`)
      .get(name) as EntityRow | undefined;
    if (!row) return undefined;
    return {
      entity: rowToEntity(row),
      observations: await this.observationsFor(name),
      out: this.relationsFrom(name),
      inn: this.relationsTo(name),
    };
  }

  deleteEntities(names: string[]): Promise<number> {
    let n = 0;
    for (const name of names) {
      n += Number(this.db.prepare("DELETE FROM entities WHERE name = ?").run(name).changes);
    }
    return Promise.resolve(n);
  }

  /**
   * Record use: bump usage counters and promote younglings that crossed
   * the worker threshold. Queens are never demoted here; drones touched
   * by recall wake back to worker.
   */
  touchEntities(names: string[]): Promise<void> {
    if (!names.length) return Promise.resolve();
    const now = Date.now();
    for (const name of names) {
      this.db
        .prepare("UPDATE entities SET usage_count = usage_count + 1, last_used_at = ? WHERE name = ?")
        .run(now, name);
    }
    this.db
      .prepare(
        `UPDATE entities SET caste = 'worker'
         WHERE name IN (${names.map(() => "?").join(",")})
           AND ((caste = 'youngling' AND usage_count >= ?) OR caste = 'drone')`,
      )
      .run(...names, this.casteDoors.workerAfterUses);
    return Promise.resolve();
  }

  /**
   * Lifecycle sweep: demote stale workers/younglings to drone.
   * Queens and `current-state`/`pointer` entries never demote.
   */
  matureEntities(): Promise<{ demoted: number }> {
    if (this.casteDoors.droneAfterDays <= 0) return Promise.resolve({ demoted: 0 });
    const cutoff = Date.now() - this.casteDoors.droneAfterDays * 86_400_000;
    const res = this.db
      .prepare(
        `UPDATE entities SET caste = 'drone'
         WHERE caste IN ('youngling', 'worker')
           AND type NOT IN ('current-state', 'pointer')
           AND MAX(last_used_at, updated_at) < ?`,
      )
      .run(cutoff);
    return Promise.resolve({ demoted: Number(res.changes) });
  }

  // ── observations ──────────────────────────────────────────────────

  /** Seal contents for storage (plaintext rows when the door is shut). */
  async sealContents(
    contents: string[],
  ): Promise<{ content: string; enc: string | null }[]> {
    const out: { content: string; enc: string | null }[] = [];
    for (const content of contents) {
      if (this.crypto) {
        const sealed = await this.crypto.seal(new TextEncoder().encode(content));
        out.push({ content: sealed.blob, enc: JSON.stringify(sealed.entry) });
      } else {
        out.push({ content, enc: null });
      }
    }
    return out;
  }

  private insertObservationRows(
    entity: string,
    rows: { content: string; enc: string | null }[],
    now: number,
  ): number[] {
    const ids: number[] = [];
    for (const row of rows) {
      const res = this.db
        .prepare("INSERT INTO observations (entity, content, enc, created_at) VALUES (?, ?, ?, ?)")
        .run(entity, row.content, row.enc, now);
      ids.push(Number(res.lastInsertRowid));
    }
    return ids;
  }

  async addObservations(entity: string, contents: string[]): Promise<Observation[]> {
    const now = Date.now();
    const rows = await this.sealContents(contents);
    const ids = this.insertObservationRows(entity, rows, now);
    return contents.map((content, i) => ({ id: ids[i], entity, content, createdAt: now }));
  }

  async observationsFor(entity: string, limit = 50): Promise<Observation[]> {
    const rows = this.db
      .prepare(
        "SELECT id, entity, content, enc, created_at FROM observations WHERE entity = ? ORDER BY created_at DESC, id DESC LIMIT ?",
      )
      .all(entity, limit) as {
        id: number;
        entity: string;
        content: string;
        enc: string | null;
        created_at: number;
      }[];
    const out: Observation[] = [];
    for (const r of rows) {
      out.push({
        id: r.id,
        entity: r.entity,
        content: await this.decodeContent(r.content, r.enc),
        createdAt: r.created_at,
      });
    }
    return out;
  }

  private async decodeContent(content: string, enc: string | null): Promise<string> {
    if (enc === null) return content;
    if (!this.crypto) return "[encrypted]";
    try {
      const entry = JSON.parse(enc) as ManifestEntry;
      return new TextDecoder().decode(await this.crypto.open(content, entry));
    } catch {
      // Tamper or wrong key: surface unreadable, never wrong plaintext.
      return "[unreadable]";
    }
  }

  deleteObservations(ids: number[]): Promise<number> {
    let n = 0;
    for (const id of ids) {
      n += Number(this.db.prepare("DELETE FROM observations WHERE id = ?").run(id).changes);
    }
    return Promise.resolve(n);
  }

  // ── relations ─────────────────────────────────────────────────────

  addRelations(rels: RelationSpec[]): Promise<Relation[]> {
    const now = Date.now();
    const out: Relation[] = [];
    for (const rel of rels) {
      this.db
        .prepare(
          "INSERT OR IGNORE INTO relations (from_entity, to_entity, rel_type, created_at) VALUES (?, ?, ?, ?)",
        )
        .run(rel.from, rel.to, rel.relType, now);
      out.push({ from: rel.from, to: rel.to, relType: rel.relType, createdAt: now });
    }
    return Promise.resolve(out);
  }

  /** Atomically make `from` point at exactly one `to` via `relType`. */
  repointRelation(from: string, to: string, relType: string): void {
    this.atomically(() => {
      this.db
        .prepare("DELETE FROM relations WHERE from_entity = ? AND rel_type = ?")
        .run(from, relType);
      this.db
        .prepare(
          "INSERT OR IGNORE INTO relations (from_entity, to_entity, rel_type, created_at) VALUES (?, ?, ?, ?)",
        )
        .run(from, to, relType, Date.now());
    });
  }

  deleteRelations(rels: RelationSpec[]): Promise<number> {
    let n = 0;
    for (const rel of rels) {
      n += Number(
        this.db
          .prepare("DELETE FROM relations WHERE from_entity = ? AND to_entity = ? AND rel_type = ?")
          .run(rel.from, rel.to, rel.relType).changes,
      );
    }
    return Promise.resolve(n);
  }

  relationsFrom(name: string): Relation[] {
    return (this.db
      .prepare("SELECT from_entity, to_entity, rel_type, created_at FROM relations WHERE from_entity = ?")
      .all(name) as { from_entity: string; to_entity: string; rel_type: string; created_at: number }[])
      .map((r) => ({ from: r.from_entity, to: r.to_entity, relType: r.rel_type, createdAt: r.created_at }));
  }

  relationsTo(name: string): Relation[] {
    return (this.db
      .prepare("SELECT from_entity, to_entity, rel_type, created_at FROM relations WHERE to_entity = ?")
      .all(name) as { from_entity: string; to_entity: string; rel_type: string; created_at: number }[])
      .map((r) => ({ from: r.from_entity, to: r.to_entity, relType: r.rel_type, createdAt: r.created_at }));
  }

  // ── search / traversal ────────────────────────────────────────────

  search(q: SearchQuery): Promise<EntityHit[]> {
    const limit = q.limit ?? 20;
    const clauses: string[] = [];
    const args: (string | number)[] = [];
    if (q.project) {
      clauses.push("project = ?");
      args.push(q.project);
    }
    if (q.type) {
      clauses.push("type = ?");
      args.push(q.type);
    }
    if (q.query) {
      clauses.push("name LIKE ? ESCAPE '\\'");
      args.push(`%${escapeLike(q.query)}%`);
    }
    const where = clauses.length ? `WHERE ${clauses.join(" AND ")}` : "";
    const rows = this.db
      .prepare(
        `SELECT ${ENTITY_COLS} FROM entities ${where} ORDER BY updated_at DESC LIMIT ?`,
      )
      .all(...args, limit) as unknown as EntityRow[];
    return Promise.resolve(
      rows.map((r) => ({ entity: rowToEntity(r), score: 1, matched: ["name"] })),
    );
  }

  neighbors(
    name: string,
    opts: { depth?: number; relType?: string; direction?: "out" | "in" | "both" } = {},
  ): Promise<NeighborsResult> {
    const depth = Math.min(opts.depth ?? 1, 4);
    const direction = opts.direction ?? "both";
    const seen = new Set<string>([name]);
    const relations: Relation[] = [];
    const relKey = (r: Relation) => `${r.from} ${r.to} ${r.relType}`;
    const seenRels = new Set<string>();
    let frontier = [name];
    for (let d = 0; d < depth && frontier.length; d++) {
      const next: string[] = [];
      for (const node of frontier) {
        const outward = direction !== "in" ? this.relationsFrom(node) : [];
        const inward = direction !== "out" ? this.relationsTo(node) : [];
        for (const rel of [...outward, ...inward]) {
          if (opts.relType && rel.relType !== opts.relType) continue;
          if (!seenRels.has(relKey(rel))) {
            seenRels.add(relKey(rel));
            relations.push(rel);
          }
          const other = rel.from === node ? rel.to : rel.from;
          if (!seen.has(other)) {
            seen.add(other);
            next.push(other);
          }
        }
      }
      frontier = next;
    }
    seen.delete(name);
    const entities: Entity[] = [];
    for (const n of seen) {
      const row = this.db.prepare(`SELECT ${ENTITY_COLS} FROM entities WHERE name = ?`)
        .get(n) as EntityRow | undefined;
      if (row) entities.push(rowToEntity(row));
    }
    return Promise.resolve({ entities, relations });
  }

  /**
   * Delete stale structural entities of a project (types listed, names
   * not in `keep`) — enumerate + delete in ONE transaction so the
   * window cannot race a concurrent writer in the other process.
   */
  sweepStale(project: string, types: string[], keep: Set<string>): Promise<number> {
    return Promise.resolve(this.atomically(() => {
      let n = 0;
      for (const type of types) {
        const rows = this.db
          .prepare("SELECT name FROM entities WHERE project = ? AND type = ?")
          .all(project, type) as unknown as { name: string }[];
        for (const row of rows) {
          if (!keep.has(row.name)) {
            n += Number(this.db.prepare("DELETE FROM entities WHERE name = ?").run(row.name).changes);
          }
        }
      }
      return n;
    }));
  }

  /** Full edge list for a project (or all) — the native/GPU traversal feed. */
  edgeList(project?: string): Promise<{ nodes: string[]; edges: [number, number][] }> {
    const rows = (project
      ? this.db
        .prepare(
          `SELECT r.from_entity, r.to_entity FROM relations r
           JOIN entities e ON e.name = r.from_entity WHERE e.project = ?`,
        )
        .all(project)
      : this.db.prepare("SELECT from_entity, to_entity FROM relations").all()) as {
        from_entity: string;
        to_entity: string;
      }[];
    const index = new Map<string, number>();
    const nodes: string[] = [];
    const idx = (n: string) => {
      let i = index.get(n);
      if (i === undefined) {
        i = nodes.length;
        nodes.push(n);
        index.set(n, i);
      }
      return i;
    };
    const edges: [number, number][] = rows.map((r) => [idx(r.from_entity), idx(r.to_entity)]);
    return Promise.resolve({ nodes, edges });
  }

  // ── embeddings ────────────────────────────────────────────────────

  putEmbedding(entity: string, vec: Float32Array): Promise<void> {
    this.db
      .prepare("INSERT OR REPLACE INTO embeddings (entity, dim, vec) VALUES (?, ?, ?)")
      .run(
        entity,
        vec.length,
        new Uint8Array(vec.buffer, vec.byteOffset, vec.byteLength).slice(),
      );
    return Promise.resolve();
  }

  embeddings(project?: string): Promise<{ entity: string; vec: Float32Array }[]> {
    const rows = (project
      ? this.db
        .prepare(
          `SELECT em.entity, em.dim, em.vec FROM embeddings em
           JOIN entities e ON e.name = em.entity WHERE e.project = ?`,
        )
        .all(project)
      : this.db.prepare("SELECT entity, dim, vec FROM embeddings").all()) as {
        entity: string;
        dim: number;
        vec: Uint8Array;
      }[];
    return Promise.resolve(rows.map((r) => ({
      entity: r.entity,
      vec: new Float32Array(r.vec.buffer, r.vec.byteOffset, r.dim),
    })));
  }

  missingEmbeddings(limit = 64): Promise<string[]> {
    const rows = this.db
      .prepare(
        `SELECT name FROM entities e WHERE NOT EXISTS
         (SELECT 1 FROM embeddings em WHERE em.entity = e.name)
         ORDER BY updated_at DESC LIMIT ?`,
      )
      .all(limit) as { name: string }[];
    return Promise.resolve(rows.map((r) => r.name));
  }

  // ── stats ─────────────────────────────────────────────────────────

  stats(): Promise<GraphStats> {
    const count = (sql: string) => Number((this.db.prepare(sql).get() as { n: number }).n);
    const castes: Record<string, number> = {};
    for (
      const r of this.db.prepare("SELECT caste, COUNT(*) AS n FROM entities GROUP BY caste")
        .all() as { caste: string; n: number }[]
    ) castes[r.caste] = r.n;
    const perProject: Record<string, number> = {};
    for (
      const r of this.db
        .prepare(
          "SELECT project, COUNT(*) AS n FROM entities WHERE project != '' GROUP BY project ORDER BY n DESC",
        )
        .all() as unknown as { project: string; n: number }[]
    ) perProject[r.project] = r.n;
    return Promise.resolve({
      entities: count("SELECT COUNT(*) AS n FROM entities"),
      observations: count("SELECT COUNT(*) AS n FROM observations"),
      relations: count("SELECT COUNT(*) AS n FROM relations"),
      projects: Object.keys(perProject),
      perProject,
      castes,
    });
  }

  close(): void {
    this.db.close();
  }
}

export function escapeLike(s: string): string {
  return s.replace(/[\\%_]/g, (c) => `\\${c}`);
}
