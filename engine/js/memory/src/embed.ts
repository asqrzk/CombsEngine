/**
 * Semantic recall (L2) over the engine's own /v1/embeddings. Vectors
 * are computed ONCE at write time (embedMissing backfill), truncated
 * matryoshka-style to `dim` and re-L2-normalized, stored as Float32
 * BLOBs; recall embeds only the query and takes cosine in-process — at
 * machine scale a linear scan beats any index it could carry.
 */

import type { GraphStore } from "./store.ts";
import { recallL1, type RecallHit, type RecallQuery } from "./recall.ts";

export class EmbedClient {
  constructor(private baseUrl: string, private opts: { dim?: number } = {}) {}

  async embed(texts: string[]): Promise<Float32Array[]> {
    const res = await fetch(`${this.baseUrl.replace(/\/$/, "")}/v1/embeddings`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ input: texts }),
    });
    if (!res.ok) {
      throw new Error(`embeddings HTTP ${res.status}: ${await res.text().catch(() => "")}`);
    }
    const body = await res.json();
    const dim = this.opts.dim ?? 0;
    return (body.data as { embedding: number[] }[]).map((d) => {
      let v = Float32Array.from(d.embedding);
      if (dim > 0 && dim < v.length) v = v.slice(0, dim);
      let norm = 0;
      for (const x of v) norm += x * x;
      norm = Math.sqrt(norm) || 1;
      return v.map((x) => x / norm);
    });
  }
}

function embedTextFor(
  entity: { name: string; type: string },
  observations: { content: string }[],
): string {
  return [`${entity.name} (${entity.type})`, ...observations.slice(0, 4).map((o) => o.content)]
    .join("\n")
    .slice(0, 1000);
}

/** Backfill vectors for entities that lack one. Returns count embedded. */
export async function embedMissing(
  store: GraphStore,
  client: EmbedClient,
  limit = 64,
): Promise<number> {
  const missing = await store.missingEmbeddings(limit);
  if (!missing.length) return 0;
  const texts: string[] = [];
  const names: string[] = [];
  for (const name of missing) {
    const got = await store.getEntity(name);
    if (!got) continue;
    names.push(name);
    texts.push(embedTextFor(got.entity, got.observations));
  }
  // The engine caps input at 64 texts per call.
  for (let i = 0; i < names.length; i += 64) {
    const vecs = await client.embed(texts.slice(i, i + 64));
    for (let j = 0; j < vecs.length; j++) {
      await store.putEmbedding(names[i + j], vecs[j]);
    }
  }
  return names.length;
}

function cosine(a: Float32Array, b: Float32Array): number {
  const n = Math.min(a.length, b.length);
  let dot = 0;
  for (let i = 0; i < n; i++) dot += a[i] * b[i];
  return dot; // both sides are L2-normalized
}

/** Pure semantic recall: cosine over stored vectors. */
export async function recallL2(
  store: GraphStore,
  client: EmbedClient,
  q: RecallQuery,
): Promise<RecallHit[]> {
  const [queryVec] = await client.embed([q.text]);
  const all = await store.embeddings(q.project);
  const scored = all
    .map((e) => ({ entity: e.entity, sim: cosine(queryVec, e.vec) }))
    .filter((s) => s.sim > 0.15)
    .sort((a, b) => b.sim - a.sim)
    .slice(0, q.k ?? 6);
  const hits: RecallHit[] = [];
  for (const s of scored) {
    const got = await store.getEntity(s.entity);
    if (!got) continue;
    hits.push({
      entity: s.entity,
      type: got.entity.type,
      caste: got.entity.caste,
      score: Math.round(s.sim * 100) / 100,
      lines: got.observations.slice(0, 2).map((o) => o.content),
    });
  }
  await store.touchEntities(hits.map((h) => h.entity));
  return hits;
}

/**
 * Hybrid recall: L1 keyword ∪ L2 semantic, score-merged (L1 scores are
 * unbounded keyword sums, L2 is cosine ≤ 1 — normalize L1 by its own
 * max so the two contribute comparably). `client` null = pure L1.
 */
export async function recallHybrid(
  store: GraphStore,
  client: EmbedClient | null,
  q: RecallQuery,
): Promise<RecallHit[]> {
  const l1 = await recallL1(store, q);
  if (!client) return l1;
  let l2: RecallHit[] = [];
  try {
    l2 = await recallL2(store, client, q);
  } catch {
    // embed worker away — L1 stands alone, honestly
    return l1;
  }
  const l1max = Math.max(...l1.map((h) => h.score), 1);
  const merged = new Map<string, RecallHit>();
  for (const h of l1) {
    merged.set(h.entity, { ...h, score: Math.round((h.score / l1max) * 100) / 100 });
  }
  for (const h of l2) {
    const prior = merged.get(h.entity);
    if (prior) prior.score = Math.round((prior.score + h.score) * 100) / 100;
    else merged.set(h.entity, h);
  }
  return [...merged.values()]
    .sort((a, b) => b.score - a.score)
    .slice(0, q.k ?? 6);
}
