/**
 * L1 recall — SQL + in-process scoring, no model involved.
 *
 * Ranking: keyword overlap (entity name ×3, type ×1, observation ×1),
 * recency decay, relation-degree bonus, all multiplied by the entity's
 * caste weight (the lifecycle doors). Top seeds expand one hop so a hit
 * carries its neighborhood. Recalled entities are touched — recall IS
 * the usage signal that matures younglings into workers.
 */

import type { GraphStore } from "./store.ts";
import type { Entity, RelationSpec } from "./types.ts";

export interface RecallQuery {
  text: string;
  project?: string;
  k?: number;
  capChars?: number;
}

export interface RecallHit {
  entity: string;
  type: string;
  caste: string;
  score: number;
  lines: string[];
}

const STOPWORDS = new Set(
  ("a an and are as at be but by for from has have how i in is it its of on or " +
    "that the this to was we what when where which who will with you your").split(" "),
);

export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9_./-]+/)
    .map((t) => t.trim())
    .filter((t) => t.length > 1 && !STOPWORDS.has(t));
}

function overlap(tokens: string[], hay: string): number {
  const lower = hay.toLowerCase();
  let n = 0;
  for (const t of tokens) if (lower.includes(t)) n++;
  return n;
}

const HALF_LIFE_MS = 14 * 86_400_000;

export async function recallL1(store: GraphStore, q: RecallQuery): Promise<RecallHit[]> {
  const tokens = tokenize(q.text);
  if (!tokens.length) return [];
  const k = q.k ?? 6;

  // Candidates: project-scoped (or global) entities via the indexed
  // paths, then scored in-process. The candidate pull is bounded, not a
  // full-table walk.
  const candidates = new Map<string, Entity>();
  for (const token of tokens.slice(0, 8)) {
    for (const hit of await store.search({ query: token, project: q.project, limit: 25 })) {
      candidates.set(hit.entity.name, hit.entity);
    }
  }
  // Recent project entities join the pool so observation-only matches
  // (name says nothing, facts say everything) are reachable.
  for (const hit of await store.search({ project: q.project, limit: 50 })) {
    candidates.set(hit.entity.name, hit.entity);
  }

  const now = Date.now();
  const weights = store.casteDoors.weights;
  const scored: { entity: Entity; score: number; lines: string[] }[] = [];
  for (const entity of candidates.values()) {
    const obs = await store.observationsFor(entity.name, 8);
    const nameScore = overlap(tokens, entity.name) * 3;
    const typeScore = overlap(tokens, entity.type);
    let obsScore = 0;
    const matchedLines: string[] = [];
    for (const o of obs) {
      const hits = overlap(tokens, o.content);
      if (hits > 0) {
        obsScore += hits;
        matchedLines.push(o.content);
      }
    }
    const raw = nameScore + typeScore + obsScore;
    if (raw === 0) continue;
    const ageMs = now - Math.max(entity.updatedAt, entity.lastUsedAt);
    const recency = Math.pow(0.5, ageMs / HALF_LIFE_MS);
    const degree = store.relationsFrom(entity.name).length +
      store.relationsTo(entity.name).length;
    const degreeBonus = 1 + Math.min(degree, 8) * 0.05;
    const caste = weights[entity.caste] ?? 1;
    const score = raw * (0.5 + 0.5 * recency) * degreeBonus * caste;
    const lines = matchedLines.length ? matchedLines : obs.slice(0, 2).map((o) => o.content);
    scored.push({ entity, score, lines: lines.slice(0, 3) });
  }

  scored.sort((a, b) => b.score - a.score);
  const top = scored.slice(0, k);
  await store.touchEntities(top.map((s) => s.entity.name));
  return top.map((s) => ({
    entity: s.entity.name,
    type: s.entity.type,
    caste: s.entity.caste,
    score: Math.round(s.score * 100) / 100,
    lines: s.lines,
  }));
}

/** Compact injection text: short fact lines under a hard char cap. */
export function formatRecall(hits: RecallHit[], capChars = 1200): string {
  const lines: string[] = [];
  let used = 0;
  for (const hit of hits) {
    const fact = hit.lines[0] ? `: ${hit.lines[0]}` : "";
    const line = `- ${hit.entity} (${hit.type})${fact}`;
    const cost = line.length + 1;
    if (used + cost > capChars) break;
    lines.push(line);
    used += cost;
    for (const extra of hit.lines.slice(1)) {
      const sub = `  ${extra}`;
      if (used + sub.length + 1 > capChars) break;
      lines.push(sub);
      used += sub.length + 1;
    }
  }
  return lines.join("\n");
}

/** One-hop context for a set of seed names (used by orientation). */
export async function hopContext(
  store: GraphStore,
  seeds: string[],
): Promise<RelationSpec[]> {
  const out: RelationSpec[] = [];
  for (const seed of seeds) {
    const { relations } = await store.neighbors(seed, { depth: 1 });
    for (const r of relations) out.push({ from: r.from, to: r.to, relType: r.relType });
  }
  return out;
}
