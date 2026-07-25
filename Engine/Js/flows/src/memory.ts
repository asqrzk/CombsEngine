/**
 * addMemory: one-call memory setup + node instrumentation.
 *
 * ```ts
 * const memory = await addMemory({ type: "kv", scope: "support-bot" });
 * // pass to createReactAgent({ memory }) / createRoleplayChat({ memory })
 * // or wrap any node:
 * const remembered = withMemory(myNode, memory, { agent: "support" });
 * ```
 */

import { KvMemoryStore, SqliteMemoryStore } from "@combs/agents";
import type { MemoryStore } from "@combs/agents";

export interface MemorySpec {
  type: "kv" | "sqlite";
  /** KV/SQLite file path (optional; defaults to Deno KV's default store). */
  path?: string;
  /** Logical scope (KV prefix). */
  scope?: string;
}

/** Creates a memory store with sane defaults. */
export async function addMemory(spec: MemorySpec = { type: "kv" }): Promise<MemoryStore> {
  if (spec.type === "sqlite") {
    return new SqliteMemoryStore(spec.path ?? "combs-memory.sqlite");
  }
  return await KvMemoryStore.open(spec.path, spec.scope ?? "memory");
}

/** Wraps a graph node: recalls relevant memories before, remembers after. */
export function withMemory<S extends Record<string, unknown>>(
  node: (state: S, ctx: unknown) => Promise<Partial<S>> | Partial<S>,
  memory: MemoryStore,
  tags: Record<string, string> = {},
  opts: { recallLimit?: number; rememberKey?: keyof S } = {},
) {
  return async (state: S, ctx: unknown): Promise<Partial<S>> => {
    const memories = await memory.recall(opts.recallLimit ?? 5, tags);
    const enriched = {
      ...state,
      __memories: memories.map((m) => m.content),
    } as S;
    const result = await node(enriched, ctx);
    if (opts.rememberKey && result && opts.rememberKey in result) {
      const content = result[opts.rememberKey];
      await memory.remember(
        typeof content === "string" ? content : JSON.stringify(content),
        tags,
      );
    }
    return result;
  };
}
