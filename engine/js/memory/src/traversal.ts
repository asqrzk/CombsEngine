/**
 * Traversal door: `sql` (the in-process BFS over SQLite rows — always
 * available), `native` (CPU spreading activation in libcombsgraph_ffi),
 * `gpu` (the same loop on Burn/wgpu inside the dylib; the dylib itself
 * falls back to CPU over its dense cap and NAMES the fallback in
 * `backend`). dlopen is lazy and failure degrades to `sql` — the door
 * never becomes a wall.
 */

import type { GraphStore } from "./store.ts";

export type TraversalBackend = "sql" | "native" | "gpu";

export interface ActivationResult {
  /** entity name → activation score */
  scores: Map<string, number>;
  backend: string;
  stepsRun?: number;
  ms: number;
}

interface GraphkitLib {
  symbols: {
    combsgraph_op_json: (input: Uint8Array) => Deno.PointerValue;
    combsgraph_string_free: (s: Deno.PointerValue) => void;
  };
  close(): void;
}

let lib: GraphkitLib | null = null;
let libTried = false;

function libCandidates(): string[] {
  const env = Deno.env.get("COMBS_GRAPHKIT_LIB");
  if (env) return [env];
  const ext = Deno.build.os === "darwin"
    ? "dylib"
    : Deno.build.os === "windows"
    ? "dll"
    : "so";
  const roots = [
    new URL("../../../core/target/release/", import.meta.url).pathname,
    new URL("../../../core/target/debug/", import.meta.url).pathname,
  ];
  return roots.map((r) => `${r}libcombsgraph_ffi.${ext}`);
}

function openLib(): GraphkitLib | null {
  if (libTried) return lib;
  libTried = true;
  for (const path of libCandidates()) {
    try {
      const opened = Deno.dlopen(path, {
        combsgraph_op_json: { parameters: ["buffer"], result: "pointer" },
        combsgraph_string_free: { parameters: ["pointer"], result: "void" },
      });
      lib = opened as unknown as GraphkitLib;
      break;
    } catch {
      // try the next candidate; absent dylib = door stays sql
    }
  }
  return lib;
}

function callOp(request: unknown): Record<string, unknown> | null {
  const l = openLib();
  if (!l) return null;
  const input = new TextEncoder().encode(JSON.stringify(request) + "\0");
  const ptr = l.symbols.combsgraph_op_json(input);
  if (ptr === null) return null;
  try {
    const view = new Deno.UnsafePointerView(ptr as Deno.PointerObject);
    return JSON.parse(view.getCString());
  } finally {
    l.symbols.combsgraph_string_free(ptr);
  }
}

/** True when the native dylib is loadable on this machine. */
export function nativeAvailable(): boolean {
  return openLib() !== null;
}

/**
 * Multi-hop relevance from seed entities. `sql` runs the decayed BFS
 * in-process; `native`/`gpu` run the convergent activation loop in the
 * dylib (which reports the backend it ACTUALLY used).
 */
export async function activate(
  store: GraphStore,
  seeds: string[],
  backend: TraversalBackend = "sql",
  opts: { project?: string; damping?: number; maxSteps?: number; depth?: number } = {},
): Promise<ActivationResult> {
  const t0 = performance.now();
  const { nodes, edges } = await store.edgeList(opts.project);
  const index = new Map(nodes.map((n, i) => [n, i]));
  const seedIdx = seeds.map((s) => index.get(s)).filter((i): i is number => i !== undefined);

  if (backend !== "sql") {
    const out = callOp({
      op: "activate",
      nodes: nodes.length,
      edges,
      seeds: seedIdx,
      damping: opts.damping,
      max_steps: opts.maxSteps,
      backend,
    });
    if (out && !out.error) {
      const raw = out.scores as number[];
      const scores = new Map<string, number>();
      for (let i = 0; i < nodes.length; i++) if (raw[i] > 1e-9) scores.set(nodes[i], raw[i]);
      return {
        scores,
        backend: String(out.backend),
        stepsRun: Number(out.steps_run),
        ms: performance.now() - t0,
      };
    }
    // dylib absent or op failed — degrade honestly to sql
  }

  const depth = opts.depth ?? 3;
  const decay = 0.5;
  const scores = new Map<string, number>();
  const adj = new Map<number, number[]>();
  for (const [u, v] of edges) {
    let a = adj.get(u);
    if (!a) adj.set(u, a = []);
    a.push(v);
  }
  let frontier = seedIdx;
  for (const i of seedIdx) scores.set(nodes[i], 1);
  let gain = 1;
  for (let d = 0; d < depth && frontier.length; d++) {
    gain *= decay;
    const next: number[] = [];
    for (const u of frontier) {
      for (const v of adj.get(u) ?? []) {
        if (!scores.has(nodes[v])) {
          scores.set(nodes[v], gain);
          next.push(v);
        }
      }
    }
    frontier = next;
  }
  return { scores, backend: "sql", ms: performance.now() - t0 };
}
