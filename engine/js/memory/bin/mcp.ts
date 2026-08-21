/**
 * Launcher: the knowledge-graph MCP stdio server over a real
 * GraphStore. Doors are env: COMBS_MEMORY_DB (store path),
 * COMBS_MEMORY_ENCRYPT (at-rest crypto), COMBS_EMBED_URL reserved for
 * the semantic-recall door. Register with an MCP client as a stdio
 * command: `deno run -A <this file>`.
 */

import { GraphStore } from "../src/store.ts";
import { fileKeyCrypto } from "../src/crypto.ts";
import { formatRecall, recallL1 } from "../src/recall.ts";
import { setCurrentState, whereAreWe } from "../src/state.ts";
import { graphify } from "../src/graphify.ts";
import { type GraphOps, MemoryMcpServer } from "../src/mcp.ts";

const crypto = Deno.env.get("COMBS_MEMORY_ENCRYPT") === "1"
  ? await fileKeyCrypto()
  : undefined;
const store = new GraphStore({
  path: Deno.env.get("COMBS_MEMORY_DB") ?? undefined,
  crypto,
});

const ops: GraphOps = {
  upsertEntities: (specs) => store.upsertEntities(specs),
  addObservations: (entity, contents) => store.addObservations(entity, contents),
  addRelations: (rels) => store.addRelations(rels),
  deleteMixed: async (sel) => {
    let n = 0;
    if (sel.entities?.length) n += await store.deleteEntities(sel.entities);
    if (sel.observationIds?.length) n += await store.deleteObservations(sel.observationIds);
    if (sel.relations?.length) n += await store.deleteRelations(sel.relations);
    return n;
  },
  getEntity: (name) => store.getEntity(name),
  search: (q) => store.search(q),
  neighbors: (name, opts) =>
    store.neighbors(name, opts as { depth?: number; relType?: string } | undefined),
  recall: async (q) => {
    const hits = await recallL1(store, q);
    return { text: formatRecall(hits, q.capChars ?? 1200), hits };
  },
  setState: (project, lines) => setCurrentState(store, project, lines),
  where: (project) => whereAreWe(store, project),
  graphify: (path, project, maxFiles) => graphify(store, path, { project, maxFiles }),
  stats: () => store.stats(),
};

await new MemoryMcpServer(ops).serveStdio();
store.close();
