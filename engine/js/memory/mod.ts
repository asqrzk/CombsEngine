/**
 * @combs/memory — knowledge-graph memory (L2).
 *
 * - `GraphStore`: SQLite entities/observations/relations/embeddings
 *   with a generic lifecycle stage (caste) on entities and an optional
 *   at-rest crypto door on observation contents.
 * - `recall.ts` / `state.ts`: ranked compact recall and the
 *   current-state / `now` orientation convention.
 * - `graphify.ts`: ingest a repository into the graph.
 * - `MemoryMcpServer`: MCP stdio server exposing the graph as tools
 *   (bin/mcp.ts is the launcher).
 */
export {
  escapeLike,
  GraphStore,
  type GraphStoreOptions,
} from "./src/store.ts";
export {
  formatRecall,
  hopContext,
  recallL1,
  type RecallHit,
  type RecallQuery,
  tokenize,
} from "./src/recall.ts";
export { setCurrentState, whereAreWe } from "./src/state.ts";
export { fileKeyCrypto } from "./src/crypto.ts";
export { graphify, type GraphifyResult } from "./src/graphify.ts";
export { type GraphOps, MemoryMcpServer } from "./src/mcp.ts";
export {
  EmbedClient,
  embedMissing,
  recallHybrid,
  recallL2,
} from "./src/embed.ts";
export {
  activate,
  type ActivationResult,
  nativeAvailable,
  type TraversalBackend,
} from "./src/traversal.ts";
export {
  type BlobCrypto,
  type Caste,
  type CasteDoors,
  DEFAULT_CASTE_DOORS,
  type Entity,
  type EntityHit,
  type EntitySpec,
  type GraphStats,
  type ManifestEntry,
  type NeighborsResult,
  type Observation,
  type Relation,
  type RelationSpec,
  type SearchQuery,
} from "./src/types.ts";
