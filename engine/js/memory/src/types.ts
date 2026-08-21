/**
 * Shared types for the knowledge-graph memory store.
 *
 * Entities carry a generic lifecycle stage (`caste`): entries begin as
 * `youngling`, promote to `worker` through use, are pinned as `queen`
 * explicitly (identity/state-class entries), and demote to `drone` when
 * stale. Recall weights castes; drones are pruning candidates. The
 * thresholds and weights are doors (see `CasteDoors`), not constants.
 */

export type Caste = "youngling" | "worker" | "queen" | "drone";

export interface Entity {
  name: string;
  type: string;
  project: string;
  caste: Caste;
  usageCount: number;
  lastUsedAt: number;
  createdAt: number;
  updatedAt: number;
}

export interface Observation {
  id: number;
  entity: string;
  content: string;
  createdAt: number;
}

export interface Relation {
  from: string;
  to: string;
  relType: string;
  createdAt: number;
}

export interface EntitySpec {
  name: string;
  type: string;
  project?: string;
  caste?: Caste;
  observations?: string[];
  /** Replace this entity's observations instead of appending (refresh semantics). */
  replaceObservations?: boolean;
}

export interface RelationSpec {
  from: string;
  to: string;
  relType: string;
}

export interface SearchQuery {
  query?: string;
  type?: string;
  project?: string;
  limit?: number;
}

export interface EntityHit {
  entity: Entity;
  score: number;
  matched: string[];
}

export interface NeighborsResult {
  entities: Entity[];
  relations: Relation[];
}

export interface GraphStats {
  entities: number;
  observations: number;
  relations: number;
  projects: string[];
  perProject: Record<string, number>;
  castes: Record<string, number>;
}

/** Lifecycle doors: promotion/demotion thresholds and recall weights. */
export interface CasteDoors {
  /** Promote youngling to worker after this many recalls/touches. */
  workerAfterUses: number;
  /** Demote to drone after this many days unused (0 disables). */
  droneAfterDays: number;
  /** Recall score multiplier per caste. */
  weights: Record<Caste, number>;
}

export const DEFAULT_CASTE_DOORS: CasteDoors = {
  workerAfterUses: 3,
  droneAfterDays: 45,
  weights: { queen: 2.0, worker: 1.4, youngling: 1.0, drone: 0.5 },
};

/** At-rest crypto seam (implemented in crypto.ts; door is the constructor). */
export interface ManifestEntry {
  nonce: string;
  wrappedKey: string;
  wrapNonce: string;
  sha256: string;
}

export interface BlobCrypto {
  seal(data: Uint8Array): Promise<{ blob: string; entry: ManifestEntry }>;
  open(blob: string, entry: ManifestEntry): Promise<Uint8Array>;
}
