/**
 * The current-state convention — "remember where we are always".
 *
 * Each project keeps a `${project}/current-state` entity (caste queen —
 * state is identity-class and never demotes) whose observations are
 * REPLACED on set: state is a snapshot, not a log. A singleton `now`
 * pointer entity holds exactly one `points-to` relation naming the
 * active project's state.
 */

import type { GraphStore } from "./store.ts";

const NOW = "now";

export async function setCurrentState(
  store: GraphStore,
  project: string,
  lines: string[],
): Promise<void> {
  const name = `${project}/current-state`;
  // Two short transactions: the upsert (with observation replacement)
  // and the repoint are each atomic, so neither merged snapshots nor
  // twin `now` pointers can exist; losing the race BETWEEN them to a
  // concurrent setCurrentState just means the later writer wins whole.
  await store.upsertEntities([
    {
      name,
      type: "current-state",
      project,
      caste: "queen",
      observations: lines.slice(0, 8),
      replaceObservations: true,
    },
    { name: NOW, type: "pointer", caste: "queen" },
  ]);
  store.repointRelation(NOW, name, "points-to");
}

export async function whereAreWe(store: GraphStore, project?: string): Promise<string> {
  let stateName: string | null = null;
  if (project) {
    stateName = `${project}/current-state`;
  } else {
    const pointed = store.relationsFrom(NOW).find((r) => r.relType === "points-to");
    stateName = pointed?.to ?? null;
  }
  if (!stateName) {
    const stats = await store.stats();
    return stats.projects.length
      ? `No current-state set. Projects known: ${stats.projects.join(", ")}.`
      : "The graph is empty.";
  }
  const got = await store.getEntity(stateName);
  if (!got) return `No state recorded at ${stateName}.`;
  const lines = got.observations
    .slice()
    .reverse()
    .map((o) => `- ${o.content}`);
  const related = [...got.out, ...got.inn]
    .filter((r) => r.relType !== "points-to")
    .slice(0, 6)
    .map((r) => `  ${r.from} —[${r.relType}]→ ${r.to}`);
  const proj = got.entity.project || stateName.replace(/\/current-state$/, "");
  return [
    `Where we are — ${proj}:`,
    ...lines,
    ...(related.length ? ["Nearby:", ...related] : []),
  ].join("\n");
}
