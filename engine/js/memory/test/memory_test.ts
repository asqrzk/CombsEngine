import { assert, assertEquals } from "jsr:@std/assert@^1.0.0";
import { GraphStore } from "../src/store.ts";

async function tempStore(opts: Record<string, unknown> = {}): Promise<GraphStore> {
  const dir = await Deno.makeTempDir({ prefix: "combs-memory-test" });
  return new GraphStore({ path: `${dir}/graph.db`, ...opts });
}

Deno.test("upsert is idempotent and preserves caste on re-upsert", async () => {
  const store = await tempStore();
  await store.upsertEntities([{ name: "p/a", type: "file", project: "p" }]);
  await store.touchEntities(["p/a"]);
  await store.touchEntities(["p/a"]);
  await store.touchEntities(["p/a"]);
  const promoted = (await store.getEntity("p/a"))!.entity;
  assertEquals(promoted.caste, "worker");
  // Re-upsert without caste (a graphify re-run) must not demote.
  await store.upsertEntities([{ name: "p/a", type: "file", project: "p" }]);
  assertEquals((await store.getEntity("p/a"))!.entity.caste, "worker");
  const stats = await store.stats();
  assertEquals(stats.entities, 1);
  store.close();
});

Deno.test("cascade delete removes observations and relations", async () => {
  const store = await tempStore();
  await store.upsertEntities([
    { name: "p/a", type: "file", project: "p", observations: ["fact one"] },
    { name: "p/b", type: "file", project: "p" },
  ]);
  await store.addRelations([{ from: "p/a", to: "p/b", relType: "imports" }]);
  await store.deleteEntities(["p/a"]);
  const stats = await store.stats();
  assertEquals(stats.entities, 1);
  assertEquals(stats.observations, 0);
  assertEquals(stats.relations, 0);
  store.close();
});

Deno.test("search scopes by project and type", async () => {
  const store = await tempStore();
  await store.upsertEntities([
    { name: "p/mod", type: "module", project: "p" },
    { name: "q/mod", type: "module", project: "q" },
    { name: "p/doc", type: "doc-section", project: "p" },
  ]);
  const hits = await store.search({ project: "p", type: "module" });
  assertEquals(hits.length, 1);
  assertEquals(hits[0].entity.name, "p/mod");
  const byName = await store.search({ query: "doc" });
  assertEquals(byName.length, 1);
  store.close();
});

Deno.test("neighbors walks depth 2 both directions", async () => {
  const store = await tempStore();
  await store.upsertEntities([
    { name: "a", type: "t" },
    { name: "b", type: "t" },
    { name: "c", type: "t" },
  ]);
  await store.addRelations([
    { from: "a", to: "b", relType: "r" },
    { from: "b", to: "c", relType: "r" },
  ]);
  const one = await store.neighbors("a", { depth: 1 });
  assertEquals(one.entities.map((e) => e.name), ["b"]);
  const two = await store.neighbors("a", { depth: 2 });
  assertEquals(two.entities.map((e) => e.name).sort(), ["b", "c"]);
  assertEquals(two.relations.length, 2);
  store.close();
});

Deno.test("maturation demotes stale entries but never state/pointer", async () => {
  const store = await tempStore({ casteDoors: { droneAfterDays: 1 } });
  await store.upsertEntities([
    { name: "old", type: "file" },
    { name: "p/current-state", type: "current-state" },
  ]);
  // Fresh rows must not demote; pinned types never demote. (Backdating
  // through the public API is impossible, so staleness demotion itself
  // is covered by the sweep SQL reading MAX(last_used_at, updated_at).)
  const res = await store.matureEntities();
  assertEquals(res.demoted, 0);
  assert((await store.getEntity("p/current-state"))!.entity.caste !== "drone");
  store.close();
});

Deno.test("edge list indexes nodes stably for traversal", async () => {
  const store = await tempStore();
  await store.upsertEntities([
    { name: "a", type: "t", project: "p" },
    { name: "b", type: "t", project: "p" },
  ]);
  await store.addRelations([{ from: "a", to: "b", relType: "r" }]);
  const { nodes, edges } = await store.edgeList("p");
  assertEquals(nodes.length, 2);
  assertEquals(edges, [[0, 1]]);
  store.close();
});

Deno.test("embeddings round-trip as float32", async () => {
  const store = await tempStore();
  await store.upsertEntities([{ name: "a", type: "t" }]);
  await store.putEmbedding("a", new Float32Array([0.25, -1, 0.5]));
  const all = await store.embeddings();
  assertEquals(all.length, 1);
  assertEquals(Array.from(all[0].vec), [0.25, -1, 0.5]);
  assertEquals(await store.missingEmbeddings(), []);
  store.close();
});

// ── recall + current-state ──────────────────────────────────────────

import { formatRecall, recallL1 } from "../src/recall.ts";
import { setCurrentState, whereAreWe } from "../src/state.ts";

Deno.test("recall ranks caste-weighted keyword hits and touches them", async () => {
  const store = await tempStore();
  await store.upsertEntities([
    { name: "p/kv-cache", type: "file", project: "p", observations: ["paged kv cache with prefix reuse"] },
    { name: "p/sampler", type: "file", project: "p", observations: ["temperature and top-p sampling"] },
    { name: "p/plain", type: "note", project: "p", observations: ["kv cache design decision record"] },
    { name: "p/royal", type: "note", project: "p", caste: "queen", observations: ["kv cache design decision record"] },
  ]);
  const hits = await recallL1(store, { text: "how does the kv cache work", project: "p" });
  assert(hits.length >= 3);
  // A name+content match legitimately dominates any caste boost...
  assertEquals(hits[0].entity, "p/kv-cache");
  // ...but on EQUAL signals the queen outranks the youngling twin.
  const royal = hits.findIndex((h) => h.entity === "p/royal");
  const plain = hits.findIndex((h) => h.entity === "p/plain");
  assert(royal !== -1 && plain !== -1 && royal < plain);
  // Recall touched the winners.
  const touched = (await store.getEntity("p/kv-cache"))!.entity;
  assertEquals(touched.usageCount, 1);
  const text = formatRecall(hits, 200);
  assert(text.startsWith("- p/kv-cache (file)"));
  assert(text.length <= 200);
  store.close();
});

Deno.test("current-state replaces observations and repoints now", async () => {
  const store = await tempStore();
  await setCurrentState(store, "alpha", ["on branch x", "next: land y"]);
  await setCurrentState(store, "beta", ["waiting on z"]);
  await setCurrentState(store, "alpha", ["on branch x2"]);
  const where = await whereAreWe(store);
  assert(where.includes("alpha"));
  assert(where.includes("x2"));
  assert(!where.includes("branch x\n"));
  const beta = await whereAreWe(store, "beta");
  assert(beta.includes("waiting on z"));
  // now has exactly one points-to.
  assertEquals(store.relationsFrom("now").filter((r) => r.relType === "points-to").length, 1);
  store.close();
});

// ── at-rest crypto door ─────────────────────────────────────────────

import { fileKeyCrypto } from "../src/crypto.ts";

Deno.test("crypto door round-trips and surfaces tamper as unreadable", async () => {
  const dir = await Deno.makeTempDir({ prefix: "combs-memory-crypt" });
  const crypto = await fileKeyCrypto(`${dir}/master.key`);
  const store = new GraphStore({ path: `${dir}/graph.db`, crypto });
  await store.upsertEntities([{ name: "s", type: "note", observations: ["the secret fact"] }]);
  const obs = await store.observationsFor("s");
  assertEquals(obs[0].content, "the secret fact");
  // On-disk row is ciphertext, not plaintext.
  const raw = new (await import("node:sqlite")).DatabaseSync(`${dir}/graph.db`);
  const row = raw.prepare("SELECT content, enc FROM observations").get() as unknown as {
    content: string;
    enc: string | null;
  };
  assert(row.enc !== null);
  assert(!row.content.includes("secret"));
  // Tamper: flip the ciphertext → read yields [unreadable], never junk.
  raw.prepare("UPDATE observations SET content = ?").run("AAAA" + row.content.slice(4));
  raw.close();
  const store2 = new GraphStore({ path: `${dir}/graph.db`, crypto });
  const tampered = await store2.observationsFor("s");
  assertEquals(tampered[0].content, "[unreadable]");
  // Without the crypto door, encrypted rows read as [encrypted].
  const store3 = new GraphStore({ path: `${dir}/graph.db` });
  assertEquals((await store3.observationsFor("s"))[0].content, "[encrypted]");
  store.close();
  store2.close();
  store3.close();
});

// ── graphify ────────────────────────────────────────────────────────

import { graphify } from "../src/graphify.ts";

Deno.test("graphify ingests this workspace with import edges and refreshes idempotently", async () => {
  const store = await tempStore();
  // The engine/js workspace itself is the fixture: real files, real git.
  const wsPath = new URL("../..", import.meta.url).pathname.replace(/\/$/, "");
  const res = await graphify(store, wsPath, { project: "jsws" });
  assert(res.files > 20, `files: ${res.files}`);
  assert(res.entities > 30);
  assert(res.relations > res.files);
  // A known import edge: mesh/src/mcp.ts imports from mesh/src/mesh.ts.
  const got = await store.getEntity("jsws/mesh/src/mcp.ts");
  assert(got, "mcp.ts entity missing");
  const imports = got!.out.filter((r) => r.relType === "imports").map((r) => r.to);
  assert(imports.includes("jsws/mesh/src/mesh.ts"), `imports: ${imports.join(", ")}`);
  // Doc sections extracted from a markdown file exist.
  const sections = await store.search({ project: "jsws", type: "doc-section", limit: 5 });
  assert(sections.length > 0);
  // Re-run converges: same structural population, hand-written survives.
  await store.upsertEntities([{ name: "jsws/hand-note", type: "note", project: "jsws" }]);
  const res2 = await graphify(store, wsPath, { project: "jsws" });
  assertEquals(res2.files, res.files);
  assert(await store.getEntity("jsws/hand-note"), "hand-written entity was dropped");
  store.close();
});

Deno.test("graphify reads python: docstring over license, import edges, maxFiles door", async () => {
  const store = await tempStore();
  const dir = await Deno.makeTempDir({ prefix: "graphify-py" });
  await Deno.mkdir(`${dir}/src/pkg`, { recursive: true });
  await Deno.writeTextFile(
    `${dir}/src/pkg/__init__.py`,
    "# Copyright 2026 The Fixture Authors.\n" +
      '# Licensed under the Apache License, Version 2.0 (the "License");\n' +
      "# you may not use this file except in compliance with the License.\n" +
      '\n"""The fixture package."""\n',
  );
  await Deno.writeTextFile(
    `${dir}/src/pkg/util.py`,
    '"""Helpers that trim the seam."""\nfrom . import config\n',
  );
  await Deno.writeTextFile(
    `${dir}/src/pkg/asf.py`,
    "# Licensed to the Apache Software Foundation (ASF) under one\n" +
      "# or more contributor license agreements.\n\nimport os\n",
  );
  await Deno.writeTextFile(`${dir}/notes.md`, "use pkgctl to inspect the arena\n");
  await Deno.writeTextFile(`${dir}/src/pkg/config.py`, "WIDTH = 3\n");
  await Deno.writeTextFile(`${dir}/main.py`, "import pkg.util\n");
  const res = await graphify(store, dir, { project: "pyfix" });
  assertEquals(res.files, 6);
  // The docstring, not the license header, is the file's observation.
  const init = await store.getEntity("pyfix/src/pkg/__init__.py");
  assertEquals(init!.observations.map((o) => o.content), ["The fixture package."]);
  // Relative and src-layout absolute imports resolve to in-repo edges.
  const util = await store.getEntity("pyfix/src/pkg/util.py");
  assert(
    util!.out.some((r) => r.relType === "imports" && r.to === "pyfix/src/pkg/config.py"),
    `util edges: ${util!.out.map((r) => r.to).join(", ")}`,
  );
  const main = await store.getEntity("pyfix/main.py");
  assert(
    main!.out.some((r) => r.relType === "imports" && r.to === "pyfix/src/pkg/util.py"),
    `main edges: ${main!.out.map((r) => r.to).join(", ")}`,
  );
  // The ASF header ("Licensed to ...") is never the observation, and a
  // markdown prose line opening with an import-like word survives.
  assertEquals((await store.getEntity("pyfix/src/pkg/asf.py"))!.observations, []);
  assertEquals(
    (await store.getEntity("pyfix/notes.md"))!.observations.map((o) => o.content),
    ["use pkgctl to inspect the arena"],
  );
  // The maxFiles door caps the walk and reports what it skipped.
  const capped = await graphify(store, dir, { project: "pycap", maxFiles: 2 });
  assert(capped.skipped >= 1, `skipped: ${capped.skipped}`);
  // A truncated re-run must NOT sweep what it did not enumerate, and
  // non-finite junk falls back to the default cap instead of slicing
  // the listing to nothing.
  const before = (await store.stats()).entities;
  await graphify(store, dir, { project: "pyfix", maxFiles: 2 });
  await graphify(store, dir, { project: "pyfix", maxFiles: Number.NaN });
  assertEquals((await store.stats()).entities, before);
  assert(await store.getEntity("pyfix/main.py"), "capped re-run swept main.py");
  store.close();
  await Deno.remove(dir, { recursive: true });
});

// ── MCP server ──────────────────────────────────────────────────────

import { MemoryMcpServer } from "../src/mcp.ts";
import { formatRecall as fmtR, recallL1 as rcl } from "../src/recall.ts";
import { setCurrentState as setSt, whereAreWe as whr } from "../src/state.ts";
import { graphify as gfy } from "../src/graphify.ts";

Deno.test("mcp server: initialize, tools/list, and tool round-trips over a real store", async () => {
  const store = await tempStore();
  const ops = {
    upsertEntities: (s: Parameters<typeof store.upsertEntities>[0]) => store.upsertEntities(s),
    addObservations: (e: string, c: string[]) => store.addObservations(e, c),
    addRelations: (r: Parameters<typeof store.addRelations>[0]) => store.addRelations(r),
    deleteMixed: async () => 0,
    getEntity: (n: string) => store.getEntity(n),
    search: (q: Parameters<typeof store.search>[0]) => store.search(q),
    neighbors: (n: string, o?: { depth?: number }) => store.neighbors(n, o),
    recall: async (q: { text: string; project?: string; k?: number }) => {
      const hits = await rcl(store, q);
      return { text: fmtR(hits), hits };
    },
    setState: (p: string, l: string[]) => setSt(store, p, l),
    where: (p?: string) => whr(store, p),
    graphify: (path: string, project?: string) => gfy(store, path, { project }),
    stats: () => store.stats(),
  };
  const server = new MemoryMcpServer(ops);
  const init = await server.handle({ jsonrpc: "2.0", id: 1, method: "initialize" });
  assertEquals((init!.result as { serverInfo: { name: string } }).serverInfo.name, "combs-memory");
  const list = await server.handle({ jsonrpc: "2.0", id: 2, method: "tools/list" });
  assertEquals((list!.result as { tools: unknown[] }).tools.length, 12);
  // Notification gets no answer.
  assertEquals(await server.handle({ jsonrpc: "2.0", method: "notifications/initialized" }), null);
  // Round-trip: upsert → state → where.
  await server.handle({
    jsonrpc: "2.0",
    id: 3,
    method: "tools/call",
    params: {
      name: "memory_upsert_entities",
      arguments: { entities: [{ name: "x/f", type: "file", project: "x" }] },
    },
  });
  await server.handle({
    jsonrpc: "2.0",
    id: 4,
    method: "tools/call",
    params: { name: "memory_set_state", arguments: { project: "x", lines: ["at the start"] } },
  });
  const where = await server.handle({
    jsonrpc: "2.0",
    id: 5,
    method: "tools/call",
    params: { name: "memory_where", arguments: {} },
  });
  const text = JSON.parse(
    (where!.result as { content: { text: string }[] }).content[0].text,
  ).text as string;
  assert(text.includes("at the start"));
  // Unknown tool surfaces as a JSON-RPC error, not a crash.
  const bad = await server.handle({
    jsonrpc: "2.0",
    id: 6,
    method: "tools/call",
    params: { name: "nope" },
  });
  assert(bad!.error);
  store.close();
});

// ── embeddings (mocked engine) ──────────────────────────────────────

import { EmbedClient, embedMissing, recallHybrid } from "../src/embed.ts";

Deno.test("embed client normalizes, backfills, and hybrid recall merges", async () => {
  const store = await tempStore();
  await store.upsertEntities([
    { name: "h/alpha", type: "note", project: "h", observations: ["speculative decoding seam"] },
    { name: "h/beta", type: "note", project: "h", observations: ["sampling temperature guard"] },
  ]);
  // Mock /v1/embeddings: axis-ish vectors keyed on content keywords.
  const realFetch = globalThis.fetch;
  globalThis.fetch = ((_url: unknown, init?: RequestInit) => {
    const body = JSON.parse(String(init?.body ?? "{}"));
    const data = (body.input as string[]).map((t) => ({
      embedding: t.includes("specul") || t.includes("draft") ? [3, 0, 0] : [0, 3, 0],
    }));
    return Promise.resolve(new Response(JSON.stringify({ data }), { status: 200 }));
  }) as typeof fetch;
  try {
    const client = new EmbedClient("http://mock:1");
    const [v] = await client.embed(["speculative things"]);
    assert(Math.abs(Math.hypot(...v) - 1) < 1e-5, "vector must be re-normalized");
    const n = await embedMissing(store, client);
    assertEquals(n, 2);
    // "draft tokens" shares no KEYWORD with the alpha note — L1 misses,
    // the mocked semantic space hits.
    const hits = await recallHybrid(store, client, { text: "draft tokens", project: "h" });
    assert(hits.length >= 1);
    assertEquals(hits[0].entity, "h/alpha");
  } finally {
    globalThis.fetch = realFetch;
  }
  store.close();
});

// ── sanitize ─────────────────────────────────────────────────────────

import { cleanText, looksBinary } from "../src/sanitize.ts";

Deno.test("sanitize strips bidi, tag, and zero-width poison at every write door", async () => {
  // Bidi override, PDF, zero-width space, tag characters, BOM.
  const poisoned = "run\u202E me\u202C no\u200Bw \u{E0041}\u{E0042}safe\uFEFF";
  assertEquals(cleanText(poisoned), "run me now safe");
  assertEquals(cleanText("plain \u00e9tude keeps accents"), "plain \u00e9tude keeps accents");
  // Escape sequences go as whole sequences — payload included.
  assertEquals(cleanText("red \x1B[31mtext\x1B[0m done"), "red text done");
  assertEquals(cleanText("\x1B]0;evil title\x07body"), "body");
  // Category strip: private use, noncharacters, variation selectors.
  assertEquals(cleanText("a\u00ADb\uE000c\uFDD0d\uFE0F"), "abcd");
  assert(looksBinary("\u0000binary"));
  assert(looksBinary("\uFFFD\uFFFD\uFFFD\uFFFD\uFFFDx"));
  assert(!looksBinary("one boundary cut \uFFFD at the tail of a long enough line"));
  // The store applies it at the write boundary, not just in graphify.
  const store = await tempStore();
  await store.upsertEntities([{
    name: "p/cl\u200Bean",
    type: "file",
    project: "p",
    observations: ["fact\u202E with poison\u202C inside"],
  }]);
  const got = await store.getEntity("p/clean");
  assert(got, "poisoned name was not cleaned to p/clean");
  assertEquals(got!.observations[0].content, "fact with poison inside");
  // Symmetry: reads, touches, deletes, and the state walk address the
  // same cleaned key space as writes.
  assert(await store.getEntity("p/cl\u200Bean"), "raw-name read missed the cleaned entity");
  const echoed = await store.addObservations("p/cl\u200Bean", ["tail\u202E fact"]);
  assertEquals(echoed[0].content, "tail fact");
  await setSt(store, "p\u200Broj", ["mid-walk"]);
  assert((await whr(store, "proj")).includes("mid-walk"), "state walk broke on poisoned project");
  // Junk castes never land — the enum is the boundary.
  await store.upsertEntities([
    { name: "p/junk", type: "file", project: "p", caste: "evil\u202E" as never },
  ]);
  assertEquals((await store.getEntity("p/junk"))!.entity.caste, "youngling");
  assertEquals(await store.deleteEntities(["p/cl\u200Bean"]), 1);
  store.close();
});
