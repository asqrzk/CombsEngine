/**
 * graphify — ingest a repository into the graph.
 *
 * Enumeration is `git ls-files` (tracked files only, so gitignored
 * private material never enters the graph from a repo walk); non-git
 * directories fall back to a bounded walk. Entities are namespaced
 * `${project}/...` because entity names are a machine-global PK.
 * Re-runs are full refreshes per project: everything seen is upserted,
 * structural entities NOT seen are deleted (cascade) — hand-written
 * entities and current-state survive via the type filter.
 */

import type { GraphStore } from "./store.ts";
import type { EntitySpec, RelationSpec } from "./types.ts";
import { cleanText, looksBinary } from "./sanitize.ts";

export interface GraphifyResult {
  project: string;
  entities: number;
  relations: number;
  files: number;
  skipped: number;
}

const DEFAULT_MAX_FILES = 5000;
const MAX_FILES_CEILING = 50000;
const MAX_FILE_BYTES = 512 * 1024;
const READ_BYTES = 64 * 1024;
const STRUCTURAL_TYPES = ["repo", "module", "file", "doc-section"];
const SOURCE_EXT = /\.(ts|js|tsx|jsx|rs|md|py)$/;
const ANCHOR_FILES = new Set(["deno.json", "Cargo.toml", "package.json", "pyproject.toml"]);

async function listFiles(
  repoPath: string,
  maxFiles: number,
): Promise<{ files: string[]; skipped: number }> {
  try {
    const out = await new Deno.Command("git", {
      args: ["ls-files"],
      cwd: repoPath,
      stdout: "piped",
      stderr: "null",
    }).output();
    if (out.success) {
      const all = new TextDecoder().decode(out.stdout).split("\n").filter(Boolean);
      return { files: all.slice(0, maxFiles), skipped: Math.max(0, all.length - maxFiles) };
    }
  } catch {
    // git absent — fall through to the bounded walk
  }
  const files: string[] = [];
  let skipped = 0;
  const EXCLUDE = new Set([".git", "node_modules", "target", "dist", ".cache", ".venv", "venv", "__pycache__"]);
  async function walk(dir: string, rel: string): Promise<void> {
    if (files.length >= maxFiles) return;
    for await (const e of Deno.readDir(dir)) {
      if (files.length >= maxFiles) {
        skipped++;
        continue;
      }
      const relPath = rel ? `${rel}/${e.name}` : e.name;
      if (e.isDirectory) {
        if (!EXCLUDE.has(e.name)) await walk(`${dir}/${e.name}`, relPath);
      } else if (e.isFile) {
        files.push(relPath);
      }
    }
  }
  await walk(repoPath, "");
  return { files, skipped };
}

async function readHead(path: string): Promise<string | null> {
  try {
    const stat = await Deno.stat(path);
    if (stat.size > MAX_FILE_BYTES) return null;
    const f = await Deno.open(path, { read: true });
    const buf = new Uint8Array(Math.min(READ_BYTES, stat.size));
    let n = 0;
    while (n < buf.length) {
      const r = await f.read(buf.subarray(n));
      if (r === null) break;
      n += r;
    }
    f.close();
    return new TextDecoder().decode(buf.subarray(0, n));
  } catch {
    return null;
  }
}

/**
 * First doc-comment, docstring, or contentful line — the file's one
 * observation. A docstring beats a leading comment (Python files open
 * with license headers; the docstring is the document), and a
 * copyright/license block is never the observation — it says nothing
 * about the file.
 */
function firstDocLine(text: string, skipImports = true): string | null {
  let comment: string | null = null;
  let inLicense = false;
  for (const raw of text.split("\n").slice(0, 40)) {
    const line = raw.trim();
    if (!line) {
      inLicense = false;
      continue;
    }
    const ds = line.match(/^[rub]{0,2}["']{3}\s*(.*?)\s*(?:["']{3})?\s*$/i);
    if (ds && ds[1]) return ds[1].slice(0, 200);
    const m = line.match(/^(?:\/\/[/!]?|\/?\*+|#)\s*(.+?)\s*\*?\/?$/);
    if (m && m[1] && !/^![[]/.test(m[1])) {
      if (/copyright\b|licen[cs]ed?\b|all rights reserved/i.test(m[1])) {
        inLicense = true;
        continue;
      }
      if (!inLicense && comment === null) comment = m[1].slice(0, 200);
      continue;
    }
    if (!/^[/*#{}[\]()'"`;,\s-]/.test(line)) {
      // An import line is not a document — skip it; honest absence
      // beats storing `import x` as the file's one fact. Only for code
      // files: markdown prose legitimately opens with these words.
      if (
        skipImports &&
        /^(?:import\b|from\s+\S+\s+import\b|export\s.*\sfrom\s|(?:pub\s+)?(?:use|mod|extern)\b|require\()/.test(line)
      ) {
        continue;
      }
      return comment ?? line.slice(0, 200);
    }
  }
  return comment;
}

function slugify(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 60);
}

interface MdSection {
  slug: string;
  heading: string;
  firstLine: string;
}

function markdownSections(text: string): MdSection[] {
  const out: MdSection[] = [];
  const lines = text.split("\n");
  for (let i = 0; i < lines.length && out.length < 20; i++) {
    const m = lines[i].match(/^#{1,2}\s+(.+)/);
    if (!m) continue;
    let first = "";
    for (let j = i + 1; j < lines.length; j++) {
      const t = lines[j].trim();
      if (t.match(/^#{1,2}\s/)) break;
      if (t) {
        first = t.slice(0, 160);
        break;
      }
    }
    out.push({ slug: slugify(m[1]), heading: m[1].trim(), firstLine: first });
  }
  return out;
}

const TSJS_IMPORT =
  /^\s*(?:import|export)\s[^'"]*['"]([^'"]+)['"]|require\(\s*['"]([^'"]+)['"]\s*\)|import\(\s*['"]([^'"]+)['"]\s*\)/;

function resolveRelative(fromFile: string, spec: string, fileSet: Set<string>): string | null {
  if (!spec.startsWith("./") && !spec.startsWith("../")) return null;
  const baseDir = fromFile.includes("/") ? fromFile.slice(0, fromFile.lastIndexOf("/")) : "";
  const parts = (baseDir ? `${baseDir}/${spec}` : spec).split("/");
  const stack: string[] = [];
  for (const part of parts) {
    if (part === "." || part === "") continue;
    else if (part === "..") {
      // Escaping the repo root is not an in-repo edge — clamping to
      // root would manufacture false imports.
      if (!stack.length) return null;
      stack.pop();
    } else stack.push(part);
  }
  const joined = stack.join("/");
  const candidates = [
    joined,
    `${joined}.ts`,
    `${joined}.js`,
    `${joined}.tsx`,
    `${joined}.jsx`,
    `${joined}/index.ts`,
    `${joined}/index.js`,
    joined.replace(/\.js$/, ".ts"),
  ];
  for (const c of candidates) if (fileSet.has(c)) return c;
  return null;
}

function pyEdges(fromFile: string, text: string, fileSet: Set<string>): string[] {
  const out: string[] = [];
  const dir = fromFile.includes("/") ? fromFile.slice(0, fromFile.lastIndexOf("/")) : "";
  const tryModule = (base: string, mod: string): string | null => {
    const rel = mod.replace(/\./g, "/");
    const joined = base ? `${base}/${rel}` : rel;
    for (const c of [`${joined}.py`, `${joined}/__init__.py`]) {
      if (fileSet.has(c)) return c;
    }
    return null;
  };
  // `from X import a, b as c` — the names may themselves be modules;
  // resolve each before falling back to the package/module edge.
  const parseNames = (list: string): string[] =>
    list.split(",").map((t) => t.trim().split(/\s+as\s+/)[0])
      .filter((n) => /^\w+$/.test(n)).slice(0, 8);
  for (const raw of text.split("\n")) {
    // Relative: `from .sibling import x`, `from ..pkg.mod import x`.
    const rel = raw.match(/^\s*from\s+(\.+)([\w.]*)\s+import\s+(.+)/);
    if (rel) {
      // One dot is the current package; each extra dot pops a level.
      // Escaping the repo root is not an in-repo edge.
      let base: string | null = dir;
      for (let i = 1; i < rel[1].length && base !== null; i++) {
        base = base === "" ? null : base.includes("/") ? base.slice(0, base.lastIndexOf("/")) : "";
      }
      if (base !== null) {
        const pkg = rel[2]
          ? [base, rel[2].replace(/\./g, "/")].filter(Boolean).join("/")
          : base;
        let found = false;
        for (const nm of parseNames(rel[3])) {
          const t = tryModule(pkg, nm);
          if (t) {
            out.push(t);
            found = true;
          }
        }
        if (!found) {
          const target = rel[2]
            ? tryModule(base, rel[2])
            : fileSet.has(pkg ? `${pkg}/__init__.py` : "__init__.py")
            ? (pkg ? `${pkg}/__init__.py` : "__init__.py")
            : null;
          if (target) out.push(target);
        }
      }
      continue;
    }
    // Absolute: `import pkg.mod`, `from pkg.mod import x` — resolved
    // at the repo root and under src/ (the setuptools src layout).
    const abs = raw.match(/^\s*(?:from\s+([\w.]+)\s+import\s+(.+)|import\s+([\w.]+))/);
    if (abs) {
      const mod = abs[1] ?? abs[3];
      let found = false;
      for (const nm of abs[2] ? parseNames(abs[2]) : []) {
        const t = tryModule("", `${mod}.${nm}`) ?? tryModule("src", `${mod}.${nm}`);
        if (t) {
          out.push(t);
          found = true;
        }
      }
      if (!found) {
        const target = tryModule("", mod) ?? tryModule("src", mod);
        if (target) out.push(target);
      }
    }
  }
  return out;
}

function rustEdges(fromFile: string, text: string, fileSet: Set<string>): string[] {
  const out: string[] = [];
  const dir = fromFile.includes("/") ? fromFile.slice(0, fromFile.lastIndexOf("/")) : "";
  const srcRoot = fromFile.match(/^(.*?src)\//)?.[1];
  for (const raw of text.split("\n")) {
    const mod = raw.match(/^\s*(?:pub\s+)?mod\s+(\w+)\s*;/);
    if (mod) {
      for (const c of [`${dir}/${mod[1]}.rs`, `${dir}/${mod[1]}/mod.rs`]) {
        if (fileSet.has(c)) out.push(c);
      }
    }
    const use = raw.match(/^\s*(?:pub\s+)?use\s+crate::(\w+)/);
    if (use && srcRoot) {
      for (const c of [`${srcRoot}/${use[1]}.rs`, `${srcRoot}/${use[1]}/mod.rs`]) {
        if (fileSet.has(c)) out.push(c);
      }
    }
  }
  return out;
}

export async function graphify(
  store: GraphStore,
  repoPath: string,
  opts: { project?: string; maxFiles?: number } = {},
): Promise<GraphifyResult> {
  const clean = repoPath.replace(/\/+$/, "");
  const project = cleanText(opts.project ?? clean.split("/").pop() ?? "project");
  // Non-finite junk (NaN from a loose client) must fall to the default:
  // Math.min/max propagate NaN, and a NaN cap slices the listing to []
  // — which the sweep would read as "everything vanished".
  const requested = Number.isFinite(opts.maxFiles) ? opts.maxFiles as number : DEFAULT_MAX_FILES;
  const maxFiles = Math.max(1, Math.min(MAX_FILES_CEILING, requested));
  const { files, skipped } = await listFiles(clean, maxFiles);
  const fileSet = new Set(files);

  // Specs dedupe by name (twin markdown headings slugify identically),
  // and every structural spec REPLACES its observations so re-runs
  // converge instead of appending the same doc line forever.
  const specByName = new Map<string, EntitySpec>();
  const specs = {
    push(spec: EntitySpec) {
      specByName.set(spec.name, { ...spec, replaceObservations: true });
    },
  };
  const rels: RelationSpec[] = [];
  const seen = new Set<string>();
  // Cleaned here too so `seen` matches what the store persists —
  // sweepStale compares the two.
  const name = (rel: string) => cleanText(`${project}/${rel}`);

  const readme = files.find((f) => f.toLowerCase() === "readme.md");
  const readmeHead = readme ? await readHead(`${clean}/${readme}`) : null;
  const topDirs = [...new Set(files.filter((f) => f.includes("/")).map((f) => f.split("/")[0]))];
  specs.push({
    name: project,
    type: "repo",
    project,
    observations: [
      readmeHead?.match(/^#\s+(.+)/m)?.[1] ?? `repository at ${project}`,
      `${files.length} tracked files; top: ${topDirs.slice(0, 8).join(", ")}`,
    ],
  });
  seen.add(project);

  // Modules: directories that hold source files, depth ≤ 3.
  const moduleDirs = new Set<string>();
  for (const f of files) {
    if (!SOURCE_EXT.test(f) && !ANCHOR_FILES.has(f.split("/").pop()!)) continue;
    const parts = f.split("/");
    for (let d = 1; d < Math.min(parts.length, 4); d++) {
      moduleDirs.add(parts.slice(0, d).join("/"));
    }
  }
  for (const dir of [...moduleDirs].sort()) {
    const n = name(dir);
    specs.push({ name: n, type: "module", project });
    seen.add(n);
    const parent = dir.includes("/") ? name(dir.slice(0, dir.lastIndexOf("/"))) : project;
    rels.push({ from: parent, to: n, relType: "contains" });
  }

  let fileCount = 0;
  let skippedFiles = skipped;
  for (const f of files) {
    const base = f.split("/").pop()!;
    if (!SOURCE_EXT.test(f) && !ANCHOR_FILES.has(base)) continue;
    const text = await readHead(`${clean}/${f}`);
    if (text === null || looksBinary(text)) {
      skippedFiles++;
      continue;
    }
    fileCount++;
    const n = name(f);
    const doc = firstDocLine(text, !f.endsWith(".md"));
    specs.push({ name: n, type: "file", project, observations: doc ? [doc] : [] });
    seen.add(n);
    const parent = f.includes("/") ? name(f.slice(0, f.lastIndexOf("/"))) : project;
    rels.push({ from: parent, to: n, relType: "contains" });

    if (/\.(ts|js|tsx|jsx)$/.test(f)) {
      for (const raw of text.split("\n")) {
        const m = raw.match(TSJS_IMPORT);
        const spec = m?.[1] ?? m?.[2] ?? m?.[3];
        if (!spec) continue;
        const target = resolveRelative(f, spec, fileSet);
        if (target) rels.push({ from: n, to: name(target), relType: "imports" });
      }
    } else if (f.endsWith(".rs")) {
      for (const target of rustEdges(f, text, fileSet)) {
        rels.push({ from: n, to: name(target), relType: "imports" });
      }
    } else if (f.endsWith(".py")) {
      for (const target of pyEdges(f, text, fileSet)) {
        rels.push({ from: n, to: name(target), relType: "imports" });
      }
    } else if (f.endsWith(".md")) {
      for (const sec of markdownSections(text)) {
        const sn = `${n}#${sec.slug}`;
        specs.push({
          name: sn,
          type: "doc-section",
          project,
          observations: [sec.firstLine ? `${sec.heading} — ${sec.firstLine}` : sec.heading],
        });
        seen.add(sn);
        rels.push({ from: n, to: sn, relType: "has-section" });
      }
    }
  }

  await store.upsertEntities([...specByName.values()]);

  // Import edges may point at files skipped by caps; keep only edges
  // whose both ends exist as entities.
  const valid = rels.filter((r) => seen.has(r.from) && seen.has(r.to));
  await store.addRelations(valid);

  // Full-refresh sweep inside one transaction: the enumerate + delete
  // window must not race a concurrent writer in the other process.
  // Only when the listing was complete — a truncated enumeration is
  // not ground truth, and sweeping against it would mass-delete
  // entities that are merely past the cap.
  if (skipped === 0) {
    await store.sweepStale(project, STRUCTURAL_TYPES, seen);
  }

  return {
    project,
    entities: specByName.size,
    relations: valid.length,
    files: fileCount,
    skipped: skippedFiles,
  };
}
