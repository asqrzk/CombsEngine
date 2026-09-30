/**
 * sand build — `.sand` → `.ts` (+ `.sand.json`, + `.js` on request).
 *
 * The emitted `.ts` imports `@combs/sand`; the `deno.json` written beside it
 * maps that name to jsr, or to `$SAND_IMPORT` when set (the tree, before a
 * publish). `--js` strips types with sucrase — the only moment the CLI reaches
 * for a dependency, and only then.
 */

import { transform } from "../mod.ts";

export interface BuildResult { dir: string; ts: string; plan: string; js: string | null; result: ReturnType<typeof transform>; name: string }

export function importTarget(): string {
  const local = Deno.env.get("SAND_IMPORT");
  if (!local) return "jsr:@combs/sand@^0.3.0";
  return local.startsWith("file://") || local.startsWith("http") ? local : `file://${local.startsWith("/") ? local : `${Deno.cwd()}/${local}`}`;
}

export async function build(file: string, opts: { outDir?: string; js?: boolean } = {}): Promise<BuildResult> {
  const src = await Deno.readTextFile(file);
  const name = file.replace(/^.*\//, "").replace(/\.sand$/, "");
  const result = transform(src, { importStyle: "jsr", file: file.replace(/^.*\//, "") });
  const dir = opts.outDir ?? await Deno.makeTempDir({ prefix: `sand-${result.plan.hash.replace(/^fnv1a:/, "")}-` });
  await Deno.mkdir(dir, { recursive: true });
  const ts = `${dir}/${name}.ts`, plan = `${dir}/${name}.sand.json`;
  await Deno.writeTextFile(ts, result.code);
  await Deno.writeTextFile(plan, JSON.stringify(result.plan, null, 2) + "\n");
  await Deno.writeTextFile(`${dir}/deno.json`, JSON.stringify({ imports: { "@combs/sand": importTarget() }, nodeModulesDir: "none" }, null, 2) + "\n");
  let js: string | null = null;
  if (opts.js) {
    const sucrase = await import("npm:sucrase@3.35.1");
    const out = sucrase.transform(result.code, { transforms: ["typescript"], disableESTransforms: true }).code;
    js = `${dir}/${name}.js`;
    await Deno.writeTextFile(js, out);
  }
  return { dir, ts, plan, js, result, name };
}
