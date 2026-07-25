/**
 * Layered configuration: defaults → preset → combs.config.json → per-call.
 *
 * `combs.config.json` is read from $COMBS_CONFIG, else ./combs.config.json,
 * else ~/.config/combs/config.json — all optional.
 */
import type { CombsConfig, SamplingParams } from "./types.ts";

/** Deep-merges sampling params (defined values win). */
export function mergeSampling(
  ...layers: (SamplingParams | undefined)[]
): SamplingParams {
  const out: SamplingParams = {};
  for (const layer of layers) {
    if (!layer) continue;
    for (const [k, v] of Object.entries(layer)) {
      if (v !== undefined) (out as Record<string, unknown>)[k] = v;
    }
  }
  return out;
}

/** Loads the optional config file; returns {} when absent. */
export async function loadConfigFile(): Promise<CombsConfig> {
  const candidates = [
    Deno.env.get("COMBS_CONFIG"),
    "combs.config.json",
    `${Deno.env.get("HOME") ?? "."}/.config/combs/config.json`,
  ].filter((p): p is string => !!p);
  for (const path of candidates) {
    try {
      const text = await Deno.readTextFile(path);
      return JSON.parse(text) as CombsConfig;
    } catch (e) {
      if (e instanceof Deno.errors.NotFound) continue;
      throw new Error(`invalid config file ${path}: ${e}`);
    }
  }
  return {};
}
