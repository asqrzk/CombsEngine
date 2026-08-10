/**
 * Combs: the high-level entry point.
 *
 * ```ts
 * import { Combs } from "@combs/core";
 *
 * const engine = await Combs.init("smollm2-135m");
 * for await (const ev of engine.stream({ messages: [{ role: "user", content: "Hi!" }] })) {
 *   if (ev.type === "delta") Deno.stdout.writeSync(new TextEncoder().encode(ev.text));
 * }
 * engine.close();
 * ```
 *
 * `Combs.init` resolves a preset, downloads the model if needed, plans the
 * engine config from device capabilities and loads the native engine — all
 * overridable via `combs.config.json` and per-call options.
 */
import { ModelCache } from "./cache.ts";
import { loadConfigFile, mergeSampling } from "./config.ts";
import { FfiEngine, queryDeviceCaps } from "./engine.ts";
import { planEngineConfig } from "./planner.ts";
import { findPreset, listPresets } from "./presets.ts";
import { RemoteEngine } from "./remote.ts";
import type {
  CombsConfig,
  DeviceCaps,
  EngineClient,
  EngineConfig,
  ModelPreset,
  SamplingParams,
} from "./types.ts";

export interface CombsInitOptions {
  /** Preset id ("smollm2-135m") or HF repo id, or a path to a local model dir. */
  model: string;
  /** Skip the planner and pass this engine config through. */
  engine?: Partial<EngineConfig>;
  /** Sampling defaults merged over the preset's. */
  sampling?: SamplingParams;
  /** Native library path override (else COMBS_LIB / bundled locations). */
  libraryPath?: string;
  /** Model store root override. */
  modelStore?: string;
  /** Progress callback during model download. */
  onProgress?: (p: { file: string; received: number; total: number | null }) => void;
}

export class Combs {
  /** Lists the built-in model presets. */
  static presets(): readonly ModelPreset[] {
    return listPresets();
  }

  /** Device capabilities from the native core (planner input). */
  static deviceCaps(libraryPath?: string): DeviceCaps {
    return queryDeviceCaps(libraryPath);
  }

  /** Loads a model end-to-end: preset → cache → planner → native engine. */
  static async init(
    modelOrOptions: string | CombsInitOptions,
    overrides: CombsConfig = {},
  ): Promise<EngineClient> {
    const options: CombsInitOptions =
      typeof modelOrOptions === "string" ? { model: modelOrOptions } : modelOrOptions;
    const fileConfig = await loadConfigFile();
    const config: CombsConfig = { ...fileConfig, ...overrides };

    // 1. Resolve the model: preset id → cached download; otherwise a path.
    const preset = findPreset(options.model);
    let modelDir: string;
    if (preset) {
      const cache = new ModelCache(options.modelStore ?? config.modelStore);
      if (!(await cache.has(preset))) {
        await cache.pull(preset, options.onProgress);
      }
      modelDir = cache.modelDir(preset.id);
    } else if (await isDir(options.model)) {
      modelDir = options.model;
    } else {
      throw new Error(
        `unknown model ${JSON.stringify(options.model)}: not a preset ` +
          `(${listPresets().map((p) => p.id).join(", ")}) and not a local directory`,
      );
    }

    // 2. Plan the engine config from device capabilities.
    const metadata = await readModelMetadata(modelDir);
    const caps = queryDeviceCaps(options.libraryPath ?? config.libraryPath);
    const engineConfig = planEngineConfig({
      caps,
      preset: preset ?? localPreset(modelDir, metadata),
      modelMaxPositionEmbeddings: metadata.max_position_embeddings,
      model: {
        num_hidden_layers: metadata.num_hidden_layers,
        num_key_value_heads: metadata.num_key_value_heads,
        head_dim: metadata.hidden_size / metadata.num_attention_heads,
      },
      overrides: { ...config.engine, ...options.engine },
    });
    engineConfig.model_dir = modelDir;

    // 3. Load the native engine.
    return FfiEngine.load(engineConfig, options.libraryPath ?? config.libraryPath);
  }

  /** Connects to a remote `combs serve` instance. */
  static remote(baseUrl: string): EngineClient {
    return new RemoteEngine(baseUrl);
  }

  /** Resolves default sampling params for a preset (+ config file). */
  static async samplingDefaults(presetId: string): Promise<SamplingParams> {
    const preset = findPreset(presetId);
    const fileConfig = await loadConfigFile();
    return mergeSampling(preset?.sampling, fileConfig.sampling);
  }
}

async function isDir(path: string): Promise<boolean> {
  try {
    return (await Deno.stat(path)).isDirectory;
  } catch {
    return false;
  }
}

/** Minimal config.json read for planner inputs. */
async function readModelMetadata(modelDir: string): Promise<{
  max_position_embeddings: number;
  num_hidden_layers: number;
  num_key_value_heads: number;
  num_attention_heads: number;
  hidden_size: number;
}> {
  const raw = JSON.parse(await Deno.readTextFile(`${modelDir}/config.json`));
  const heads = raw.num_attention_heads ?? 1;
  return {
    max_position_embeddings: raw.max_position_embeddings ?? 2048,
    num_hidden_layers: raw.num_hidden_layers ?? 1,
    num_key_value_heads: raw.num_key_value_heads ?? heads,
    num_attention_heads: heads,
    hidden_size: raw.hidden_size ?? 1,
  };
}

/** Synthesizes a preset for a local (non-registry) model dir. */
function localPreset(
  modelDir: string,
  metadata: { max_position_embeddings: number },
): ModelPreset {
  return {
    id: modelDir.split("/").pop() ?? "local",
    hfRepo: "",
    files: [],
    architecture: "llama",
    description: "local model directory",
    sizeMb: 0,
    defaultMaxSeqLen: metadata.max_position_embeddings,
    sampling: {},
    chatTemplate: "chatml",
  };
}
