/**
 * DevicePlanner: turns raw device capabilities + model metadata into an
 * EngineConfig. This is where Android-vs-desktop differences are handled —
 * memory limits, buffer sharding constraints, prefill chunking — so the
 * rest of the app never thinks about them.
 *
 * Every decision is overridable: planner output is merged
 *   defaults → preset → combs.config.json → per-call overrides.
 */
import type { DeviceCaps, EngineConfig, ModelPreset } from "./types.ts";

export interface PlannerInput {
  caps: DeviceCaps;
  preset: ModelPreset;
  /** Model context window from config.json (upper bound). */
  modelMaxPositionEmbeddings: number;
  /** Architecture constants for KV VRAM estimation. */
  model: {
    num_hidden_layers: number;
    num_key_value_heads: number;
    head_dim: number;
    /** Total weight bytes (safetensors size), for sharding checks. */
    weightBytes?: number;
  };
  overrides?: Partial<EngineConfig>;
}

/** Fraction of the estimated usable GPU memory the KV arena may occupy. */
const KV_BUDGET_FRACTION = 0.25;
/** Conservative usable-memory estimate relative to max_buffer_size. */
const USABLE_MEMORY_FRACTION = 0.5;
/** Small-buffer threshold below which we assume a mobile-class GPU. */
const MOBILE_BUFFER_LIMIT = 256 * 1024 * 1024;

export function planEngineConfig(input: PlannerInput): EngineConfig {
  const { caps, preset, model } = input;

  // --- KV cache capacity ------------------------------------------------
  // KV bytes per token (K + V, f32) across all layers.
  const kvBytesPerToken =
    model.num_hidden_layers * model.num_key_value_heads * model.head_dim * 2 * 4;
  const kvBudget = caps.max_buffer_size * USABLE_MEMORY_FRACTION * KV_BUDGET_FRACTION;
  const maxSeqByMemory = Math.floor(kvBudget / kvBytesPerToken);
  const max_seq_len = Math.max(
    512,
    Math.min(preset.defaultMaxSeqLen, input.modelMaxPositionEmbeddings, maxSeqByMemory),
  );

  // --- Prefill chunking ---------------------------------------------------
  // Mobile-class GPUs (small storage-buffer limit) get smaller chunks.
  const mobile = caps.max_storage_buffer_binding_size < MOBILE_BUFFER_LIMIT;
  const prefill_chunk_size = mobile ? 256 : 512;

  const config: EngineConfig = {
    model_dir: "", // filled by the caller
    max_seq_len,
    page_size: 16,
    kv_cache: "paged",
    prefill_chunk_size,
  };

  return { ...config, ...input.overrides };
}

/** Checks whether the model's largest single weight fits one GPU buffer;
 * if not, weight sharding (a ring buffer, not yet built) is required and we
 * warn — in a browser this is the difference between a refusal at selection
 * time and a crash mid-load. */
export function needsWeightSharding(caps: DeviceCaps, weightBytes: number): boolean {
  return weightBytes > caps.max_storage_buffer_binding_size;
}
