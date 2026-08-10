/** Shared types for @combs/core. */

/** Hardware capabilities reported by the Rust core (combs_device_caps_json). */
export interface DeviceCaps {
  name: string;
  backend: string;
  device_type: string;
  max_storage_buffer_binding_size: number;
  max_buffer_size: number;
  max_compute_workgroup_size_x: number;
  max_compute_invocations_per_workgroup: number;
  features: string;
}

/** Engine creation config sent across the FFI boundary. */
export interface EngineConfig {
  model_dir: string;
  max_seq_len?: number;
  page_size?: number;
  kv_cache?: "paged" | "contiguous";
  prefill_chunk_size?: number;
}

/** Model metadata reported by a loaded engine. */
export interface EngineMetadata {
  architecture: string;
  vocab_size: number;
  max_position_embeddings: number;
  max_seq_len: number;
  page_size: number;
  eos_token_ids: number[];
  im_end_id: number | null;
}

export interface ChatMessage {
  role: "system" | "user" | "assistant" | string;
  content: string;
}

/** Sampling + generation parameters; absent fields fall back to the
 * model's generation_config.json defaults. */
export interface SamplingParams {
  max_tokens?: number;
  temperature?: number;
  top_k?: number;
  top_p?: number;
  repetition_penalty?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  seed?: number;
  stop?: string[];
  stop_token_ids?: number[];
  prefill_chunk_size?: number;
}

/** A chat request: messages (ChatML applied) or a raw prompt. */
export interface ChatRequest extends SamplingParams {
  messages?: ChatMessage[];
  prompt?: string;
}

/** Streaming events from the engine. */
export type StreamEvent =
  | { type: "delta"; text: string; token_id: number }
  | { type: "done"; finish_reason: "stop" | "length" | "cancelled"; stats: GenerationStats }
  | { type: "error"; message: string };

export interface GenerationStats {
  prompt_tokens: number;
  generated_tokens: number;
  ttft_ms: number;
  decode_tokens_per_second: number;
  prefill_tokens_per_second: number;
  cache_pages_used: number;
}

/** The transport-agnostic engine contract (FfiEngine, RemoteEngine, and
 * the Phase-5 WorkerEngine all implement this). */
export interface EngineClient {
  readonly kind: "ffi" | "remote" | "worker";
  metadata(): Promise<EngineMetadata>;
  /** Streaming chat; yields delta events and ends with done/error. */
  stream(request: ChatRequest, requestId?: string): AsyncIterable<StreamEvent>;
  /** Non-streaming convenience wrapper over stream(). */
  complete(
    request: ChatRequest,
  ): Promise<{ text: string; stats: GenerationStats; finishReason: string }>;
  cancel(requestId: string): void;
  close(): void;
}

/** A downloadable, ready-to-run model preset. */
export interface ModelPreset {
  /** Preset id, e.g. "smollm2-135m". */
  id: string;
  /** HuggingFace repo, e.g. "HuggingFaceTB/SmolLM2-135M". */
  hfRepo: string;
  /** Files required from the repo. */
  files: string[];
  /** Model architecture (must be registered in the Rust core). */
  architecture: string;
  /** Human-readable description. */
  description: string;
  /** Approximate download size, MB. */
  sizeMb: number;
  /** Default KV cache capacity (planner may lower it for small devices). */
  defaultMaxSeqLen: number;
  /** Default sampling parameters. */
  sampling: SamplingParams;
  /** Chat template identifier ("chatml" | "raw"). */
  chatTemplate: string;
}

/** Top-level resolved Combs configuration. */
export interface CombsConfig {
  /** Model store root (default: ~/.cache/combs/models). */
  modelStore?: string;
  /** Override the native library path. */
  libraryPath?: string;
  /** Per-target engine overrides merged over planner output. */
  engine?: Partial<EngineConfig>;
  /** Sampling overrides merged over preset defaults. */
  sampling?: SamplingParams;
}
