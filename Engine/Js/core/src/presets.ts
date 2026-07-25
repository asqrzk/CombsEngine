/**
 * Model presets: one-line access to known-good models.
 *
 * Presets capture everything a "normal user" shouldn't have to think about:
 * which HF repo + files, which architecture, sensible cache and sampling
 * defaults. The DevicePlanner may still lower `defaultMaxSeqLen` on
 * memory-constrained devices.
 *
 * NOTE: presets are limited to architectures registered in the Rust core
 * (currently the Llama family: llama/smollm2 with GQA + RMSNorm + SwiGLU).
 */
import type { ModelPreset } from "./types.ts";

export const PRESETS: readonly ModelPreset[] = [
  {
    id: "smollm2-135m",
    hfRepo: "HuggingFaceTB/SmolLM2-135M-Instruct",
    files: [
      "config.json",
      "generation_config.json",
      "tokenizer.json",
      "tokenizer_config.json",
      "model.safetensors",
    ],
    architecture: "llama",
    description: "SmolLM2 135M — tiny chat model, great for smoke tests",
    sizeMb: 270,
    defaultMaxSeqLen: 8192,
    sampling: { temperature: 0.6, top_p: 0.95, max_tokens: 256 },
    chatTemplate: "chatml",
  },
  {
    id: "smollm2-360m",
    hfRepo: "HuggingFaceTB/SmolLM2-360M-Instruct",
    files: [
      "config.json",
      "generation_config.json",
      "tokenizer.json",
      "tokenizer_config.json",
      "model.safetensors",
    ],
    architecture: "llama",
    description: "SmolLM2 360M — small chat model",
    sizeMb: 730,
    defaultMaxSeqLen: 8192,
    sampling: { temperature: 0.6, top_p: 0.95, max_tokens: 256 },
    chatTemplate: "chatml",
  },
  {
    id: "smollm2-1.7b",
    hfRepo: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    files: [
      "config.json",
      "generation_config.json",
      "tokenizer.json",
      "tokenizer_config.json",
      "model.safetensors",
    ],
    architecture: "llama",
    description: "SmolLM2 1.7B — capable small chat model",
    sizeMb: 3400,
    defaultMaxSeqLen: 8192,
    sampling: { temperature: 0.6, top_p: 0.95, max_tokens: 256 },
    chatTemplate: "chatml",
  },
] as const;

/** Finds a preset by id (case-insensitive). */
export function findPreset(id: string): ModelPreset | undefined {
  const norm = id.toLowerCase();
  return PRESETS.find((p) => p.id === norm || p.hfRepo.toLowerCase() === norm);
}

/** Lists all presets. */
export function listPresets(): readonly ModelPreset[] {
  return PRESETS;
}
