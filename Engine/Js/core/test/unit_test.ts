/** Unit tests: planner + config merging (no FFI needed). */
import { assertEquals, assertThrows } from "jsr:@std/assert";
import { planEngineConfig } from "../src/planner.ts";
import { mergeSampling } from "../src/config.ts";
import { findPreset } from "../src/presets.ts";
import type { DeviceCaps } from "../src/types.ts";

const desktopCaps: DeviceCaps = {
  name: "Apple M3 Pro",
  backend: "Metal",
  device_type: "IntegratedGpu",
  max_storage_buffer_binding_size: 2_147_483_648,
  max_buffer_size: 8_589_934_592,
  max_compute_workgroup_size_x: 1024,
  max_compute_invocations_per_workgroup: 1024,
  features: "",
};

const mobileCaps: DeviceCaps = {
  ...desktopCaps,
  name: "Adreno 740",
  backend: "Vulkan",
  max_storage_buffer_binding_size: 134_217_728,
  max_buffer_size: 268_435_456,
};

const model = {
  num_hidden_layers: 30,
  num_key_value_heads: 3,
  head_dim: 64,
};

Deno.test("planner: desktop gets full context and large chunks", () => {
  const cfg = planEngineConfig({
    caps: desktopCaps,
    preset: findPreset("smollm2-135m")!,
    modelMaxPositionEmbeddings: 8192,
    model,
  });
  assertEquals(cfg.max_seq_len, 8192);
  assertEquals(cfg.prefill_chunk_size, 512);
  assertEquals(cfg.kv_cache, "paged");
});

Deno.test("planner: mobile gets smaller prefill chunks and memory-capped context", () => {
  const cfg = planEngineConfig({
    caps: mobileCaps,
    preset: findPreset("smollm2-135m")!,
    modelMaxPositionEmbeddings: 8192,
    model,
  });
  assertEquals(cfg.prefill_chunk_size, 256);
  // KV budget = 268MB * 0.5 * 0.25 = 33.5MB; per-token = 30*3*64*2*4 = 46080B
  // -> ~728 tokens, clamped to >= 512.
  assertEquals(cfg.max_seq_len! >= 512 && cfg.max_seq_len! < 2048, true);
});

Deno.test("planner: overrides always win", () => {
  const cfg = planEngineConfig({
    caps: desktopCaps,
    preset: findPreset("smollm2-135m")!,
    modelMaxPositionEmbeddings: 8192,
    model,
    overrides: { max_seq_len: 1024, kv_cache: "contiguous" },
  });
  assertEquals(cfg.max_seq_len, 1024);
  assertEquals(cfg.kv_cache, "contiguous");
});

Deno.test("mergeSampling: later layers win, undefined is ignored", () => {
  const merged = mergeSampling(
    { temperature: 0.6, top_p: 0.95 },
    { temperature: 0.0, top_k: 40 },
    { temperature: undefined, seed: 42 },
  );
  assertEquals(merged, { temperature: 0.0, top_p: 0.95, top_k: 40, seed: 42 });
});

Deno.test("presets: lookup by id and repo", () => {
  assertEquals(findPreset("smollm2-135m")?.hfRepo, "HuggingFaceTB/SmolLM2-135M");
  assertEquals(findPreset("HuggingFaceTB/SmolLM2-360M")?.id, "smollm2-360m");
  assertEquals(findPreset("nonexistent"), undefined);
});
