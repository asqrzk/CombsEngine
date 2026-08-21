/**
 * Integration test: loads the real model through Deno FFI and generates.
 *
 * Requires the native library (cargo xtask build --release) and the
 * SmolLM2-135M model dir. Set COMBS_TEST_MODEL to override the model path.
 */
import { assert, assertEquals, assertStringIncludes } from "jsr:@std/assert";
import { Combs } from "../src/combs.ts";
import type { StreamEvent } from "../src/types.ts";

const MODEL = Deno.env.get("COMBS_TEST_MODEL") ??
  new URL("../../../../models/SmolLM2-135M", import.meta.url).pathname;

// The default model dir is gitignored, so it is absent on CI runners —
// skip instead of failing when no model is available.
const MODEL_EXISTS = (() => {
  try {
    Deno.statSync(MODEL);
    return true;
  } catch {
    return false;
  }
})();

Deno.test({
  name: "ffi engine: metadata + streaming generation",
  ignore: Deno.env.get("COMBS_SKIP_INTEGRATION") === "1" || !MODEL_EXISTS,
  async fn() {
    const engine = await Combs.init({ model: MODEL, engine: { max_seq_len: 1024 } });
    try {
      const meta = await engine.metadata();
      assertEquals(meta.architecture, "llama");
      assert(meta.vocab_size > 0);

      const events: StreamEvent[] = [];
      let text = "";
      for await (
        const ev of engine.stream({
          prompt: "The capital of France is",
          max_tokens: 12,
          temperature: 0,
          seed: 1,
        })
      ) {
        events.push(ev);
        if (ev.type === "delta") text += ev.text;
      }

      const done = events.find((e) => e.type === "done");
      assert(done, "expected a done event");
      assert(text.length > 0, "expected generated text");
      // 5 text tokens + the BOS the engine prepends before prefill.
      assertEquals(done.stats.prompt_tokens, 6);
      assert(done.stats.generated_tokens > 0);

      // Chat mode applies ChatML.
      const chat = await engine.complete({
        messages: [{ role: "user", content: "Say the word 'apple'." }],
        max_tokens: 8,
        temperature: 0,
      });
      assert(chat.text.length > 0);
    } finally {
      engine.close();
    }
  },
});

Deno.test({
  name: "device caps are queryable without loading a model",
  ignore: Deno.env.get("COMBS_SKIP_INTEGRATION") === "1",
  fn() {
    const caps = Combs.deviceCaps();
    assert(caps.name.length > 0);
    assert(caps.max_buffer_size > 0);
    assertStringIncludes(caps.backend, "");
  },
});
