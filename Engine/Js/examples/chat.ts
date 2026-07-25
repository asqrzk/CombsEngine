/**
 * Demo: high-level Combs API against a local model (raw prompt + chat),
 * with streaming. Run:
 *   deno run --allow-ffi --allow-read --allow-env --allow-net examples/chat.ts [model-dir]
 */
import { Combs } from "../core/mod.ts";

const model = Deno.args[0] ?? new URL("../../../models/SmolLM2-135M", import.meta.url).pathname;
const encoder = new TextEncoder();
const write = (s: string) => Deno.stdout.writeSync(encoder.encode(s));

const engine = await Combs.init({ model, engine: { max_seq_len: 2048 } });
const meta = await engine.metadata();
write(`loaded: ${meta.architecture} (ctx ${meta.max_seq_len}, eos ${meta.eos_token_ids})\n\n`);

write("> The capital of France is");
for await (
  const ev of engine.stream({
    prompt: "The capital of France is",
    max_tokens: 32,
    temperature: 0,
    repetition_penalty: 1.2,
  })
) {
  if (ev.type === "delta") write(ev.text);
  if (ev.type === "done") {
    write(
      `\n[${ev.finish_reason} | ${ev.stats.generated_tokens} tokens @ ${
        ev.stats.decode_tokens_per_second.toFixed(1)
      } tok/s | ttft ${ev.stats.ttft_ms.toFixed(0)}ms]\n\n`,
    );
  }
}

write("> [chat] What is 2+2? Answer with one word.\n");
const chat = await engine.complete({
  messages: [{ role: "user", content: "What is 2+2? Answer with one word." }],
  max_tokens: 12,
  temperature: 0,
});
write(`${chat.text}\n[${chat.finishReason}]\n`);

engine.close();
