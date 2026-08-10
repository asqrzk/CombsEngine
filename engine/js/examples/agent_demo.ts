/**
 * Demo: a ReAct agent with tools over the REAL native engine.
 *
 *   deno run --allow-ffi --allow-read --allow-env --allow-net --unstable-kv examples/agent_demo.ts
 */
import { Combs } from "../core/mod.ts";
import { createReactAgent, tool } from "../agents/mod.ts";
import { getLogger } from "../telemetry/mod.ts";

const log = getLogger("demo.agent");
const model = "smollm2-135m";

const calculator = tool({
  name: "calculator",
  description: "Evaluate a simple arithmetic expression like '12 * (3 + 4)'.",
  schema: {
    type: "object",
    properties: { expression: { type: "string", description: "arithmetic expression" } },
    required: ["expression"],
  },
  invoke: (args) => {
    const expr = String((args as { expression: string }).expression);
    if (!/^[\d\s+\-*/().]+$/.test(expr)) return "error: unsafe expression";
    return String(Function(`"use strict"; return (${expr})`)());
  },
});

const engine = await Combs.init({ model, engine: { max_seq_len: 2048 } });
log.info("engine loaded");

const agent = createReactAgent({
  engine,
  tools: [calculator],
  systemPrompt: "You are a precise math assistant. Use the calculator tool for any arithmetic.",
  maxRounds: 3,
});

const out = await agent.invoke({
  messages: [{ role: "user", content: "What is 12 * (3 + 4)? Use the calculator." }],
});
const encoder = new TextEncoder();
for (const m of out.messages) {
  Deno.stdout.writeSync(encoder.encode(`\n[${m.role}] ${m.content.slice(0, 300)}`));
}
Deno.stdout.writeSync(encoder.encode("\n"));
engine.close();
