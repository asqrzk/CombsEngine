/** @combs/agents tests: tools, structured output, memory, react loop (mock engine). */
import { assert, assertEquals, assertStringIncludes } from "jsr:@std/assert";
import {
  createReactAgent,
  extractJson,
  KvMemoryStore,
  parseToolCalls,
  tool,
  ToolRegistry,
  validateAgainstSchema,
} from "../mod.ts";
import type { EngineClient, StreamEvent } from "@combs/core";
import type { ChatRequest } from "@combs/core";

/** Canned-response mock engine. */
function mockEngine(responses: string[]): EngineClient {
  let i = 0;
  return {
    kind: "remote",
    metadata: () => Promise.resolve({} as never),
    stream: async function* (_r: ChatRequest): AsyncIterable<StreamEvent> {
      yield { type: "done", finish_reason: "stop", stats: {} as never };
    },
    complete: () => {
      const text = responses[Math.min(i++, responses.length - 1)];
      return Promise.resolve({ text, stats: {} as never, finishReason: "stop" });
    },
    cancel: () => {},
    close: () => {},
  };
}

Deno.test("parseToolCalls: fenced, bare, and plain text", () => {
  const fenced = parseToolCalls('Let me check.\n```json\n{"tool_calls":[{"name":"search","args":{"q":"x"}}]}\n```');
  assertEquals(fenced.length, 1);
  assertEquals(fenced[0].name, "search");
  assertEquals(fenced[0].args, { q: "x" });

  const bare = parseToolCalls('{"tool_calls":[{"name":"calc","args":{"expr":"1+1"}}]}');
  assertEquals(bare[0].name, "calc");

  assertEquals(parseToolCalls("Just a normal answer."), []);
});

Deno.test("structured output: extract + validate", () => {
  const value = extractJson('```json\n{"name":"x","age":3}\n```');
  assertEquals(value, { name: "x", age: 3 });

  const schema = {
    type: "object",
    required: ["name"],
    properties: { name: { type: "string" }, age: { type: "integer" } },
    additionalProperties: false,
  };
  assertEquals(validateAgainstSchema({ name: "x", age: 3 }, schema), []);
  assert(validateAgainstSchema({ age: 3 }, schema).length > 0);
  assert(validateAgainstSchema({ name: "x", age: 3, extra: 1 }, schema).length > 0);
  assert(validateAgainstSchema({ name: "x", age: 3.5 }, schema).length > 0);
});

Deno.test("react agent: tool call loop ends with a final answer", async () => {
  const search = tool({
    name: "search",
    description: "Search the web",
    schema: { type: "object", properties: { q: { type: "string" } } },
    invoke: (args) => `results for ${(args as { q: string }).q}: paris, france`,
  });

  const engine = mockEngine([
    '```json\n{"tool_calls":[{"name":"search","args":{"q":"capital of france"}}]}\n```',
    "The capital of France is Paris.",
  ]);

  const agent = createReactAgent({ engine, tools: [search] });
  const out = await agent.invoke({ messages: [{ role: "user", content: "capital of france?" }] });
  const last = out.messages[out.messages.length - 1];
  assertStringIncludes(last.content, "Paris");
  // user + assistant(tool_calls) + tool result + assistant(final)
  assertEquals(out.messages.length, 4);
  assertEquals(out.messages[2].role, "tool");
});

Deno.test({
  name: "kv memory: remember / recall / forget",
  ignore: Deno.env.get("COMBS_SKIP_KV") === "1",
  async fn() {
    const store = await KvMemoryStore.open(undefined, `test-${crypto.randomUUID().slice(0, 8)}`);
    try {
      await store.remember("user likes tea", { user: "u1" });
      await store.remember("user is allergic to nuts", { user: "u1" });
      await store.remember("other user fact", { user: "u2" });

      const all = await store.recall(10, { user: "u1" });
      assertEquals(all.length, 2);
      assertStringIncludes(all[0].content, "allergic"); // latest first

      await store.forget(all[0].id);
      const after = await store.recall(10, { user: "u1" });
      assertEquals(after.length, 1);
    } finally {
      await store.clear();
    }
  },
});

Deno.test("tool registry: prompt block + duplicate guard", () => {
  const registry = new ToolRegistry();
  registry.register(tool({
    name: "t1",
    description: "d",
    schema: {},
    invoke: () => null,
  }));
  assertStringIncludes(registry.toPromptBlock(), '"t1"');
  let threw = false;
  try {
    registry.register(tool({ name: "t1", description: "d", schema: {}, invoke: () => null }));
  } catch {
    threw = true;
  }
  assert(threw);
});
