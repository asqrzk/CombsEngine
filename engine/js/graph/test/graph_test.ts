/** @combs/graph semantics tests (no FFI/GPU needed). */
import { assert, assertEquals, assertRejects } from "jsr:@std/assert";
import {
  append,
  binaryOp,
  CompiledGraph,
  END,
  lastValue,
  MemoryCheckpointer,
  messages,
  Send,
  START,
  StateGraph,
} from "../mod.ts";
import type { GraphMessage } from "../mod.ts";

interface S extends Record<string, unknown> {
  log: string[];
  value: number;
  messages: GraphMessage[];
}

function makeGraph() {
  return new StateGraph<S>({
    log: append<string>([]),
    value: lastValue<number>(0),
    messages: messages(),
  });
}

Deno.test("linear graph flows state through nodes", async () => {
  const g = makeGraph()
    .addNode("a", () => ({ log: ["a"], value: 1 }))
    .addNode("b", (s) => ({ log: ["b"], value: (s.value as number) + 1 }))
    .addEdge(START, "a")
    .addEdge("a", "b")
    .addEdge("b", END)
    .compile();
  const out = await g.invoke({});
  assertEquals(out.log, ["a", "b"]);
  assertEquals(out.value, 2);
});

Deno.test("parallel fan-out + join barrier", async () => {
  const g = makeGraph()
    .addNode("b", async () => {
      await new Promise((r) => setTimeout(r, 20));
      return { log: ["b"] };
    })
    .addNode("c", () => ({ log: ["c"] }))
    .addNode("d", (s) => ({ value: s.log.length }))
    .addEdge(START, "b")
    .addEdge(START, "c")
    .addEdge(["b", "c"], "d")
    .addEdge("d", END)
    .compile();
  const out = await g.invoke({});
  assertEquals(out.log.sort(), ["b", "c"]);
  assertEquals(out.value, 2);
});

Deno.test("conditional edges route by state", async () => {
  const g = makeGraph()
    .addNode("decide", () => ({ value: 42 }))
    .addNode("big", () => ({ log: ["big"] }))
    .addNode("small", () => ({ log: ["small"] }))
    .addEdge(START, "decide")
    .addConditionalEdges("decide", (s) => (s.value > 10 ? "big" : "small"))
    .addEdge("big", END)
    .addEdge("small", END)
    .compile();
  const out = await g.invoke({});
  assertEquals(out.log, ["big"]);
});

Deno.test("Command.goto and Send fan-out", async () => {
  const g = makeGraph()
    .addNode("start", () => ({
      goto: [new Send("worker", { n: 1 }), new Send("worker", { n: 2 })],
    }))
    .addNode("worker", (_s, ctx) => {
      const { n } = ctx.sendArgs as { n: number };
      return { log: [`w${n}`] };
    })
    .addEdge(START, "start")
    .addEdge("worker", END)
    .compile();
  const out = await g.invoke({});
  assertEquals(out.log.sort(), ["w1", "w2"]);
});

Deno.test("binaryOp reducer merges parallel writes; lastValue rejects doubles", async () => {
  interface R extends Record<string, unknown> {
    sum: number;
    single: string;
  }
  const g = new StateGraph<R>({
    sum: binaryOp<number>(0, (a, b) => a + b),
    single: lastValue<string>(""),
  })
    .addNode("a", () => ({ sum: 1 }))
    .addNode("b", () => ({ sum: 2 }))
    .addNode("x", () => ({ single: "one" }))
    .addNode("y", () => ({ single: "two" }))
    .addEdge(START, "a")
    .addEdge(START, "b")
    .addEdge(["a", "b"], "x")
    .addEdge("x", END)
    .compile();
  const out = await g.invoke({});
  assertEquals(out.sum, 3);

  const bad = new StateGraph<R>({
    sum: binaryOp<number>(0, (a, b) => a + b),
    single: lastValue<string>(""),
  })
    .addNode("x", () => ({ single: "one" }))
    .addNode("y", () => ({ single: "two" }))
    .addEdge(START, "x")
    .addEdge(START, "y")
    .addEdge(["x", "y"], "a")
    .addNode("a", () => ({}))
    .addEdge("a", END)
    .compile();
  await assertRejects(() => bad.invoke({}));
});

Deno.test("checkpoint resume + history + updateState", async () => {
  const cp = new MemoryCheckpointer();
  const g = makeGraph()
    .addNode("a", () => ({ log: ["a"], value: 1 }))
    .addNode("b", (s) => ({ log: ["b"], value: (s.value as number) * 10 }))
    .addEdge(START, "a")
    .addEdge("a", "b")
    .addEdge("b", END)
    .compile({ checkpointer: cp });

  const out = await g.invoke({}, { threadId: "t1" });
  assertEquals(out.value, 10);

  const history = await g.getStateHistory("t1");
  assert(history.length >= 2, "expected multiple checkpoints");

  const snap = await g.getState("t1");
  assertEquals(snap?.values.value, 10);

  await g.updateState("t1", { value: 999 });
  const edited = await g.getState("t1");
  assertEquals(edited?.values.value, 999);
});

Deno.test("interrupt pauses; resume re-executes with the human value", async () => {
  const cp = new MemoryCheckpointer();
  const calls: string[] = [];
  const g = makeGraph()
    .addNode("draft", () => ({ log: ["draft"] }))
    .addNode("human", async (s, ctx) => {
      calls.push("human");
      const approved = await ctx.interrupt({ question: "approve draft?" });
      return { log: [`approved=${approved}`] };
    })
    .addEdge(START, "draft")
    .addEdge("draft", "human")
    .addEdge("human", END)
    .compile({ checkpointer: cp });

  await assertRejects(() => g.invoke({}, { threadId: "t2" }));
  const paused = await g.getState("t2");
  assertEquals(paused?.pendingInterrupts?.length, 1);
  assertEquals(calls, ["human"]);

  const out = await g.invoke({}, { threadId: "t2", resume: "yes" });
  assertEquals(out.log, ["draft", "approved=yes"]);
  // The node re-executed from the top on resume.
  assertEquals(calls, ["human", "human"]);
});

Deno.test("interruptBefore breakpoint halts before the node", async () => {
  const cp = new MemoryCheckpointer();
  const g = makeGraph()
    .addNode("a", () => ({ log: ["a"] }))
    .addNode("b", () => ({ log: ["b"] }))
    .addEdge(START, "a")
    .addEdge("a", "b")
    .addEdge("b", END)
    .compile({ checkpointer: cp, interruptBefore: ["b"] });

  await assertRejects(() => g.invoke({}, { threadId: "t3" }));
  const paused = await g.getState("t3");
  assertEquals(paused?.values.log, ["a"]);

  const out = await g.invoke({}, { threadId: "t3" });
  assertEquals(out.log, ["a", "b"]);
});

Deno.test("retry policy retries failed nodes", async () => {
  let attempts = 0;
  const g = makeGraph()
    .addNode("flaky", () => {
      attempts++;
      if (attempts < 3) throw new Error("boom");
      return { value: 7 };
    }, { retry: { maxAttempts: 3, backoffMs: 1 } })
    .addEdge(START, "flaky")
    .addEdge("flaky", END)
    .compile();
  const out = await g.invoke({});
  assertEquals(out.value, 7);
  assertEquals(attempts, 3);
});

Deno.test("stream emits updates and values", async () => {
  const g = makeGraph()
    .addNode("a", () => ({ log: ["a"] }))
    .addEdge(START, "a")
    .addEdge("a", END)
    .compile();
  const modes: string[] = [];
  for await (const ev of g.stream({})) modes.push(ev.mode);
  assert(modes.includes("updates"));
  assert(modes.includes("values"));
});
