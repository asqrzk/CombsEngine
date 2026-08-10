/** @combs/flows tests: workflows with loops/checks, roleplay with mock engine. */
import { assert, assertEquals, assertRejects, assertStringIncludes } from "jsr:@std/assert";
import { addMemory, createRoleplayChat, createWorkflow, withMemory } from "../mod.ts";
import { append, lastValue } from "@combs/graph";
import type { EngineClient, StreamEvent } from "@combs/core";

interface Doc extends Record<string, unknown> {
  topic: string;
  draft: string;
  reviewed: boolean;
  __loop_polish?: number;
}

Deno.test("workflow: linear steps + checks", async () => {
  const wf = createWorkflow<Doc>(
    {
      steps: [
        { name: "draft", run: (s) => ({ draft: `draft about ${s.topic}` }) },
        { name: "review", run: () => ({ reviewed: true }) },
      ],
      checks: [{ after: "draft", assert: (s) => s.draft.length > 0, message: "empty" }],
    },
    {
      topic: lastValue(""),
      draft: lastValue(""),
      reviewed: lastValue(false),
    },
  );
  const out = await wf.invoke({ topic: "edge AI" });
  assertStringIncludes(out.draft, "edge AI");
  assertEquals(out.reviewed, true);
});

Deno.test("workflow: loops until condition, then continues", async () => {
  const wf = createWorkflow<Doc>(
    {
      steps: [
        {
          name: "polish",
          run: (s) => ({ draft: `${s.draft}.` }),
          loops: { max: 5, until: (s) => s.draft.length >= 3 },
        },
        { name: "review", run: () => ({ reviewed: true }) },
      ],
    },
    {
      topic: lastValue(""),
      draft: lastValue(""),
      reviewed: lastValue(false),
      __loop_polish: lastValue(0),
    },
  );
  const out = await wf.invoke({ draft: "x" });
  assertEquals(out.draft, "x..");
  assertEquals(out.__loop_polish, 2);
  assertEquals(out.reviewed, true);
});

Deno.test("workflow: failing check throws", async () => {
  const wf = createWorkflow<Doc>(
    {
      steps: [
        { name: "draft", run: () => ({ draft: "" }) },
        { name: "review", run: () => ({ reviewed: true }) },
      ],
      checks: [{ after: "draft", assert: (s) => s.draft.length > 0, message: "empty draft" }],
    },
    {
      topic: lastValue(""),
      draft: lastValue(""),
      reviewed: lastValue(false),
    },
  );
  await assertRejects(() => wf.invoke({}));
});

function scriptEngine(lines: Record<string, string[]>): EngineClient {
  const counts: Record<string, number> = {};
  return {
    kind: "remote",
    metadata: () => Promise.resolve({} as never),
    stream: async function* (): AsyncIterable<StreamEvent> {},
    complete: (req) => {
      const system = req.messages?.[0]?.content ?? "";
      const name = Object.keys(lines).find((n) => system.includes(`You are ${n},`)) ?? "?";
      const i = counts[name] ?? 0;
      counts[name] = i + 1;
      const pool = lines[name] ?? ["..."];
      return Promise.resolve({
        text: pool[Math.min(i, pool.length - 1)],
        stats: {} as never,
        finishReason: "stop",
      });
    },
    cancel: () => {},
    close: () => {},
  };
}

Deno.test("roleplay: two agents alternate for N rounds", async () => {
  const engine = scriptEngine({
    sherlock: ["The mud on the boot tells the story."],
    watson: ["How on earth do you know that?"],
  });
  const chat = createRoleplayChat({
    agents: [
      { name: "sherlock", role: "detective" },
      { name: "watson", role: "companion" },
    ],
    engine,
    rounds: 4,
  });
  const { transcript } = await chat.run("A mysterious boot was found.");
  // 1 scene + 4 turns
  assertEquals(transcript.length, 5);
  assertEquals(transcript[1].name, "sherlock");
  assertEquals(transcript[2].name, "watson");
  assertEquals(transcript[3].name, "sherlock");
});

Deno.test({
  name: "addMemory + withMemory: recall-before, remember-after",
  ignore: Deno.env.get("COMBS_SKIP_KV") === "1",
  async fn() {
    const memory = await addMemory({ type: "kv", scope: `test-${crypto.randomUUID().slice(0, 8)}` });
    try {
      await memory.remember("prior fact", { agent: "a" });
      let sawMemories: string[] = [];
      const node = withMemory(
        (state: { __memories?: string[]; out?: string }) => {
          sawMemories = state.__memories ?? [];
          return { out: "new insight" };
        },
        memory,
        { agent: "a" },
        { rememberKey: "out" },
      );
      const result = await node({}, null);
      assertEquals(sawMemories, ["prior fact"]);
      assertEquals(result, { out: "new insight" });
      const recalled = await memory.recall(10, { agent: "a" });
      assertEquals(recalled.length, 2);
      assertStringIncludes(recalled[0].content, "new insight");
    } finally {
      await memory.clear();
    }
  },
});
