/** @combs/runtime tests: primitives, agent server + orchestrator round trip. */
import { assert, assertEquals } from "jsr:@std/assert";
import {
  createAgentServer,
  findFreePort,
  generateToken,
  KeyedMutex,
  Orchestrator,
  Semaphore,
  SessionStore,
} from "../mod.ts";

Deno.test("findFreePort returns a bindable port", () => {
  const port = findFreePort();
  assert(port > 0 && port < 65536);
  const listener = Deno.listen({ port });
  listener.close();
});

Deno.test("tokens are unique and url-safe", () => {
  const a = generateToken();
  const b = generateToken();
  assert(a !== b);
  assert(/^[A-Za-z0-9_-]+$/.test(a));
});

Deno.test("KeyedMutex serializes per key", async () => {
  const mutex = new KeyedMutex();
  const order: string[] = [];
  await Promise.all([
    mutex.lock("k", async () => {
      await new Promise((r) => setTimeout(r, 30));
      order.push("a");
    }),
    mutex.lock("k", async () => {
      order.push("b");
    }),
  ]);
  assertEquals(order, ["a", "b"]);
});

Deno.test("Semaphore bounds concurrency", async () => {
  const sem = new Semaphore(2);
  let running = 0;
  let peak = 0;
  await Promise.all(
    Array.from({ length: 6 }, () =>
      sem.run(async () => {
        running++;
        peak = Math.max(peak, running);
        await new Promise((r) => setTimeout(r, 10));
        running--;
      }),
    ),
  );
  assertEquals(peak, 2);
});

Deno.test("agent server + orchestrator: delegate round trip, auth, events", async () => {
  const server = await createAgentServer({
    name: "echo",
    handler: async (input, emit) => {
      emit({ progress: 0.5 });
      await new Promise((r) => setTimeout(r, 10));
      return { echoed: input.text ?? null };
    },
  });
  try {
    // HTTP auth is enforced.
    const unauthorized = await fetch(`${server.url}/v1/chat/completions`, {
      method: "POST",
      body: "{}",
    });
    assertEquals(unauthorized.status, 401);

    const orch = new Orchestrator();
    await orch.register({ name: "echo", url: server.url, token: server.token });
    assertEquals(orch.listAgents(), ["echo"]);

    const events: unknown[] = [];
    const result = await orch.delegate("echo", { text: "hello" }, { onEvent: (e) => events.push(e) });
    assert(result.ok, `delegation failed: ${result.error}`);
    assertEquals(result.data, { echoed: "hello" });
    assertEquals(events.length, 1);

    await orch.close();
  } finally {
    await server.close();
  }
});

Deno.test("session store: save / get / list / delete", () => {
  const path = `/tmp/combs-test-sessions-${crypto.randomUUID().slice(0, 8)}.sqlite`;
  const store = new SessionStore(path);
  try {
    store.save({ id: "s1", agent: "a", threadId: "t1", data: { step: 1 } });
    store.save({ id: "s2", agent: "a", threadId: "t2", data: { step: 2 } });
    store.save({ id: "s1", agent: "a", threadId: "t1", data: { step: 3 } }); // update

    const s1 = store.get("s1");
    assertEquals(s1?.data.step, 3);
    assertEquals(store.list("a").length, 2);

    store.delete("s2");
    assertEquals(store.list("a").length, 1);
  } finally {
    store.close();
    Deno.removeSync(path);
  }
});
