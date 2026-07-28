import { assert, assertEquals, assertRejects } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { createAgentServer, KeyRing, Orchestrator } from "../mod.ts";

Deno.test("secure channel: key exchange + sealed delegate round-trip", async () => {
  const agentKeys = await KeyRing.create();
  const orchKeys = await KeyRing.create();

  const server = await createAgentServer({
    name: "secure-echo",
    keyring: agentKeys,
    handler: (input, emit) => {
      emit({ progress: 1 });
      return Promise.resolve({ echo: input, secret: "s3cr3t" });
    },
  });
  const orch = new Orchestrator({ keyring: orchKeys });
  try {
    await orch.register({ name: "secure-echo", url: server.url, token: server.token });

    // keys exchanged in both directions
    assertEquals(orchKeys.peerCount(), 1);
    assertEquals(agentKeys.peerCount(), 1);

    const events: unknown[] = [];
    const result = await orch.delegate(
      "secure-echo",
      { task: "classified" },
      { onEvent: (e) => events.push(e) },
    );
    assert(result.ok, `delegate failed: ${result.error}`);
    assertEquals((result.data as { secret: string }).secret, "s3cr3t");
    assertEquals(events, [{ progress: 1 }]);
  } finally {
    await orch.close();
  }
});

Deno.test("secure channel: envelope from an impostor is rejected", async () => {
  const agentKeys = await KeyRing.create();
  const orchKeys = await KeyRing.create();
  const mallory = await KeyRing.create();

  const server = await createAgentServer({
    name: "vault",
    keyring: agentKeys,
    handler: (input) => Promise.resolve(input),
  });
  const orch = new Orchestrator({ keyring: orchKeys });
  try {
    await orch.register({ name: "vault", url: server.url, token: server.token });

    // Mallory knows the agent's PUBLIC keys (they're public) and seals a
    // message with HER keys — but the agent never keyed her as a peer,
    // so openFrom must reject it.
    const agentFp = orchKeys.peer(agentKeys.identity.fingerprint);
    assert(agentFp);
    mallory.addPeer(agentFp);
    const forged = await mallory.sealFor(agentFp.fingerprint, { task: "evil" });
    await assertRejects(() => agentKeys.openFrom(forged), Error, "unknown peer");
  } finally {
    await orch.close();
  }
});

Deno.test("secure channel: plaintext delegate still works without keyring", async () => {
  const server = await createAgentServer({
    name: "plain",
    handler: (input) => Promise.resolve({ ok: input }),
  });
  const orch = new Orchestrator();
  try {
    await orch.register({ name: "plain", url: server.url, token: server.token });
    const result = await orch.delegate("plain", { a: 1 });
    assert(result.ok);
    assertEquals(result.data, { ok: { a: 1 } });
  } finally {
    await orch.close();
  }
});
