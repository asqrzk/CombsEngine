/**
 * Demo: runtime orchestration — two agent servers on free ports, an
 * orchestrator delegating over authenticated WebSockets, sessions in SQLite.
 *
 *   deno run --allow-ffi --allow-read --allow-write --allow-env --allow-net examples/orchestration_demo.ts
 */
import { Combs } from "../core/mod.ts";
import {
  createAgentServer,
  Orchestrator,
  SessionStore,
} from "../runtime/mod.ts";
import { getLogger } from "../telemetry/mod.ts";

const log = getLogger("demo.orchestration");
const model = new URL("../../../models/SmolLM2-135M", import.meta.url).pathname;

// 1. Boot one native engine (shared in-process for the demo; subprocesses
//    would each boot their own via AgentPool).
const engine = await Combs.init({ model, engine: { max_seq_len: 1024 } });
log.info("engine loaded");

// 2. Two persona agent servers on free ports (auth tokens auto-minted).
async function personaServer(name: string, persona: string) {
  return await createAgentServer({
    name,
    handler: async (input) => {
      const question = String(input.text ?? "");
      const { text } = await engine.complete({
        messages: [
          { role: "system", content: persona },
          { role: "user", content: question },
        ],
        max_tokens: 40,
        temperature: 0.7,
      });
      return { name, reply: text };
    },
    metadata: { persona: persona.slice(0, 60) },
  });
}

const poet = await personaServer("poet", "You answer in one short rhyming couplet.");
const scientist = await personaServer("scientist", "You answer in one precise factual sentence.");
log.info("agent servers up", { poet: poet.port, scientist: scientist.port });

// 3. Orchestrator delegates over authenticated WebSockets.
const orch = new Orchestrator();
await orch.register({ name: "poet", url: poet.url, token: poet.token });
await orch.register({ name: "scientist", url: scientist.url, token: scientist.token });

const question = "What is photosynthesis?";
const results = await orch.broadcast(["poet", "scientist"], { text: question });
for (const [name, result] of results) {
  const reply = (result.data as { reply?: string })?.reply ?? result.error;
  log.info(`answer from ${name}`, { reply: reply?.slice(0, 140) });
}

// 4. Persist the session to SQLite (resumable).
const sessions = new SessionStore("/tmp/combs-demo-sessions.sqlite");
const session = sessions.save({
  id: crypto.randomUUID(),
  agent: "orchestrator",
  threadId: "demo-1",
  data: {
    question,
    answers: Object.fromEntries(
      [...results].map(([n, r]) => [n, (r.data as { reply?: string })?.reply ?? r.error]),
    ),
  },
});
log.info("session saved", { id: session.id, resumed: sessions.get(session.id)?.threadId });

await orch.close();
await poet.close();
await scientist.close();
sessions.close();
engine.close();
