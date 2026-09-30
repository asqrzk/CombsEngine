/**
 * The child protocol v1, pinned against a real child: a built program is
 * spawned exactly as `cli/run.ts` spawns it — `deno run --no-prompt` with the
 * fence the plan implies — and speaks JSON lines to a `Deno.serve` fake API on
 * 127.0.0.1. Asserted: the ready line and one envelope per input, a control
 * line answering a `sand.confirm`, the session in the envelope and carried into
 * the next input, and that a bogus permission flag ends the command with
 * `E-FENCE` and exit 2 rather than running wider.
 */
import { assert, assertEquals, assertStringIncludes } from "jsr:@std/assert";
import { transform } from "../mod.ts";

const here = new URL(".", import.meta.url).pathname;
const SAND = here.replace(/\/test\/$/, "");
const enc = new TextEncoder(), dec = new TextDecoder();

/** One pass per request; a yes/no question is true when the state asks something. Plus a chat completion. */
function fakeApi() {
  const seen: { path: string; body: any }[] = [];
  const server = Deno.serve({ hostname: "127.0.0.1", port: 0, onListen: () => {} }, async (req) => {
    const url = new URL(req.url);
    const body = await req.json().catch(() => ({}));
    seen.push({ path: url.pathname, body });
    if (url.pathname.endsWith("/v1/classify")) {
      const answers = (body.questions ?? []).map((q: any) => ({
        name: q.name, type: q.type, top: q.type === "noul" ? String(String(body.state ?? "").includes("?")) : (q.options ?? q.levels)[0].id,
        p_top: 0.9, distribution: { true: 0.9, false: 0.1 }, decision: "act", reason: null, confidence: 0.9, margin: 0.8, implications: [],
      }));
      return Response.json({ answers, model: "fake", usage: { forward_passes: 1 } }, { headers: { "x-combs-passes": "1", "x-combs-balance": "9", "x-combs-request": "r1" } });
    }
    if (url.pathname.endsWith("/chat/completions")) {
      return Response.json({ choices: [{ message: { content: "here is the reply" }, finish_reason: "stop" }], usage: { total_tokens: 12 } });
    }
    return new Response("no", { status: 404 });
  });
  return { server, seen, host: `127.0.0.1:${(server.addr as Deno.NetAddr).port}` };
}

/** Builds a `.sand` source the way `cli/build.ts` does, into a temp dir. */
async function buildInto(source: string, name: string) {
  const dir = await Deno.makeTempDir({ prefix: `sand-protocol-${name}-` });
  const { code, plan } = transform(source, { file: `${name}.sand` });
  await Deno.writeTextFile(`${dir}/${name}.ts`, code);
  await Deno.writeTextFile(`${dir}/deno.json`, JSON.stringify({ imports: { "@combs/sand": `file://${SAND}/mod.ts` }, nodeModulesDir: "none" }, null, 2));
  return { dir, ts: `${dir}/${name}.ts`, plan };
}

/** The child, spawned under exactly the fence the plan and the config imply. */
function spawn(ts: string, dir: string, permissions: string[]) {
  return new Deno.Command("deno", {
    args: ["run", "--no-prompt", ...permissions, "--config", `${dir}/deno.json`, ts],
    env: { SAND_CHILD: "1" }, stdin: "piped", stdout: "piped", stderr: "piped",
  }).spawn();
}

/** Reads whole JSON lines off a stream and hands each to `onLine`. */
async function pump(stream: ReadableStream<Uint8Array>, onLine: (l: string) => void) {
  let buf = "";
  for await (const chunk of stream) {
    buf += dec.decode(chunk, { stream: true });
    let k;
    while ((k = buf.indexOf("\n")) >= 0) { const l = buf.slice(0, k); buf = buf.slice(k + 1); if (l.trim()) onLine(l); }
  }
  if (buf.trim()) onLine(buf);
}

const CHAT_BOT = `settle({ passes: 4, turns: 4, wait: 5000 });
const desk = chat("You answer support mail in one sentence.");
if (the {{input}} is a question unless it is only a greeting) {
  emit(desk.say({{input}}));
  if (confirm("File a ticket?") === true) { emit("filed"); }
  return "asked";
}
return "greeted";
`;

Deno.test("child protocol v1: the ready line, one envelope per input, a control line, and the session", async () => {
  const api = fakeApi();
  const built = await buildInto(CHAT_BOT, "bot");
  const config = {
    models: {
      combs: { kind: "decision", url: `http://${api.host}`, key: "k", road: "seat" },
      "combs-chat": { kind: "chat", url: `http://${api.host}/v1`, key: "k", model: "default" },
    },
    default: { decide: "combs", ask: "combs-chat", memory: "none" },
  };
  const child = spawn(built.ts, built.dir, [`--allow-net=${api.host}`, "--allow-env=SAND_CHILD"]);
  const writer = child.stdin.getWriter();
  const tell = (o: unknown) => writer.write(enc.encode(JSON.stringify(o) + "\n"));

  const envelopes: any[] = [];
  const events: any[] = [];
  let ready: any = null;
  const outDone = pump(child.stdout, (l) => { const j = JSON.parse(l); if (ready === null) ready = j; else envelopes.push(j); });
  const errDone = pump(child.stderr, (l) => {
    let ev: any = null;
    try { ev = JSON.parse(l); } catch { return; }
    events.push(ev);
    // a person is asked: answer it on the child's stdin while the next input queues
    if (ev.name === "sand.confirm" && ev.kind === "ask") tell({ confirm: { id: ev.attrs.id, answer: { answer: "y", by: "tty" } } });
  });

  await tell({ sand: "1", config, limits: { passes: 8, tokens: 5000, ms: 20_000, turns: 8, wait: 5000 }, file: "bot.sand", traceId: "t-protocol" });
  await tell({ input: "where is my order?", session: { id: "s1" } });
  // a bare JSON string is still an input
  await tell("hello there");
  while (envelopes.length < 2) await new Promise((r) => setTimeout(r, 20));
  await writer.close();
  const status = await child.status;
  await Promise.all([outDone, errDone]);
  api.server.shutdown();

  assertEquals(ready, { sand: "1", ready: true });
  assertEquals(status.code, 0);
  assertEquals(envelopes.length, 2);
  assertEquals(envelopes[0].result, "asked");
  assertEquals(envelopes[0].session.id, "s1");
  assertEquals(envelopes[0].session.state.v, 1);
  assertEquals(envelopes[0].session.state.chats.desk.length, 2, "one user turn and one assistant turn are kept");
  assertEquals(envelopes[1].result, "greeted");
  const asked = events.filter((e) => e.name === "sand.confirm" && e.kind === "ask");
  assertEquals(asked.length, 1);
  assertEquals(asked[0].traceId, "t-protocol");
  const answered = events.find((e) => e.name === "sand.confirm" && e.kind === "span.end");
  assertEquals(answered.attrs.answer, "true");
  assertEquals(answered.attrs.by, "tty");
  assert(events.some((e) => e.name === "sand.output"), "emitted pieces travel as sand.output");
  assert(events.some((e) => e.name === "sand.turn"), "a say is one turn");
  await Deno.remove(built.dir, { recursive: true });
});

Deno.test("child protocol v1: an error is an envelope of its own and the run keeps going", async () => {
  const api = fakeApi();
  const built = await buildInto(`settle({ passes: 1 });\nif (the {{input}} is a question unless it is a greeting) { return "q" }\nreturn "no";\n`, "err");
  const child = spawn(built.ts, built.dir, [`--allow-net=${api.host}`, "--allow-env=SAND_CHILD"]);
  const writer = child.stdin.getWriter();
  const tell = (o: unknown) => writer.write(enc.encode(JSON.stringify(o) + "\n"));
  const envelopes: any[] = [];
  let ready: any = null;
  const outDone = pump(child.stdout, (l) => { const j = JSON.parse(l); if (ready === null) ready = j; else envelopes.push(j); });
  const errDone = pump(child.stderr, () => {});
  // no decision model at all: every input fails the same way, and each failure is its own envelope
  await tell({ sand: "1", config: { models: {}, default: { decide: "none", ask: "none", memory: "none" } }, limits: { passes: 1 } });
  await tell({ input: "where is my order?" });
  await tell({ input: "and again?" });
  while (envelopes.length < 2) await new Promise((r) => setTimeout(r, 20));
  await writer.close();
  const status = await child.status;
  await Promise.all([outDone, errDone]);
  api.server.shutdown();
  assertEquals(envelopes.length, 2);
  assertEquals(envelopes[0].error.code, "E-MODEL");
  assertStringIncludes(envelopes[0].error.message, "E-MODEL");
  assertEquals(envelopes[1].error.code, "E-MODEL");
  assertEquals(status.code, 3, "a failed input is exit 3");
  await Deno.remove(built.dir, { recursive: true });
});

Deno.test("sand run: a bogus permission flag ends the command with E-FENCE and exit 2, never wider", async () => {
  const api = fakeApi();
  const dir = await Deno.makeTempDir({ prefix: "sand-fence-" });
  await Deno.writeTextFile(`${dir}/tiny.sand`, `settle({ passes: 1 });\nif (the {{input}} is a question unless it is a greeting) { return "q" }\nreturn "no";\n`);
  const run = async (extra: string[]) =>
    await new Deno.Command("deno", {
      args: ["run", "-A", `${SAND}/cli/main.ts`, "run", `${dir}/tiny.sand`, "--input", "where is it?", "--quiet", ...extra],
      env: {
        SAND_IMPORT: `${SAND}/mod.ts`, COMBS_API_KEY: "k", COMBS_API_URL: `http://${api.host}`,
        HOME: dir, PATH: Deno.env.get("PATH") ?? "", SAND_CONFIG: "",
      },
      clearEnv: true, stdout: "piped", stderr: "piped",
    }).output();

  const bad = await run(["--allow", "--allow-nonsense"]);
  api.server.shutdown();
  assertEquals(bad.code, 2, dec.decode(bad.stderr));
  assertStringIncludes(dec.decode(bad.stderr), "E-FENCE");
  assert(!dec.decode(bad.stdout).includes('"q"'), "a refused fence never runs the program");
  await Deno.remove(dir, { recursive: true });
});
