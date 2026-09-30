/**
 * @combs/sand — a Deno Worker body for a host that runs programs in-process.
 *
 * Receives `{id, module, input, config, limits?, session?, traceId?}`, imports
 * the built program, runs it over the input with its own context, and posts
 * `{id, result, session, events}` back — the same envelope the child protocol
 * writes on stdout, minus the pipe. Nothing is shared between workers but the
 * messages, so a person cannot be asked from here: `confirm` is unsure.
 */

import { run, configure, setPorts } from "./verbs.js";

setPorts({ env: (n) => { try { return Deno.env.get(n); } catch { return undefined; } }, log: () => {} });

self.onmessage = async ({ data }) => {
  const { id, module, input, config, limits, session, traceId } = data;
  const events = [];
  const bus = { events, publish(p) { const ev = { id: `w${events.length + 1}`, ts: Date.now(), ...p }; events.push(ev); return ev; } };
  try {
    if (config) configure(config);
    if (limits) configure({ limits });
    const mod = await import(module);
    const out = await run(mod.default, { input, bus, session, traceId });
    self.postMessage({ id, result: out.result === undefined ? null : out.result, session: out.session ?? null, events });
  } catch (e) {
    self.postMessage({ id, error: { code: e?.code ?? "E-RUNTIME", message: String(e?.message ?? e), at: e?.at ?? null }, session: e?.session ?? null, events });
  }
};
