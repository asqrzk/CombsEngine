/**
 * @combs/sand — a Deno Worker body for `sand run --workers N`.
 *
 * Receives `{ module, input, config }`, imports the built program, runs it over
 * the input with its own context, and posts `{ result, events, error }` back.
 * Nothing is shared between workers but the messages.
 */

import { run, configure, setPorts } from "./verbs.js";

setPorts({ env: (n) => { try { return Deno.env.get(n); } catch { return undefined; } }, log: () => {} });

self.onmessage = async ({ data }) => {
  const { id, module, input, config } = data;
  try {
    if (config) configure(config);
    const mod = await import(module);
    const main = mod.default;
    const events = [];
    const bus = { events, publish(p) { const ev = { id: `w${events.length + 1}`, ts: Date.now(), ...p }; events.push(ev); return ev; } };
    const { result } = await run(main, { input, bus });
    self.postMessage({ id, result: result === undefined ? null : result, events });
  } catch (e) {
    self.postMessage({ id, error: { code: e?.code ?? "E-RUNTIME", message: String(e?.message ?? e), at: e?.at ?? null }, events: e?.trace?.events ?? [] });
  }
};
