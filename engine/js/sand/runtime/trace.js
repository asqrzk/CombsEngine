/**
 * @combs/sand — the trace.
 *
 * The runtime publishes events shaped like @combs/observe's ObsEvent
 * (`{id, ts, source, kind, name, traceId, attrs, input, output, status}`) to an
 * injected bus with a `publish` method — an @combs/observe EventBus, or the
 * built-in ring below. Attrs are kept to eight keys with values under 120
 * characters so a platform run can mirror every event to its timeline rail
 * unchanged; bodies and answers travel in `input`/`output`, never in attrs.
 */

let counter = 0;
const newId = () => `s${(++counter).toString(36)}${Date.now().toString(36).slice(-4)}`;

/** The built-in ring: the fallback bus, and what `trace()` reads back. */
export function createRing(capacity = 5000) {
  const events = [];
  return {
    events,
    publish(partial) {
      const ev = { id: partial.id ?? newId(), ts: partial.ts ?? Date.now(), ...partial };
      events.push(ev);
      if (events.length > capacity) events.splice(0, events.length - capacity);
      return ev;
    },
  };
}

/** Clamps attrs to the rail's limits: ≤8 keys, string values ≤120 chars. */
export function clampAttrs(attrs = {}) {
  const out = {}; let n = 0;
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null) continue;
    if (n++ >= 8) break;
    const s = typeof v === "string" ? v : typeof v === "number" || typeof v === "boolean" ? String(v) : JSON.stringify(v);
    out[k] = s.length > 120 ? s.slice(0, 117) + "…" : s;
  }
  return out;
}

/** Makes an emitter bound to a bus and a trace id. */
export function emitter(bus, traceId) {
  const ring = bus ?? createRing();
  return {
    bus: ring,
    emit(name, kind, attrs, extra = {}) {
      return ring.publish({ source: "sand", kind, name, traceId, attrs: clampAttrs(attrs), ...extra });
    },
  };
}

/** A human line per event (what `sand log` and the playground print). */
export function traceLines(events) {
  const out = [];
  for (const e of events) {
    const a = e.attrs ?? {};
    const t = new Date(e.ts).toISOString().slice(11, 23);
    switch (e.name) {
      case "sand.run": out.push(`${t}  run       ${e.kind === "span.start" ? "start" : `end · ${a.status ?? ""} · ${a.passes ?? 0} pass${a.passes === "1" ? "" : "es"} · ${a.tokens ?? 0} tokens · ${a.ms ?? ""} ms`}`); break;
      case "sand.request": out.push(`${t}  request   ${a.model ?? ""}${a.road ? "/" + a.road : ""} · ${a.questions ?? ""}q · ${a.passes != null ? a.passes + " pass" + (a.passes === "1" ? "" : "es") : (a.tokens ?? "") + " tokens"} · ${a.ms ?? ""} ms${a.balance ? " · balance " + a.balance : ""}${a.status !== "ok" && a.status ? " · " + a.status : ""}`); break;
      case "sand.decision": out.push(`${t}  decision  ${a.at ?? ""} · ${a.how ?? ""} ${a.top ?? ""}${a.p_top ? " " + a.p_top : ""} · ${a.decision ?? ""}${a.reason ? " (" + a.reason + ")" : ""}${a.memo === "true" ? " · memo" : ""}`); break;
      case "sand.effect": out.push(`${t}  effect    ${a.at ?? ""} · ${a.effect ?? ""} ${a.name ?? ""}${a.value ? " = " + a.value : ""}`); break;
      case "sand.budget": out.push(`${t}  budget    ${a.limit} ${a.used}/${a.max} · ${a.action}`); break;
      case "sand.log": out.push(`${t}  log       ${a.at ? a.at + " · " : ""}${typeof e.output === "string" ? e.output : a.message ?? ""}`); break;
      default: out.push(`${t}  ${e.name}`);
    }
  }
  return out;
}
