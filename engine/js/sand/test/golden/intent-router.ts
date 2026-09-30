import { kind, scale, shape, text, settle, configure, decide, decideAll, classify, ask, extract, chat, assistant, tool, memory, confirm, emit, unsure, and, or, not, parallel, race, timeout, retry, using, named, log, run, trace } from "@combs/sand";
export default async function main(input: string) {
const Mode = kind("what does this turn need from the assistant", {
  chat:  "a question or conversation answerable from knowledge alone",
  tools: "an action: memory lookup, web, files, commands",
  plan:  "a multi-step task needing several actions in sequence",
}, { name: "Mode", fallback: "tools" });                       // a wrong "tools" is the cheap error, as before
settle({ floor: 0.5, passes: 1, ms: 1500 });    // the original's budgetMs, now enforced by the runtime

const FORCE = new Map([["/chat", "chat"], ["/tools", "tools"], ["/plan", "plan"], ["/agent", "plan"]]);
const GREETING = /^(hi|hello|hey|yo|thanks|thank you|ok|okay|good (morning|evening|night))[.!\s]*$/i;
const ACT_HINTS = /\b(search|browse|fetch|look up|read (the |this )?(url|page|site|link)|run|execute|command|install|create (a )?file|write (a )?file|todo|remember|recall|my notes|memory|screenshot|open http)\b/i;
const PLAN_HINTS = /\b(plan (and|then|out)|step by step build|research .{0,40}(and|then) (write|build|create|summarize)|multi[- ]?step)\b/i;

const t = input.trim();
if (!t) return { mode: "chat", via: "rules" };
const slash = t.split(/\s+/, 1)[0].toLowerCase();
if (FORCE.has(slash)) return { mode: FORCE.get(slash), via: "rules" };
if (GREETING.test(t)) return { mode: "chat", via: "rules" };
if (t.length < 12 && !ACT_HINTS.test(t)) return { mode: "chat", via: "rules" };

try {
  // A sentence can carry an action's words without asking for it. The word-matchers
  // below only run when the message is asking, not quoting, translating or defining.
  const d1 = await decide(input, "the message only mentions an action rather than asking for one", { not: ["they are asking us to do it now","they are giving an instruction"], noun: "the message", at: "23:3", speculate: [Mode] });  if ((not(d1.value)) === true) {
    if (PLAN_HINTS.test(t)) return { mode: "plan", via: "rules" };
    if (ACT_HINTS.test(t)) return { mode: "tools", via: "rules" };
  }
  switch ((await decide(input, Mode, { at: "27:11" })).value) {
    case Mode.chat:  return { mode: "chat",  via: "model" };
    case Mode.plan:  return { mode: "plan",  via: "model" };
    case Mode.tools: return { mode: "tools", via: "model" };
    case unsure:     return { mode: "tools", via: "fallback" };
  }
} catch {
  return { mode: "tools", via: "fallback" };   // E-BUDGET (1500 ms) or E-WIRE: fall open, as the original did
}

}
if (import.meta.main) run(main);
