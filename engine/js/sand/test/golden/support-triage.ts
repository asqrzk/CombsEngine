import { kind, scale, shape, text, settle, configure, decide, decideAll, classify, ask, extract, chat, assistant, tool, memory, confirm, emit, unsure, and, or, not, parallel, race, timeout, retry, using, named, log, run, trace } from "@combs/sand";
export default async function main(input: string) {
const Ticket = kind("what is the person writing in about", {
  refund: "the person wants money they paid returned",
  cancel: "the person wants their subscription ended now",
  other:  "none of these",
}, { name: "Ticket", fallback: "other" });
const Mood = scale("how does the person sound", ["furious", "annoyed", "neutral", "pleased"], { name: "Mood" });
settle({ floor: 0.55, gap: 0.4, confidence: 0.3 });

const ticket = (await decide(input, Ticket, { at: "9:16", speculate: [Mood] }));
switch (ticket.value) {
  case Ticket.refund:
    const d1 = await decide(input, Mood, { at: "12:5" });  if ((d1.atLeast("annoyed")) === true) { log("refund, escalate") }
    else { log("refund, standard") }
    break;
  case "cancel":
    log("cancel flow");
    break;
  case unsure:
    log("could not tell:", ticket.reason);
    break;
  default:
    log("general inbox");
}

}
if (import.meta.main) run(main);
