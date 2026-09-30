import { kind, scale, shape, text, settle, configure, decide, decideAll, classify, ask, extract, unsure, and, or, not, parallel, race, timeout, retry, using, named, log, run, trace } from "@combs/sand";
export default async function main(input: string) {
const Topic = kind("what is the ticket about", { billing: "a charge or invoice", outage: "something is down or broken", feature: "they want something new", other: "none of these" }, { name: "Topic", fallback: "other" });
const Ticket = shape({ topic: Topic, product: text("the product named, or 'unknown'"), summary: text("one sentence in the customer's words") }, { name: "Ticket" });
settle({ floor: 0.6, passes: 4, tokens: 4000 });

const d1 = await decide(input, "the message asks for a status update", { not: ["they are reporting the problem for the first time"], noun: "the message", at: "5:1", speculate: [Ticket] });  if ((d1.value) === true) {
  const t = (await extract(input, Ticket, { at: "6:13" }));
  const reply = (await ask(`Write a two-sentence status acknowledgement for ${t.product} in the customer's tone: ${input}`, { at: "7:17" }));
  log(reply);
} else if ((d1.value) !== unsure) {
  log("filed", (await extract(input, Ticket, { at: "10:16", using: "ask" })));
} else {
  log("could not tell; filed for a human");
}

}
if (import.meta.main) run(main);
