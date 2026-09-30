import { kind, scale, shape, text, settle, configure, decide, decideAll, classify, ask, extract, chat, assistant, tool, memory, confirm, emit, unsure, and, or, not, parallel, race, timeout, retry, using, named, log, run, trace } from "@combs/sand";
export default async function main(input: string) {
// Support desk. One message comes in. The Combs model answers the yes/no
// questions, the chat model writes the reply, a person says whether to file a
// ticket, and the program files it. Needs: the Combs model and a chat model.

const Topic = kind("what is the message about", {
  billing: "a charge, an invoice or a payment",
  outage: "something is down, broken or failing",
  feature: "they want something that does not exist yet",
  other: "none of these",
}, { name: "Topic", fallback: "other" });
const Urgency = kind("how urgent is it", { now: "it must happen today", soon: "this week is fine", whenever: "no rush is expressed" }, { name: "Urgency", fallback: "whenever" });
const Details = shape({ topic: Topic, urgency: Urgency, summary: text("what they need, in one sentence") }, { name: "Details" });
const TicketArgs = shape({ title: text("a short title for the ticket"), summary: text("what happened, in one paragraph"), urgency: Urgency }, { name: "TicketArgs" });
settle({ floor: 0.6, passes: 4, turns: 4, tokens: 6000, wait: 600_000 });

// A tool is a plain function with a shape for its arguments. The program runs it, never the model.
const createTicket = tool("file a support ticket", TicketArgs, async (args) => {
  log("ticket filed:", args.title);
  return { id: "T-" + Date.now().toString(36), ...args };
}, { name: "createTicket", at: "17:7" });

const d1 = await decide(input, "the message is a support issue", { not: ["they are only saying thanks or making small talk"], noun: "the message", at: "22:1", speculate: [Details] });  if ((d1.value) === true) {
  const details = (await extract(input, Details, { at: "23:19" }));
  const desk = chat("You are Acme's support desk. Be brief and warm. Never promise a refund or a date.", { name: "desk", at: "24:9" });
  let reply = (await desk.say(`Reply to this customer. What they need: ${details.summary}. Their message: ${input}`, { at: "25:15" }));
  // A guardrail is a yes/no question about the reply itself.
  const d2 = await decide(reply, "the reply promises a refund or a date", { not: ["it only says what happens next"], noun: "the reply", at: "27:3" });  if ((d2.value) === true) {
    reply = (await desk.say("Rewrite it without promising a refund or a date.", { at: "28:13" }));
  }
  emit(reply, { at: "30:3" });
  if ((await confirm("File a ticket for this message?", { at: "31:7" })) === true) {
    const ticket = (await createTicket.run({ title: details.summary, summary: input, urgency: details.urgency }));
    emit(`Filed as ${ticket.id}.`, { at: "33:5" });
  }
} else if ((d1.value) !== unsure) {
  emit((await ask(`Answer briefly and kindly: ${input}`, { at: "36:8" })), { at: "36:3" });
} else {
  emit("Thanks. A person will read this and get back to you.", { at: "38:3" });
}

}
if (import.meta.main) run(main);
