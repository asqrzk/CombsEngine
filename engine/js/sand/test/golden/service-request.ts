import { kind, scale, shape, text, settle, configure, decide, decideAll, classify, ask, extract, chat, assistant, tool, memory, confirm, emit, unsure, and, or, not, parallel, race, timeout, retry, using, named, log, run, trace } from "@combs/sand";
export default async function main(input: string) {
const Urgency = kind("how urgent is this", { now: "it must happen today", soon: "this week is fine", whenever: "no rush is expressed" }, { name: "Urgency" });
const Channel = kind("how do they want to be reached", { email: "they ask for a written reply", phone: "they ask for a call", any: "no preference is expressed" }, { name: "Channel" });
const ServiceRequest = shape({ urgency: Urgency, channel: Channel }, { name: "ServiceRequest" });

const d1 = await decide(input, "the message is a service request", { not: ["they are describing something that already happened"], noun: "the message", at: "5:1", speculate: [ServiceRequest] });  if ((d1.value) === true)
{
(await extract(input, ServiceRequest, { at: "7:1" }))
}

}
if (import.meta.main) run(main);
