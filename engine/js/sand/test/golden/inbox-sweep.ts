import { kind, scale, shape, text, settle, configure, decide, decideAll, classify, ask, extract, chat, assistant, tool, memory, confirm, emit, unsure, and, or, not, parallel, race, timeout, retry, using, named, log, run, trace } from "@combs/sand";
export default async function main(input: string) {
// Inbox sweep. A list of messages, all at once, one pass each, under a budget.
// Needs: the Combs model only.

const Category = kind("what is this message", {
  complaint: "something went wrong and they are unhappy about it",
  question: "they want to know something",
  praise: "they are pleased and say so",
  noise: "an ad, a bounce, or nothing that needs a reply",
}, { name: "Category", fallback: "noise" });
const Heat = scale("how upset does the writer sound", ["calm", "annoyed", "furious"], { name: "Heat" });
settle({ floor: 0.55, passes: 200, ms: 30_000 });

const messages: string[] = JSON.parse(input);
const sorted = (await parallel(messages, async (message) => {
  const [d1, d2] = await decideAll(message, [Category, Heat], { at: "15:3" });  if ((and(d1.is("complaint"), d2.atLeast("annoyed"))) === true) {
    return { message, action: "call back today" };
  } else if ((or(d1.is("complaint"), d1.is("question"))) === true) {
    return { message, action: "reply this week" };
  } else if ((and(d1.is("complaint"), d2.atLeast("annoyed"))) !== unsure && (or(d1.is("complaint"), d1.is("question"))) !== unsure) {
    return { message, action: "file" };
  } else {
    return { message, action: "a person reads it" };
  }
}, { at: "14:16", limit: 8, tolerate: true }));

const counts: Record<string, number> = {};
for (const r of sorted) {
  const action = r instanceof Error ? "not sorted: out of passes" : r.action;
  counts[action] = (counts[action] ?? 0) + 1;
}
log(counts);
return counts;

}
if (import.meta.main) run(main);
