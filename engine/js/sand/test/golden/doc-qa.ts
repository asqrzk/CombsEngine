import { kind, scale, shape, text, settle, configure, decide, decideAll, classify, ask, extract, chat, assistant, tool, memory, confirm, emit, unsure, and, or, not, parallel, race, timeout, retry, using, named, log, run, trace } from "@combs/sand";
export default async function main(input: string) {
// Doc Q&A. Teach it with "remember: <passage>"; then ask. Recall finds the
// passages, the chat model answers from them and cites them, and a yes/no
// question checks that the answer stayed inside the passages.
// Needs: the Combs model, a chat model, and a memory (the platform's, or a local file).

settle({ floor: 0.6, passes: 2, turns: 1, tokens: 8000 });
const handbook = memory("handbook", { name: "handbook", at: "7:7" });

if (input.startsWith("remember:")) {
  const id = (await handbook.remember(input.slice(9).trim(), { at: "10:14" }));
  emit(`Noted (${id}).`, { at: "11:3" });
  return;
}

const question = input;
const d1 = await decide(question, "the question asks how the company works", { not: ["it is small talk"], noun: "the question", at: "16:1" });  if ((d1.value) === true) {
  const hits = (await handbook.recall(question, { at: "17:16", limit: 4 }));
  if (!hits.length) { emit("The handbook has nothing on that yet.", { at: "18:23" }); return; }
  const passages = hits.map((h, i) => "[" + (i + 1) + "] " + h.text).join("\n\n");
  const answer = (await ask(`Answer the question from the passages only, citing them as [1], [2]. If they do not cover it, say so.\n\nPassages:\n${passages}\n\nQuestion: ${question}`, { at: "20:18", maxTokens: 300 }));
  const pair = `Passages:\n${passages}\n\nAnswer:\n${answer}`;
  const check = pair;
  const d2 = await decide(check, "the answer with its passages has an answer that claims something the passages do not say", { not: ["the answer says the passages do not cover it"], noun: "the answer with its passages", at: "23:3" });  if ((d2.value) === true) {
    emit("The handbook does not clearly cover that. The closest passage: " + hits[0].text, { at: "24:5" });
  } else if ((d2.value) !== unsure) {
    emit(answer, { at: "26:5" });
  } else {
    emit(answer + "\n\n(Please check this against the handbook.)", { at: "28:5" });
  }
} else {
  emit("Ask me how things work here; teach me with \"remember: …\".", { at: "31:3" });
}

}
if (import.meta.main) run(main);
