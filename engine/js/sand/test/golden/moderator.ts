import { kind, scale, shape, text, settle, configure, decide, decideAll, classify, ask, extract, chat, assistant, tool, memory, confirm, emit, unsure, and, or, not, parallel, race, timeout, retry, using, named, log, run, trace } from "@combs/sand";
export default async function main(input: string) {
// Moderator. One post. A yes/no question decides; when the model is unsure a
// person is asked, and if nobody is there the post waits in a queue. The
// counts live in a memory too. Needs: the Combs model and a memory.

settle({ floor: 0.7, gap: 0.3, passes: 1, wait: 30_000 });
const queue = memory("moderation-queue", { name: "queue", at: "6:7" });
const tally = memory("moderation-tally", { name: "tally", at: "7:7" });

const d1 = await decide(input, "the message attacks a person or a group", { not: ["it quotes, reports or condemns such an attack"], noun: "the message", at: "9:1" });  if ((d1.value) === true) {
  emit("removed", { at: "10:3" });
  (await tally.remember({ action: "removed", post: input }, { at: "11:3" }));
} else if ((d1.value) !== unsure) {
  emit("kept", { at: "13:3" });
  (await tally.remember({ action: "kept", post: input }, { at: "14:3" }));
} else {
  const answer = (await confirm("The model was not sure about this post. Remove it?", { at: "16:18" }));
  if (answer === true) { emit("removed by a person", { at: "17:26" }); (await tally.remember({ action: "removed", by: "person", post: input }, { at: "17:55" })); }
  else if (answer === false) { emit("kept by a person", { at: "18:32" }); (await tally.remember({ action: "kept", by: "person", post: input }, { at: "18:58" })); }
  else { (await queue.remember({ post: input, held: new Date().toISOString() }, { at: "19:10" })); emit("held for review", { at: "19:79" }); }
}

const waiting = (await queue.recall("held posts", { at: "22:17", limit: 100 }));
const removed = (await tally.recall("removed", { at: "23:17", limit: 1000 }));
log(waiting.length, "waiting for a person ·", removed.length, "removed so far");

}
if (import.meta.main) run(main);
