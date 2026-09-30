import { kind, scale, shape, text, settle, configure, decide, decideAll, classify, ask, extract, chat, assistant, tool, memory, confirm, emit, unsure, and, or, not, parallel, race, timeout, retry, using, named, log, run, trace } from "@combs/sand";
export default async function main(input: string) {
// Order assistant. The chat model picks the tools and their arguments; the
// program runs them. A refund needs a person's yes. Needs: the Combs model and
// a chat model that supports tools.

const OrderArgs = shape({ orderId: text("the order number, like A-1234") }, { name: "OrderArgs" });
const RefundArgs = shape({ orderId: text("the order number"), reason: text("why, in the customer's own words") }, { name: "RefundArgs" });
settle({ floor: 0.6, passes: 4, turns: 12, tokens: 8000, wait: 600_000 });

// Stand-in data; a real program would call the shop's API from these tools.
const orders: Record<string, { status: string; total: number }> = {
  "A-1234": { status: "shipped", total: 42 },
  "A-2000": { status: "delivered", total: 129 },
};
const lookupOrder = tool("look up an order by its number", OrderArgs, ({ orderId }) => orders[orderId] ?? { error: "no such order" }, { name: "lookupOrder", at: "14:7" });
const refundOrder = tool("refund an order in full", RefundArgs, async ({ orderId, reason }) => {
  const order = orders[orderId];
  if (!order) return { error: "no such order" };
  if (order.status === "refunded") return { error: "already refunded" };
  if ((await confirm(`Refund ${orderId} for ${order.total}? Reason: ${reason}`, { at: "19:7" })) !== true) return { refunded: false, note: "a person did not approve this refund" };
  order.status = "refunded";
  return { refunded: true, total: order.total };
}, { name: "refundOrder", at: "15:7" });
const bot = assistant("You are Acme's order assistant. Use the tools for any order fact; never invent one. Two sentences at most.", { name: "bot", at: "23:7", tools: [lookupOrder, refundOrder], settle: { turns: 4 } });

const d1 = await decide(input, "the message is about an existing order", { not: ["they are asking about something they have not bought yet"], noun: "the message", at: "25:1" });  if ((d1.value) === true) {
  const reply = (await bot.say(input, { at: "26:17" }));
  const refunded = reply.calls.some((c) => c.name === "refundOrder" && (c.result as { refunded?: boolean })?.refunded === true);
  // The guardrail: the reply may not claim a refund the tool did not make.
  const d2 = await decide(reply, "the reply says a refund was made", { not: ["it says one was requested or declined"], noun: "the reply", at: "29:3" });  if ((and(d2.value, not((refunded)))) === true) {
    emit((await bot.say("Correct yourself: no refund was made. Say what actually happened.", { at: "30:10" })), { at: "30:5" });
  } else if ((and(d2.value, not((refunded)))) !== unsure) {
    emit(reply, { at: "32:5" });
  } else {
    emit(reply, { at: "34:5" });
  }
} else if ((d1.value) !== unsure) {
  emit("I can help with an order you have placed. What is the order number?", { at: "37:3" });
} else {
  emit((await bot.say(input, { at: "39:8" })), { at: "39:3" });
}

}
if (import.meta.main) run(main);
