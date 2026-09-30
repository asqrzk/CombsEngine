import { kind, scale, shape, text, settle, configure, decide, decideAll, classify, ask, extract, unsure, and, or, not, parallel, race, timeout, retry, using, named, log, run, trace } from "@combs/sand";
export default async function main(input: string) {
const Sentiment = scale("how does the review sound", { hostile: "angry, demanding a remedy", cold: "disappointed", warm: "satisfied", glowing: "delighted and says so" }, { name: "Sentiment" });
settle({ passes: 100, ms: 20_000 });

const reviews: string[] = JSON.parse(input);
const flagged = await parallel(reviews, async (review) => {
  const [d1, d2] = await decideAll(review, [Sentiment, { means: "the review names a specific employee", not: ["they name a product"], noun: "the review" }], { at: "6:3" });  if ((and(d1.atMost("cold"), d2.value)) === true) {
    return review;
  }
  return null;
}, { at: "5:23", limit: 8 });
log(flagged.filter(Boolean).length, "reviews need a manager");

}
if (import.meta.main) run(main);
