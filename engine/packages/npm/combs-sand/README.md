# @combs/sand

**Semantic abstraction of neurolinguistic determinism.** SAND is TypeScript plus three things the TypeScript checker cannot express: a *binding* that carries a noun, a *prose predicate* that is a model decision, and a third truth value, *unsure*. Everything else is TypeScript and stays TypeScript.

```
const Urgency = kind("how urgent is this", { now: "it must happen today", soon: "this week is fine", whenever: "no rush is expressed" });
const Channel = kind("how do they want to be reached", { email: "they ask for a written reply", phone: "they ask for a call", any: "no preference is expressed" });
const ServiceRequest = shape({ urgency: Urgency, channel: Channel });

if(the {{input}} is a service request )
{
extract ({{input}})
}
```

```sh
export COMBS_API_KEY=cmb_…          # from https://api.combs.network/account
deno install -g -A -n sand jsr:@combs/sand/cli
sand check service-request.sand      # what it asks, what it costs, whether it type-checks
sand run   service-request.sand --input "My password reset never arrived; please email me today"
```

That program is **one forward pass**: the condition and the two questions the extraction needs travel in one request, and the extraction is answered from memory.

## The three extensions

**Bindings `{{name}}`.** `{{ident}}` or `{{ident.path}}` names a variable in scope. Inside a string or template literal it interpolates (`"hello {{name}}"` → `` `hello ${name}` ``); as an expression it is the value (`JSON.parse({{input}})`); inside a condition it is the *state* the model reads. A binding carries a noun for prose: `input` is "the message"; `const email = {{input}} as "the email"` renames; any other binding's noun is derived from its name (`serviceRequest` → "the service request").

**Prose predicates.** A condition of `if`, `else if` or `while` that contains a binding outside a string is a natural-language decision:

```
if (the {{input}} is a refund && !(the {{input}} is at least annoyed on Mood)) { … }
while (the {{draft}} sounds rude unless it is only firm; it quotes the customer) { draft = ask("kinder: {{draft}}") }
```

Split at `&&`, `||`, `!` and parentheses into atoms; every atom with a binding is asked in **one request**; atoms combine under strong Kleene logic. An atom resolves, first match wins, as:

1. a **kind test** — `{{x}} is a refund` where `refund` is an option of exactly one declared kind (or `… on Ticket`): reads the one `choice` question for that (state, kind);
2. a **scale test** — `{{x}} is at least annoyed on Mood` (`at least` / `at most` / `above` / `below` compare the expected level; `is annoyed on Mood` compares the pick);
3. a **noul** — anything else: the atom text with the noun in place is the proposition; `unless c1; c2` lists what it can be mistaken for. Without `unless`, a generic contrary is supplied and `W-GENERIC-CONTRARY` says so.

**`unsure`.** The value of a decision the model abstained on, and a third branch:

```
if (the {{input}} is a refund) { refund() } else { file() } unsure { escalate() }
switch (decide({{input}}, Ticket)) { case Ticket.refund: …; case unsure: log("could not tell"); break; default: … }
```

Without an `unsure` block, unsure takes the `else` path: abstain is never a label, it is "do not act".

## Declarations and verbs

```ts
kind(instructions, { id: "what it means", … }, { fallback?: id, name?: string })   // a choice
scale(instructions, ["low", "high"] | { low: "…", high: "…" }, { name? })           // an ordered score, 2–10 levels
shape({ slot: Kind | Scale | text("…") | Shape }, { name? })                        // what extract fills
text(instructions)                                                                  // a free-text slot (needs a chat model)
settle({ floor?, gap?, confidence?, passes?, tokens?, ms? })                        // thresholds and the budget

decide(state, Kind | Scale | "proposition", { not?, floor?, model?, … })  → Decision
decideAll(state, [specs], opts)                                             → Decision[]   (one request, ≤16 questions)
classify({ state, questions, thresholds? })                                → the wire's answer, unwrapped
extract(state, Shape, { using?: "decide" | "ask" })                         → { slot: id | unsure | string }
ask(prompt, { as?: Shape, system?, maxTokens?, temperature?, model? })      → string | record
log(...parts) · parallel(items, fn, { limit?, tolerate? }) · race(...) · timeout(ms, p) · retry(fn, { times?, on? }) · using(model, fn) · named(text, noun)
```

Model verbs never need `await` inside a `.sand` file; your own async functions do. A `Decision` carries `value` (`true`/`false`, an id, or `unsure`), `top` (the model's honest pick even when it abstained), `p`, `distribution`, `decision` (`act` | `abstain`), `reason` (in the wire's own words), `confidence`, `margin`, `expected_level` for scales, `cut` (the option text was truncated on the wire), and `request` (which request answered it, and whether from memory).

## Two model classes, one program

Which model answers is configuration, not syntax.

| construct | decision model (the Combs classify wire) | chat model (OpenAI Chat Completions) |
|---|---|---|
| prose predicate | one `noul` with contraries | one completion, `response_format` json_schema `{answer: true \| false \| unsure}`; `p` from `logprobs` when the provider returns them |
| kind / scale test | one `choice` / `score` | an enum over the ids plus `"unsure"` |
| `extract` | one question per kinded slot; a `text` slot needs a chat model | one completion with the shape as a JSON Schema |
| `ask` | refused by name | a completion |
| cost | one pass per request of up to 16 questions | tokens |

### `sand.json`

```jsonc
{
  "models": {
    "combs":  { "kind": "decision", "url": "$COMBS_API_URL", "key": "$COMBS_API_KEY", "road": "seat" },
    "openai": { "kind": "chat", "url": "$OPENAI_BASE_URL", "key": "$OPENAI_API_KEY", "model": "$OPENAI_MODEL", "logprobs": "auto" },
    "local":  { "kind": "chat", "url": "http://127.0.0.1:8080/v1", "model": "default" }
  },
  "default": { "decide": "combs", "ask": "openai" },
  "settle":  { "floor": 0.55 },
  "limits":  { "passes": 64, "tokens": 20000, "ms": 60000, "concurrency": 4 }
}
```

Layers, lowest first: defaults → `~/.config/combs/sand.json` → the nearest `sand.json` upward from the program → `$SAND_CONFIG` → `configure({...})` in code → what a call names. `$NAME` reads the environment; an unset name makes that model unavailable and `E-MODEL` names the variable the moment it is used. **Zero config**: `COMBS_API_KEY` alone yields the decision model `combs` (`COMBS_API_URL` defaults to `https://api.combs.network`); `OPENAI_API_KEY` alone the chat model `openai` (`OPENAI_BASE_URL`, `OPENAI_MODEL`). Roads: `seat`, `id-means`, `label-means`, `label-anchors` cost one pass per request; `modernbert` and `deberta` cost one pass per option and cannot answer a prose predicate.

Another wire is one function: `registerKind("mine", (model, ports) => adapter)`, where the adapter answers in the decision wire's vocabulary (`answers[{name, top, p_top, distribution, decision, reason}]`).

## Cost, batching, budget

- Questions on one state are asked together: the compiler hoists consecutive same-binding conditions into one request, and the runtime coalesces decisions issued in the same tick on the same (model, state), splitting at 16. `Promise.all` and `parallel` batch for free.
- Questions nested below a decision on the same binding ride along as *speculation* on a model priced per request, and the later call is answered from memory. On per-option or per-token models they are asked when reached. This never changes what a program does.
- A question asked twice on the same state is answered from memory.
- `settle({ passes, tokens, ms })` is the budget: a call that would cross it rejects with `E-BUDGET` and the trace shows what was spent. A program can lower `sand.json`'s `limits`, never raise them.
- `sand check` prints `cost  min … · max …`: `min` is the cheapest path; `max` sums every site outside loops (an upper bound) and reports loops as `n ×`; a `while` that asks a model without `settle({ passes })` is reported unbounded.

## Concurrency

A program instance is one JavaScript thread: `await` is the only place another task runs, so there are no data races. I/O concurrency is `Promise.all` and `parallel(items, fn, { limit })`; the batcher keeps each state at one request. CPU parallelism is `sand run --workers N`: N isolated children, each with its own budget and trace, sharing nothing but the list of inputs. `timeout` and `race` never cancel the loser; pass `signal` for real cancellation. `retry` backs off and never retries `E-BUDGET`.

## The command line

```
sand check <file.sand> [--strict]
sand build <file.sand> [-o dir] [--js]
sand run   <file.sand> (--input "…" | --stdin | --jsonl in.jsonl) [--workers N] [--trace out.ndjson] [--model name] [--road seat] [--timeout ms] [--allow <deno flag>] [--quiet]
sand log   <trace.ndjson>
```

`sand run` spawns `deno run --no-prompt --allow-net=<the model hosts> --allow-env=<the variables the config reads>` and nothing else; keys travel in the environment, never on the command line. The child speaks JSON lines: one output per input on stdout, the trace on stderr. Exit codes: 0 ok · 1 compile or type error · 2 usage · 3 budget stop or a failed input · 4 refused by the API · 5 timeout.

## The trace

Every event is `{id, ts, source: "sand", kind, name, traceId, attrs, input?, output?}` with at most eight attrs of at most 120 characters, the shape `@combs/observe` and the platform's timeline rail accept. Names: `sand.run` (span), `sand.request` (`status, model, road, questions, passes, tokens, balance, ms`), `sand.decision` (`at, how, reads, top, p_top, decision, reason, memo`), `sand.effect`, `sand.budget`, `sand.log`.

## Errors

Every refusal reads `CODE line:col · what the program carries, then the fix`. Compile time: `E-BINDING` (two bindings in one predicate), `E-OPTION` (an id of no kind, or of two), `E-UNLESS` (contraries on a kind or scale test), `E-UNSURE` (an `unsure` after a plain condition), `E-SHAPE` (`extract` with no shape, or several), `E-ASYNC` (a predicate in a function that is not async), `E-BLOCK` (a prose `if` without braces), `E-PROSE-PLACE` (a predicate outside a condition), `E-RESERVED` (`export`, or a SAND name redeclared), `E-PARSE`. Run time: `E-MODEL`, `E-STATE`, `E-WIRE` (the provider's message, verbatim), `E-AUTH`, `E-PASSES`, `E-SUSPENDED`, `E-HOST`, `E-ANSWER-SHAPE`, `E-BUDGET`, `E-TIMEOUT`. Warnings never stop a run: `W-GENERIC-CONTRARY`, `W-UNSETTLED`, `W-UNBOUNDED-LOOP`, `W-NEAR-OPTION`, `W-BATCH-SPLIT`, `W-SYNC-VERB`, `W-ROAD`, `W-UNCALIBRATED`, `W-CUT`.

## Compile target

`.sand` → `.ts` → `.js`. The transform is a token-level rewriter, never a parser: output stays line-for-line with input after a two-line prologue, so TypeScript's own errors point at your lines. Beside the `.ts` it writes a `.sand.json` plan: declarations, every predicate with its line, the batches, the cost range, the warnings. Deno runs the `.ts` as is; `sand build --js` strips types; in the browser the same transform runs with `importStyle: "global"`.

## In a page or a host

```js
import { transform, run, configure } from "@combs-edge/combs-sand";
configure({ models: { combs: { kind: "decision", url: location.origin, road: "seat", key: async () => token } }, default: { decide: "combs", ask: "none" } });
const { code } = transform(source, { importStyle: "global" });
const { result, trace } = await run(main, { input, fetchImpl, bus });
```

Zero dependencies. Runs in browsers, Deno and Cloudflare Workers. Published as `@combs/sand` (jsr) and `@combs-edge/combs-sand` (npm); the examples in `examples/` are the learning ladder, from the four lines to `parallel` under a budget.
