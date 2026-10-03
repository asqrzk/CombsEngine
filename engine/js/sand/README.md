# @combs/sand

**Semantic abstraction of neurolinguistic determinism.** SAND is TypeScript plus three things the TypeScript checker cannot express: a *binding* that carries a noun, a *prose predicate* that is a model decision, and a third truth value, *unsure*. Everything else is TypeScript and stays TypeScript.

One line teaches the whole language: **decide** with the fast model, **say** with the chat model, **act** with tools you write, **remember** with a memory, and **ask a person** when nobody should guess.

## Start

```
const Urgency = kind("how urgent is this", { now: "it must happen today", soon: "this week is fine", whenever: "no rush is expressed" });
const Channel = kind("how do they want to be reached", { email: "they ask for a written reply", phone: "they ask for a call", any: "no preference is expressed" });
const ServiceRequest = shape({ urgency: Urgency, channel: Channel });

if (the {{input}} is a service request unless they are describing something that already happened)
{
extract({{input}}, ServiceRequest)
}
```

```sh
export COMBS_API_KEY=cmb_…          # from https://api.combs.network/account
deno install -g -A -n sand jsr:@combs/sand/cli
sand check service-request.sand      # what it asks, what it costs, what it needs, whether it type-checks
sand run   service-request.sand --input "My password reset never arrived; please email me today"
```

That program is **one pass**: the condition and the two questions the extraction needs travel in one request, and the extraction is answered from memory. The bare form runs too — `if (the {{input}} is a service request) { extract({{input}}) }` under the same three declarations is a whole program; the compiler supplies a generic contrary and says so with `W-GENERIC-CONTRARY`, and `extract` takes the one shape in scope.

A chat bot is two lines:

```
const bot = chat("You are Acme's support desk. Be brief.");
emit(bot.say({{input}}));
```

A program is a message handler. `input` is the message, the session is the conversation, and the host runs the program once per message and streams what it emits.

## The language

**Bindings `{{name}}`.** `{{ident}}` or `{{ident.path}}` names a variable in scope. Inside a string or template literal it interpolates (`"hello {{name}}"` → `` `hello ${name}` ``); as an expression it is the value (`JSON.parse({{input}})`); inside a condition it is the *text* the model reads. A binding carries a noun for prose: `input` is "the message"; `const email = {{input}} as "the email"` renames; any other binding's noun comes from its name (`serviceRequest` → "the service request").

**Prose predicates.** A condition of `if`, `else if` or `while` that contains a binding outside a string is a natural-language decision:

```
if (the {{input}} is a refund && !(the {{input}} is at least annoyed on Mood)) { … }
while (the {{draft}} sounds rude unless it is only firm; it quotes the customer) { draft = ask("kinder: {{draft}}") }
```

Split at `&&`, `||`, `!` and parentheses into atoms; every atom with a binding is asked in **one request**; atoms combine under strong Kleene logic. An atom resolves, first match wins, as:

1. a **pick** — `{{x}} is a refund` where `refund` is an option of exactly one declared `kind` (or `… on Ticket`): reads the one pick question for that (text, kind);
2. a **rating** — `{{x}} is at least annoyed on Mood` (`at least` / `at most` / `above` / `below` compare the expected level; `is annoyed on Mood` compares the pick);
3. a **yes/no question** — anything else: the atom's words with the noun in place are the proposition, and `unless c1; c2` lists what it can be mistaken for. Without `unless`, a generic contrary is supplied and `W-GENERIC-CONTRARY` says so.

**`unsure`.** The value of a decision the model declined to make, and a third branch:

```
if (the {{input}} is a refund) { refund() } else { file() } unsure { escalate() }
if (confirm("refund this order?")) { refund() } else { hold() } unsure { escalate() }
switch (decide({{input}}, Ticket)) { case Ticket.refund: …; case unsure: log("could not tell"); break; default: … }
```

Without an `unsure` block, unsure takes the `else` path: not sure is never a label, it is "do not act".

**Guardrails are predicates on what the model wrote.** A reply is text, so a yes/no question reads it like any other:

```
let reply = desk.say("Reply to this customer: {{input}}");
if (the {{reply}} promises a refund or a date unless it only says what happens next) { reply = desk.say("Rewrite it without promising a refund or a date."); }
```

## Verbs

```ts
kind(instructions, { id: "what it means", … }, { fallback?: id, name? })   // a pick
scale(instructions, ["low", "high"] | { low: "…", … }, { name? })          // an ordered rating, 2–10 levels
shape({ slot: Kind | Scale | text("…") | Shape }, { name? })               // what extract fills
text(instructions)                                                          // a free-text slot (needs a chat model)
settle({ floor?, gap?, confidence?, passes?, tokens?, ms?, turns?, wait? }) // confidence rules and the budget
```

### decide

```ts
decide(text, Kind | Scale | "proposition", { not?, floor?, model?, … })  → Decision
decideAll(text, [specs], opts)                                           → Decision[]   (one request, ≤16 questions)
classify({ state, questions, thresholds? })                              → the wire's answer, unwrapped
extract(text, Shape, { using?: "decide" | "ask" })                       → { slot: id | unsure | string }
```

A `Decision` carries `value` (`true`/`false`, an id, or `unsure`), `top` (the model's honest pick even when it declined), `p`, `distribution`, `decision` (`act` | `abstain`), `reason` in the wire's own words, `confidence`, `margin`, `expected_level` for ratings, `cut` (an option's words were truncated on the wire) and `request` (which request answered it, and whether from memory).

### ask and chat

```ts
ask(prompt, { as?: Shape, system?, maxTokens?, temperature?, model? })   → string | record
chat(system, { keep?, model?, name? })                                   → Conversation
assistant(system, { tools, settle?: { turns? }, keep?, model? })          → Conversation
```

```ts
const desk = chat("You answer support mail in three sentences.");
const reply = desk.say("{{input}}");                               // one turn; the history is kept
for await (const piece of desk.stream("and again?")) emit(piece);  // breaking the loop stops the model
desk.history; desk.last; desk.forget();
```

A **`Reply`** is an object, as a `Decision` is: `text`, `value` (the text, or `unsure` when the turns ran out, the answer was empty, or it was stopped), `stopped` (`null | "turns" | "length" | "empty" | "aborted"`), `turns`, `tokens`, `calls`, `reason`. `toString()` and `toJSON()` are the text, so `"…{{reply}}…"`, `emit(reply)`, `log(reply)` and a prose predicate on `{{reply}}` all read it — but `typeof reply === "string"` is **false**.

A **`Conversation`** keeps its history; `keep` (40 by default) bounds what goes on the wire, not what `history` holds, and a tool answer is never cut away from the turn that asked for it. `chat` never calls tools (`E-STATE` if given any).

### tools and agents

```ts
tool(description, Shape, fn, { name?, net? })   → Tool      // fn is yours and runs where the program runs
```

```ts
const fileTicket = tool("file a ticket", shape({ summary: text("one line") }), ({ summary }) => queue.push(summary));
const agent = assistant("You file tickets when asked.", { tools: [fileTicket], settle: { turns: 4 } });
const done = agent.say("the printer on floor two is jammed");   // the tool loop runs inside this say
done.calls;   // [{ id, name, args, result, error, ms }]
fileTicket.run({ summary: "…" });                               // or call it yourself
```

A tool sends the shape's JSON Schema and nothing else; the model can only ever ask for it by name with arguments. `assistant` needs at least one tool, runs them **one after another** (so a tool may ask a person), appends a `tool` message each, and repeats up to `settle({ turns })` — 8 by default. Past that bound the reply is `value: unsure, stopped: "turns"`. A tool that throws is fed back to the model as `{"error": "…"}` and the round continues; a budget stop, a timeout and an abort are the run's and stop it. `net: ["host"]` declares what the body reaches, and `sand run` builds the child's network permission from exactly that list plus the model hosts — nothing wider.

The shape must be one declared above the tool (or written inline), and an assistant's `tools` must be a list written out, so the plan and the fence can read both: anything else is `E-TOOL-SHAPE`.

### memory

```ts
memory(name, { model? })                        → Memory
notes.remember(text | Reply | record)           → id
notes.recall(query, { limit = 5 })              → [{ id, text, record, score, at }]
notes.forget(id | [ids] | "*")                  → how many went
```

A memory is a named store that outlives the run: the platform's (`POST {url}/v1/memory/{name}/remember|recall|forget`, metered like any other request), or a local JSON file per store under a `file:<dir>` url, recalled by idf-weighted lexical overlap. A local store needs the host's `writeFile` port; a page without one is told to use the platform store (`E-MEMORY`). `sand run --memory <dir>` gives a program a local store and permits exactly that directory.

### people, control and budget

```ts
confirm(question, { wait? })    → true | false | unsure
parallel(items, fn, { limit?, tolerate? }) · race(…) · timeout(ms, p) · retry(fn, { times?, on? })
using(model, fn) · named(text, noun) · log(…parts)
```

`confirm` asks a person and answers a verdict, so `and`/`or`/`not` compose and an `unsure` block catches "nobody was there". At a terminal `sand run` prompts `[y/n/?]`; with several workers or no terminal, `--yes`, `--no` or `--unsure` answers for everyone; in a page it is the `confirm` port; hosted, the run pauses and a resume answers it. Nobody, or a `wait` that elapses, is `unsure` and one `W-NOBODY` line. **The budget clock stops for the whole wait**, so a slow answer never spends the `ms` limit; what it took is `waited` in the trace.

`settle({ passes, tokens, ms, turns, wait })` is the budget: a call that would cross it rejects with `E-BUDGET` and the trace shows what was spent. A program can lower what `sand.json` allows, never raise it. One `say` is one **turn** (32 by default); `wait` bounds how long a `confirm` waits for a person.

### output

```ts
emit(piece)     // a piece of output for whoever listens; sync, and it never fails
```

In process it is the `output` port; under `sand run` it is a `sand.output` event printed as it arrives; on the platform it is server-sent events. Pieces are coalesced over 50 ms, so a token stream is about twenty events a second rather than one per token. The program's return value is still the run's result.

**Sessions.** `run(main, { session })` loads each conversation's history from `session.state.chats[name]` before its first `say` and hands the state back in `{ result, trace, session }`; the name is the `const` the conversation was assigned to; `{ session: false }` opts out. `sand run --session <id>` keeps one across runs in `.sand-sessions/<id>.json` beside the program.

## Two model classes, one program

Which model answers is configuration, not syntax.

| construct | decision model (the Combs classify wire) | chat model (OpenAI Chat Completions) |
|---|---|---|
| prose predicate | one yes/no question with its contraries | one completion, `response_format` json_schema `{answer: true \| false \| unsure}`; `p` from `logprobs` when the provider returns them |
| pick / rating test | one `choice` / `score` | an enum over the ids plus `"unsure"` |
| `extract` | one question per kinded slot; a `text` slot needs a chat model | one completion with the shape as a JSON Schema |
| `ask` · `chat` · `assistant` | refused by name | a completion; streaming with `stream_options.include_usage`, tools by `tool_calls` |
| `memory` | — | a memory store, its own kind |
| cost | one pass per request of up to 16 questions | tokens, and one **turn** per `say` |

### `sand.json`

```jsonc
{
  "models": {
    "combs":  { "kind": "decision", "url": "$COMBS_API_URL", "key": "$COMBS_API_KEY", "road": "seat" },
    "openai": { "kind": "chat", "url": "$OPENAI_BASE_URL", "key": "$OPENAI_API_KEY", "model": "$OPENAI_MODEL", "logprobs": "auto" },
    "local":  { "kind": "chat", "url": "http://127.0.0.1:8080/v1", "model": "default" },
    "notes":  { "kind": "memory", "url": "file:./.sand-memory" }
  },
  "default": { "decide": "combs", "ask": "openai", "memory": "notes" },
  "settle":  { "floor": 0.55 },
  "limits":  { "passes": 64, "tokens": 20000, "ms": 60000, "turns": 32, "concurrency": 4 }
}
```

Layers, lowest first: defaults → `~/.config/combs/sand.json` → the nearest `sand.json` upward from the program → `$SAND_CONFIG` → `configure({…})` in code → what a call names. `$NAME` reads the environment; an unset name makes that model unavailable and `E-MODEL` names the variable the moment it is used. **Zero config**: `COMBS_API_KEY` alone yields the decision model `combs`, the chat model `combs-chat` and the memory store `combs-memory` (`COMBS_API_URL` defaults to `https://api.combs.network`); `OPENAI_API_KEY` alone the chat model `openai`, which keeps precedence for `ask`.

A decision model's `road` field says how it asks. Two of its values cost one pass **per option** and cannot answer a prose predicate at all; `sand check` sees that and says `W-ROAD` rather than letting the program cost a surprise. Another wire is one function: `registerKind("mine", (model, ports) => adapter)`, where the adapter answers in the decision wire's vocabulary (`answers[{name, top, p_top, distribution, decision, reason}]`).

## Running

```
sand check <file.sand> [--strict]
sand build <file.sand> [-o dir] [--js]
sand run   <file.sand> (--input "…" | --stdin | --jsonl in.jsonl) [--session id] [--workers N]
                       [--yes | --no | --unsure] [--memory dir] [--trace out.ndjson]
                       [--model name] [--road name] [--timeout ms] [--allow <deno flag>] [--quiet]
sand log   <trace.ndjson>
```

`sand check` prints every predicate with its line, the batches, the cost, the `needs` line (what the program asks of its host) and then type-checks the emitted TypeScript, mapping errors back to your lines. `--strict` makes a `W-` warning an exit 1.

**The fence.** `sand run` spawns `deno run --no-prompt` with exactly the permissions the program implies and nothing else: `--allow-net=` the model hosts plus every host a tool's `net` declares, `--allow-env=` the variables the config reads, and `--allow-read`/`--allow-write` on the memory directory when there is one. Keys travel in the environment, never on the command line. **If the fence cannot be applied — `deno` is missing, a flag is refused, or the child dies before it acknowledges its preamble — the run ends `E-FENCE` and exit 2, and it is never retried wider.**

Stdout carries one result per input; stderr carries the trace, the decision lines, and each `emit` piece as it arrives. At a terminal a `confirm` is asked there and the kill clock stops until the person answers.

Exit codes: 0 ok · 1 a compile or type error · 2 usage, no decision model, or `E-FENCE` · 3 a failed input or a budget stop, each written to stdout as `{error: {code, message, at}}`. The built program run on its own (`deno run <built>.ts "text"`) exits 3 on a budget stop, 4 when the API refused it (`E-AUTH`, `E-PASSES`, `E-SUSPENDED`), 5 on `E-TIMEOUT` and 1 on anything else.

**The child protocol (v1).** A built program under `SAND_CHILD=1` speaks JSON lines, one object per line, so any host can drive it:

| direction | line |
|---|---|
| in | `{sand:"1", config, limits, file?, memoryDir?, traceId?}` — the preamble, first and once |
| out | `{sand:"1", ready:true}` — it read the preamble and the fence held |
| in | `{input, session?, traceId?}` — one run (a bare JSON string is still an input) |
| in | `{confirm:{id, answer}}` — at any time: a person's answer |
| out | `{result, session?}` or `{error:{code,message,at}, session?}` — one envelope per input, in order |
| err | one trace event per line, including the `sand.confirm` ask that carries the `id` |

Stdin is read continuously, so inputs queue behind the run in flight while a control line resolves a pending `confirm` at once, and it stays open until the parent has every envelope.

**In a page or a host:**

```js
import { transform, run, configure } from "@combs-edge/combs-sand";
configure({ models: { combs: { kind: "decision", url: location.origin, road: "seat", key: async () => token } }, default: { decide: "combs", ask: "none" } });
const { code } = transform(source, { importStyle: "global" });
const { result, trace, session } = await run(main, { input, fetchImpl, bus, output, confirm, session });
```

## Cost

- Questions on one text are asked together: the compiler hoists consecutive same-binding conditions into one request, and the runtime coalesces decisions issued in the same tick on the same (model, text), splitting at 16. `Promise.all` and `parallel` batch for free.
- Questions nested below a decision on the same binding ride along as *asked in advance*, and the later call is answered from memory. On models priced per option or per token they are asked when reached. This never changes what a program does.
- A question asked twice on the same text is answered from memory.
- `sand check` prints the range, and the plan holds it as numbers:

```
cost  min 1 pass · max 2 passes, 4 turns · asks a person up to 1 time
cost  min 0 passes · max 0 passes + n × 1 pass (capped at 200 by settle)
```

`min` is the cheapest path — what an unconditional site costs. `max` sums every site outside loops, counts one turn per `say` on a `chat` and up to `settle({ turns })` on an `assistant`, counts each `remember`/`recall`/`forget` as a memory call, and reports loops as `n ×`. A `while` that asks a model with no `settle({ passes })` is reported unbounded. Nothing here is money: a pass, a turn and a memory call are units of work, and what they cost is your account's business.

## Diagnostics

Every refusal reads `CODE line:col · what the program carries, then the fix`.

**Compile time, refusals.** `E-PARSE` (an unterminated string, template or bracket) · `E-RESERVED` (`export`, or one of SAND's names redeclared — including `chat`, `assistant`, `tool`, `memory`, `confirm` and `emit`) · `E-BINDING` (two bindings in one predicate) · `E-OPTION` (an id of no kind, or of two) · `E-UNLESS` (contraries on a pick or rating test) · `E-UNSURE` (an `unsure` block after a condition that cannot be unsure) · `E-SHAPE` (`extract` with no shape, or several) · `E-TOOL-SHAPE` (a tool given something that is not a shape declared above it, or an assistant's `tools` that is not a list) · `E-ASYNC` (a predicate or a `confirm` in a function that is not async) · `E-BLOCK` (a prose `if` without braces) · `E-PROSE-PLACE` (a predicate outside a condition).

**Compile time, warnings** — they never stop a run, and `--strict` makes them an exit 1: `W-GENERIC-CONTRARY` (no `unless`, so the decision will be weak) · `W-UNSETTLED` (no `settle()`, so nothing can be unsure and every unsure branch is dead) · `W-UNBOUNDED-LOOP` (a `while` asks a model every turn with no `settle({ passes })`) · `W-UNBOUNDED-CHAT` (a `say`, a `stream` or an `ask` inside a loop with no `settle({ turns })`) · `W-NO-CONFIRM-BUDGET` (a `confirm` with no `settle({ wait })`, so an unattended run hangs instead of answering unsure) · `W-NEAR-OPTION` (a phrase that reads like a declared option but is asked as a bare predicate) · `W-BATCH-SPLIT` (more than 16 questions at one site) · `W-SYNC-VERB` (a model verb in a function that is not async, so its promise is left unawaited) · `W-ROAD` (from `sand check`: the configured decision model cannot answer this program's predicates).

**Compile time, notes** — printed, never counted against `--strict`: `N-EXTRACT-SHAPE` (`extract` named no shape and the one in scope was used) · `N-UNUSED` (a kind or scale nothing reads) · `N-TOOL-UNUSED` (a tool no assistant lists and nothing runs).

**Run time.** `E-MODEL` (no model, the wrong class, or an unset variable) · `E-STATE` · `E-SHAPE` · `E-OPTION` · `E-RESERVED` · `E-TOOL-SHAPE` · `E-WIRE` (the provider's message, verbatim) · `E-AUTH` · `E-PASSES` · `E-SUSPENDED` · `E-HOST` · `E-ANSWER-SHAPE` · `E-BUDGET` · `E-TIMEOUT` · `E-STREAM` (a stream carried an error frame or broke mid-answer) · `E-MEMORY` (a store refused, answered the wrong shape, or cannot be reached from this host) · `E-FENCE` (the fence a run asked for could not be built, so the run did not happen) · `E-RUNTIME` (an error of the program's own, carrying no code, as a child or a worker reports it) · `E-NO-MODEL` (`sand run` with no decision model configured).

**Said once as a `sand.log` line rather than a code:** `W-UNCALIBRATED` (a model returned no logprobs; `p` is null and the floor is not applied) · `W-UNMETERED` (a model reported no usage; tokens are estimated at one per four characters) · `W-NOBODY` (nobody was there to answer a `confirm`, or the wait elapsed).

**Said in a decision's own `reason`:** `W-PARSE` — a chat model answered a decision with JSON that named no option, so the decision is unsure and its `reason` carries what it said.

### The trace

Every event is `{id, ts, source: "sand", kind, name, traceId, attrs, input?, output?}` with at most eight attrs of at most 120 characters — the shape `@combs/observe` and the platform's timeline accept. Names: `sand.run` (span; its end carries `passes, tokens, turns, requests, ms, status`) · `sand.request` (`status, model, road, questions, passes, tokens, balance, ms`) · `sand.decision` (`at, how, reads, top, p_top, decision, reason, memo`) · `sand.effect` · `sand.budget` · `sand.log` · `sand.turn` (`at, chat, turn, model, stop, tokens, ms, calls`) · `sand.tool` (span; `at, chat, tool, id, status, ms`, the arguments in `input` and the result in `output`) · `sand.memory` (`at, memory, op, hits, ms, model`) · `sand.output` (`at, chars`, the pieces in `output`) · `sand.thinking` (`at, chars`, the model's reasoning in `output`, ahead of the answer's `sand.output`; a model that thinks first hands its reasoning over apart from its answer, and `Reply.thinking` holds it) · `sand.confirm` (span; `at, answer, waited, by`, and an `ask` line carrying the `id` a control line answers).

## Cookbook

`examples/` is the ladder, from the four lines to a whole system; `examples/index.json` carries each one's input, what to expect, and what it needs.

| # | file | what it shows | needs |
|---|---|---|---|
| 0 | `service-request.sand` | decide → extract from the same pass | decide |
| 1 | `support-triage.sand` | a kind, a rating, a `switch` with `case unsure`, one `settle()` | decide |
| 2 | `extract-shape.sand` | a shape with kinded and text slots across both model classes | decide · ask |
| 3 | `intent-router.sand` | a regex router rewritten: the mention-versus-ask hazard is one predicate | decide |
| 4 | `review-sweep.sand` | `parallel` over a list under a pass budget | decide |
| 5 | `support-desk.sand` | decide → extract → a chat reply → a guardrail on that reply → a person's yes → a tool files the ticket | decide · ask · chat · tools · confirm · emit |
| 6 | `order-assistant.sand` | an `assistant` picking its own tools; `confirm` inside the refund tool; a predicate that catches a claimed refund | decide · chat · tools · confirm · emit |
| 7 | `inbox-sweep.sand` | `parallel` over a JSON list, one pass each, `tolerate`, under `settle({ passes: 200 })` | decide |
| 8 | `doc-qa.sand` | a memory taught with "remember: …", recalled into a prompt, and an answer checked against its passages | decide · ask · memory · emit |
| 9 | `moderator.sand` | an `unsure` branch that asks a person; unattended, the post waits in a memory | decide · memory · confirm · emit |

## Concurrency

A program instance is one JavaScript thread: `await` is the only place another task runs, so there are no data races. I/O concurrency is `Promise.all` and `parallel(items, fn, { limit })`; the batcher keeps each text at one request. CPU parallelism is `sand run --workers N`: N isolated children, each with its own budget and trace, sharing nothing but the list of inputs — and no person is asked from them, so give them `--yes`, `--no` or `--unsure`. `timeout` and `race` never cancel the loser; pass `signal` for real cancellation. `retry` backs off and never retries `E-BUDGET`.

## Compile target

`.sand` → `.ts` → `.js`. The transform is a token-level rewriter, never a parser: every edit is a splice into the original text, so the output stays line-for-line with the input after a two-line prologue and TypeScript's own errors point at your lines. Beside the `.ts` it writes a `.sand.json` plan (`sand: "0.2"`): what the program uses, its declarations, its conversations, tools and memories, every predicate with its line, the batches, the hosts a fence must allow, the cost range, and the warnings. Deno runs the `.ts` as is; `sand build --js` strips types; in a browser the same transform runs with `importStyle: "global"`.

## Glossary

| word | what it means |
|---|---|
| **pass** | one run of the decision model on one text, up to 16 questions in the same request |
| **turn** | one call to the chat model — one `say`, or one round of an assistant's tool loop |
| **memory call** | one `remember`, `recall` or `forget` on a store |
| **yes/no question** | a question the model answers true, false, or not sure |
| **what it means** | the sentence the model compares your text against |
| **not to be confused with** | the sentences that look like a match but are not — what `unless` writes |
| **unsure** | the model declined to pick; your program decides what to do |
| **confidence rules** | `settle({ floor, gap, confidence })`: how sure the model must be before your code acts |
| **asked in advance** | questions bundled into a request so a later line costs nothing |
| **the fence** | the exact permissions a run is given, built from the program's plan and from nothing else |
| **the plan** | the `.sand.json` beside the built program: what it asks, what it needs, what it costs |

Zero dependencies. Runs in browsers, Deno and Cloudflare Workers. Published as `@combs/sand` (jsr) and `@combs-edge/combs-sand` (npm).
