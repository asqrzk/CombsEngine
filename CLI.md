# The `combs` command line

Combs Engine is one binary. It loads a model onto your GPU and serves it —
text, images, speech — with no Python, no runtime download, and no service
in the background unless you start one.

This page is the reference for the command line: every subcommand, what it
does, the flags it takes and their defaults, what it prints, how it refuses,
and the environment doors you can turn. For installation and prerequisites,
see [INSTALL.md](INSTALL.md); for what the engine is and which model formats
it reads, see [README.md](README.md).

Everything here is read from the shipped binary's own definitions, and every
flag and default on this page was checked against `combs <command> --help` as
the binary actually renders it. Where a number matters (a default, a cap, a
limit), it is the number the code uses.

---

## Quick start

```sh
combs pull qwen3-0.6b-gguf                 # download a model into the cache
combs run --model ~/.cache/combs/models/qwen3-0.6b-gguf/model.gguf \
          --prompt "Explain attention in two sentences."
combs serve --model ~/.cache/combs/models/qwen3-0.6b-gguf/model.gguf
```

`serve` then answers OpenAI-compatible requests on `http://127.0.0.1:8080`.

Two commands worth running before anything else when a machine is new:

```sh
combs devices     # what GPU the engine can see
combs doctor      # whether it can be trusted with real work
```

---

## The commands

Fifteen subcommands (thirteen on a CPU-only build, where `generate-image`
and `serve-images` are compiled out).

### Running a model

#### `combs run` — one prompt, one answer

| flag | type | default |
|---|---|---|
| `--model` | path | required |
| `--prompt` | string | required |
| `--max-tokens` | number | the model's own `generation_config.json`, else 64 |
| `--temperature`, `--top-k`, `--top-p` | numbers | the model's own defaults |
| `--repetition-penalty`, `--frequency-penalty`, `--presence-penalty` | numbers | the model's own defaults |
| `--stop` | string, repeatable | none |
| `--seed` | number | random |
| `--chat` | flag | off — wraps the prompt in the model's chat template |
| `--image` | path | none — needs a vision model |
| `--prefill-chunk-size` | number | 512 |

Generated text streams to stdout. A statistics line goes to **stderr**, so
`combs run … > answer.txt` keeps the answer clean and still shows you the
speed: prompt tokens, prefill tok/s, generated tokens, time to first token,
decode tok/s, total seconds.

```sh
combs run --model models/qwen3-4b-gguf/model.gguf \
          --prompt "Name three uses for a paperclip." --chat --max-tokens 128
```

`--image` against a model whose `config.json` has no `vision_config` is
refused by name rather than ignored.

#### `combs serve` — an OpenAI-compatible server

| flag | type | default |
|---|---|---|
| `--model` | path — a directory (safetensors layout) or a `.gguf` file | required |
| `--port` | number | 8080 |
| `--context-size` | number | the model's maximum, capped (see below) |
| `--page-size` | number | 16 |
| `--prefill-chunk-size` | number | 512 (`0` = prefill the whole prompt in one call) |
| `--kv-auto` | flag | off — size the KV arena from free memory |
| `--lora`, `--lora-scale` | path or cached adapter id, number | none, 1.0 |
| `--anonymous-sessions` | `keep` \| `off` | `keep` |

The KV arena defaults to the smaller of the model's positional maximum and
**32,768 tokens** (65,536 on an f16 build). `--kv-auto` sizes it instead from
what the machine actually has free, minus the weight file, minus recurrent
state, minus 2 GiB of headroom, clamped to 4,096–131,072 tokens.

`--kv-auto` is **ignored when `--context-size` names a number** — an explicit
size is the operator's word. On a platform with no memory probe it falls back
to the default.

`--page-size` must be between 1 and the arena size, or the command refuses
and says so.

**`--anonymous-sessions`** decides what a request that names no `session_id`
gets. `keep` (the default) gives every such request *one shared rolling KV
session*: fast for a single user, but one caller can reuse another's prompt
prefix and learn from `cached_tokens` how much of it matched. `off` gives
each anonymous request a fresh cache that is never saved back. **Serving more
than one person: use `off`, or name a `session_id` per conversation.**

Ctrl-C (or `SIGTERM`) closes the listener and drains the engine before the
process exits, so a generation in flight finishes its GPU work rather than
being cut mid-submit.

##### Routes

| method | path | what it does |
|---|---|---|
| GET | `/health` | liveness — answers `{"status": "ok"}` |
| GET | `/v1/models` | the one model this process serves |
| GET | `/v1/model/info` | capabilities: context, vision, tools, embeddings |
| GET | `/v1/stats` | totals, last-request timings, live KV sessions, GPU pool, build |
| GET | `/v1/sessions` | live KV sessions and their page counts |
| DELETE | `/v1/sessions` or `/v1/sessions/{id}` | drop one session or all, freeing their arenas |
| POST | `/v1/chat/completions` | chat, streaming or not, tools, JSON mode, logprobs, `n` |
| POST | `/v1/completions` | raw prompt / fill-in-the-middle |
| POST | `/v1/embeddings` | pooled embeddings, matryoshka `dimensions`, base64 |
| POST | `/v1/tokenize` | the token ids this model's own chat path would prefill |
| GET | `/v1/kv/events` | the KV pages this engine holds, as a sequenced log |

Notes that save debugging time:

- **Streaming** ends with `data: [DONE]`, and sends a `: keepalive` comment
  every 100 ms during decode gaps, so a client that disappears is noticed
  within about a second.
- **`n`** is capped at 1–4, and `n > 1` is refused together with `stream`.
- **Images** must arrive as base64 `data:` URLs; remote URLs are refused
  (fetch them client-side, deliberately — the server makes no outbound calls).
- **Tools** against a model whose chat template cannot express them are
  refused by name rather than silently dropped.
- Every response carries `Access-Control-Allow-Origin: *`.
- A `session_id` is truncated to 64 characters; an empty string names no
  session.
- The listener binds only **after** the model has loaded, so there is no
  half-ready state: until then nothing answers on the port at all.

##### `POST /v1/tokenize`

Takes either a chat body or a raw prompt, and answers **ids only** — it never
echoes the rendered text back:

```sh
curl -s localhost:8080/v1/tokenize -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}]}'
# {"count":9,"object":"tokenize","token_ids":[…]}   (count is model-dependent)
```

`{"prompt":"hello"}` tokenizes a raw string instead. An image turn is refused:
its token count depends on pixels this route never reads.

This exists so a cache-aware router can count a prompt's tokens *the way this
engine will*, rather than guessing with a different tokenizer.

##### `GET /v1/kv/events`

A sequenced log of the KV pages this engine currently holds, for routers that
send a conversation back to the worker that already has its prefix.

```sh
curl -s 'localhost:8080/v1/kv/events?since=0&wait_ms=500'
```

`since` (default 0) is the last sequence number you saw; `wait_ms` (default 0,
capped at 30,000) long-polls for the next event. Events are `stored`,
`removed` and `cleared`. **Page keys travel as decimal strings, not JSON
numbers** — they are 64-bit, and a reader whose numbers are doubles would
lose the low bits and mismatch every key.

The log is **off by default**. Without `COMBS_KV_EVENTS=1` the route answers
`503` with type `kv_events_off` and tells you which door to open. It costs
memory proportional to what your sessions hold, and it publishes token ids to
whoever can reach the port — so it is opt-in, and it belongs on a private
network.

#### `combs perplexity` — how well a model predicts a text

```sh
combs perplexity --model models/qwen3-0.6b-gguf/model.gguf --file sample.txt
```

`--file` or `--text`, exactly one (it says so if you pass both or neither);
`--chunk` (default 256) and `--max-tokens` (default 0 = every token that fits
the context) bound the work. Prints one line: perplexity over the scored
tokens, the mean negative log-likelihood in nats per token, and tok/s — the
instrument for "did this quantization hurt?".

### Knowing your machine

#### `combs devices`

Prints the GPU the engine will use: name, backend, device type, driver, and
the two limits that decide what fits — maximum storage binding and maximum
buffer size.

#### `combs doctor`

Four checks — `device`, `wgsl`, `batched`, `e2e` — each printed as
`name status detail (N ms)`, ending in a verdict.

| flag | default |
|---|---|
| `--json` | off |
| `--model` | none — a preset id or a path; without it the end-to-end check reports `skip`, never `pass` |
| `--skip-e2e` | off |
| `--e2e-tokens` | 16 |

**Exits 1 if any check fails**, so it works as a gate in a script. It refuses
software rasterizers (llvmpipe, swiftshader, lavapipe, softpipe, warp) by
name: a CPU rasterizer pretending to be a GPU produces plausible garbage very
slowly, which is worse than an honest failure.

#### `combs inspect` — read a GGUF header

```sh
combs inspect --model models/qwen3-4b-gguf/model.gguf
```

`--model` takes a `.gguf` path **or a preset id already in the cache**.

One line: architecture, context length, vocabulary, tensor count, whether a
chat template is present and whether it supports tools and thinking. `--json`
gives the full header including the per-quantization tensor census. GGUF only;
anything else is refused by name.

#### `combs build-info`

Which binary you are actually running: version, dtype (f16/f32), profile,
commit, whether the tree was dirty, build time, target triple. `--json` for
the full manifest. Worth capturing in any bug report, and worth checking when
two machines disagree about speed.

### Getting models

#### `combs pull`

```sh
combs pull qwen3-0.6b-gguf          # a preset
combs pull Qwen/Qwen3-8B-GGUF       # any Hugging Face repo
```

The argument is **positional**, not a flag. There are 18 presets: SmolLM2,
Qwen3 (and Qwen3.5), Phi-3.1, Gemma-3 and SmolVLM for text and vision,
Stable Diffusion 1.5 and FLUX.2-klein for images, Kokoro for speech, and
Whisper for transcription. Models
land in `$COMBS_HOME/models`, defaulting to `~/.cache/combs/models`, and the
command prints `cached at: <dir>`.

Gated repositories read `HF_TOKEN` (or `HUGGINGFACE_TOKEN`).

Repositories that publish only *split* GGUF shards (`…-00001-of-0000N`) are
refused with a named message — multi-file GGUF loading is not supported, and
a half-downloaded model failing later is worse than a refusal now.

#### `combs convert`

Not implemented. It prints that and exits 2. (Exit code 2 is also what
`clap` returns for any command-line error, on every subcommand — see the
exit-code table.)

### Images

Both image commands are compiled out of a CPU-only build.

#### `combs generate-image`

| flag | default |
|---|---|
| `--model` | required |
| `--prompt` | required |
| `--negative-prompt` | none |
| `--width`, `--height` | 512, 512 |
| `--steps` | 20 |
| `--guidance-scale` | 7.5 |
| `--seed` | random |
| `--scheduler` (`ddpm` \| `ddim` \| `dpm++2m`) | `dpm++2m` |
| `--output` | `output.png` |
| `--lora`, `--lora-scale` | none, 1.0 |
| `--llm`, `--vae` | none — a pair, for recipe-style checkouts |

Prints `saved <path> (<sampler>, seed <n>)`, and reports **the sampler that
actually ran**: pipelines with a fixed schedule ignore `--scheduler`, and the
line tells you so rather than echoing your request back.

Seeded noise is generated on the host, so the same seed gives byte-identical
images across GPUs and backends.

#### `combs serve-images`

| flag | default |
|---|---|
| `--model` | required |
| `--port` | 8082 |
| `--lora`, `--lora-scale` | none, 1.0 |
| `--preview-every` | 0 (off) |
| `--llm`, `--vae` | none — a pair |

Routes: `GET /health`, `/v1/model/info`, `/v1/stats`, `/v1/preview`, and
`POST /v1/images/generations`.

`/v1/stats` reports a live generation: step, total, phase (encode → denoise →
decode → png), a measured ETA, and elapsed time. With `--preview-every N`,
`/v1/preview` serves the emerging image as PNG with an
`X-Combs-Preview-Step` header.

Requests are validated **before** the pipeline is taken, so a bad request
never queues behind a running render: width and height must be multiples of 8
between 64 and 1024, steps 1–1000, guidance finite and 0–50.

### Speech

#### `combs generate-audio`

`--model`, `--text`, `--voice` (default `af_heart`), `--speed` (1.0),
`--lang`, `--output` (default `output.wav`), and `--list-voices` to print the
voices in the model directory and exit. Requires **espeak-ng** on the system
and says how to install it when missing.

#### `combs serve-audio`

`--model`, `--port` (8083), `--transcribe-model`, `--language` (`en`).

Routes: `GET /health`, `/v1/model/info`, `/v1/stats`, `/v1/audio/voices`,
`POST /v1/audio/speech`, `POST /v1/audio/transcriptions`.

Transcription is only mounted when you pass `--transcribe-model`; without it
the route answers `501 not_configured` and names the flag. `speed` is clamped
to 0.25–4.0 rather than refused. Bodies are capped: 1 MB for speech, 64 MB
for transcription (about 35 minutes of 16 kHz mono).

#### `combs transcribe`

```sh
combs transcribe --model whisper-base --file speech.wav
```

`--model` takes a checkpoint directory or a preset id; `--file` is the WAV
and is required; `--language` is the Whisper language token (default `en`).
16-bit PCM WAV in, transcript to stdout, timing to stderr.

### `combs chew` — scaffold a local app

Seven modes: `chat-ui`, `debate-ui`, `roleplay-ui`, `multi-turn-ui`,
`orchestration-observe-ui`, `kv-cache-ui`, `debate-kv-ui`. Flags include
`--dir`, `--model`, `--server` (default `http://localhost:8080`), `--theme`,
`--agents`, `--topic`, `--reasoning`, `--vision`, `--audio`, `--save-chats`
and `--yes`.

All options are flags — there is no positional directory argument; the target
directory is `--dir`. `--yes` requires `--model`, so a scaffolded app always
names the model it expects. Authentication is on in every scaffolded app —
there is no flag to turn it off; edit `combs.ui.json` afterwards.

---

## Exit codes

| code | meaning |
|---|---|
| 0 | success |
| 1 | any error (printed to stderr as `Error: …`), including a failing `doctor` |
| 2 | a command-line error (`clap`: unknown flag, missing required argument, unknown subcommand) on any command — and `combs convert`, which is not implemented |

---

## When the engine refuses

Four refusals are worth recognizing, because each names its own arithmetic and
each has a specific way out.

**The device cannot hold one tensor.** Names the device, its allocation cap,
the offending tensor and what it needed. Use a smaller or more quantized
model, or a machine with a larger binding limit.

**Not enough memory to serve this model.** Prints weights, KV arena and
headroom against what is free. Ways out: `--context-size` (a smaller arena),
`COMBS_KV_QUANT=1` (int8 KV), a smaller quantization, or `--kv-auto` to size
the arena from what the machine actually has.

**Not enough free memory for this canvas** (`HTTP 507`, images). Prints the
estimated peak and what is free. Ask for a smaller image, or free memory.
This gate exists because an oversized render can take a desktop down, and a
refusal is better than a freeze.

**Software rasterizer refused** (`doctor`). Your Vulkan/Metal stack is
answering with a CPU renderer. Install the real GPU driver — output through a
rasterizer is slow *and* wrong.

---

## Configuration: the environment doors

The engine is configured by flags for everything a user chooses per run, and
by environment variables for everything that belongs to a machine. The
complete table (69 entries, with what each one changes and where it is read)
is printed by the measurement kit if you build from source:

```sh
cd engine/core                      # the cargo workspace lives here
cargo build --release -p combs-kit
./target/release/combs-kit doors
```

The ones most worth knowing:

### Memory and the KV cache

| door | values | what it does |
|---|---|---|
| `COMBS_KV` | `paged` (default), `contiguous` | `contiguous` turns off prefix reuse and speculative decode |
| `COMBS_KV_QUANT` | `1` | int8 KV on global layers — roughly 3.5× less KV memory on an f32 build (sliding-window layers stay in float). Refuses out loud on a model whose head dimension is not a multiple of 32 |
| `COMBS_KV_SCRUB` | `0` to turn off | zero every KV buffer before it returns to the pool, so a freed session's tokens cannot be read by the next allocation. On by default |
| `COMBS_KV_IDLE_SWEEP_S` | seconds | drop KV sessions nobody has touched; off by default |
| `COMBS_VRAM_BUDGET_MB` | number | refuse to mount a model whose weights + KV + slack exceed this |
| `COMBS_LLM_PREFLIGHT` | `0` | open the host-memory gate (`_FREE_MB`, `_HEADROOM_MB` override its inputs) |
| `COMBS_IMAGE_PREFLIGHT` | `0` | close the image memory check (`_FREE_MB`, `_HEADROOM_MB` likewise) |

### Speed

| door | values | what it does |
|---|---|---|
| `COMBS_SPEC` | `1` | speculative decoding, greedy requests only |
| `COMBS_ATTN` | `manual` | use the unfused attention path instead of flash |
| `COMBS_QMATMUL_OPERAND` | `f16` | f16 operands into the batched prefill matmul (f32 accumulate). Measurably faster prefill on several models; off by default |
| `COMBS_MEM_POOLS` | `subslices` | restore sub-slice memory pools |
| `CUBECL_WGPU_MAX_TASKS` | number | dispatches batched per submit. Read by the GPU layer; your value wins over the engine's default |

### Observability

| door | values | what it does |
|---|---|---|
| `COMBS_PROGRESS` | `json` | one `@combs {…}` line per load and pull event on stderr — the only channel open while a server is still loading |
| `COMBS_PROVENANCE` | `off`, `startup` (default), `turn`, `debug` | how much the engine says about what it did |
| `COMBS_PROVENANCE_FORMAT` | `json` | one object per line |
| `COMBS_KV_EVENTS` | `1` | publish the KV page log on `/v1/kv/events` |

### Paths

| door | what it does |
|---|---|
| `COMBS_HOME` | the model cache root (default `~/.cache/combs`) |
| `HF_TOKEN` / `HUGGINGFACE_TOKEN` | authenticate `combs pull` against gated repositories |
| `COMBS_UI_TEMPLATE` | serve a `chew` UI template from disk instead of the embedded one |

---

## Beyond the CLI: the measurement kit

Built from source, `combs-kit` is a second binary that measures rather than
serves. It is how every performance claim about this engine is produced, and
it is worth having when you are deciding what your machine can hold:

```sh
cd engine/core                                    # the cargo workspace lives here
cargo build --release -p combs-kit
./target/release/combs-kit system                 # what this machine is
./target/release/combs-kit model <model.gguf>     # tok/s, TTFT, memory, at two depths
./target/release/combs-kit doors                  # every door, its default, what it changes
./target/release/combs-kit ops                    # what works on this platform, and why not
./target/release/combs-kit diff a.json b.json     # did a change make it slower?
```

Two habits it enforces, because both have produced wrong numbers before: it
**refuses to run when it is older than the engine source it measures** (build
it again, or pass `--allow-stale` on purpose), and every figure it reports
carries how it was obtained — measured, or derived from measured values with
the arithmetic named.

It is not included in the release archives today; build it from source.

---

## Getting help

`combs --help` and `combs <command> --help` print the flags this binary
actually has, which is the authority if this page and your binary ever
disagree. Include `combs build-info` output in any report — the single most
useful line for telling two builds apart.
