# Combs UI — Svelte 5 chat & debate apps

Sleek, responsive web UIs for the Combs Engine, scaffolded by the CLI:

```sh
# Interactive (Firebase-CLI style: SPACE to toggle, arrows, ENTER):
combs chew chat-ui
combs chew debate-ui

# Or fully flagged (scriptable) — `--yes` makes `--model` mandatory:
combs chew chat-ui --yes \
  --reasoning=true --vision=false --audio=false \
  --save-chats=true \
  --theme=dark --model smollm2-135m --server http://localhost:8080

combs chew debate-ui --yes --model smollm2-135m --agents "Ada,Grace" \
  --topic "Is Rust better than C++?" --turns 8

cd chat-ui        # or your --dir
npm install
npm run dev       # UI talks to `combs serve` on the configured server URL
```

## What's in the template

`template/` is the Vite + Svelte 5 + Tailwind v4 source the CLI copies and
configures (`combs.ui.json`):

- **First-run auth** (`authentication: true` in `combs.ui.json`): ECDSA P-256
  keypair generated on-device (WebCrypto), public fingerprint shown,
  private-key backup download ritual gated before the app unlocks. `combs
  chew` always scaffolds with authentication on — there is no flag to turn
  it off; edit `combs.ui.json` afterwards.
- **Fine-grained permissions**: before any network download, inference
  call, or local caching, a dialog asks **allow once / allow this session /
  allow always / deny**. "Always" and "deny" grants are persisted by the
  backend proxy to `server/permissions.json`; session grants live in the
  proxy's memory. The client (`src/lib/permissions.svelte.ts`) renders the
  dialog and retries — the proxy is what enforces.
- **Realtime network & storage monitor** in the top bar: live ↓/↑ byte
  counters (fed by the API layer).
- **Dark / light / system themes** (class-based, persisted, toggle in the
  top bar).
- **chat-ui**: streaming chat with any `combs serve` model, an optional
  reasoning view, saved sessions (permission-gated). The vision and audio
  toggles add buttons that currently answer "coming soon".
- **debate-ui**: multi-agent turn-taking debates — agent names, stances
  (pro/against), behaviors, topic and turn count, driven client-side over
  the same server.
- Responsive layout (mobile → desktop), shadcn-style minimal components
  (Button/Card/Badge/Dialog) — no heavy UI dependency.

## Architecture

The browser talks to `combs serve` (OpenAI-compatible HTTP/SSE) through a
small Node proxy the template ships in `server/` — `npm run dev` starts it
(`node server/dev.mjs`), and it owns permissions, storage and the agent
sandbox. `combs.ui.json` (written by `combs chew`) selects the mode,
features, theme, model and server; every value can also be edited directly.
