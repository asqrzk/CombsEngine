# Combs UI — Svelte 5 chat & debate apps

Sleek, responsive web UIs for the Combs Engine, scaffolded by the CLI:

```sh
# Interactive (Firebase-CLI style: SPACE to toggle, arrows, ENTER):
combs chew-chat-ui
combs chew-debate-ui

# Or fully flagged (scriptable):
combs chew-chat-ui --yes \
  --reasoning=true --vision=false --audio=false \
  --save-chats=true --authentication=false \
  --theme=dark --model smollm2-135m --server http://localhost:8080

combs chew-debate-ui --yes --agents "Ada,Grace" \
  --topic "Is Rust better than C++?" --turns 8

cd chat-ui        # or your --dir
npm install
npm run dev       # UI talks to `combs serve` on the configured server URL
```

## What's in the template

`template/` is the Vite + Svelte 5 + Tailwind v4 source the CLI copies and
configures (`combs.ui.json`):

- **First-run auth** (`authentication: true`): ECDSA P-256 keypair generated
  on-device (WebCrypto), public fingerprint shown, private-key backup
  download ritual gated before the app unlocks. `--authentication=false`
  = incognito (no auth, no persistence).
- **Fine-grained permissions**: before any network download, inference
  call, or local caching, a dialog asks **allow once / allow this session /
  allow always / deny**. Grants persist in localStorage and are checked
  before every connection (`src/lib/permissions.ts` + guarded fetch).
- **Realtime network & storage monitor** in the top bar: live ↓/↑ byte
  counters (fed by the API layer) + `navigator.storage.estimate()` quota.
- **Dark / light / system themes** (class-based, persisted, toggle in the
  top bar).
- **chat-ui**: streaming chat with any `combs serve` model, optional
  reasoning/vision/audio panels, saved sessions (permission-gated).
- **debate-ui**: multi-agent turn-taking debates — agent names, stances
  (pro/against), behaviors, topic and turn count, driven client-side over
  the same server.
- Responsive layout (mobile → desktop), shadcn-style minimal components
  (Button/Card/Badge/Dialog) — no heavy UI dependency.

## Architecture

The UI is a pure client of `combs serve` (OpenAI-compatible HTTP/SSE) —
no server-side code. `combs.ui.json` (written by `combs chew`) selects the
mode, features, theme, model and server; every value can also be edited
directly.
