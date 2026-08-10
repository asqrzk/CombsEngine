# combs-engine (npm)

Installs the prebuilt [`combs`](https://github.com/asqrzk/CombsEngine) CLI
binary for your platform (macOS arm64/x86_64, Linux x86_64, Windows x86_64).

```bash
npm install -g combs-engine
combs devices
combs run --model <path-or-gguf> --prompt "Hello" --chat
combs chew chat-ui my-app --yes     # scaffold + install + launch a chat UI
combs serve --model <path> --port 8080
```

The binary is downloaded from the matching GitHub Release at install time.
Useful env vars:

| Var | Effect |
|---|---|
| `COMBS_BIN` | Use an existing `combs` binary instead of the downloaded one |
| `COMBS_SKIP_BINARY_DOWNLOAD=1` | Skip the postinstall download |
| `COMBS_RELEASE_REPO` | Override the GitHub repo releases are fetched from |

For the TypeScript agent framework (`@combs/core`, `@combs/graph`, ...), see
the JSR packages: https://jsr.io/@combs
