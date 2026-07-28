# combs-engine (PyPI)

Installs the prebuilt [`combs`](https://github.com/asqrzk/CombsEngine) CLI
binary for your platform (macOS arm64/x86_64, Linux x86_64, Windows x86_64).

```bash
pip install combs-engine
combs devices
combs run --model <path-or-gguf> --prompt "Hello" --chat
combs chew chat-ui my-app --yes     # scaffold + install + launch a chat UI
combs serve --model <path> --port 8080
```

On first run the binary is fetched from the matching GitHub Release into
`~/.cache/combs/bin` (override with `COMBS_HOME`). CI-built platform wheels
bundle the binary directly, so no download is needed for those.

| Env var | Effect |
|---|---|
| `COMBS_BIN` | Use an existing `combs` binary |
| `COMBS_HOME` | Override the cache directory (default `~/.cache/combs`) |
| `COMBS_RELEASE_REPO` | Override the GitHub repo releases are fetched from |
