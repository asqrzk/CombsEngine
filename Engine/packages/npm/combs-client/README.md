# combs-client

Official JavaScript client for [Combs Engine](https://github.com/asqrzk/CombsEngine)
servers (`combs serve`, OpenAI-compatible HTTP/SSE). Zero dependencies;
runs in browsers, Node 18+, Deno and Bun.

```bash
npm install combs-client
```

```js
import { CombsClient } from "combs-client";

const client = new CombsClient({ baseUrl: "http://localhost:8080" });

// streaming
await client.streamChatCompletion(
  { messages: [{ role: "user", content: "Hello" }] },
  { onDelta: (t) => process.stdout.write(t), onDone: () => console.log() },
);

// non-streaming
const reply = await client.chatCompletion({
  messages: [{ role: "user", content: "Hello" }],
});

// discovery
console.log(await client.listModels(), await client.health());
```

All I/O goes through an injectable `fetchImpl`, so apps can add auth
headers, permission gates or mocks; `onDownload(bytes)` reports received
bytes for bandwidth monitors. `baseUrl` is a required constructor option —
the library contains no hardcoded server address; the host application
always decides where to connect.

The Combs UI template (`combs chew chat-ui`) depends on this package and
routes its `fetchImpl` through a backend permission proxy, so every byte
the app sends or receives is permission-checked server-side.
