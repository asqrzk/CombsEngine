/** Demo: RemoteEngine against `combs serve`. Run: deno run -A examples/remote.ts */
import { Combs } from "../core/mod.ts";
const engine = Combs.remote(Deno.args[0] ?? "http://localhost:8472");
const r = await engine.complete({
  messages: [{ role: "user", content: "Say hello" }],
  max_tokens: 10,
  temperature: 0,
});
console.log("remote:", JSON.stringify(r.text), `[${r.finishReason}]`);
engine.close();
