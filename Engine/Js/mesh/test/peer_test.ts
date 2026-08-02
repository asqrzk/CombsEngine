/**
 * MeshPeer test: real HTTP server on a free port, two registered emojis,
 * client fetches by name and hash with integrity verification, plus
 * 401/404/tamper paths. FFI-gated like the other integration tests.
 */
import { assert, assertEquals, assertRejects, assertStringIncludes } from "jsr:@std/assert";
import { Mesh, MeshPeer } from "../mod.ts";

const IGNORE = Deno.env.get("COMBS_SKIP_INTEGRATION") === "1";

Deno.test({
  name: "peer: serve + fetch round-trip with integrity, auth, 404, tamper",
  ignore: IGNORE,
  sanitizeOps: false, // the WS probe below leaves an op in flight at assertion time
  async fn() {
    const home = await Deno.makeTempDir({ prefix: "combs-mesh-peer-" });
    Deno.env.set("COMBS_HOME", home);
    const mesh = Mesh.open();
    const peer = MeshPeer.serve({ mesh, token: "secret-token" });
    try {
      // Register two emojis.
      const a = mesh.buildEmoji({ name: "peer-a", description: "first" });
      const b = mesh.buildEmoji({ name: "peer-b", description: "second" });
      const hashA = mesh.registryRegister(a.binary);
      const hashB = mesh.registryRegister(b.binary);
      assert(hashA !== hashB);

      // Announce + list.
      const announce = await (await fetch(`${peer.baseUrl()}/mesh/v1/announce`, {
        headers: { authorization: "Bearer secret-token" },
      })).json();
      assertEquals(announce.count, 2);
      assertEquals(announce.peer_id.length, 64);
      const list = await (await fetch(`${peer.baseUrl()}/mesh/v1/list`, {
        headers: { authorization: "Bearer secret-token" },
      })).json();
      assertEquals(list.entries.length, 2);

      // Fetch by name and by hash; byte-equality with the local binary.
      const byName = await MeshPeer.fetch(peer.baseUrl(), "peer-a", { token: "secret-token" });
      assertEquals(byName.binary, a.binary);
      assertEquals(byName.emojiJson.name, "peer-a");
      assertEquals(byName.hash, hashA);
      const byHash = await MeshPeer.fetch(peer.baseUrl(), hashB, { token: "secret-token" });
      assertEquals(byHash.binary, b.binary);
      assertEquals(byHash.emojiJson.name, "peer-b");

      // Wrong token → 401.
      const e401 = await assertRejects(
        () => MeshPeer.fetch(peer.baseUrl(), "peer-a", { token: "wrong" }),
        Error,
        "HTTP 401",
      );
      assertStringIncludes(e401.message, "401");

      // Unknown emoji → 404.
      await assertRejects(
        () => MeshPeer.fetch(peer.baseUrl(), "no-such-emoji", { token: "secret-token" }),
        Error,
        "HTTP 404",
      );

      // Tamper path: a rogue server advertising hash A but serving flipped
      // bytes must fail the integrity check.
      const tampered = a.binary.slice();
      tampered[tampered.length - 1] ^= 0xff;
      const rogue = Deno.serve({ hostname: "127.0.0.1", port: 0 }, () =>
        new Response(JSON.stringify({
          emoji: a.emoji,
          hash: hashA,
          binary_b64: btoa(String.fromCharCode(...tampered)),
        })));
      const rogueAddr = rogue.addr as Deno.NetAddr;
      try {
        const err = await assertRejects(
          () => MeshPeer.fetch(`http://127.0.0.1:${rogueAddr.port}`, "peer-a"),
          Error,
          "integrity mismatch",
        );
        assertStringIncludes(err.message, "integrity mismatch");
      } finally {
        await rogue.shutdown();
      }

      // WS endpoint: announce on connect. (WebSocket can't set custom
      // headers, so probe with a token-less throwaway peer.)
      const openPeer = MeshPeer.serve({ mesh });
      try {
        const wsMsg = await new Promise<Record<string, unknown>>((resolve, reject) => {
          const ws = new WebSocket(`ws://127.0.0.1:${openPeer.port}/mesh/v1/ws`);
          const timer = setTimeout(() => reject(new Error("ws timeout")), 5000);
          ws.onmessage = (ev) => {
            clearTimeout(timer);
            resolve(JSON.parse(String(ev.data)));
            ws.close();
          };
          ws.onerror = () => {
            clearTimeout(timer);
            reject(new Error("ws error"));
          };
        });
        assertEquals(wsMsg.type, "announce");
        assertEquals(wsMsg.count, 2);
      } finally {
        await openPeer.close();
      }
    } finally {
      await peer.close();
      mesh.close();
      await Deno.remove(home, { recursive: true });
    }
  },
});
