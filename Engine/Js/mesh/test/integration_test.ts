/**
 * Integration test: real libcombsmesh_ffi through Deno FFI.
 *
 * Requires the native library (cargo build --release -p combs-mesh-ffi).
 * Skip with COMBS_SKIP_INTEGRATION=1. The registry is isolated in a temp
 * COMBS_HOME.
 *
 * The parity test is the critical one: the pure-TS unicode codec must
 * produce byte-identical output to the Rust core (via the FFI toUnicode
 * op) for the same emoji.
 */
import { assert, assertEquals, assertStringIncludes } from "jsr:@std/assert";
import {
  type BlockTag,
  encodeBlocks,
  EmojiMcpServer,
  type EmojiJson,
  Mesh,
} from "../mod.ts";

const IGNORE = Deno.env.get("COMBS_SKIP_INTEGRATION") === "1";

// Isolate the registry before any Mesh call touches it.
const COMBS_HOME = await Deno.makeTempDir({ prefix: "combs-mesh-test-" });
Deno.env.set("COMBS_HOME", COMBS_HOME);

/** A fixed emoji definition used by the parity test (img atlas: 2x2 RGBA). */
const FIXED = {
  name: "parity-emoji",
  description: "unicode parity check",
  blocks: [
    {
      type: "tdo",
      items: [{ key: "t1", value: "check parity", status: "Pending", depends_on: [] }],
    },
    {
      type: "img",
      name: "",
      atlas: {
        width: 2,
        height: 2,
        frame_width: 2,
        frame_height: 2,
        frame_count: 1,
        rgba: [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 9, 9, 9, 255],
      },
    },
  ],
};

/** Splits an internally-tagged block ({type, ...fields}) into {tag, payload}. */
function splitBlocks(emoji: EmojiJson): { tag: BlockTag; payload: unknown }[] {
  return emoji.blocks.map((b) => {
    const { type, ...payload } = b as Record<string, unknown>;
    return { tag: type as BlockTag, payload };
  });
}

Deno.test({
  name: "mesh: init, crypto round-trip, build/register/render via ops",
  ignore: IGNORE,
  fn() {
    const mesh = Mesh.open();
    try {
      mesh.init(new TextEncoder().encode("deno integration master key!"));

      // Direct C symbols: encrypt/decrypt round-trip.
      const data = new TextEncoder().encode("hello mesh ffi");
      const ct = mesh.encrypt(data);
      assert(ct.length > data.length);
      assertEquals(mesh.decrypt(ct), data);

      // op_json: build → register → list/resolve → render.
      const built = mesh.buildEmoji(FIXED);
      assertEquals(built.emoji.name, "parity-emoji");
      const hash = mesh.registryRegister(built.binary);
      assertEquals(hash.length, 64);

      const entries = mesh.registryList();
      assert(entries.some((e) => e.name === "parity-emoji" && e.hash === hash));

      const resolved = mesh.registryResolve("parity-emoji");
      assertEquals(resolved.emoji.name, "parity-emoji");
      assertEquals(resolved.binary, built.binary);

      const rendered = mesh.renderSprite(resolved.binary, 0);
      assertEquals(rendered.width, 2);
      assertEquals(rendered.height, 2);
      assertEquals(rendered.rgba.length, 2 * 2 * 4);
      assertEquals(rendered.rgba[0], 255);
    } finally {
      mesh.close();
    }
  },
});

Deno.test({
  name: "parity: pure-TS unicode codec === Rust core (FFI) byte-for-byte",
  ignore: IGNORE,
  fn() {
    const mesh = Mesh.open();
    try {
      // Rust side: build the emoji and encode it via the FFI op.
      const built = mesh.buildEmoji(FIXED);

      // TS side: encode the SAME emoji JSON with the pure-TS codec.
      const tsUnicode = encodeBlocks(splitBlocks(built.emoji));

      // Byte-for-byte equality with Rust's output (built.unicode comes
      // from EmojiExporter::to_unicode inside the Rust core).
      assertEquals(tsUnicode, built.unicode);
      assertEquals(tsUnicode, mesh.toUnicode(built.emoji));

      // And the FFI decoder accepts the TS encoding.
      const decoded = mesh.fromUnicode(tsUnicode);
      assertEquals(decoded, built.emoji);
    } finally {
      mesh.close();
    }
  },
});

Deno.test({
  name: "mcp server: initialize/tools/resources round-trip over handler",
  ignore: IGNORE,
  fn() {
    const mesh = Mesh.open();
    try {
      const server = new EmojiMcpServer(mesh);

      // initialize
      const init = server.handle({ jsonrpc: "2.0", id: 1, method: "initialize", params: {} });
      assertEquals((init?.result as { serverInfo: { name: string } }).serverInfo.name, "combs-mesh");

      // notifications get no response
      assertEquals(
        server.handle({ jsonrpc: "2.0", method: "notifications/initialized" }),
        null,
      );

      // tools/list
      const list = server.handle({ jsonrpc: "2.0", id: 2, method: "tools/list" });
      const tools = (list?.result as { tools: { name: string }[] }).tools.map((t) => t.name);
      assertEquals(tools, ["emoji_build", "emoji_render", "emoji_list", "emoji_to_unicode"]);

      // tools/call emoji_build → emoji_list → emoji_render → emoji_to_unicode
      const built = server.handle({
        jsonrpc: "2.0",
        id: 3,
        method: "tools/call",
        params: { name: "emoji_build", arguments: { name: "mcp-emoji", description: "via mcp" } },
      });
      const builtPayload = JSON.parse(toolText(built));
      assertEquals(builtPayload.hash.length, 64);
      assert(builtPayload.unicode_len > 0);

      const listed = server.handle({
        jsonrpc: "2.0",
        id: 4,
        method: "tools/call",
        params: { name: "emoji_list", arguments: {} },
      });
      const entries = JSON.parse(toolText(listed)).entries as { name: string }[];
      assert(entries.some((e) => e.name === "mcp-emoji"));

      const toUnicode = server.handle({
        jsonrpc: "2.0",
        id: 5,
        method: "tools/call",
        params: { name: "emoji_to_unicode", arguments: { name_or_hash: "mcp-emoji" } },
      });
      assert(JSON.parse(toolText(toUnicode)).chars > 0);

      // resources/list + resources/read
      const resList = server.handle({ jsonrpc: "2.0", id: 6, method: "resources/list" });
      const resources = (resList?.result as { resources: { uri: string }[] }).resources;
      assert(resources.some((r) => r.uri === "emoji://mcp-emoji"));

      const read = server.handle({
        jsonrpc: "2.0",
        id: 7,
        method: "resources/read",
        params: { uri: "emoji://mcp-emoji" },
      });
      const contents = (read?.result as { contents: { blob?: string; mimeType: string }[] })
        .contents;
      assertEquals(contents[0].mimeType, "application/vnd.combs.cmse");
      assert((contents[0].blob ?? "").length > 0);

      // unknown method → JSON-RPC error
      const bad = server.handle({ jsonrpc: "2.0", id: 8, method: "bogus/method" });
      assert(bad?.error);
      assertStringIncludes(bad.error!.message, "unknown method");
    } finally {
      mesh.close();
    }
  },
});

function toolText(response: unknown): string {
  const result = (response as { result: { content: { text: string }[] } }).result;
  return result.content[0].text;
}
