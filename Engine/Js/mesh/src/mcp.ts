/**
 * MCP stdio SERVER: serves emojis as MCP tools + resources to any agent
 * (complements @combs/agents' MCP *client*). JSON-RPC 2.0,
 * newline-delimited — the exact framing the agents client speaks.
 *
 * The server is dependency-injected against the `MeshOps` interface
 * (implemented by `Mesh`), so tests can drive it without FFI.
 */

import type { EmojiJson, RegistryEntry } from "./mesh.ts";

const PROTOCOL_VERSION = "2024-11-05";
const SERVER_INFO = { name: "combs-mesh", version: "0.2.0" };

interface JsonRpcRequest {
  jsonrpc: "2.0";
  id?: number;
  method: string;
  params?: Record<string, unknown>;
}

interface JsonRpcResponse {
  jsonrpc: "2.0";
  id: number | null;
  result?: unknown;
  error?: { code: number; message: string };
}

/** The Mesh surface the server needs (structural — `Mesh` satisfies it). */
export interface MeshOps {
  buildEmoji(input: { name: string; description?: string }): {
    emoji: EmojiJson;
    binary: Uint8Array;
    unicode: string;
  };
  registryRegister(binary: Uint8Array, name?: string): string;
  registryResolve(nameOrHash: string): { emoji: EmojiJson; binary: Uint8Array };
  registryList(): RegistryEntry[];
  toUnicode(emoji: EmojiJson): string;
  renderSprite(cmse: Uint8Array, frame?: number): {
    rgba: Uint8Array;
    width: number;
    height: number;
  };
}

const TOOLS = [
  {
    name: "emoji_build",
    description: "Build an emoji and register it in the content-addressed store",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string" },
        description: { type: "string" },
      },
      required: ["name"],
    },
  },
  {
    name: "emoji_render",
    description: "Render one sprite frame of a registered emoji to RGBA8 (base64)",
    inputSchema: {
      type: "object",
      properties: {
        name_or_hash: { type: "string" },
        frame: { type: "number" },
      },
      required: ["name_or_hash"],
    },
  },
  {
    name: "emoji_list",
    description: "List registered emojis",
    inputSchema: { type: "object", properties: {} },
  },
  {
    name: "emoji_to_unicode",
    description: "Encode a registered emoji to its unicode envelope string",
    inputSchema: {
      type: "object",
      properties: { name_or_hash: { type: "string" } },
      required: ["name_or_hash"],
    },
  },
];

function b64(bytes: Uint8Array): string {
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(bin);
}

/** MCP server for the emoji registry. */
export class EmojiMcpServer {
  constructor(private mesh: MeshOps) {}

  /**
   * Handles one JSON-RPC message. Returns the response, or null for
   * notifications (which must not be answered).
   */
  handle(request: JsonRpcRequest): JsonRpcResponse | null {
    const id = request.id ?? null;
    try {
      const result = this.dispatch(request.method, request.params ?? {});
      // Notifications (no id) get no response, per JSON-RPC.
      if (request.id === undefined) return null;
      return { jsonrpc: "2.0", id, result };
    } catch (e) {
      if (request.id === undefined) return null;
      return {
        jsonrpc: "2.0",
        id,
        error: { code: -32603, message: e instanceof Error ? e.message : String(e) },
      };
    }
  }

  private dispatch(method: string, params: Record<string, unknown>): unknown {
    switch (method) {
      case "initialize":
        return {
          protocolVersion: PROTOCOL_VERSION,
          capabilities: { tools: {}, resources: {} },
          serverInfo: SERVER_INFO,
        };
      case "notifications/initialized":
        return {};
      case "tools/list":
        return { tools: TOOLS };
      case "tools/call":
        return this.callTool(params);
      case "resources/list":
        return { resources: this.resources() };
      case "resources/read":
        return this.readResource(params);
      default:
        throw new Error(`unknown method: ${method}`);
    }
  }

  private toolResult(value: unknown): unknown {
    return { content: [{ type: "text", text: JSON.stringify(value) }] };
  }

  private callTool(params: Record<string, unknown>): unknown {
    const name = String(params.name ?? "");
    const args = (params.arguments ?? {}) as Record<string, unknown>;
    switch (name) {
      case "emoji_build": {
        const emojiName = String(args.name ?? "");
        if (!emojiName) throw new Error("emoji_build needs `name`");
        const built = this.mesh.buildEmoji({
          name: emojiName,
          description: args.description as string | undefined,
        });
        const hash = this.mesh.registryRegister(built.binary, emojiName);
        return this.toolResult({
          hash,
          binary_b64: b64(built.binary),
          unicode_len: Array.from(built.unicode).length,
        });
      }
      case "emoji_render": {
        const { binary } = this.mesh.registryResolve(String(args.name_or_hash ?? ""));
        const frame = typeof args.frame === "number" ? args.frame : 0;
        const rendered = this.mesh.renderSprite(binary, frame);
        return this.toolResult({
          rgba_b64: b64(rendered.rgba),
          width: rendered.width,
          height: rendered.height,
        });
      }
      case "emoji_list":
        return this.toolResult({ entries: this.mesh.registryList() });
      case "emoji_to_unicode": {
        const { emoji } = this.mesh.registryResolve(String(args.name_or_hash ?? ""));
        const unicode = this.mesh.toUnicode(emoji);
        return this.toolResult({ unicode, chars: Array.from(unicode).length });
      }
      default:
        throw new Error(`unknown tool: ${name}`);
    }
  }

  private resources() {
    return this.mesh.registryList().map((entry) => ({
      uri: `emoji://${entry.name}`,
      name: entry.name,
      description: `emoji ${entry.name} (${entry.bytes} bytes, sha256 ${entry.hash.slice(0, 12)}…)`,
      mimeType: "application/vnd.combs.cmse",
    }));
  }

  private readResource(params: Record<string, unknown>): unknown {
    const uri = String(params.uri ?? "");
    if (!uri.startsWith("emoji://")) throw new Error(`unsupported uri: ${uri}`);
    const nameOrHash = uri.slice("emoji://".length);
    const { emoji, binary } = this.mesh.registryResolve(nameOrHash);
    return {
      contents: [
        {
          uri,
          mimeType: "application/vnd.combs.cmse",
          blob: b64(binary),
        },
        {
          uri: `${uri}#meta`,
          mimeType: "application/json",
          text: JSON.stringify({ name: emoji.name, blocks: emoji.blocks.length }),
        },
      ],
    };
  }

  /** Serves newline-delimited JSON-RPC on stdin/stdout forever. */
  async serveStdio(): Promise<void> {
    const decoder = new TextDecoder();
    const encoder = new TextEncoder();
    const writer = Deno.stdout.writable.getWriter();
    let buffer = "";
    for await (const chunk of Deno.stdin.readable) {
      buffer += decoder.decode(chunk, { stream: true });
      let idx: number;
      while ((idx = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (!line) continue;
        let response: JsonRpcResponse | null = null;
        try {
          response = this.handle(JSON.parse(line));
        } catch {
          response = {
            jsonrpc: "2.0",
            id: null,
            error: { code: -32700, message: "parse error" },
          };
        }
        if (response) {
          await writer.write(encoder.encode(JSON.stringify(response) + "\n"));
        }
      }
    }
    writer.close();
  }
}
