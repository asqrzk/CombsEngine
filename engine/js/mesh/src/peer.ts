/**
 * MeshPeer — the v1 connectivity connector: serves a registry over
 * HTTP+WS and fetches emojis from other peers with sha256 integrity
 * verification. Zero-dep, Deno-only.
 *
 * Composable with (but separate from) the MCP server in mcp.ts: MCP is
 * the agent-facing connector, MeshPeer is the peer-to-peer one — both sit
 * on the same `Mesh`.
 *
 * Endpoints (optional `Authorization: Bearer <token>` gate):
 *   GET /mesh/v1/list                 → {entries: [{name, hash, bytes}]}
 *   GET /mesh/v1/emoji/:nameOrHash    → {emoji, hash, binary_b64}
 *   GET /mesh/v1/announce             → {peer_id, name, count}
 *   GET /mesh/v1/ws                   → WS upgrade; sends announce on
 *                                       connect (seam for future push
 *                                       events)
 */

import type { EmojiJson, Mesh, RegistryEntry } from "./mesh.ts";

const encoder = new TextEncoder();

/** SHA-256 hex digest (integrity hash, same digest family as the Rust registry). */
export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes as BufferSource);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

function b64(bytes: Uint8Array): string {
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(bin);
}

function unb64(s: string): Uint8Array {
  const bin = atob(s);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

function json(value: unknown, init?: ResponseInit): Response {
  return new Response(JSON.stringify(value), {
    ...init,
    headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
  });
}

/** A serving peer: local registry over HTTP+WS. */
export class MeshPeer {
  private constructor(
    private server: Deno.HttpServer,
    /** This peer's id (sha256 hex). */
    readonly peerId: Promise<string>,
    /** The port actually bound. */
    readonly port: number,
  ) {}

  /** Serves the mesh registry; `port: 0`/undefined picks a free port. */
  static serve(options: {
    mesh: Mesh;
    port?: number;
    token?: string;
    name?: string;
  }): MeshPeer {
    const { mesh, token } = options;
    const name = options.name ?? "combs-mesh-peer";
    const random = crypto.getRandomValues(new Uint8Array(16));
    const peerId = sha256Hex(options.name ? encoder.encode(options.name) : random);

    const announce = async () => ({
      peer_id: await peerId,
      name,
      count: mesh.registryList().length,
    });

    const handler = async (request: Request): Promise<Response> => {
      if (token) {
        const auth = request.headers.get("authorization");
        if (auth !== `Bearer ${token}`) {
          return json({ error: "unauthorized" }, { status: 401 });
        }
      }
      const url = new URL(request.url);
      if (request.method !== "GET") {
        return json({ error: "method not allowed" }, { status: 405 });
      }
      switch (url.pathname) {
        case "/mesh/v1/list":
          return json({
            entries: mesh.registryList().map((e: RegistryEntry) => ({
              name: e.name,
              hash: e.hash,
              bytes: e.bytes,
            })),
          });
        case "/mesh/v1/announce":
          return json(await announce());
        case "/mesh/v1/ws": {
          if (request.headers.get("upgrade") !== "websocket") {
            return json({ error: "expected websocket upgrade" }, { status: 400 });
          }
          const { socket, response } = Deno.upgradeWebSocket(request);
          socket.onopen = async () => {
            socket.send(JSON.stringify({ type: "announce", ...(await announce()) }));
          };
          return response;
        }
        default: {
          const m = url.pathname.match(/^\/mesh\/v1\/emoji\/(.+)$/);
          if (!m) return json({ error: "not found" }, { status: 404 });
          try {
            const { emoji, binary } = mesh.registryResolve(decodeURIComponent(m[1]));
            return json({
              emoji,
              hash: await sha256Hex(binary),
              binary_b64: b64(binary),
            });
          } catch {
            return json({ error: "emoji not found" }, { status: 404 });
          }
        }
      }
    };

    const server = Deno.serve(
      { hostname: "127.0.0.1", port: options.port ?? 0 },
      (request) => handler(request),
    );
    const addr = server.addr as Deno.NetAddr;
    return new MeshPeer(server, peerId, addr.port);
  }

  /** The base URL of this peer. */
  baseUrl(): string {
    return `http://127.0.0.1:${this.port}`;
  }

  /** Stops serving. */
  close(): Promise<void> {
    return this.server.shutdown();
  }

  /**
   * Fetches an emoji from a peer and VERIFIES its integrity:
   * sha256(binary) must equal the advertised hash — throws on mismatch,
   * on HTTP errors, and on malformed responses.
   */
  static async fetch(
    baseUrl: string,
    nameOrHash: string,
    options?: { token?: string },
  ): Promise<{ emojiJson: EmojiJson; binary: Uint8Array; hash: string }> {
    const url = `${baseUrl}/mesh/v1/emoji/${encodeURIComponent(nameOrHash)}`;
    const response = await fetch(url, {
      headers: options?.token ? { authorization: `Bearer ${options.token}` } : {},
    });
    if (!response.ok) {
      throw new Error(`mesh fetch ${nameOrHash}: HTTP ${response.status}`);
    }
    const body = await response.json();
    if (!body?.binary_b64 || !body?.hash || !body?.emoji) {
      throw new Error("mesh fetch: malformed response");
    }
    const binary = unb64(body.binary_b64);
    const actual = await sha256Hex(binary);
    if (actual !== body.hash) {
      throw new Error(
        `mesh fetch: integrity mismatch (advertised ${body.hash.slice(0, 12)}…, got ${actual.slice(0, 12)}…)`,
      );
    }
    return { emojiJson: body.emoji, binary, hash: body.hash };
  }
}
