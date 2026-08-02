/**
 * AgentServer: one agent exposed over HTTP + WebSocket on its own port.
 *
 * - `GET  /health`                  — liveness (no auth)
 * - `GET  /info`                    — agent name/metadata (no auth)
 * - `POST /v1/chat/completions`     — OpenAI-shaped chat (bearer auth)
 * - `WS   /ws`                      — orchestration channel (bearer auth):
 *     client → {type:"delegate", id, input}
 *     server → {type:"event", id, event}* then {type:"result", id, ok, data|error}
 *
 * The server wraps a CompiledGraph (preferred — full state/checkpoint
 * support) or any async handler. Authorization uses a per-server token
 * minted at creation (see primitives.generateToken).
 */

import { getLogger } from "@combs/telemetry";
import type { CompiledGraph } from "@combs/graph";
import { checkBearer, findFreePort, generateToken } from "./primitives.ts";
import type { KeyRing } from "./keys.ts";
import type { Envelope } from "@combs/zerotrust";

const log = getLogger("combs.runtime.server");

/** What an agent server runs per request. */
export type AgentHandler = (
  input: { messages?: { role: string; content: string }[]; threadId?: string; [k: string]: unknown },
  emit: (event: unknown) => void,
) => Promise<unknown>;

export interface AgentServerOptions {
  name: string;
  /** A compiled graph: input = partial state, streams superstep events. */
  graph?: CompiledGraph<Record<string, unknown>>;
  /** Or a raw handler. */
  handler?: AgentHandler;
  port?: number;
  token?: string;
  /** Extra metadata reported by /info. */
  metadata?: Record<string, unknown>;
  /**
   * Zero-trust keyring. When set, the server exposes GET/POST /keys for
   * the one-time peer exchange, and WS messages sealed as envelopes are
   * opened before dispatch (replies are sealed back). Plaintext delegates
   * still work for unkeyed legacy peers.
   */
  keyring?: KeyRing;
}

export interface AgentServerHandle {
  name: string;
  port: number;
  token: string;
  url: string;
  wsUrl: string;
  close(): Promise<void>;
}

/** Adapts a compiled graph to the handler contract. */
function graphHandler(graph: CompiledGraph<Record<string, unknown>>): AgentHandler {
  return async (input, emit) => {
    const config: { threadId?: string } = {};
    if (typeof input.threadId === "string") config.threadId = input.threadId;
    for await (const event of graph.stream(input, config)) {
      emit(event);
    }
    return await graph.invoke(input, config);
  };
}

/** Starts an agent server; resolves with its handle once listening. */
export async function createAgentServer(options: AgentServerOptions): Promise<AgentServerHandle> {
  const port = options.port ?? findFreePort();
  const token = options.token ?? generateToken();
  const handler = options.handler ??
    (options.graph ? graphHandler(options.graph) : undefined);
  if (!handler) throw new Error("createAgentServer needs a graph or a handler");

  const sockets = new Set<WebSocket>();

  const server = Deno.serve({ port, hostname: "127.0.0.1" }, (request) => {
    const url = new URL(request.url);

    if (url.pathname === "/health") {
      return Response.json({ status: "ok", name: options.name });
    }
    if (url.pathname === "/info") {
      return Response.json({ name: options.name, metadata: options.metadata ?? {} });
    }

    // Authorization: bearer header for HTTP, `?token=` for the WS upgrade
    // (the WebSocket API can't set custom headers).
    const authed = checkBearer(request, token) ||
      url.searchParams.get("token") === token;
    if (!authed) {
      return Response.json({ error: "unauthorized" }, { status: 401 });
    }

    if (url.pathname === "/v1/chat/completions" && request.method === "POST") {
      return (async () => {
        const body = await request.json();
        const result = await handler(body, () => {});
        const text = typeof result === "object" && result !== null && "messages" in result
          ? (result.messages as { role: string; content: string }[]).at(-1)?.content ?? ""
          : String(result);
        return Response.json({
          id: `chatcmpl-${crypto.randomUUID().slice(0, 8)}`,
          object: "chat.completion",
          created: Math.floor(Date.now() / 1000),
          model: options.name,
          choices: [{ index: 0, message: { role: "assistant", content: text }, finish_reason: "stop" }],
        });
      })();
    }

    // --- zero-trust key exchange (token-authed, one time per peer) -------
    if (url.pathname === "/keys" && options.keyring) {
      if (request.method === "GET") {
        return Response.json(options.keyring.publicKeys());
      }
      if (request.method === "POST") {
        return (async () => {
          const keys = await request.json();
          if (!keys?.signPub || !keys?.encPub || !keys?.fingerprint) {
            return Response.json({ error: "need {signPub, encPub, fingerprint}" }, { status: 400 });
          }
          options.keyring!.addPeer(keys);
          log.info("peer key registered", { name: options.name, peer: keys.fingerprint });
          return Response.json({ ok: true });
        })();
      }
    }

    if (url.pathname === "/ws") {
      const { socket, response } = Deno.upgradeWebSocket(request);
      sockets.add(socket);
      socket.onclose = () => sockets.delete(socket);
      socket.onmessage = async (ev) => {
        let msg: {
          type: string;
          id?: string;
          input?: Record<string, unknown>;
          sealed?: Envelope;
        };
        try {
          msg = JSON.parse(String(ev.data));
        } catch {
          return;
        }
        if (msg.type === "delegate" && msg.id) {
          const id = msg.id;
          const send = (payload: unknown) => {
            if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(payload));
          };
          try {
            // Zero-trust: sealed input is opened (sig + hash verified)
            // before dispatch; events/results are sealed back to the
            // same peer.
            const peerFp = msg.sealed?.from;
            const input = msg.sealed && options.keyring
              ? await options.keyring.openFrom(msg.sealed) as Record<string, unknown>
              : msg.input ?? {};
            const sealBack = async (payload: unknown): Promise<Record<string, unknown>> =>
              peerFp && options.keyring
                ? { sealed: await options.keyring.sealFor(peerFp, payload) }
                : payload as Record<string, unknown>;

            // Serialize sealed sends through a per-request chain: emit() is
            // fire-and-forget for handlers, but sealFor is async — without
            // chaining, the result's seal can win the race and reach the
            // wire before an earlier event (observed as flaky ordering on
            // CI runners). Events must always precede the result.
            let sendChain: Promise<void> = Promise.resolve();
            const sendEvent = (event: unknown): Promise<void> => {
              sendChain = sendChain.then(async () => {
                send({ type: "event", id, ...(await sealBack({ event })) });
              }).catch(() => {}); // a failed event send must not break the chain
              return sendChain;
            };

            const result = await handler(input, sendEvent);
            await sendChain; // flush pending event sends before the result
            send({ type: "result", id, ok: true, ...(await sealBack({ data: result })) });
          } catch (err) {
            send({
              type: "result",
              id,
              ok: false,
              error: err instanceof Error ? err.message : String(err),
            });
          }
        }
      };
      return response;
    }

    return Response.json({ error: "not found" }, { status: 404 });
  });

  await new Promise<void>((resolve) => {
    // Deno.serve starts asynchronously; poll health until it answers.
    const check = async () => {
      for (let i = 0; i < 100; i++) {
        try {
          const res = await fetch(`http://127.0.0.1:${port}/health`);
          if (res.ok) return resolve();
        } catch {
          // not up yet
        }
        await new Promise((r) => setTimeout(r, 20));
      }
      resolve();
    };
    check();
  });

  log.info("agent server up", { name: options.name, port });
  return {
    name: options.name,
    port,
    token,
    url: `http://127.0.0.1:${port}`,
    wsUrl: `ws://127.0.0.1:${port}/ws`,
    close: async () => {
      for (const socket of sockets) socket.close();
      await server.shutdown();
    },
  };
}
