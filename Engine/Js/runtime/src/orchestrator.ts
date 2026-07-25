/**
 * Orchestrator: registers agent servers, holds authenticated WebSocket
 * channels to them, delegates work and routes events/results.
 *
 * Also provides AgentPool — spawning agent servers as subprocesses
 * (isolated Deno processes, each with its own GPU engine instance) or
 * in-process servers for dev.
 */

import { getLogger } from "@combs/telemetry";
import { KeyedMutex, Semaphore } from "./primitives.ts";
import type { AgentServerHandle } from "./server.ts";

const log = getLogger("combs.runtime.orch");

export interface DelegateResult {
  ok: boolean;
  data?: unknown;
  error?: string;
  events: unknown[];
}

interface AgentConnection {
  name: string;
  handle?: AgentServerHandle;
  url: string;
  token: string;
  ws?: WebSocket;
  pending: Map<string, {
    resolve: (r: DelegateResult) => void;
    events: unknown[];
    onEvent?: (e: unknown) => void;
  }>;
}

/** The delegation hub. */
export class Orchestrator {
  private agents = new Map<string, AgentConnection>();
  private locks = new KeyedMutex();
  /** Bounds total in-flight delegations across all agents. */
  readonly gate: Semaphore;

  constructor(opts: { maxConcurrent?: number } = {}) {
    this.gate = new Semaphore(opts.maxConcurrent ?? 16);
  }

  /** Registers an already-running agent server (by URL + token). */
  async register(agent: { name: string; url: string; token: string; handle?: AgentServerHandle }): Promise<void> {
    const conn: AgentConnection = { ...agent, pending: new Map() };
    this.agents.set(agent.name, conn);
    await this.connect(conn);
    log.info("agent registered", { name: agent.name });
  }

  private connect(conn: AgentConnection): Promise<void> {
    const wsUrl = conn.url.replace(/^http/, "ws") +
      `/ws?token=${encodeURIComponent(conn.token)}`;
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(wsUrl);
      const timer = setTimeout(() => reject(new Error(`ws connect timeout: ${conn.name}`)), 5000);
      ws.onopen = () => {
        conn.ws = ws;
        clearTimeout(timer);
        resolve();
      };
      ws.onerror = (e) => {
        clearTimeout(timer);
        reject(new Error(`ws connect failed for ${conn.name}: ${e}`));
      };
      ws.onmessage = (ev) => {
        let msg: { type: string; id?: string; ok?: boolean; data?: unknown; error?: string; event?: unknown };
        try {
          msg = JSON.parse(String(ev.data));
        } catch {
          return;
        }
        if ((msg.type === "event" || msg.type === "result") && msg.id) {
          const slot = conn.pending.get(msg.id);
          if (!slot) return;
          if (msg.type === "event") {
            slot.events.push(msg.event);
            slot.onEvent?.(msg.event);
          } else {
            conn.pending.delete(msg.id);
            slot.resolve({ ok: msg.ok ?? false, data: msg.data, error: msg.error, events: slot.events });
          }
        }
      };
      ws.onclose = () => {
        for (const [, slot] of conn.pending) {
          slot.resolve({ ok: false, error: "connection closed", events: slot.events });
        }
        conn.pending.clear();
        conn.ws = undefined;
      };
    });
  }

  /** Delegates a task to an agent; resolves with the result + events. */
  async delegate(
    name: string,
    input: Record<string, unknown>,
    opts: { onEvent?: (e: unknown) => void; serializePerAgent?: boolean } = {},
  ): Promise<DelegateResult> {
    const conn = this.agents.get(name);
    if (!conn) throw new Error(`unknown agent "${name}"`);
    if (!conn.ws || conn.ws.readyState !== WebSocket.OPEN) {
      await this.connect(conn);
    }
    const id = crypto.randomUUID();
    const run = () =>
      new Promise<DelegateResult>((resolve) => {
        conn.pending.set(id, { resolve, events: [], onEvent: opts.onEvent });
        conn.ws!.send(JSON.stringify({ type: "delegate", id, input }));
      });
    if (opts.serializePerAgent ?? true) {
      // One in-flight task per agent by default (GPU single-flight).
      return this.locks.lock(name, () => this.gate.run(run));
    }
    return this.gate.run(run);
  }

  /** Delegates the same input to several agents in parallel. */
  async broadcast(
    names: string[],
    input: Record<string, unknown>,
  ): Promise<Map<string, DelegateResult>> {
    const out = new Map<string, DelegateResult>();
    await Promise.all(
      names.map(async (name) => out.set(name, await this.delegate(name, input))),
    );
    return out;
  }

  listAgents(): string[] {
    return [...this.agents.keys()];
  }

  async close(): Promise<void> {
    for (const conn of this.agents.values()) {
      conn.ws?.close();
      await conn.handle?.close();
    }
    this.agents.clear();
  }
}

export interface SpawnAgentSpec {
  name: string;
  /** Module URL/path of the agent server entrypoint (must call
   * createAgentServer reading COMBS_AGENT_* env vars, or accept --port/--token). */
  module: string;
  /** Extra deno run flags (permissions). */
  allow?: string[];
  env?: Record<string, string>;
}

/** Spawns agent servers as subprocesses and registers them. */
export class AgentPool {
  private processes = new Map<string, Deno.ChildProcess>();

  constructor(private orchestrator: Orchestrator) {}

  /** Spawns one agent subprocess: finds a free port, mints a token,
   * passes both via env, waits for health, registers with the orchestrator. */
  async spawn(spec: SpawnAgentSpec, port: number, token: string): Promise<void> {
    const allow = spec.allow ?? [
      "--allow-read",
      "--allow-env",
      "--allow-net",
      "--allow-ffi",
    ];
    const child = new Deno.Command("deno", {
      args: ["run", ...allow, spec.module],
      env: {
        ...spec.env,
        COMBS_AGENT_NAME: spec.name,
        COMBS_AGENT_PORT: String(port),
        COMBS_AGENT_TOKEN: token,
      },
      stdout: "inherit",
      stderr: "inherit",
    }).spawn();
    this.processes.set(spec.name, child);

    // Wait for health, then register.
    const url = `http://127.0.0.1:${port}`;
    for (let i = 0; i < 200; i++) {
      try {
        const res = await fetch(`${url}/health`);
        if (res.ok) {
          await this.orchestrator.register({ name: spec.name, url, token });
          return;
        }
      } catch {
        // not up yet
      }
      await new Promise((r) => setTimeout(r, 50));
    }
    throw new Error(`agent ${spec.name} did not become healthy on port ${port}`);
  }

  async kill(name: string): Promise<void> {
    const proc = this.processes.get(name);
    try {
      proc?.kill("SIGTERM");
    } catch {
      // already dead
    }
    this.processes.delete(name);
    await proc?.status;
  }

  async killAll(): Promise<void> {
    for (const name of [...this.processes.keys()]) await this.kill(name);
  }
}
