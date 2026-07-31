/**
 * Orchestration engine: N roles (2–4), each on its OWN engine process,
 * conversing round-robin inside a user-defined scenario.
 *
 * Every role gets its own `combs serve` subprocess (spawned via the proxy
 * with a per-role tag), and every turn is published to the observe bus as a
 * span — so the Control Tower shows one source per agent (`agent:<name>`)
 * with the full input messages and output text inspectable per turn.
 * All inference traffic crosses the permission proxy (relay) as usual.
 */

import { postObserve, spawnEngine, stopEngine, streamChat } from "./api";
import type { UiConfig } from "./config";

export interface OrchRole {
  name: string;
  persona: string;
  behaviour: string;
}

export interface OrchTurn {
  role: string;
  content: string;
  streaming?: boolean;
}

interface RoleEngine {
  url: string;
  port: number;
}

export const MAX_ROLES = 4;

let idCounter = 0;
function rid(prefix: string): string {
  idCounter = (idCounter + 1) & 0xffff;
  return `${prefix}-${Date.now().toString(36)}-${idCounter.toString(36)}`;
}

function blankRole(): OrchRole {
  return { name: "", persona: "", behaviour: "" };
}

export class OrchestrationStore {
  phase = $state<"setup" | "starting" | "running" | "done">("setup");
  scenario = $state("");
  roles = $state<OrchRole[]>([blankRole(), blankRole()]);
  turnCount = $state(12);
  turns = $state<OrchTurn[]>([]);
  currentSpeaker = $state<string | null>(null);
  busy = $state(false);
  error = $state<string | null>(null);
  /** role name → its dedicated engine process. */
  engines = $state<Record<string, RoleEngine>>({});

  private traceId = "";
  private stopRequested = false;

  constructor(private config: UiConfig) {}

  get canAddRole(): boolean {
    return this.roles.length < MAX_ROLES;
  }

  get ready(): boolean {
    return this.roles.length >= 2 && this.roles.every((r) => r.name.trim() !== "");
  }

  addRole(): void {
    if (this.canAddRole) this.roles.push(blankRole());
  }

  removeRole(index: number): void {
    if (this.roles.length > 2) this.roles.splice(index, 1);
  }

  /** Spawns one engine per role, then runs the conversation loop. */
  async begin(): Promise<void> {
    if (this.busy || !this.ready) return;
    this.error = null;
    this.busy = true;
    this.phase = "starting";
    this.turns = [];
    this.engines = {};
    this.stopRequested = false;
    this.traceId = rid("orch");
    const roles = this.roles.map((r) => ({
      name: r.name.trim(),
      persona: r.persona.trim(),
      behaviour: r.behaviour.trim(),
    }));
    try {
      // One engine process per role (tagged) — each shows up separately in
      // the tower and in the per-process stats.
      for (const role of roles) {
        this.engines[role.name] = await spawnEngine(this.config.model, role.name);
      }
      void postObserve({
        source: "orchestration",
        kind: "event",
        name: "orch.session.start",
        traceId: this.traceId,
        attrs: { agents: roles.length, turns: this.turnCount },
        context: { scenario: this.scenario.trim(), roles },
      });
      this.phase = "running";
      await this.runLoop(roles);
      void postObserve({
        source: "orchestration",
        kind: "event",
        name: "orch.session.end",
        traceId: this.traceId,
        attrs: { turns: this.turns.length, stopped: this.stopRequested },
      });
      if (!this.stopRequested) this.phase = "done";
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
      this.phase = "setup";
    } finally {
      this.busy = false;
      this.currentSpeaker = null;
    }
  }

  private async runLoop(roles: OrchRole[]): Promise<void> {
    for (let t = 0; t < this.turnCount && !this.stopRequested; t++) {
      const role = roles[t % roles.length];
      const others = roles.filter((r) => r !== role).map((r) => r.name).join(", ");
      const engine = this.engines[role.name];
      this.currentSpeaker = role.name;
      this.turns.push({ role: role.name, content: "", streaming: true });
      // mutate only through the $state proxy (indexed access)
      const current = this.turns[this.turns.length - 1];

      const transcript = this.turns
        .slice(0, -1)
        .map((x) => `${x.role}: ${x.content}`)
        .join("\n\n");
      const system = [
        this.scenario.trim() ? `Scenario: ${this.scenario.trim()}` : "You are in an improvised scene.",
        `You are ${role.name}.${role.persona ? ` ${role.persona}` : ""}`,
        role.behaviour ? `Your behaviour: ${role.behaviour}` : "",
        `You are speaking with ${others}. Stay fully in character.`,
        "Keep each message under 80 words and respond to the last message directly.",
      ].filter(Boolean).join("\n");
      const messages = [
        { role: "system", content: system },
        {
          role: "user",
          content: transcript
            ? `The scene so far:\n${transcript}\n\nYour turn as ${role.name}:`
            : `Begin the scene as ${role.name}.`,
        },
      ];

      const source = `agent:${role.name}`;
      const spanId = rid("sp");
      const started = Date.now();
      void postObserve({
        source, kind: "span.start", name: "agent.turn",
        traceId: this.traceId, spanId,
        attrs: { turn: t, agent: role.name, port: engine?.port },
        input: messages,
      });

      let turnError: string | null = null;
      try {
        await streamChat(
          engine?.url ?? this.config.server,
          messages,
          { model: this.config.model, temperature: 0.85, maxTokens: 160 },
          {
            onDelta: (d) => {
              current.content += d;
            },
            onDone: () => {
              current.streaming = false;
            },
            onError: (err) => {
              turnError = err.message;
            },
          },
        );
      } catch (e) {
        turnError = e instanceof Error ? e.message : String(e);
      }
      // Always settle the turn — covers streams that abort without onDone.
      current.streaming = false;

      if (turnError) {
        // Permission denial is a user decision — stop the session.
        if (turnError.startsWith("permission denied")) {
          this.error = turnError;
          void postObserve({
            source, kind: "span.end", name: "agent.turn",
            traceId: this.traceId, spanId, status: "error", error: turnError,
            attrs: { turn: t, agent: role.name, durationMs: Date.now() - started },
          });
          break;
        }
        // Transient failure: record it on the turn and keep going.
        current.content = current.content || `*[turn failed: ${turnError}]*`;
      }
      void postObserve({
        source, kind: "span.end", name: "agent.turn",
        traceId: this.traceId, spanId,
        status: turnError ? "error" : "ok",
        error: turnError ?? undefined,
        attrs: { turn: t, agent: role.name, durationMs: Date.now() - started },
        output: current.content,
      });
    }
    this.currentSpeaker = null;
  }

  /** Stops the loop and all spawned engine subprocesses. */
  async end(): Promise<void> {
    this.stopRequested = true;
    for (const e of Object.values(this.engines)) await stopEngine(e.port);
    this.engines = {};
    this.turns = [];
    this.currentSpeaker = null;
    this.error = null;
    this.busy = false;
    this.phase = "setup";
  }
}
