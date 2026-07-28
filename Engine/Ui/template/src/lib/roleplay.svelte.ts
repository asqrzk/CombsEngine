/**
 * Roleplay engine: two roles, TWO engine processes.
 *
 * The first role talks to the engine from the app config (started by
 * `combs chew`). When both roles are defined, the UI asks the proxy to
 * spawn a SECOND `combs serve` subprocess on its own port — a separate
 * process with its own GPU device, so each role generates independently.
 * All traffic crosses the permission proxy (relay) exactly like chat.
 */

import { spawnEngine, stopEngine, streamChat } from "./api";
import type { UiConfig } from "./config";

export interface Role {
  name: string;
  persona: string;
}

export interface RoleplayTurn {
  role: string;
  content: string;
  streaming?: boolean;
}

export class RoleplayStore {
  phase = $state<"setup" | "starting" | "running">("setup");
  roles = $state<[Role, Role] | null>(null);
  turns = $state<RoleplayTurn[]>([]);
  currentSpeaker = $state<string | null>(null);
  busy = $state(false);
  error = $state<string | null>(null);
  /** Second engine URL (spawned subprocess); null until started. */
  engineB = $state<string | null>(null);
  engineBPort: number | null = null;

  constructor(private config: UiConfig) {}

  /** Role defined → once both exist, spin up the second engine. */
  async begin(roleA: Role, roleB: Role, turns: number): Promise<void> {
    if (this.busy) return;
    this.error = null;
    this.busy = true;
    this.phase = "starting";
    this.roles = [roleA, roleB];
    this.turns = [];
    try {
      // second engine subprocess on its own port (permission-gated)
      const engine = await spawnEngine(this.config.model);
      this.engineB = engine.url;
      this.engineBPort = engine.port;
      this.phase = "running";
      await this.runLoop(turns);
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
      this.phase = "setup";
    } finally {
      this.busy = false;
      this.currentSpeaker = null;
    }
  }

  private engineFor(roleIndex: number): string {
    return roleIndex === 0 ? this.config.server : (this.engineB ?? this.config.server);
  }

  private async runLoop(totalTurns: number): Promise<void> {
    if (!this.roles) return;
    for (let t = 0; t < totalTurns; t++) {
      const roleIndex = t % 2;
      const role = this.roles[roleIndex];
      const other = this.roles[1 - roleIndex];
      this.currentSpeaker = role.name;
      this.turns.push({ role: role.name, content: "", streaming: true });
      // mutate only through the $state proxy (indexed access)
      const current = this.turns[this.turns.length - 1];

      const transcript = this.turns
        .slice(0, -1)
        .map((x) => `${x.role}: ${x.content}`)
        .join("\n\n");
      const system = [
        `You are ${role.name}. ${role.persona}`,
        `You are in a roleplay with ${other.name}. Stay fully in character.`,
        "Keep each message under 80 words and respond to the last message directly.",
      ].join("\n");
      const messages = [
        { role: "system", content: system },
        {
          role: "user",
          content: transcript
            ? `Roleplay so far:\n${transcript}\n\nYour turn as ${role.name}:`
            : `Begin the roleplay as ${role.name}.`,
        },
      ];

      let turnError: string | null = null;
      try {
        await streamChat(
          this.engineFor(roleIndex),
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
          break;
        }
        // Transient failure (engine still loading, relay hiccup): record it
        // on the turn and keep the roleplay going.
        current.content = current.content || `*[turn failed: ${turnError}]*`;
      }
    }
    this.currentSpeaker = null;
  }

  /** Ends the session and stops the second engine subprocess. */
  async end(): Promise<void> {
    if (this.engineBPort !== null) await stopEngine(this.engineBPort);
    this.engineBPort = null;
    this.engineB = null;
    this.turns = [];
    this.roles = null;
    this.currentSpeaker = null;
    this.error = null;
    this.busy = false;
    this.phase = "setup";
  }
}
