/** Debate engine: multi-agent turn-taking, client-side over combs serve. */

import { streamChat } from "./api";
import type { UiConfig } from "./config";

export interface DebateTurn {
  agent: string;
  stance: string;
  content: string;
  streaming?: boolean;
}

export class DebateStore {
  turns = $state<DebateTurn[]>([]);
  currentAgent = $state<string | null>(null);
  running = $state(false);
  done = $state(false);
  error = $state<string | null>(null);

  constructor(private config: UiConfig) {}

  async run(): Promise<void> {
    const debate = this.config.debate;
    if (!debate || this.running) return;
    this.running = true;
    this.done = false;
    this.error = null;
    this.turns = [];

    const totalTurns = debate.turns;
    for (let turn = 0; turn < totalTurns; turn++) {
      const agent = debate.agents[turn % debate.agents.length];
      this.currentAgent = agent.name;
      const entry: DebateTurn = { agent: agent.name, stance: agent.stance, content: "", streaming: true };
      this.turns.push(entry);

      const transcript = this.turns
        .slice(0, -1)
        .map((t) => `${t.agent}: ${t.content}`)
        .join("\n\n");
      const system = [
        `You are ${agent.name}, debating ${agent.stance} the topic: "${debate.topic}".`,
        agent.behavior,
        "Keep each turn under 80 words. Address your opponent's last point directly.",
      ].join("\n");

      const messages = [
        { role: "system", content: system },
        { role: "user", content: transcript ? `Debate so far:\n${transcript}\n\nYour turn:` : `Open the debate on: ${debate.topic}` },
      ];

      let failed: Error | null = null;
      await streamChat(
        this.config.server,
        messages,
        { model: this.config.model, temperature: 0.8, maxTokens: 180 },
        {
          onDelta: (d) => {
            entry.content += d;
          },
          onDone: () => {
            entry.streaming = false;
          },
          onError: (err) => {
            entry.streaming = false;
            failed = err;
          },
        },
      );
      if (failed) {
        this.error = failed.message;
        break;
      }
    }

    this.currentAgent = null;
    this.running = false;
    this.done = true;
  }

  reset(): void {
    this.turns = [];
    this.done = false;
    this.error = null;
  }
}
