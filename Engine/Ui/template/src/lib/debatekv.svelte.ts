/**
 * Debate + KV cache: the debate loop (per-agent stances, alternating turns)
 * where each agent runs under its OWN named KV session on the engine
 * (`session_id = agent name`). Agent turns are interleaved, but each
 * agent's prompt extends its own previous prompt, so the shared prefix is
 * served straight from the KV cache — recorded here per turn for the KV
 * panel.
 */

import { streamChatWithUsage, type CompletionUsage } from "./api";
import type { UiConfig } from "./config";

export interface DebateTurn {
  agent: string;
  stance: string;
  content: string;
  streaming?: boolean;
}

export interface DebateKvStat {
  agent: string;
  promptTokens: number;
  cachedTokens: number;
  completionTokens: number;
  ttftMs: number;
}

export class DebateKvStore {
  turns = $state<DebateTurn[]>([]);
  stats = $state<DebateKvStat[]>([]);
  currentAgent = $state<string | null>(null);
  running = $state(false);
  done = $state(false);
  error = $state<string | null>(null);

  constructor(private config: UiConfig) {}

  get totalSaved(): number {
    return this.stats.reduce((a, s) => a + s.cachedTokens, 0);
  }

  get totalPrompt(): number {
    return this.stats.reduce((a, s) => a + s.promptTokens, 0);
  }

  get avgHitRate(): number {
    return this.totalPrompt > 0 ? this.totalSaved / this.totalPrompt : 0;
  }

  async run(): Promise<void> {
    const debate = this.config.debate;
    if (!debate || this.running) return;
    this.running = true;
    this.done = false;
    this.error = null;
    this.turns = [];
    this.stats = [];

    for (let turn = 0; turn < debate.turns; turn++) {
      const agent = debate.agents[turn % debate.agents.length];
      const opponents = debate.agents.filter((a) => a !== agent).map((a) => a.name).join(", ");
      this.currentAgent = agent.name;
      this.turns.push({ agent: agent.name, stance: agent.stance, content: "", streaming: true });
      // Mutate ONLY through the $state proxy (indexed access).
      const current = this.turns[this.turns.length - 1];

      // Chat-native history (NOT a transcript dumped into one user message):
      // the agent's own past turns are `assistant`, opponents' are `user` —
      // append-only across turns, so the named KV session still serves the
      // whole prefix. This keeps instruct models in "answer as the
      // character" mode instead of "continue the document" mode (which is
      // what produced transcript echoes and role-prefix rambling).
      const system = [
        `You are ${agent.name}, in a formal debate. Motion: "${debate.topic}".`,
        `You argue ${agent.stance === "pro" ? "FOR" : "AGAINST"} the motion — your side never changes, no matter what ${opponents} ${debate.agents.length > 2 ? "say" : "says"}.`,
        agent.behavior ? `Your style: ${agent.behavior}.` : "",
        "Rules: start your reply with the argument itself — never meta-talk (\"I'm ready\", \"let's debate\"), never questions back, no role prefixes, no quoting. Address the last point directly and keep it under 80 words.",
      ].filter(Boolean).join("\n");

      const history = this.turns.slice(0, -1).map((t) => ({
        role: t.agent === agent.name ? "assistant" : "user",
        // Name-tag opponents only when more than two agents debate.
        content: debate.agents.length > 2 && t.agent !== agent.name
          ? `${t.agent}: ${t.content}`
          : t.content,
      }));
      const messages = [
        { role: "system", content: system },
        ...(history.length
          ? history
          : [{ role: "user", content: `The motion is: "${debate.topic}". Open the debate for your side.` }]),
      ];

      const t0 = performance.now();
      let ttftMs = 0;
      let usage: CompletionUsage | null = null;
      let failed: Error | null = null;

      await streamChatWithUsage(
        this.config.server,
        messages,
        // frequency_penalty breaks the verbatim echo loops small models
        // fall into on long multi-turn contexts.
        { model: this.config.model, temperature: 0.7, maxTokens: 180, frequencyPenalty: 0.5, sessionId: agent.name },
        {
          onDelta: (d) => {
            if (ttftMs === 0) ttftMs = Math.round(performance.now() - t0);
            current.content += d;
          },
          onUsage: (u) => {
            usage = u;
          },
          onDone: () => {
            current.streaming = false;
          },
          onError: (err) => {
            current.streaming = false;
            failed = err;
          },
        },
      );
      if (failed) {
        this.error = (failed as Error).message;
        break;
      }
      if (usage) {
        const u: CompletionUsage = usage;
        this.stats.push({
          agent: agent.name,
          promptTokens: u.prompt_tokens,
          cachedTokens: u.prompt_tokens_details?.cached_tokens ?? 0,
          completionTokens: u.completion_tokens,
          ttftMs,
        });
      }
    }

    this.currentAgent = null;
    this.running = false;
    this.done = true;
  }

  reset(): void {
    this.turns = [];
    this.stats = [];
    this.done = false;
    this.error = null;
  }
}
