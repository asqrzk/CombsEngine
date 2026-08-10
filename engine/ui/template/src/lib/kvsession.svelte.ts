/**
 * KV cache demo chat store — a plain multi-turn chat that sends the FULL
 * transcript every turn. The engine's rolling-session prefix reuse serves
 * the shared prefix straight from the KV cache (`cached_tokens` in the
 * final SSE usage), so each turn only prefills the new messages. This store
 * records per-turn cache stats for the KV panel.
 */

import { streamChatWithUsage, type CompletionUsage } from "./api";
import type { UiConfig } from "./config";

export interface KvTurn {
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
}

export interface TurnStat {
  promptTokens: number;
  cachedTokens: number;
  completionTokens: number;
  /** Client-measured time to first token. */
  ttftMs: number;
}

const SYSTEM = "You are a helpful assistant. Answer concisely.";

export class KvSessionStore {
  turns = $state<KvTurn[]>([]);
  stats = $state<TurnStat[]>([]);
  busy = $state(false);
  error = $state<string | null>(null);
  private abort: AbortController | null = null;

  constructor(private config: UiConfig) {}

  /** Total prompt tokens served from the KV cache instead of prefill. */
  get totalSaved(): number {
    return this.stats.reduce((a, s) => a + s.cachedTokens, 0);
  }

  get totalPrompt(): number {
    return this.stats.reduce((a, s) => a + s.promptTokens, 0);
  }

  /** Fraction of all prompt tokens that were cache hits (0..1). */
  get avgHitRate(): number {
    return this.totalPrompt > 0 ? this.totalSaved / this.totalPrompt : 0;
  }

  /** Full transcript (append-only) — that's what makes the prefix reusable. */
  private get messages(): { role: string; content: string }[] {
    const history = this.turns
      .filter((t) => !t.streaming)
      .map((t) => ({ role: t.role, content: t.content }));
    return [{ role: "system", content: SYSTEM }, ...history];
  }

  async send(text: string): Promise<void> {
    if (this.busy || !text.trim()) return;
    this.error = null;
    this.busy = true;
    this.turns.push({ role: "user", content: text });
    this.turns.push({ role: "assistant", content: "", streaming: true });

    const messages = this.messages;
    const last = this.turns[this.turns.length - 1];
    this.abort = new AbortController();

    const t0 = performance.now();
    let ttftMs = 0;
    let usage: CompletionUsage | null = null;

    await streamChatWithUsage(
      this.config.server,
      messages,
      { model: this.config.model, temperature: 0.7, maxTokens: 256 },
      {
        onDelta: (d) => {
          if (ttftMs === 0) ttftMs = Math.round(performance.now() - t0);
          last.content += d;
        },
        onUsage: (u) => {
          usage = u;
        },
        onDone: () => {
          last.streaming = false;
          this.busy = false;
          if (usage) {
            const u: CompletionUsage = usage;
            this.stats.push({
              promptTokens: u.prompt_tokens,
              cachedTokens: u.prompt_tokens_details?.cached_tokens ?? 0,
              completionTokens: u.completion_tokens,
              ttftMs,
            });
          }
        },
        onError: (err) => {
          last.streaming = false;
          this.busy = false;
          this.error = err.message;
        },
      },
      this.abort.signal,
    );
  }

  stop(): void {
    this.abort?.abort();
    const last = this.turns[this.turns.length - 1];
    if (last?.streaming) last.streaming = false;
    this.busy = false;
  }
}
