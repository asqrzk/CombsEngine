/**
 * Multi-turn chat store — like chat, but with a bounded context window and a
 * live "context sent next turn" preview for the Control Tower. The window
 * keeps prompts short so small models stay coherent across many turns.
 */

import { streamChat } from "./api";
import type { UiConfig } from "./config";

export interface Turn {
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
}

const SYSTEM = "You are a helpful assistant. Answer concisely.";

export class MultiTurnStore {
  turns = $state<Turn[]>([]);
  busy = $state(false);
  error = $state<string | null>(null);
  windowSize = $state(12);
  private abort: AbortController | null = null;

  constructor(private config: UiConfig) {}

  /** The messages that will be sent on the next turn (system + window). */
  get contextMessages(): { role: string; content: string }[] {
    const history = this.turns
      .filter((t) => !t.streaming)
      .slice(-this.windowSize)
      .map((t) => ({ role: t.role, content: t.content }));
    return [{ role: "system", content: SYSTEM }, ...history];
  }

  get contextPreview(): string {
    return this.contextMessages
      .map((m) => `${m.role}: ${m.content}`)
      .join("\n\n") || "(empty)";
  }

  async send(text: string): Promise<void> {
    if (this.busy || !text.trim()) return;
    this.error = null;
    this.busy = true;
    this.turns.push({ role: "user", content: text });
    this.turns.push({ role: "assistant", content: "", streaming: true });

    const messages = this.contextMessages;
    const last = this.turns[this.turns.length - 1];
    this.abort = new AbortController();

    await streamChat(
      this.config.server,
      messages,
      { model: this.config.model, temperature: 0.7, maxTokens: 256, repetitionPenalty: 1.15 },
      {
        onDelta: (d) => {
          last.content += d;
        },
        onDone: () => {
          last.streaming = false;
          this.busy = false;
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
