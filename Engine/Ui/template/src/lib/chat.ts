/** Chat session store: streaming messages, optional local persistence. */

import { streamChat } from "./api";
import { permissions } from "./permissions";
import type { UiConfig } from "./config";

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
}

const CHATS_KEY = "combs.chats";

export class ChatStore {
  turns = $state<ChatTurn[]>([]);
  busy = $state(false);
  error = $state<string | null>(null);
  private abort: AbortController | null = null;

  constructor(private config: UiConfig) {
    if (config.features.save_chats) {
      try {
        this.turns = JSON.parse(localStorage.getItem(CHATS_KEY) ?? "[]");
      } catch {
        this.turns = [];
      }
    }
  }

  private persist(): void {
    if (!this.config.features.save_chats) return;
    permissions
      .require("storage:chats", "save this chat on this device")
      .then((ok) => {
        if (ok) localStorage.setItem(CHATS_KEY, JSON.stringify(this.turns));
      });
  }

  async send(text: string): Promise<void> {
    if (this.busy || !text.trim()) return;
    this.error = null;
    this.busy = true;
    this.turns.push({ role: "user", content: text });
    this.turns.push({ role: "assistant", content: "", streaming: true });
    this.persist();

    const history = this.turns.slice(0, -1).map((t) => ({ role: t.role, content: t.content }));
    const last = this.turns[this.turns.length - 1];
    this.abort = new AbortController();

    await streamChat(
      this.config.server,
      history,
      { model: this.config.model, temperature: 0.7 },
      {
        onDelta: (delta) => {
          last.content += delta;
        },
        onDone: () => {
          last.streaming = false;
          this.busy = false;
          this.persist();
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

  clear(): void {
    this.turns = [];
    localStorage.removeItem(CHATS_KEY);
  }
}
