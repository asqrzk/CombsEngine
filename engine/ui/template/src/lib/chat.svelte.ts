/** Chat session store: streaming messages, optional persistence (via proxy). */

import { deleteFile, readFile, streamChat, writeFile } from "./api";
import type { UiConfig } from "./config";

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
}

const CHATS_FILE = "chats.json";

export class ChatStore {
  turns = $state<ChatTurn[]>([]);
  busy = $state(false);
  error = $state<string | null>(null);
  private abort: AbortController | null = null;

  constructor(private config: UiConfig) {
    if (config.features.save_chats) {
      // Persisted by the proxy on the backend (permission-gated on write).
      readFile(CHATS_FILE).then((raw) => {
        if (!raw) return;
        try {
          this.turns = JSON.parse(raw);
        } catch {
          this.turns = [];
        }
      });
    }
  }

  private persist(): void {
    if (!this.config.features.save_chats) return;
    // Fire-and-forget; the proxy asks for storage:chats permission when
    // needed and enforces it server-side.
    void writeFile(CHATS_FILE, JSON.stringify(this.turns), "storage:chats");
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
    void deleteFile(CHATS_FILE, "storage:chats");
  }
}
