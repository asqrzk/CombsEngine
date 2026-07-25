/** UiConfig — written by `combs chew` into combs.ui.json at the app root. */

export interface UiConfig {
  mode: "chat-ui" | "debate-ui";
  features: {
    reasoning: boolean;
    vision: boolean;
    audio: boolean;
    save_chats: boolean;
  };
  authentication: boolean;
  theme: "system" | "dark" | "light";
  model: string;
  server: string;
  debate?: {
    agents: { name: string; stance: string; behavior: string }[];
    topic: string;
    turns: number;
  };
}

export const DEFAULT_CONFIG: UiConfig = {
  mode: "chat-ui",
  features: { reasoning: false, vision: false, audio: false, save_chats: true },
  authentication: true,
  theme: "system",
  model: "smollm2-135m",
  server: "http://localhost:8080",
};

let cached: UiConfig | null = null;

/** Loads combs.ui.json once (falls back to defaults when absent). */
export async function loadConfig(): Promise<UiConfig> {
  if (cached) return cached;
  try {
    const res = await fetch("/combs.ui.json");
    if (res.ok) {
      cached = { ...DEFAULT_CONFIG, ...(await res.json()) };
      return cached!;
    }
  } catch {
    // fall through to defaults
  }
  cached = DEFAULT_CONFIG;
  return cached;
}
