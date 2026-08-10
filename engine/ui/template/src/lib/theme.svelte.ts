/** Theme store: dark / light / system, persisted (encrypted), class-based. */

import { secureGet, secureSet } from "./secureStore";

export type Theme = "system" | "dark" | "light";

const KEY = "combs.theme";

function isTheme(v: string | null): v is Theme {
  return v === "dark" || v === "light" || v === "system";
}

function systemDark(): boolean {
  return globalThis.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
}

class ThemeStore {
  theme = $state<Theme>("system");

  init(preset: Theme): void {
    this.theme = preset;
    this.apply();
    // encrypted read: upgrades the theme once the identity is available
    void secureGet(KEY).then((saved) => {
      if (isTheme(saved)) {
        this.theme = saved;
        this.apply();
      }
    });
    globalThis.matchMedia?.("(prefers-color-scheme: dark)").addEventListener("change", () => {
      this.apply();
    });
  }

  get dark(): boolean {
    return this.theme === "dark" || (this.theme === "system" && systemDark());
  }

  set(theme: Theme): void {
    this.theme = theme;
    void secureSet(KEY, theme); // encrypted at rest (fire-and-forget)
    this.apply();
  }

  toggle(): void {
    this.set(this.dark ? "light" : "dark");
  }

  private apply(): void {
    document.documentElement.classList.toggle("dark", this.dark);
  }
}

export const themeStore = new ThemeStore();
