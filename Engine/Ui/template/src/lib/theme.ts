/** Theme store: dark / light / system, persisted, class-based. */

export type Theme = "system" | "dark" | "light";

const KEY = "combs.theme";

function initial(preset: Theme): Theme {
  const saved = localStorage.getItem(KEY);
  if (saved === "dark" || saved === "light" || saved === "system") return saved;
  return preset;
}

function systemDark(): boolean {
  return globalThis.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
}

class ThemeStore {
  theme = $state<Theme>("system");

  init(preset: Theme): void {
    this.theme = initial(preset);
    globalThis.matchMedia?.("(prefers-color-scheme: dark)").addEventListener("change", () => {
      this.apply();
    });
    this.apply();
  }

  get dark(): boolean {
    return this.theme === "dark" || (this.theme === "system" && systemDark());
  }

  set(theme: Theme): void {
    this.theme = theme;
    localStorage.setItem(KEY, theme);
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
