<script lang="ts">
  import { monitor } from "../monitor";
  import { themeStore } from "../theme";
  import Badge from "./ui/Badge.svelte";
</script>

<header class="sticky top-0 z-40 border-b bg-[rgb(var(--bg))]/80 backdrop-blur">
  <div class="mx-auto flex max-w-5xl items-center gap-3 px-4 py-2.5">
    <div class="flex items-center gap-2">
      <div class="flex h-7 w-7 items-center justify-center rounded-lg bg-[rgb(var(--accent))] text-sm font-bold text-[rgb(var(--accent-fg))]">
        C
      </div>
      <span class="text-sm font-semibold tracking-tight">Combs</span>
    </div>

    <div class="ml-auto flex items-center gap-2 text-xs">
      <!-- realtime network monitor -->
      <Badge tone="muted" title="network down / up">
        ↓ {monitor.downLabel} · ↑ {monitor.upLabel}
      </Badge>
      <!-- realtime storage monitor -->
      <Badge tone="muted" title="local storage used (quota %)">
        ⛁ {monitor.storageLabel}
      </Badge>
      <button
        class="rounded-lg border px-2.5 py-1.5 text-sm hover:bg-[rgb(var(--border))]"
        title="toggle theme"
        onclick={() => themeStore.toggle()}
      >
        {themeStore.dark ? "☀️" : "🌙"}
      </button>
    </div>
  </div>
</header>
