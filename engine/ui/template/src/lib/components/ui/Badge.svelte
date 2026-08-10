<script lang="ts">
  let { tone = "muted", children }: { tone?: "muted" | "accent" | "green" | "red"; children: import("svelte").Snippet } = $props();
  const tones: Record<string, string> = {
    muted: "bg-[rgb(var(--border))] text-[rgb(var(--muted))]",
    accent: "bg-[rgb(var(--accent))]/15 text-[rgb(var(--accent))]",
    green: "bg-green-500/15 text-green-600 dark:text-green-400",
    red: "bg-red-500/15 text-red-600 dark:text-red-400",
  };
</script>

<span class="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium {tones[tone]}">
  {@render children()}
</span>
