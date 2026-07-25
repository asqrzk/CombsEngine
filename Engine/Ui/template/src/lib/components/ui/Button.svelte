<script lang="ts">
  import type { Snippet } from "svelte";

  let {
    variant = "default",
    disabled = false,
    onclick,
    children,
  }: {
    variant?: "default" | "accent" | "ghost" | "danger";
    disabled?: boolean;
    onclick?: () => void;
    children: Snippet;
  } = $props();

  const styles: Record<string, string> = {
    default: "bg-[rgb(var(--card))] border hover:bg-[rgb(var(--border))]",
    accent: "bg-[rgb(var(--accent))] text-[rgb(var(--accent-fg))] hover:opacity-90 border-transparent",
    ghost: "border-transparent hover:bg-[rgb(var(--border))]",
    danger: "bg-red-600 text-white hover:bg-red-700 border-transparent",
  };
</script>

<button
  class="rounded-lg border px-3.5 py-2 text-sm font-medium transition-colors disabled:opacity-40 disabled:pointer-events-none {styles[variant]}"
  {disabled}
  {onclick}
>
  {@render children()}
</button>
