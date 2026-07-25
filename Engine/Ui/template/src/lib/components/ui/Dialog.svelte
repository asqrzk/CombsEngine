<script lang="ts">
  import type { Snippet } from "svelte";
  let {
    open = false,
    title,
    children,
    onclose,
  }: {
    open: boolean;
    title: string;
    children: Snippet;
    onclose?: () => void;
  } = $props();
</script>

{#if open}
  <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4" role="dialog" aria-modal="true">
    <div class="w-full max-w-md rounded-xl border bg-[rgb(var(--card))] p-5 shadow-xl">
      <h2 class="mb-3 text-lg font-semibold">{title}</h2>
      {@render children()}
      {#if onclose}
        <div class="mt-4 flex justify-end">
          <button class="text-sm text-[rgb(var(--muted))] hover:text-[rgb(var(--fg))]" onclick={onclose}>
            Close
          </button>
        </div>
      {/if}
    </div>
  </div>
{/if}
