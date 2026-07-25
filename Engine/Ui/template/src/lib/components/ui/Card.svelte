<script lang="ts">
  import type { Snippet } from "svelte";
  let { children, padded = true }: { children: Snippet; padded?: boolean } = $props();
</script>

<div class="rounded-xl border bg-[rgb(var(--card))] shadow-sm {padded ? 'p-4' : ''}">
  {@render children()}
</div>
