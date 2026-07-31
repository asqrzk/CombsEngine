<script lang="ts">
  import type { UiConfig } from "../config";
  import { ChatStore } from "../chat.svelte";
  import Button from "./ui/Button.svelte";
  import Badge from "./ui/Badge.svelte";

  let { config }: { config: UiConfig } = $props();

  const chat = new ChatStore(config);
  let input = $state("");
  let scrollEl: HTMLDivElement | undefined = $state();

  $effect(() => {
    // autoscroll on new content
    chat.turns.length;
    chat.turns[chat.turns.length - 1]?.content;
    scrollEl?.scrollTo({ top: scrollEl.scrollHeight, behavior: "smooth" });
  });

  function submit() {
    const text = input;
    input = "";
    chat.send(text);
  }
</script>

<div class="mx-auto flex h-full max-w-3xl flex-col">
  <div class="flex-1 overflow-y-auto px-4 py-4" bind:this={scrollEl}>
    {#if chat.turns.length === 0}
      <div class="mt-16 text-center text-[rgb(var(--muted))]">
        <p class="text-lg">Start a conversation</p>
        <p class="mt-1 text-sm">model: {config.model} · server: {config.server}</p>
        {#if config.features.reasoning}
          <div class="mt-2"><Badge tone="accent">reasoning mode on</Badge></div>
        {/if}
      </div>
    {/if}

    <div class="flex flex-col gap-3">
      {#each chat.turns as turn}
        <div class="flex {turn.role === 'user' ? 'justify-end' : 'justify-start'}">
          <div
            class="max-w-[85%] rounded-2xl px-4 py-2.5 text-sm whitespace-pre-wrap
              {turn.role === 'user'
                ? 'bg-[rgb(var(--accent))] text-[rgb(var(--accent-fg))] rounded-br-md'
                : 'bg-[rgb(var(--card))] border rounded-bl-md'}"
          >
            {turn.content}{#if turn.streaming}<span class="animate-pulse">▍</span>{/if}
          </div>
        </div>
      {/each}
    </div>

    {#if chat.error}
      <p class="mt-3 text-sm text-red-500">error: {chat.error}</p>
    {/if}
  </div>

  <div class="border-t px-4 py-3">
    <form class="flex gap-2" onsubmit={(e) => { e.preventDefault(); submit(); }}>
      {#if config.features.vision}
        <Button variant="ghost" onclick={() => alert("vision attachments: coming soon")}>📷</Button>
      {/if}
      {#if config.features.audio}
        <Button variant="ghost" onclick={() => alert("audio capture: coming soon")}>🎤</Button>
      {/if}
      <input
        class="flex-1 rounded-lg border bg-[rgb(var(--card))] px-3.5 py-2 text-sm outline-none focus:ring-2 ring-[rgb(var(--accent))]"
        placeholder="Message {config.model}…"
        bind:value={input}
        disabled={chat.busy}
      />
      {#if chat.busy}
        <Button variant="ghost" onclick={() => chat.stop()}>Stop</Button>
      {:else}
        <Button variant="accent" onclick={submit}>Send</Button>
      {/if}
    </form>
  </div>
</div>
