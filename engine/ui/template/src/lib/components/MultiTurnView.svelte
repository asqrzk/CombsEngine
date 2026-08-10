<script lang="ts">
  import type { UiConfig } from "../config";
  import { MultiTurnStore } from "../multiturn.svelte";
  import TowerView from "./tower/TowerView.svelte";
  import Button from "./ui/Button.svelte";
  import Badge from "./ui/Badge.svelte";

  let { config }: { config: UiConfig } = $props();

  const mt = new MultiTurnStore(config);
  let input = $state("");
  let scrollEl: HTMLDivElement | undefined = $state();
  let showTower = $state(true);
  let showContext = $state(true);
  /** Small screens show ONE pane at a time; lg+ shows panes side by side. */
  let mobileTab = $state<"chat" | "context" | "tower">("chat");

  // All column variants written literally so Tailwind compiles them.
  const colsClass = $derived(
    showContext && showTower
      ? "lg:grid-cols-[minmax(0,1fr)_320px_minmax(0,1.2fr)]"
      : showContext
        ? "lg:grid-cols-[minmax(0,1fr)_320px]"
        : showTower
          ? "lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]"
          : "lg:grid-cols-[minmax(0,1fr)]",
  );

  $effect(() => {
    mt.turns.length;
    mt.turns[mt.turns.length - 1]?.content;
    scrollEl?.scrollTo({ top: scrollEl.scrollHeight, behavior: "smooth" });
  });

  function submit() {
    const text = input.trim();
    if (!text) return;
    input = "";
    void mt.send(text);
  }
</script>

<div class="flex h-full flex-col">
  <!-- toolbar -->
  <div class="flex flex-wrap items-center gap-2 border-b px-4 py-2">
    <h1 class="text-sm font-semibold">Multi-turn</h1>
    <Badge tone="accent">{config.model}</Badge>
    <!-- mobile pane switcher -->
    <div class="flex items-center gap-1 text-xs lg:hidden">
      {#each [["chat", "chat"], ["context", "context"], ["tower", "tower"]] as [tab, label]}
        <button
          class="rounded-md px-2 py-1 hover:bg-[rgb(var(--border))]"
          class:bg-[rgb(var(--border))]={mobileTab === tab}
          class:font-semibold={mobileTab === tab}
          onclick={() => (mobileTab = tab as typeof mobileTab)}
        >{label}</button>
      {/each}
    </div>
    <div class="ml-auto hidden items-center gap-2 lg:flex">
      <Button variant="ghost" onclick={() => (showContext = !showContext)}>
        {showContext ? "hide context" : "show context"}
      </Button>
      <Button variant="ghost" onclick={() => (showTower = !showTower)}>
        {showTower ? "hide tower" : "show tower"}
      </Button>
    </div>
  </div>

  <div class="grid min-h-0 flex-1 grid-cols-1 {colsClass}">
    <!-- chat -->
    <div class="min-h-0 min-w-0 flex-col border-r {mobileTab === 'chat' ? 'flex' : 'hidden'} lg:flex">
      <div class="min-h-0 flex-1 overflow-y-auto px-4 py-3" bind:this={scrollEl}>
        <div class="flex flex-col gap-3">
          {#each mt.turns as turn}
            <div class="flex {turn.role === 'user' ? 'justify-end' : 'justify-start'}">
              <div class="max-w-[85%]">
                <div class="mb-1 text-xs font-medium text-[rgb(var(--muted))]">{turn.role}</div>
                <div class="rounded-2xl border bg-[rgb(var(--card))] px-4 py-2.5 text-sm whitespace-pre-wrap
                  {turn.role === 'user' ? 'rounded-br-md' : 'rounded-bl-md'}">
                  {turn.content}{#if turn.streaming}<span class="animate-pulse">▍</span>{/if}
                </div>
              </div>
            </div>
          {/each}
          {#if mt.turns.length === 0}
            <p class="py-8 text-center text-sm text-[rgb(var(--muted))]">
              start a conversation — every turn's context and run shows up in the tower
            </p>
          {/if}
        </div>
        {#if mt.error}
          <p class="mt-2 text-sm text-red-500">error: {mt.error}</p>
        {/if}
      </div>
      <div class="border-t p-3">
        <div class="flex gap-2">
          <textarea
            class="h-14 flex-1 resize-none rounded-md border bg-transparent px-3 py-2 text-sm"
            placeholder="message…"
            bind:value={input}
            onkeydown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } }}
          ></textarea>
          <div class="flex flex-col gap-2">
            <Button variant="accent" disabled={mt.busy || !input.trim()} onclick={submit}>Send</Button>
            {#if mt.busy}<Button variant="ghost" onclick={() => mt.stop()}>Stop</Button>{/if}
          </div>
        </div>
      </div>
    </div>

    <!-- live context -->
    {#if showContext}
      <div class="min-h-0 min-w-0 overflow-y-auto border-r bg-[rgb(var(--card))] p-3 {mobileTab === 'context' ? 'block' : 'hidden'} lg:block">
        <div class="mb-2 text-xs font-semibold uppercase tracking-wide text-[rgb(var(--muted))]">
          context sent next turn
        </div>
        <pre class="whitespace-pre-wrap rounded-md bg-[rgb(var(--bg))] p-2 text-xs">{mt.contextPreview}</pre>
        <div class="mt-3 text-xs text-[rgb(var(--muted))]">
          {mt.turns.length} turns · window last {mt.windowSize}
        </div>
      </div>
    {/if}

    <!-- control tower -->
    {#if showTower}
      <div class="min-h-0 min-w-0 overflow-hidden {mobileTab === 'tower' ? 'block' : 'hidden'} lg:block">
        <TowerView compact />
      </div>
    {/if}
  </div>
</div>
