<script lang="ts">
  import type { UiConfig } from "../config";
  import { KvSessionStore } from "../kvsession.svelte";
  import Button from "./ui/Button.svelte";
  import Badge from "./ui/Badge.svelte";

  let { config }: { config: UiConfig } = $props();

  const kv = new KvSessionStore(config);
  let input = $state("");
  let scrollEl: HTMLDivElement | undefined = $state();
  /** Small screens show ONE pane at a time; lg+ shows chat | kv side by side. */
  let mobileTab = $state<"chat" | "kv">("chat");

  $effect(() => {
    kv.turns.length;
    kv.turns[kv.turns.length - 1]?.content;
    scrollEl?.scrollTo({ top: scrollEl.scrollHeight, behavior: "smooth" });
  });

  function submit() {
    const text = input.trim();
    if (!text) return;
    input = "";
    void kv.send(text);
  }

  function hitPct(cached: number, prompt: number): number {
    return prompt > 0 ? Math.round((cached / prompt) * 100) : 0;
  }
</script>

<div class="flex h-full flex-col">
  <!-- toolbar -->
  <div class="flex flex-wrap items-center gap-2 border-b px-4 py-2">
    <h1 class="text-sm font-semibold">KV cache</h1>
    <Badge tone="accent">{config.model}</Badge>
    <span class="hidden text-xs text-[rgb(var(--muted))] sm:inline">
      rolling-session prefix reuse — every turn shares its prefix with the last
    </span>
    <!-- mobile pane switcher -->
    <div class="flex items-center gap-1 text-xs lg:hidden">
      {#each ["chat", "kv"] as tab}
        <button
          class="rounded-md px-2 py-1 hover:bg-[rgb(var(--border))]"
          class:bg-[rgb(var(--border))]={mobileTab === tab}
          class:font-semibold={mobileTab === tab}
          onclick={() => (mobileTab = tab as typeof mobileTab)}
        >{tab === "kv" ? "kv stats" : tab}</button>
      {/each}
    </div>
    <div class="ml-auto">
      <a href="#tower" class="rounded-md border px-2 py-1 text-xs hover:bg-[rgb(var(--border))]">Tower →</a>
    </div>
  </div>

  <div class="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[minmax(0,1fr)_360px]">
    <!-- chat -->
    <div class="min-h-0 min-w-0 flex-col border-r {mobileTab === 'chat' ? 'flex' : 'hidden'} lg:flex">
      <div class="min-h-0 flex-1 overflow-y-auto px-4 py-3" bind:this={scrollEl}>
        <div class="flex flex-col gap-3">
          {#each kv.turns as turn}
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
          {#if kv.turns.length === 0}
            <p class="py-8 text-center text-sm text-[rgb(var(--muted))]">
              chat away — the first turn is a cold prefill; from turn two the
              prompt prefix comes straight from the KV cache
            </p>
          {/if}
        </div>
        {#if kv.error}
          <p class="mt-2 text-sm text-red-500">error: {kv.error}</p>
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
            <Button variant="accent" disabled={kv.busy || !input.trim()} onclick={submit}>Send</Button>
            {#if kv.busy}<Button variant="ghost" onclick={() => kv.stop()}>Stop</Button>{/if}
          </div>
        </div>
      </div>
    </div>

    <!-- kv stats panel -->
    <div class="min-h-0 min-w-0 overflow-y-auto bg-[rgb(var(--card))] p-3 {mobileTab === 'kv' ? 'block' : 'hidden'} lg:block">
      <div class="mb-2 text-xs font-semibold uppercase tracking-wide text-[rgb(var(--muted))]">
        kv cache session
      </div>

      <!-- aggregate -->
      <div class="rounded-xl border p-3">
        <div class="text-2xl font-semibold tabular-nums">{kv.totalSaved.toLocaleString()}</div>
        <div class="text-xs text-[rgb(var(--muted))]">prompt tokens served from cache</div>
        <div class="mt-3 flex items-center gap-2">
          <div class="h-1.5 flex-1 overflow-hidden rounded-full bg-[rgb(var(--border))]">
            <div class="h-full bg-green-500" style="width: {Math.round(kv.avgHitRate * 100)}%"></div>
          </div>
          <span class="text-xs tabular-nums">{Math.round(kv.avgHitRate * 100)}% hit rate</span>
        </div>
      </div>

      <!-- per-turn rows -->
      <div class="mt-3 flex flex-col gap-2">
        {#each [...kv.stats].reverse() as s, ri}
          {@const turn = kv.stats.length - ri}
          {@const pct = hitPct(s.cachedTokens, s.promptTokens)}
          <div class="rounded-lg border p-2 text-xs">
            <div class="flex items-center justify-between">
              <span class="font-medium">turn {turn}</span>
              <Badge tone={pct > 80 ? "green" : pct > 0 ? "accent" : "muted"}>{pct}% cached</Badge>
            </div>
            <div class="mt-1.5 h-1.5 overflow-hidden rounded-full bg-[rgb(var(--border))]">
              <div class="h-full bg-green-500" style="width: {pct}%"></div>
            </div>
            <div class="mt-1.5 text-[rgb(var(--muted))]">
              {s.cachedTokens}/{s.promptTokens} prompt tokens cached · +{s.completionTokens} generated
              {#if s.ttftMs > 0} · ttft {s.ttftMs}ms{/if}
            </div>
          </div>
        {/each}
        {#if kv.stats.length === 0}
          <p class="text-xs text-[rgb(var(--muted))]">
            no completed turns yet — send a couple of messages and watch the
            hit rate climb (turn 1 is always cold).
          </p>
        {/if}
      </div>
    </div>
  </div>
</div>
