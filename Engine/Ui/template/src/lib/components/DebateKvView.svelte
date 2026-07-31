<script lang="ts">
  import type { UiConfig } from "../config";
  import { DebateKvStore } from "../debatekv.svelte";
  import Button from "./ui/Button.svelte";
  import Badge from "./ui/Badge.svelte";
  import Card from "./ui/Card.svelte";

  let { config }: { config: UiConfig } = $props();

  const dk = new DebateKvStore(config);
  let scrollEl: HTMLDivElement | undefined = $state();
  /** Small screens show ONE pane at a time; lg+ shows debate | kv side by side. */
  let mobileTab = $state<"debate" | "kv">("debate");

  $effect(() => {
    dk.turns.length;
    dk.turns[dk.turns.length - 1]?.content;
    scrollEl?.scrollTo({ top: scrollEl.scrollHeight, behavior: "smooth" });
  });

  function hitPct(cached: number, prompt: number): number {
    return prompt > 0 ? Math.round((cached / prompt) * 100) : 0;
  }
</script>

<div class="flex h-full flex-col">
  <!-- toolbar -->
  <div class="flex flex-wrap items-center gap-2 border-b px-4 py-2">
    <h1 class="text-sm font-semibold">{config.debate?.topic ?? "Debate"}</h1>
    <Badge tone="accent">{config.model}</Badge>
    <!-- mobile pane switcher -->
    <div class="flex items-center gap-1 text-xs lg:hidden">
      {#each ["debate", "kv"] as tab}
        <button
          class="rounded-md px-2 py-1 hover:bg-[rgb(var(--border))]"
          class:bg-[rgb(var(--border))]={mobileTab === tab}
          class:font-semibold={mobileTab === tab}
          onclick={() => (mobileTab = tab as typeof mobileTab)}
        >{tab === "kv" ? "kv stats" : tab}</button>
      {/each}
    </div>
    <div class="ml-auto flex items-center gap-2">
      {#each config.debate?.agents ?? [] as agent}
        <Badge tone={agent.stance === "pro" ? "green" : "red"}>
          {agent.name} · {agent.stance}
        </Badge>
      {/each}
      {#if !dk.running && !dk.done}
        <Button variant="accent" onclick={() => dk.run()}>Start debate</Button>
      {/if}
      {#if dk.done}
        <Button variant="ghost" onclick={() => dk.reset()}>Reset</Button>
      {/if}
      <a href="#tower" class="rounded-md border px-2 py-1 text-xs hover:bg-[rgb(var(--border))]">Tower →</a>
    </div>
  </div>
  {#if dk.currentAgent}
    <p class="border-b px-4 py-1.5 text-xs text-[rgb(var(--muted))]">
      {dk.currentAgent} is speaking… (own KV session)
    </p>
  {/if}

  <div class="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[minmax(0,1fr)_360px]">
    <!-- debate transcript -->
    <div class="min-h-0 min-w-0 overflow-y-auto border-r px-4 py-3 {mobileTab === 'debate' ? 'block' : 'hidden'} lg:block" bind:this={scrollEl}>
      <div class="flex flex-col gap-3 pb-4">
        {#each dk.turns as turn}
          <div class="flex {turn.stance === 'pro' ? 'justify-start' : 'justify-end'}">
            <div class="max-w-[85%]">
              <div class="mb-1 text-xs font-medium text-[rgb(var(--muted))]">{turn.agent}</div>
              <div
                class="rounded-2xl border bg-[rgb(var(--card))] px-4 py-2.5 text-sm whitespace-pre-wrap
                  {turn.stance === 'pro' ? 'rounded-bl-md border-l-4 border-l-green-500' : 'rounded-br-md border-r-4 border-r-red-500'}"
              >
                {turn.content}{#if turn.streaming}<span class="animate-pulse">▍</span>{/if}
              </div>
            </div>
          </div>
        {/each}
        {#if dk.turns.length === 0}
          <p class="py-8 text-center text-sm text-[rgb(var(--muted))]">
            start the debate — each agent keeps its own named KV session, so
            from its second turn on, its prefix comes straight from the cache
          </p>
        {/if}
      </div>
      {#if dk.error}
        <p class="text-sm text-red-500">error: {dk.error}</p>
      {/if}
    </div>

    <!-- kv stats panel -->
    <div class="min-h-0 min-w-0 overflow-y-auto bg-[rgb(var(--card))] p-3 {mobileTab === 'kv' ? 'block' : 'hidden'} lg:block">
      <div class="mb-2 text-xs font-semibold uppercase tracking-wide text-[rgb(var(--muted))]">
        kv sessions (per agent)
      </div>

      <div class="rounded-xl border p-3">
        <div class="text-2xl font-semibold tabular-nums">{dk.totalSaved.toLocaleString()}</div>
        <div class="text-xs text-[rgb(var(--muted))]">prompt tokens served from cache</div>
        <div class="mt-3 flex items-center gap-2">
          <div class="h-1.5 flex-1 overflow-hidden rounded-full bg-[rgb(var(--border))]">
            <div class="h-full bg-green-500" style="width: {Math.round(dk.avgHitRate * 100)}%"></div>
          </div>
          <span class="text-xs tabular-nums">{Math.round(dk.avgHitRate * 100)}% hit rate</span>
        </div>
      </div>

      <div class="mt-3 flex flex-col gap-2">
        {#each [...dk.stats].reverse() as s, ri}
          {@const turn = dk.stats.length - ri}
          {@const pct = hitPct(s.cachedTokens, s.promptTokens)}
          <div class="rounded-lg border p-2 text-xs">
            <div class="flex items-center justify-between gap-2">
              <span class="truncate font-medium">turn {turn} · {s.agent}</span>
              <Badge tone={pct > 60 ? "green" : pct > 0 ? "accent" : "muted"}>{pct}% cached</Badge>
            </div>
            <div class="mt-1.5 h-1.5 overflow-hidden rounded-full bg-[rgb(var(--border))]">
              <div class="h-full bg-green-500" style="width: {pct}%"></div>
            </div>
            <div class="mt-1.5 text-[rgb(var(--muted))]">
              {s.cachedTokens}/{s.promptTokens} cached · +{s.completionTokens} generated
              {#if s.ttftMs > 0} · ttft {s.ttftMs}ms{/if}
            </div>
          </div>
        {/each}
        {#if dk.stats.length === 0}
          <p class="text-xs text-[rgb(var(--muted))]">
            each agent's first turn is cold; watch its later turns go green as
            its named session serves the prefix.
          </p>
        {/if}
      </div>
    </div>
  </div>
</div>
