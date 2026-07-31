<script lang="ts">
  import type { UiConfig } from "../config";
  import { DebateStore } from "../debate.svelte";
  import Button from "./ui/Button.svelte";
  import Badge from "./ui/Badge.svelte";
  import Card from "./ui/Card.svelte";

  let { config }: { config: UiConfig } = $props();

  const debate = new DebateStore(config);
  let scrollEl: HTMLDivElement | undefined = $state();

  $effect(() => {
    debate.turns.length;
    debate.turns[debate.turns.length - 1]?.content;
    scrollEl?.scrollTo({ top: scrollEl.scrollHeight, behavior: "smooth" });
  });
</script>

<div class="mx-auto flex h-full max-w-3xl flex-col px-4 py-4">
  <Card>
    <div class="flex flex-wrap items-center gap-2">
      <h1 class="text-lg font-semibold">{config.debate?.topic ?? "Debate"}</h1>
      <div class="ml-auto flex items-center gap-2">
        {#each config.debate?.agents ?? [] as agent}
          <Badge tone={agent.stance === "pro" ? "green" : "red"}>
            {agent.name} · {agent.stance}
          </Badge>
        {/each}
        {#if !debate.running && !debate.done}
          <Button variant="accent" onclick={() => debate.run()}>Start debate</Button>
        {/if}
        {#if debate.done}
          <Button variant="ghost" onclick={() => debate.reset()}>Reset</Button>
        {/if}
      </div>
    </div>
    {#if debate.currentAgent}
      <p class="mt-2 text-sm text-[rgb(var(--muted))]">
        {debate.currentAgent} is speaking…
      </p>
    {/if}
  </Card>

  <div class="mt-4 flex-1 overflow-y-auto" bind:this={scrollEl}>
    <div class="flex flex-col gap-3 pb-4">
      {#each debate.turns as turn}
        <div class="flex {turn.stance === 'pro' ? 'justify-start' : 'justify-end'}">
          <div class="max-w-[85%]">
            <div class="mb-1 text-xs font-medium text-[rgb(var(--muted))]">
              {turn.agent}
            </div>
            <div
              class="rounded-2xl border bg-[rgb(var(--card))] px-4 py-2.5 text-sm whitespace-pre-wrap
                {turn.stance === 'pro' ? 'rounded-bl-md border-l-4 border-l-green-500' : 'rounded-br-md border-r-4 border-r-red-500'}"
            >
              {turn.content}{#if turn.streaming}<span class="animate-pulse">▍</span>{/if}
            </div>
          </div>
        </div>
      {/each}
    </div>
    {#if debate.error}
      <p class="text-sm text-red-500">error: {debate.error}</p>
    {/if}
  </div>
</div>
