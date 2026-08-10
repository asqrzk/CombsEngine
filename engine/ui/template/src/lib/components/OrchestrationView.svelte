<script lang="ts">
  import type { UiConfig } from "../config";
  import { OrchestrationStore } from "../orchestration.svelte";
  import Button from "./ui/Button.svelte";
  import Badge from "./ui/Badge.svelte";
  import Card from "./ui/Card.svelte";
  import StatsBar from "./tower/StatsBar.svelte";

  let { config }: { config: UiConfig } = $props();

  const orch = new OrchestrationStore(config);
  let scrollEl: HTMLDivElement | undefined = $state();
  /** transcript filter tab: null = all agents. */
  let tab = $state<string | null>(null);

  const TONES = ["green", "red", "accent", "muted"] as const;
  const ACCENTS = ["border-l-green-500", "border-l-red-500", "border-l-blue-500", "border-l-amber-500"];

  const roleNames = $derived(orch.roles.map((r) => r.name.trim()).filter(Boolean));
  const visibleTurns = $derived(
    tab === null ? orch.turns : orch.turns.filter((t) => t.role === tab),
  );

  function toneOf(role: string): (typeof TONES)[number] {
    return TONES[Math.max(0, roleNames.indexOf(role)) % TONES.length];
  }
  function accentOf(role: string): string {
    return ACCENTS[Math.max(0, roleNames.indexOf(role)) % ACCENTS.length];
  }

  $effect(() => {
    orch.turns.length;
    orch.turns[orch.turns.length - 1]?.content;
    scrollEl?.scrollTo({ top: scrollEl.scrollHeight, behavior: "smooth" });
  });
</script>

<div class="mx-auto flex h-full max-w-3xl flex-col px-4 py-4">
  {#if orch.phase === "setup"}
    <div class="min-h-0 flex-1 overflow-y-auto">
      <Card>
        <h1 class="text-lg font-semibold">Orchestration</h1>
        <p class="mt-1 text-sm text-[rgb(var(--muted))]">
          Define a scenario and the roles — each agent runs on its own engine
          process, and every turn streams into the Control Tower
          (<a class="underline" href="#tower">#tower</a>).
        </p>

        <textarea
          class="mt-4 h-20 w-full resize-none rounded-md border bg-transparent px-3 py-2 text-sm"
          placeholder="scenario (e.g. a starship crew arguing about whether to land on a dying planet)"
          bind:value={orch.scenario}
        ></textarea>

        <div class="mt-4 grid gap-4 sm:grid-cols-2">
          {#each orch.roles as role, i}
            <div class="rounded-lg border p-3">
              <div class="flex items-center justify-between">
                <Badge tone={TONES[i % TONES.length]}>agent {i + 1} · own engine</Badge>
                {#if orch.roles.length > 2}
                  <button class="text-xs text-[rgb(var(--muted))]" onclick={() => orch.removeRole(i)}>✕</button>
                {/if}
              </div>
              <input
                class="mt-2 w-full rounded-md border bg-transparent px-2 py-1.5 text-sm"
                placeholder="name (e.g. Captain Imani)"
                bind:value={role.name}
              />
              <textarea
                class="mt-2 h-16 w-full rounded-md border bg-transparent px-2 py-1.5 text-sm"
                placeholder="persona — who are they?"
                bind:value={role.persona}
              ></textarea>
              <textarea
                class="mt-2 h-16 w-full rounded-md border bg-transparent px-2 py-1.5 text-sm"
                placeholder="behaviour — how do they act / speak?"
                bind:value={role.behaviour}
              ></textarea>
            </div>
          {/each}
        </div>

        <div class="mt-4 flex flex-wrap items-center gap-3">
          {#if orch.canAddRole}
            <Button variant="ghost" onclick={() => orch.addRole()}>+ add agent</Button>
          {/if}
          <label class="text-sm text-[rgb(var(--muted))]">
            turns
            <input
              type="number" min="2" max="40"
              class="ml-2 w-16 rounded-md border bg-transparent px-2 py-1 text-sm"
              bind:value={orch.turnCount}
            />
          </label>
          <Button variant="accent" disabled={!orch.ready} onclick={() => orch.begin()}>
            Start
          </Button>
        </div>
        {#if !orch.ready}
          <p class="mt-2 text-xs text-[rgb(var(--muted))]">name at least two agents to start</p>
        {/if}
        {#if orch.error}
          <p class="mt-3 text-sm text-red-500">error: {orch.error}</p>
        {/if}
      </Card>
    </div>
  {:else}
    <Card>
      <div class="flex flex-wrap items-center gap-2">
        <h1 class="text-lg font-semibold">Orchestration</h1>
        <div class="ml-auto flex flex-wrap items-center gap-2">
          {#each roleNames as name}
            <Badge tone={toneOf(name)}>{name}{orch.engines[name] ? ` :${orch.engines[name].port}` : ""}</Badge>
          {/each}
          <a href="#tower" class="rounded-md border px-2 py-1 text-xs hover:bg-[rgb(var(--border))]">Tower →</a>
          <Button variant="ghost" onclick={() => orch.end()}>End</Button>
        </div>
      </div>
      {#if orch.phase === "starting"}
        <p class="mt-2 text-sm text-[rgb(var(--muted))]">
          starting one engine per agent (model load can take a moment)…
        </p>
      {:else if orch.phase === "done"}
        <p class="mt-2 text-sm text-[rgb(var(--muted))]">session complete — End to reset</p>
      {:else if orch.currentSpeaker}
        <p class="mt-2 text-sm text-[rgb(var(--muted))]">{orch.currentSpeaker} is speaking…</p>
      {/if}
    </Card>

    <div class="mt-3">
      <StatsBar />
    </div>

    <!-- per-agent tabs -->
    <div class="mt-3 flex flex-wrap items-center gap-1 text-xs">
      <button
        class="rounded-md px-2 py-1 hover:bg-[rgb(var(--border))]"
        class:bg-[rgb(var(--border))]={tab === null}
        class:font-semibold={tab === null}
        onclick={() => (tab = null)}
      >all</button>
      {#each roleNames as name}
        <button
          class="rounded-md px-2 py-1 hover:bg-[rgb(var(--border))]"
          class:bg-[rgb(var(--border))]={tab === name}
          class:font-semibold={tab === name}
          onclick={() => (tab = name)}
        >{name}</button>
      {/each}
    </div>

    <div class="mt-3 min-h-0 flex-1 overflow-y-auto" bind:this={scrollEl}>
      <div class="flex flex-col gap-3 pb-4">
        {#each visibleTurns as turn}
          <div class="flex justify-start">
            <div class="max-w-[85%]">
              <div class="mb-1 text-xs font-medium text-[rgb(var(--muted))]">{turn.role}</div>
              <div
                class="rounded-2xl rounded-bl-md border border-l-4 bg-[rgb(var(--card))] px-4 py-2.5 text-sm whitespace-pre-wrap {accentOf(turn.role)}"
              >
                {turn.content}{#if turn.streaming}<span class="animate-pulse">▍</span>{/if}
              </div>
            </div>
          </div>
        {/each}
        {#if visibleTurns.length === 0 && orch.phase === "running"}
          <p class="py-8 text-center text-sm text-[rgb(var(--muted))]">waiting for the first turn…</p>
        {/if}
      </div>
      {#if orch.error}
        <p class="text-sm text-red-500">error: {orch.error}</p>
      {/if}
    </div>
  {/if}
</div>
