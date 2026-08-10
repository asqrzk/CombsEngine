<script lang="ts">
  import type { UiConfig } from "../config";
  import { RoleplayStore } from "../roleplay.svelte";
  import Button from "./ui/Button.svelte";
  import Badge from "./ui/Badge.svelte";
  import Card from "./ui/Card.svelte";

  let { config }: { config: UiConfig } = $props();

  const rp = new RoleplayStore(config);
  let scrollEl: HTMLDivElement | undefined = $state();

  // role setup form
  let nameA = $state("");
  let personaA = $state("");
  let nameB = $state("");
  let personaB = $state("");
  let turnCount = $state(8);

  const rolesReady = $derived(
    nameA.trim() !== "" && personaA.trim() !== "" &&
    nameB.trim() !== "" && personaB.trim() !== "",
  );

  $effect(() => {
    rp.turns.length;
    rp.turns[rp.turns.length - 1]?.content;
    scrollEl?.scrollTo({ top: scrollEl.scrollHeight, behavior: "smooth" });
  });

  function begin() {
    void rp.begin(
      { name: nameA.trim(), persona: personaA.trim() },
      { name: nameB.trim(), persona: personaB.trim() },
      turnCount,
    );
  }
</script>

<div class="mx-auto flex h-full max-w-3xl flex-col px-4 py-4">
  {#if rp.phase === "setup"}
    <Card>
      <h1 class="text-lg font-semibold">Define the two roles</h1>
      <p class="mt-1 text-sm text-[rgb(var(--muted))]">
        Each role gets its own engine process — when both are defined, a second
        model instance is started on a new port (with your permission).
      </p>

      <div class="mt-4 grid gap-4 sm:grid-cols-2">
        <div class="rounded-lg border p-3">
          <Badge tone="green">role one · engine A</Badge>
          <input
            class="mt-2 w-full rounded-md border bg-transparent px-2 py-1.5 text-sm"
            placeholder="name (e.g. Detective Mara)"
            bind:value={nameA}
          />
          <textarea
            class="mt-2 h-24 w-full rounded-md border bg-transparent px-2 py-1.5 text-sm"
            placeholder="persona — who are they, how do they speak?"
            bind:value={personaA}
          ></textarea>
        </div>
        <div class="rounded-lg border p-3">
          <Badge tone="red">role two · engine B (new port)</Badge>
          <input
            class="mt-2 w-full rounded-md border bg-transparent px-2 py-1.5 text-sm"
            placeholder="name (e.g. Suspect Ren)"
            bind:value={nameB}
          />
          <textarea
            class="mt-2 h-24 w-full rounded-md border bg-transparent px-2 py-1.5 text-sm"
            placeholder="persona — who are they, how do they speak?"
            bind:value={personaB}
          ></textarea>
        </div>
      </div>

      <div class="mt-4 flex items-center gap-3">
        <label class="text-sm text-[rgb(var(--muted))]">
          turns
          <input
            type="number" min="2" max="40"
            class="ml-2 w-16 rounded-md border bg-transparent px-2 py-1 text-sm"
            bind:value={turnCount}
          />
        </label>
        <Button variant="accent" disabled={!rolesReady} onclick={begin}>
          Begin roleplay
        </Button>
      </div>
      {#if rp.error}
        <p class="mt-3 text-sm text-red-500">error: {rp.error}</p>
      {/if}
    </Card>
  {:else}
    <Card>
      <div class="flex flex-wrap items-center gap-2">
        <h1 class="text-lg font-semibold">Roleplay</h1>
        <div class="ml-auto flex items-center gap-2">
          {#each rp.roles ?? [] as role, i}
            <Badge tone={i === 0 ? "green" : "red"}>{role.name}</Badge>
          {/each}
          <Button variant="ghost" onclick={() => rp.end()}>End</Button>
        </div>
      </div>
      {#if rp.phase === "starting"}
        <p class="mt-2 text-sm text-[rgb(var(--muted))]">
          starting the second engine on a new port (model load can take a moment)…
        </p>
      {:else if rp.currentSpeaker}
        <p class="mt-2 text-sm text-[rgb(var(--muted))]">{rp.currentSpeaker} is speaking…</p>
      {/if}
    </Card>

    <div class="mt-4 flex-1 overflow-y-auto" bind:this={scrollEl}>
      <div class="flex flex-col gap-3 pb-4">
        {#each rp.turns as turn, i}
          <div class="flex {i % 2 === 0 ? 'justify-start' : 'justify-end'}">
            <div class="max-w-[85%]">
              <div class="mb-1 text-xs font-medium text-[rgb(var(--muted))]">{turn.role}</div>
              <div
                class="rounded-2xl border bg-[rgb(var(--card))] px-4 py-2.5 text-sm whitespace-pre-wrap
                  {i % 2 === 0 ? 'rounded-bl-md border-l-4 border-l-green-500' : 'rounded-br-md border-r-4 border-r-red-500'}"
              >
                {turn.content}{#if turn.streaming}<span class="animate-pulse">▍</span>{/if}
              </div>
            </div>
          </div>
        {/each}
      </div>
      {#if rp.error}
        <p class="text-sm text-red-500">error: {rp.error}</p>
      {/if}
    </div>
  {/if}
</div>
