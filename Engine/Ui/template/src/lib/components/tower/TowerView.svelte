<script lang="ts">
  import { onMount } from "svelte";
  import { tower, type ObsEvent } from "../../tower.svelte";
  import Badge from "../ui/Badge.svelte";
  import Button from "../ui/Button.svelte";

  onMount(() => tower.start());

  const KIND_TONE: Record<string, "muted" | "accent" | "green" | "red"> = {
    "span.start": "accent",
    "span.end": "muted",
    event: "green",
    metric: "accent",
    log: "muted",
    context: "muted",
  };

  function fmtTs(ts: number): string {
    return new Date(ts).toLocaleTimeString([], { hour12: false }) +
      "." + String(ts % 1000).padStart(3, "0");
  }

  function preview(e: ObsEvent): string {
    const bits: string[] = [];
    if (e.attrs) {
      for (const [k, v] of Object.entries(e.attrs)) {
        if (v !== undefined && ["status", "durationMs", "bytes", "port", "model", "agent", "method", "scope", "grant"].includes(k)) {
          bits.push(`${k}=${v}`);
        }
      }
    }
    if (e.error) bits.push(`err=${e.error.slice(0, 60)}`);
    return bits.slice(0, 5).join(" ");
  }
</script>

<div class="flex h-full flex-col gap-3 p-4">
  <!-- header -->
  <div class="flex flex-wrap items-center gap-2">
    <h1 class="text-lg font-semibold">Control Tower</h1>
    <Badge tone={tower.connected ? "green" : "red"}>
      {tower.connected ? "live" : "reconnecting…"}
    </Badge>
    <span class="text-xs text-[rgb(var(--muted))]">{tower.filtered.length} / {tower.events.length} events</span>
    <div class="ml-auto flex items-center gap-2">
      <input
        class="w-44 rounded-md border bg-transparent px-2 py-1 text-sm"
        placeholder="filter…"
        bind:value={tower.textFilter}
      />
      <Button variant="ghost" onclick={() => tower.clear()}>Clear</Button>
    </div>
  </div>

  <div class="grid min-h-0 flex-1 grid-cols-1 gap-3 lg:grid-cols-[220px_1fr_340px]">
    <!-- sources -->
    <div class="min-h-0 overflow-y-auto rounded-xl border bg-[rgb(var(--card))] p-3">
      <div class="mb-2 text-xs font-semibold uppercase tracking-wide text-[rgb(var(--muted))]">Sources</div>
      <button
        class="mb-1 w-full rounded-md px-2 py-1 text-left text-sm hover:bg-[rgb(var(--border))]"
        class:bg-[rgb(var(--border))]={tower.sourceFilter === null}
        onclick={() => (tower.sourceFilter = null)}
      >
        all sources
      </button>
      {#each tower.sourceList() as s}
        <button
          class="mb-1 w-full rounded-md px-2 py-1 text-left text-sm hover:bg-[rgb(var(--border))]"
          class:bg-[rgb(var(--border))]={tower.sourceFilter === s.id}
          onclick={() => (tower.sourceFilter = s.id)}
        >
          <div class="font-medium">{s.id}</div>
          <div class="text-xs text-[rgb(var(--muted))]">
            {#if s.state.port}:{s.state.port}{/if}
            {#if s.state.model} · {s.state.model}{/if}
          </div>
        </button>
      {/each}
      {#if tower.sourceList().length === 0}
        <p class="text-xs text-[rgb(var(--muted))]">no activity yet</p>
      {/if}
    </div>

    <!-- event stream -->
    <div class="flex min-h-0 flex-col rounded-xl border bg-[rgb(var(--card))]">
      <div class="flex items-center gap-1 border-b p-2">
        {#each [null, "span.start", "span.end", "event", "metric", "context"] as k}
          <button
            class="rounded-md px-2 py-0.5 text-xs hover:bg-[rgb(var(--border))]"
            class:bg-[rgb(var(--border))]={tower.kindFilter === k}
            onclick={() => (tower.kindFilter = k)}
          >
            {k ?? "all"}
          </button>
        {/each}
      </div>
      <div class="min-h-0 flex-1 overflow-y-auto p-2">
        {#each [...tower.filtered].reverse() as e (e.id)}
          <button
            class="mb-1 flex w-full items-center gap-2 rounded-md px-2 py-1 text-left text-xs hover:bg-[rgb(var(--border))]"
            class:bg-[rgb(var(--border))]={tower.selected?.id === e.id}
            onclick={() => tower.select(e)}
          >
            <span class="text-[rgb(var(--muted))]">{fmtTs(e.ts)}</span>
            <Badge tone={KIND_TONE[e.kind] ?? "muted"}>{e.kind}</Badge>
            <span class="font-medium">{e.source}</span>
            <span>{e.name}</span>
            {#if e.status === "error"}<Badge tone="red">err</Badge>{/if}
            <span class="truncate text-[rgb(var(--muted))]">{preview(e)}</span>
          </button>
        {/each}
        {#if tower.filtered.length === 0}
          <p class="p-3 text-sm text-[rgb(var(--muted))]">waiting for events…</p>
        {/if}
      </div>
    </div>

    <!-- detail inspector -->
    <div class="min-h-0 overflow-y-auto rounded-xl border bg-[rgb(var(--card))] p-3">
      {#if tower.selected}
        {@const e = tower.selected}
        <div class="mb-2 flex items-center justify-between">
          <div class="text-sm font-semibold">{e.name}</div>
          <button class="text-xs text-[rgb(var(--muted))]" onclick={() => tower.select(null)}>✕</button>
        </div>
        <div class="space-y-2 text-xs">
          <div><span class="text-[rgb(var(--muted))]">source</span> {e.source}</div>
          <div><span class="text-[rgb(var(--muted))]">kind</span> {e.kind}</div>
          {#if e.traceId}<div><span class="text-[rgb(var(--muted))]">trace</span> {e.traceId}</div>{/if}
          {#if e.status}<div><span class="text-[rgb(var(--muted))]">status</span> {e.status}</div>{/if}
          {#if e.error}<div class="text-red-500">{e.error}</div>{/if}
          {#if e.attrs && Object.keys(e.attrs).length}
            <div>
              <div class="font-semibold text-[rgb(var(--muted))]">attrs</div>
              <pre class="mt-1 overflow-x-auto rounded-md bg-[rgb(var(--bg))] p-2">{JSON.stringify(e.attrs, null, 2)}</pre>
            </div>
          {/if}
          {#if e.context !== undefined}
            <div>
              <div class="font-semibold text-[rgb(var(--muted))]">context</div>
              <pre class="mt-1 max-h-64 overflow-auto rounded-md bg-[rgb(var(--bg))] p-2">{JSON.stringify(e.context, null, 2)}</pre>
            </div>
          {/if}
          {#if e.input !== undefined}
            <div>
              <div class="font-semibold text-[rgb(var(--muted))]">input</div>
              <pre class="mt-1 max-h-64 overflow-auto rounded-md bg-[rgb(var(--bg))] p-2">{JSON.stringify(e.input, null, 2)}</pre>
            </div>
          {/if}
          {#if e.output !== undefined}
            <div>
              <div class="font-semibold text-[rgb(var(--muted))]">output</div>
              <pre class="mt-1 max-h-64 overflow-auto rounded-md bg-[rgb(var(--bg))] p-2">{JSON.stringify(e.output, null, 2)}</pre>
            </div>
          {/if}
        </div>
      {:else}
        <p class="text-sm text-[rgb(var(--muted))]">select an event to inspect input / context / output</p>
      {/if}
    </div>
  </div>
</div>
