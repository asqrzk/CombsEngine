<script lang="ts">
  import { monitor } from "../../monitor.svelte";
  import Badge from "../ui/Badge.svelte";

  const sys = $derived(monitor.sys);
  const memPct = $derived(sys ? Math.min(100, Math.round((sys.memUsedMb / sys.memTotalMb) * 100)) : 0);

  function barTone(pct: number): string {
    return pct > 85 ? "bg-red-500" : pct > 60 ? "bg-amber-500" : "bg-green-500";
  }
  function fmtMb(mb: number): string {
    return mb >= 1024 ? `${(mb / 1024).toFixed(1)}G` : `${mb}M`;
  }
</script>

<div class="flex flex-wrap items-center gap-x-4 gap-y-1.5 rounded-xl border bg-[rgb(var(--card))] px-3 py-2 text-xs">
  {#if sys}
    <!-- system cpu -->
    <div class="flex items-center gap-1.5" title="system CPU busy, all cores (proxy process: {sys.cpuPct}%)">
      <span class="text-[rgb(var(--muted))]">cpu</span>
      <div class="h-1.5 w-14 overflow-hidden rounded-full bg-[rgb(var(--border))]">
        <div class="h-full {barTone(Math.min(100, sys.sysCpuPct))}" style="width: {Math.min(100, sys.sysCpuPct)}%"></div>
      </div>
      <span class="tabular-nums">{sys.sysCpuPct}%</span>
    </div>
    <!-- system memory -->
    <div class="flex items-center gap-1.5" title="system memory in use / total (reclaimable pages excluded on macOS)">
      <span class="text-[rgb(var(--muted))]">mem</span>
      <div class="h-1.5 w-14 overflow-hidden rounded-full bg-[rgb(var(--border))]">
        <div class="h-full {barTone(memPct)}" style="width: {memPct}%"></div>
      </div>
      <span class="tabular-nums">{fmtMb(sys.memUsedMb)}/{fmtMb(sys.memTotalMb)}</span>
    </div>
    <!-- gpu (best effort) -->
    <div class="flex items-center gap-1.5" title="GPU busy % — exposed only on Linux DRM; wgpu/macOS have no unprivileged counter">
      <span class="text-[rgb(var(--muted))]">gpu</span>
      {#if sys.gpuPct === null}
        <span class="text-[rgb(var(--muted))]">n/a</span>
      {:else}
        <div class="h-1.5 w-14 overflow-hidden rounded-full bg-[rgb(var(--border))]">
          <div class="h-full {barTone(sys.gpuPct)}" style="width: {sys.gpuPct}%"></div>
        </div>
        <span class="tabular-nums">{sys.gpuPct}%</span>
      {/if}
    </div>
    <span class="text-[rgb(var(--muted))]" title="load average (1m) · proxy RSS">load {sys.loadAvg1} · proxy {fmtMb(sys.rssMb)}</span>
    <!-- per-engine process stats -->
    {#each sys.engines as e}
      <Badge tone="accent" title="engine '{e.tag ?? e.model}' pid {e.pid ?? '?'}">
        :{e.port}{e.tag ? ` ${e.tag}` : ""}
        {e.cpuPct === null ? "" : ` · ${e.cpuPct}%`}
        {e.rssMb === null ? "" : ` · ${fmtMb(e.rssMb)}`}
      </Badge>
    {/each}
    <span class="ml-auto text-[rgb(var(--muted))]">↓ {monitor.downLabel} · ↑ {monitor.upLabel}</span>
  {:else}
    <span class="text-[rgb(var(--muted))]">sampling system stats…</span>
  {/if}
</div>
