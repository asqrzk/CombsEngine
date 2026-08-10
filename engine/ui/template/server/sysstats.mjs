/**
 * System stats sampler — CPU / memory / per-engine-process / GPU (best
 * effort), zero dependencies.
 *
 * A 2s interval keeps a latest snapshot for GET /api/monitor (`sys` field);
 * every 5th sample (~10s) is also published to the observe bus as a
 * `system.stats` metric so the Control Tower stream shows a heartbeat
 * without flooding it (and the NDJSON log stays small).
 *
 * GPU utilization: there is no cross-platform, unprivileged API (wgpu does
 * not expose it; macOS needs `sudo powermetrics`). We read Linux DRM
 * `gpu_busy_percent` when present and report `null` ("n/a") elsewhere.
 */

import os from "node:os";
import fs from "node:fs";
import { execFile } from "node:child_process";
import { bus as obsBus } from "./observe.mjs";
import { engineProcesses } from "./engine.mjs";

const SAMPLE_MS = 2000;
const PUBLISH_EVERY = 5;

let timer = null;
let samples = 0;
let latest = null;
let lastCpu = null; // {usage, time} for proxy cpu% deltas

function proxyCpuPct() {
  const usage = process.cpuUsage();
  const now = Date.now();
  let pct = 0;
  if (lastCpu) {
    const wall = (now - lastCpu.time) / 1000;
    if (wall > 0) pct = ((usage.user + usage.system) - (lastCpu.usage.user + lastCpu.usage.system)) / 1e6 / wall * 100;
  }
  lastCpu = { usage, time: now };
  return Math.max(0, Math.round(pct * 10) / 10);
}

/** System-wide CPU busy % from os.cpus() time deltas (all cores aggregate). */
let lastCpuTimes = null;
function systemCpuPct() {
  const now = os.cpus().map((c) => c.times);
  let pct = 0;
  if (lastCpuTimes && lastCpuTimes.length === now.length) {
    let busy = 0;
    let total = 0;
    for (let i = 0; i < now.length; i++) {
      const a = lastCpuTimes[i];
      const b = now[i];
      const idle = b.idle - a.idle;
      const all = (b.user - a.user) + (b.nice - a.nice) + (b.sys - a.sys) + idle + (b.irq - a.irq);
      busy += all - idle;
      total += all;
    }
    pct = total > 0 ? (busy / total) * 100 : 0;
  }
  lastCpuTimes = now;
  return Math.round(pct * 10) / 10;
}

/**
 * macOS memory in-use. `os.freemem()` counts only truly-free pages, so a
 * healthy Mac always looks ~100% used (the kernel keeps inactive/cached
 * pages around). `vm_stat` lets us exclude reclaimable pages, matching
 * what Activity Monitor shows. Returns null off-darwin.
 */
function macUsedMemMb() {
  return new Promise((resolve) => {
    if (process.platform !== "darwin") return resolve(null);
    execFile("vm_stat", [], { timeout: 3000 }, (err, stdout) => {
      if (err) return resolve(null);
      const pageSize = Number(stdout.match(/page size of (\d+) bytes/)?.[1]) || 16384;
      const pages = (name) => Number(stdout.match(new RegExp(`${name}:\\s+(\\d+)`))?.[1]) || 0;
      const reclaimable =
        (pages("Pages free") + pages("Pages inactive") + pages("Pages speculative")) *
        pageSize / 1024 / 1024;
      resolve(Math.max(0, Math.round(os.totalmem() / 1024 / 1024 - reclaimable)));
    });
  });
}

/** pid → {cpuPct, rssMb} via `ps` (darwin/linux); null on win32/failure. */
function psStat(pid) {
  return new Promise((resolve) => {
    if (!pid || process.platform === "win32") return resolve(null);
    execFile("ps", ["-o", "%cpu=", "-o", "rss=", "-p", String(pid)], { timeout: 3000 }, (err, stdout) => {
      if (err) return resolve(null);
      const m = stdout.trim().match(/([\d.]+)\s+(\d+)/);
      if (!m) return resolve(null);
      resolve({ cpuPct: Math.round(parseFloat(m[1]) * 10) / 10, rssMb: Math.round(parseInt(m[2], 10) / 1024) });
    });
  });
}

/** Linux DRM busy percent (max across cards); null when unavailable. */
let gpuFiles;
function gpuPct() {
  if (gpuFiles === undefined) {
    try {
      gpuFiles = fs.readdirSync("/sys/class/drm")
        .filter((d) => /^card\d+$/.test(d))
        .map((d) => `/sys/class/drm/${d}/device/gpu_busy_percent`)
        .filter((f) => fs.existsSync(f));
    } catch {
      gpuFiles = [];
    }
  }
  let best = null;
  for (const f of gpuFiles) {
    try {
      const v = parseInt(fs.readFileSync(f, "utf8").trim(), 10);
      if (Number.isFinite(v)) best = Math.max(best ?? 0, v);
    } catch { /* card gone */ }
  }
  return best;
}

async function sample() {
  const totalMb = Math.round(os.totalmem() / 1024 / 1024);
  const freeMb = Math.round(os.freemem() / 1024 / 1024);
  const [engines, macUsed] = await Promise.all([
    Promise.all(
      engineProcesses().map(async (e) => {
        const stat = await psStat(e.pid);
        return { port: e.port, pid: e.pid, model: e.model, tag: e.tag, cpuPct: stat?.cpuPct ?? null, rssMb: stat?.rssMb ?? null };
      }),
    ),
    macUsedMemMb(),
  ]);
  latest = {
    ts: Date.now(),
    cpuPct: proxyCpuPct(),
    sysCpuPct: systemCpuPct(),
    loadAvg1: Math.round(os.loadavg()[0] * 100) / 100,
    cpus: os.cpus().length,
    // On macOS use vm_stat (reclaimable pages excluded); elsewhere
    // total-free is a reasonable approximation (Linux free includes little).
    memUsedMb: macUsed ?? totalMb - freeMb,
    memTotalMb: totalMb,
    rssMb: Math.round(process.memoryUsage().rss / 1024 / 1024),
    gpuPct: gpuPct(),
    engines,
  };
  samples += 1;
  if (samples % PUBLISH_EVERY === 0) {
    obsBus.metric("proxy", "system.stats", latest.cpuPct, { attrs: { ...latest } });
  }
}

/** Latest snapshot (null until the first sample lands). */
export function getSysStats() {
  return latest;
}

/** Starts the sampler (idempotent). */
export function startSysStats() {
  if (timer) return;
  timer = setInterval(() => {
    sample().catch(() => { /* stats never break the proxy */ });
  }, SAMPLE_MS);
  timer.unref();
  void sample();
}
