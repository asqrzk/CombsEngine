/** sand log — a trace file, one line per event, totals at the end. */
import { traceLines } from "../mod.ts";

export async function logCommand(path: string): Promise<number> {
  const text = await Deno.readTextFile(path);
  const events = text.split("\n").filter((l) => l.trim()).map((l) => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean);
  const byTrace = new Map<string, any[]>();
  for (const e of events) { const k = e.traceId ?? "-"; if (!byTrace.has(k)) byTrace.set(k, []); byTrace.get(k)!.push(e); }
  let passes = 0, tokens = 0, requests = 0, decisions = 0;
  for (const [id, evs] of byTrace) {
    console.log(`── ${id} (${evs.length} events)`);
    for (const line of traceLines(evs)) console.log("  " + line);
    for (const e of evs) {
      if (e.name === "sand.request" && e.attrs?.status === "ok") { requests++; passes += Number(e.attrs?.passes ?? 0); tokens += Number(e.attrs?.tokens ?? 0); }
      if (e.name === "sand.decision") decisions++;
    }
  }
  console.log(`── ${byTrace.size} run${byTrace.size === 1 ? "" : "s"} · ${requests} request${requests === 1 ? "" : "s"} · ${passes} pass${passes === 1 ? "" : "es"} · ${tokens} tokens · ${decisions} decision${decisions === 1 ? "" : "s"}`);
  return 0;
}
