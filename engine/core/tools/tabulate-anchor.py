#!/usr/bin/env python3
"""tabulate-anchor.py — read a sitting's own JSON and print its cells.

A sitting directory holds two kinds of JSON: llama-bench's (a LIST of
cell objects, each carrying `samples_ns`) and combs-kit's (an OBJECT
whose `schema` starts `combs-kit`). This reads both and prints one
table, so a figure in a `table.md` can be re-derived from the record
rather than copied from a console line.

Why it exists. §127, §128 and §130 say "medians of three" in their text,
and the combs arm is exactly that — three rounds, median taken. The
llama.cpp arm is not: `llama-bench` was run WITHOUT `-r`, so it took its
own default of FIVE repetitions per cell and reported `avg_ts`, which is
a MEAN. The ratios in those tables sit inside the round-to-round spread
either way, but the tables name a statistic they did not compute. This
tool computes both and says which it used:

    --stat median   n * 1e9 / median(samples_ns)      [the default]
    --stat mean     llama-bench's own `avg_ts`        [what was printed]

`n_runs` (the length of `samples_ns`) is printed beside every llama cell,
which is the proof of how many repetitions actually ran — the field the
sealed records carried all along and nobody read.

Rounds. A file whose stem ends `-r<N>` or `_r<N>` is one round of an arm;
the arm is the rest of the stem. Cells are medianed ACROSS rounds, never
within them. A `warm*` file is the discarded warm round and is skipped
unless --include-warm; a file with no round marker (a drift check, a
one-off) is printed on its own and never folded into a median.

Rates. Where a kit run records `packed_device_bytes`, a decode GB/s is
derived over DEVICE bytes — what the repack actually placed on the card,
not what the GGUF file holds. The two differ: Q8_0's block scale is
widened f16 to f32 at repack (36 B per 32 weights against the file's 34,
+5.88%) and Q4_K's superblock is 148 B against 144 (+2.78%). A GB/s over
file bytes therefore reads low — by 5.56% on Q8_0, 2.70% on Q4_K — which
is how §130's residual was read before §135 restated it.

Usage:
    tabulate-anchor.py <sitting-dir> [...] [--stat median|mean]
                       [--include-warm] [--json] [--selftest]
"""

import argparse
import json
import os
import re
import statistics
import sys

ROUND = re.compile(r"[-_]r(\d+)$")


# --- reading -----------------------------------------------------------


def classify(path):
    """('llama', doc) | ('kit', doc) | ('other', None) — by CONTENT.

    Never by filename: a record's naming is the runner's habit, and a
    habit is not a guarantee.
    """
    try:
        with open(path) as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return "other", None
    if isinstance(doc, list) and doc and isinstance(doc[0], dict) and "samples_ns" in doc[0]:
        return "llama", doc
    if isinstance(doc, dict) and str(doc.get("schema", "")).startswith("combs-kit"):
        return "kit", doc
    return "other", None


def arm_and_round(stem):
    """`kit16-qwen3-4b-gguf-r2` -> ('kit16-qwen3-4b-gguf', 2); no marker -> (stem, None)."""
    m = ROUND.search(stem)
    if not m:
        return stem, None
    return stem[: m.start()], int(m.group(1))


def llama_cells(doc, stat):
    """One (cell, tok/s, n_runs, meta) per entry in a llama-bench JSON."""
    out = []
    for e in doc:
        samples = e.get("samples_ns") or []
        n_gen, n_prompt = e.get("n_gen", 0), e.get("n_prompt", 0)
        depth = e.get("n_depth", 0)
        if n_gen:
            cell, n = f"decode n{n_gen}@d{depth}", n_gen
        elif n_prompt:
            cell, n = f"prefill p{n_prompt}@d{depth}", n_prompt
        else:
            continue
        if stat == "mean":
            # llama-bench's own figure, so a record printed from the
            # console reproduces to the digit: the arithmetic mean of
            # samples_ts, which is NOT n/mean(samples_ns).
            ts = e.get("avg_ts")
            if ts is None:
                ts = statistics.fmean(n * 1e9 / s for s in samples) if samples else 0.0
        else:
            ts = n * 1e9 / statistics.median(samples) if samples else 0.0
        out.append(
            (
                cell,
                ts,
                len(samples),
                {
                    "model": os.path.basename(os.path.dirname(e.get("model_filename", "") or "")) or "?",
                    "build": e.get("build_commit", "?"),
                },
            )
        )
    return out


KIT_VALUES = (
    "prefill_tok_s",
    "decode_tok_s",
    "ttft_ms",
    "dispatches_per_token",
    "submits_per_token",
    "decode_tokens_timed",
    "packed_device_bytes",
    "pool_in_use_mb",
)


def kit_cells(doc):
    """One (cell, {name: value}, meta) per run in a combs-kit record."""
    out = []
    for run in doc.get("runs", []):
        vals = {v["name"]: v["value"] for v in run.get("values", []) if "name" in v}
        cfg = run.get("config") or {}
        out.append(
            (
                run.get("id", "?"),
                {k: vals[k] for k in KIT_VALUES if k in vals},
                {
                    "model": run.get("subject", "?"),
                    "build": (doc.get("build") or {}).get("commit", "?"),
                    "dtype": (doc.get("build") or {}).get("dtype", "?"),
                    "fusion": (doc.get("build") or {}).get("fusion", "?"),
                    "arena": cfg.get("arena"),
                    "matmul": cfg.get("matmul"),
                },
            )
        )
    return out


def gather(dirs, stat, include_warm):
    """{'llama': {...}, 'kit': {...}, 'loose': [...]} keyed (arm, cell)."""
    llama, kit, loose, skipped = {}, {}, [], []
    for d in dirs:
        for name in sorted(os.listdir(d)):
            if not name.endswith(".json"):
                continue
            stem = name[: -len(".json")]
            kind, doc = classify(os.path.join(d, name))
            if kind == "other":
                continue
            arm, rnd = arm_and_round(stem)
            if arm.startswith("warm") and not include_warm:
                skipped.append(name)
                continue
            if kind == "llama":
                for cell, ts, n_runs, meta in llama_cells(doc, stat):
                    rec = {"file": name, "round": rnd, "tok_s": ts, "n_runs": n_runs, **meta}
                    (loose if rnd is None else llama.setdefault((arm, cell), [])).append(rec)
            else:
                for cell, vals, meta in kit_cells(doc):
                    rec = {"file": name, "round": rnd, "values": vals, **meta}
                    (loose if rnd is None else kit.setdefault((arm, cell), [])).append(rec)
    return {"llama": llama, "kit": kit, "loose": loose, "skipped": skipped}


# --- reporting ---------------------------------------------------------


def spread(xs):
    """Peak-to-peak as a percentage of the median — the honesty column."""
    if len(xs) < 2:
        return None
    med = statistics.median(xs)
    return 0.0 if med == 0 else (max(xs) - min(xs)) / med * 100.0


def fmt(x, places=2):
    return "—" if x is None else f"{x:,.{places}f}"


def report(g, stat, out=sys.stdout):
    p = lambda s="": print(s, file=out)
    if g["llama"]:
        p(f"## llama.cpp cells — {stat} of samples within a round, median across rounds")
        p()
        p("| arm | cell | model | tok/s | rounds | n_runs/round | spread |")
        p("|---|---|---|---|---|---|---|")
        for (arm, cell), recs in sorted(g["llama"].items()):
            xs = [r["tok_s"] for r in recs]
            runs = sorted({r["n_runs"] for r in recs})
            p(
                f"| {arm} | {cell} | {recs[0]['model']} | {fmt(statistics.median(xs))} "
                f"| {len(xs)} | {','.join(str(n) for n in runs)} | {fmt(spread(xs))}% |"
            )
        p()
        stated = {n for recs in g["llama"].values() for r in recs for n in (r["n_runs"],)}
        if stated - {3}:
            p(
                f"**`n_runs` per cell: {sorted(stated)}.** llama-bench's default is 5; a record "
                "whose text says \"medians of three\" is describing the combs arm only."
            )
            p()
    if g["kit"]:
        p("## combs-kit cells — median across rounds")
        p()
        p("| arm | cell | model | prefill | decode | ttft ms | disp/tok | timed | dev GB/s | spread(dec) |")
        p("|---|---|---|---|---|---|---|---|---|---|")
        for (arm, cell), recs in sorted(g["kit"].items()):
            col = lambda k: [r["values"][k] for r in recs if k in r["values"]]
            med = lambda k: statistics.median(col(k)) if col(k) else None
            dec = med("decode_tok_s")
            dev = med("packed_device_bytes")
            gbs = dec * dev / 1e9 if (dec and dev) else None
            timed = med("decode_tokens_timed")
            p(
                f"| {arm} | {cell} | {recs[0]['model']} | {fmt(med('prefill_tok_s'))} | {fmt(dec)} "
                f"| {fmt(med('ttft_ms'), 0)} | {fmt(med('dispatches_per_token'), 1)} "
                f"| {fmt(timed, 0)} | {fmt(gbs)} | {fmt(spread(col('decode_tok_s')))}% |"
            )
        p()
        builds = {(r["build"], r["dtype"], r["fusion"]) for recs in g["kit"].values() for r in recs}
        for b, dt, fu in sorted(builds):
            p(f"- build `{b}` · dtype {dt} · fusion {fu}")
        mm = {r["matmul"] for recs in g["kit"].values() for r in recs}
        for m in sorted(x for x in mm if x):
            p(f"- matmul: {m}")
        if not any(mm):
            p(
                "- matmul: **not recorded** — this record predates the `config.matmul` stamp, "
                "so the effective prefill operand cannot be read back from it."
            )
        if not any("decode_tokens_timed" in r["values"] for recs in g["kit"].values() for r in recs):
            p(
                "- decode steps timed: **not recorded** — `decode_tok_s` is "
                "(generated − 1)/decode_time, so `--tokens 16` times 15 steps against "
                "llama-bench's `-n 16`."
            )
        p()
    if g["loose"]:
        p("## files with no round marker (never folded into a median)")
        p()
        for r in g["loose"]:
            if "tok_s" in r:
                p(f"- `{r['file']}` — {fmt(r['tok_s'])} tok/s over {r['n_runs']} runs")
            else:
                v = r["values"]
                p(
                    f"- `{r['file']}` — prefill {fmt(v.get('prefill_tok_s'))} · "
                    f"decode {fmt(v.get('decode_tok_s'))}"
                )
        p()
    if g["skipped"]:
        p(f"_warm files skipped: {', '.join(g['skipped'])}_")


# --- the red -----------------------------------------------------------


def selftest():
    """Synthetic records, checked against hand arithmetic.

    Committed rather than pointed at a sealed sitting because records
    live outside the repository; the real s128 run is in the ledger.
    """
    import tempfile

    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got == want or (
            isinstance(got, float) and isinstance(want, float) and abs(got - want) < 1e-9
        )
        ok = ok and good
        print(f"{'ok  ' if good else 'FAIL'} {name}: {got!r}" + ("" if good else f" != {want!r}"))

    # samples chosen so mean and median differ visibly: one slow run.
    lb = [
        {
            "samples_ns": [100_000_000, 100_000_000, 100_000_000, 100_000_000, 200_000_000],
            "samples_ts": [160.0, 160.0, 160.0, 160.0, 80.0],
            "avg_ts": 144.0,
            "n_gen": 16,
            "n_prompt": 0,
            "n_depth": 0,
            "model_filename": "/m/seat/model.gguf",
            "build_commit": "abc1234",
        }
    ]
    check("median reads the median sample", llama_cells(lb, "median")[0][1], 160.0)
    check("mean reads llama-bench's own avg_ts", llama_cells(lb, "mean")[0][1], 144.0)
    check("n_runs is the sample count", llama_cells(lb, "median")[0][2], 5)
    check("the cell names its shape", llama_cells(lb, "median")[0][0], "decode n16@d0")

    check("a round marker splits arm from round", arm_and_round("kit16-seat-r2"), ("kit16-seat", 2))
    check("no marker means no round", arm_and_round("dense-drift"), ("dense-drift", None))
    check("an underscore marker counts too", arm_and_round("f16op_r11"), ("f16op", 11))
    check("a stem that merely ends in r is untouched", arm_and_round("anchor")[1], None)

    def kit_doc(dec, timed=None, dev=None):
        vals = [{"name": "decode_tok_s", "value": dec}, {"name": "prefill_tok_s", "value": 10.0}]
        if timed is not None:
            vals.append({"name": "decode_tokens_timed", "value": timed})
        if dev is not None:
            vals.append({"name": "packed_device_bytes", "value": dev})
        return {
            "schema": "combs-kit.v1",
            "build": {"commit": "abc1234", "dtype": "f16", "fusion": False},
            "runs": [{"id": "d4", "subject": "seat", "config": {"arena": 16384}, "values": vals}],
        }

    with tempfile.TemporaryDirectory() as d:
        for i, dec in enumerate([30.0, 31.0, 90.0], start=1):
            with open(os.path.join(d, f"kit-r{i}.json"), "w") as fh:
                json.dump(kit_doc(dec, timed=16, dev=2_000_000_000), fh)
        with open(os.path.join(d, "warm-kit.json"), "w") as fh:
            json.dump(kit_doc(1.0), fh)
        with open(os.path.join(d, "llama-r1.json"), "w") as fh:
            json.dump(lb, fh)
        with open(os.path.join(d, "notes.json"), "w") as fh:
            json.dump({"unrelated": True}, fh)
        g = gather([d], "median", include_warm=False)
        recs = g["kit"][("kit", "d4")]
        check("warm is skipped", g["skipped"], ["warm-kit.json"])
        check("three rounds gathered", len(recs), 3)
        check(
            "the median survives one wild round",
            statistics.median([r["values"]["decode_tok_s"] for r in recs]),
            31.0,
        )
        check("a foreign json is ignored", ("notes", "d4") in g["kit"], False)
        check("the llama arm is separate", list(g["llama"]), [("llama", "decode n16@d0")])
        # 31 tok/s over 2 GB of device weights = 62 GB/s
        dev = statistics.median([r["values"]["packed_device_bytes"] for r in recs])
        check("device GB/s is the rate over packed bytes", 31.0 * dev / 1e9, 62.0)

        g2 = gather([d], "median", include_warm=True)
        check("warm is kept when asked", len(g2["loose"]), 1)
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*", help="sitting directories")
    ap.add_argument("--stat", choices=("median", "mean"), default="median",
                    help="the statistic WITHIN a llama-bench cell (default: median)")
    ap.add_argument("--include-warm", action="store_true", help="fold the discarded warm round in too")
    ap.add_argument("--json", action="store_true", help="emit the gathered cells instead of a table")
    ap.add_argument("--selftest", action="store_true", help="run the arithmetic red and exit")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.dirs:
        ap.error("name at least one sitting directory (or --selftest)")
    for d in a.dirs:
        if not os.path.isdir(d):
            print(f"not a directory: {d}", file=sys.stderr)
            return 2
    g = gather(a.dirs, a.stat, a.include_warm)
    if not g["llama"] and not g["kit"] and not g["loose"]:
        print("no llama-bench or combs-kit JSON found", file=sys.stderr)
        return 3
    if a.json:
        json.dump({"stat": a.stat,
                   "llama": {f"{k[0]}|{k[1]}": v for k, v in g["llama"].items()},
                   "kit": {f"{k[0]}|{k[1]}": v for k, v in g["kit"].items()},
                   "loose": g["loose"]}, sys.stdout, indent=2)
        print()
    else:
        report(g, a.stat)
    return 0


if __name__ == "__main__":
    sys.exit(main())
