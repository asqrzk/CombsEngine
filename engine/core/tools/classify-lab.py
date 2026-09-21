#!/usr/bin/env python3
"""classify-lab.py — re-derive a classify record's every figure from its own cases.

A SUMMARY.json is a claim. This re-computes each of its numbers from the
per-case records beside it and refuses any that does not reproduce, the
same way tools/tabulate-anchor.py re-derives a sitting's cells from its
own JSON rather than trusting a table somebody typed.

Four things it does, each answerable from a record alone:

  verify <dir>            every SUMMARY figure, re-derived from the cases
  parity <dir>            every distribution re-derived from the wire's own
                          `similarities` at the answer's own temperature
  ece    <dir> --fixture  ECE three ways, with bin counts, and a bar only
                          where every populated bin carries >= 5 items
  auroc  <dir> --fixture --question <name>
                          separation by rank on one noul's p(true), split
                          into the items that ask and the items that only
                          mention

Never talks to a server: a record that needs the seat back to be read is
not a record. Stdlib only.

  classify-lab.py --selftest     the checks, shown to catch what they claim
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

# The near-miss kinds, mirroring combs-classify/src/suite.rs::NEAR_MISS_KINDS
# and classify_automator.py. Three copies is two too many, and the guard
# against them drifting is that all three are exercised on the same
# fixtures by the reds that read them.
NEAR_MISS_PREFIXES = ("contrast-of:", "translation-of:", "definition-of:", "past-of:", "negation-of:")


def is_near_miss(item: dict) -> bool:
    return any(item["kind"].startswith(p) for p in NEAR_MISS_PREFIXES)


def near_miss_mode(item: dict) -> str | None:
    return item["kind"].split("-of:", 1)[0] if is_near_miss(item) else None


def forbidden(item: dict) -> tuple[str, str] | None:
    if item.get("never"):
        return item["never"]["question"], item["never"]["option"]
    if is_near_miss(item) and not item["expect"]["fires"]:
        return item["expect"]["question"], item["expect"]["option"]
    return None


# --- reading a record --------------------------------------------------------

def load_record(d: Path) -> tuple[dict, dict]:
    """(SUMMARY, {item id: the raw wire answer}). Every JSON but SUMMARY is a case."""
    summary_path = d / "SUMMARY.json"
    if not summary_path.exists():
        raise SystemExit(f"{d}: no SUMMARY.json — is this a suite record?")
    summary = json.loads(summary_path.read_text())
    cases = {}
    for f in sorted(d.glob("*.json")):
        if f.stem in ("SUMMARY", "SWEEP-SUMMARY", "MANIFEST"):
            continue
        cases[f.stem] = json.loads(f.read_text())
    if not cases:
        raise SystemExit(f"{d}: a SUMMARY with no cases under it proves nothing")
    return summary, cases


def load_fixture(path: Path) -> tuple[dict, list[dict]]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    header = next((r for r in rows if r.get("kind") == "header"), None)
    if header is None:
        raise SystemExit(f"{path}: no header line")
    return header, [r for r in rows if r.get("kind") != "header"]


def answers_of(case: dict) -> dict[str, dict]:
    return {a["name"]: a for a in case.get("answers", [])}


# --- the arithmetic, written once --------------------------------------------

def softmax(xs: list[float], t: float) -> list[float]:
    z = [x / t for x in xs]
    top = max(z)
    e = [math.exp(v - top) for v in z]
    s = sum(e)
    return [v / s for v in e]


def ece_equal_width(pairs: list[tuple[float, bool]], bins: int) -> tuple[float, list[int]]:
    """The classic: bins of equal CONFIDENCE width, most of them usually empty."""
    counts = [0] * bins
    acc = [0] * bins
    conf = [0.0] * bins
    for c, hit in pairs:
        b = min(bins - 1, max(0, int(c * bins)))
        counts[b] += 1
        acc[b] += 1 if hit else 0
        conf[b] += c
    n = len(pairs)
    e = 0.0
    for b in range(bins):
        if counts[b]:
            e += (counts[b] / n) * abs(conf[b] / counts[b] - acc[b] / counts[b])
    return e, counts


def ece_equal_mass(pairs: list[tuple[float, bool]], bins: int) -> tuple[float, list[int]]:
    """Bins of equal COUNT. With 24 items the equal-width form puts nearly
    everything in one bin and reports that bin's error as the suite's;
    this one cannot."""
    if not pairs:
        return float("nan"), []
    ordered = sorted(pairs, key=lambda p: p[0])
    n = len(ordered)
    edges = [round(n * i / bins) for i in range(bins + 1)]
    counts, e = [], 0.0
    for i in range(bins):
        chunk = ordered[edges[i]:edges[i + 1]]
        counts.append(len(chunk))
        if chunk:
            c = sum(x for x, _ in chunk) / len(chunk)
            a = sum(1 for _, h in chunk if h) / len(chunk)
            e += (len(chunk) / n) * abs(c - a)
    return e, counts


def auroc(pos: list[float], neg: list[float]) -> float | None:
    """By RANK (Mann-Whitney U), so ties count a half and no threshold is
    chosen. None when either side is empty — an AUROC of one class is not
    0.5, it is undefined, and printing 0.5 would read as chance."""
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if p > q else (0.5 if p == q else 0.0)
    return wins / (len(pos) * len(neg))


# --- the subcommands ---------------------------------------------------------

def cmd_verify(args) -> int:
    d = Path(args.dir)
    summary, cases = load_record(d)
    by_id = {iid: answers_of(c) for iid, c in cases.items()}
    fixture = Path(args.fixture) if args.fixture else None
    if fixture is None:
        name = summary.get("fixture")
        if not name:
            raise SystemExit("this SUMMARY does not name its fixture; pass --fixture")
        fixture = Path(name)
    header, items = load_fixture(fixture)

    bad = []

    def check(what: str, got, want) -> None:
        same = got == want if not isinstance(want, float) else abs(got - want) < 5e-4
        print(f"  {'ok ' if same else 'NO '} {what:<28} re-derived {got}   claimed {want}")
        if not same:
            bad.append(what)

    fired = []
    for it in items:
        pair = forbidden(it)
        if pair is None:
            continue
        q, opt = pair
        a = by_id.get(it["id"], {}).get(q)
        if a and a["top"] == opt and a["decision"] == "act":
            fired.append(it["id"])
    c = summary.get("contrast", {})
    check("near misses", len([i for i in items if forbidden(i)]), c.get("total"))
    check("near misses held", len([i for i in items if forbidden(i)]) - len(fired), c.get("held"))

    hits = sum(
        1
        for it in items
        if it["kind"] == "base"
        and (by_id.get(it["id"], {}).get(it["expect"]["question"], {}).get("top") == it["expect"]["option"])
    )
    check("top-1 on bases", hits, summary.get("top1_on_bases", {}).get("hits"))

    pairs = []
    for it in items:
        a = by_id.get(it["id"], {}).get(it["expect"]["question"])
        if not a:
            continue
        hit = (a["top"] == it["expect"]["option"]) if it["expect"]["fires"] else (a["top"] != it["expect"]["option"])
        pairs.append((a["confidence"], hit))
    if pairs:
        acc = sum(1 for _, h in pairs if h) / len(pairs)
        brier = sum((c - (1.0 if h else 0.0)) ** 2 for c, h in pairs) / len(pairs)
        cal = summary.get("calibration", {})
        check("accuracy", round(acc, 3), cal.get("accuracy"))
        check("brier", round(brier, 3), cal.get("brier"))

    print(f"\n{len(cases)} cases re-derived from {d}")
    if bad:
        print(f"MISMATCH in {len(bad)}: {', '.join(bad)} — the summary does not follow from its cases")
        return 1
    print("every figure follows from the cases")
    return 0


def cmd_parity(args) -> int:
    """Each distribution is softmax(similarities / T). If it is not, the
    number a caller acts on is not the number the scorer produced."""
    d = Path(args.dir)
    _, cases = load_record(d)
    worst, worst_at, checked, skipped = 0.0, "", 0, 0
    for iid, case in sorted(cases.items()):
        for a in case.get("answers", []):
            sims, dist = a.get("similarities"), a.get("distribution")
            if not sims or not dist:
                skipped += 1
                continue
            keys = list(dist.keys())
            want = softmax([sims[k] for k in keys], a["temperature"])
            for k, w in zip(keys, want):
                gap = abs(dist[k] - w)
                checked += 1
                if gap > worst:
                    worst, worst_at = gap, f"{iid}.{a['name']}.{k}"
    print(f"{checked} probabilities re-derived from their own similarities ({skipped} answers carried none)")
    if not checked:
        print("nothing to check — this record predates `similarities` on the wire")
        return 1
    print(f"worst gap {worst:.2e} at {worst_at}   bound {args.tol:.0e}")
    return 0 if worst <= args.tol else 1


def cmd_ece(args) -> int:
    d = Path(args.dir)
    _, cases = load_record(d)
    header, items = load_fixture(Path(args.fixture))
    by_id = {iid: answers_of(c) for iid, c in cases.items()}
    pairs = []
    for it in items:
        a = by_id.get(it["id"], {}).get(it["expect"]["question"])
        if not a:
            continue
        hit = (a["top"] == it["expect"]["option"]) if it["expect"]["fires"] else (a["top"] != it["expect"]["option"])
        pairs.append((a["confidence"], hit))
    if not pairs:
        print("no scored items")
        return 1

    print(f"{len(pairs)} scored items, accuracy {sum(1 for _, h in pairs if h) / len(pairs):.3f}\n")
    rows = [
        ("equal-width, 10 bins", *ece_equal_width(pairs, 10)),
        ("equal-width, 15 bins", *ece_equal_width(pairs, 15)),
        ("equal-mass, 4 bins", *ece_equal_mass(pairs, 4)),
    ]
    graded = False
    for name, value, counts in rows:
        live = [c for c in counts if c]
        ok = live and min(live) >= 5
        graded = graded or ok
        mark = "  (a bar: every populated bin >= 5)" if ok else "  (reported only: a bin under 5 items)"
        print(f"  {name:<22} ECE {value:.3f}   bins {counts}{mark}")
    if not graded:
        print("\nNo binning leaves every populated bin with 5 items, so no ECE here is a bar.")
        print("It is reported because a calibration figure nobody can reproduce is worse than one nobody grades.")
    return 0


def cmd_auroc(args) -> int:
    d = Path(args.dir)
    _, cases = load_record(d)
    header, items = load_fixture(Path(args.fixture))
    by_id = {iid: answers_of(c) for iid, c in cases.items()}
    q = args.question
    asks, mentions = [], []
    for it in items:
        a = by_id.get(it["id"], {}).get(q)
        if not a:
            continue
        p = a["distribution"].get("true")
        if p is None:
            continue
        (mentions if is_near_miss(it) else asks).append((it["id"], p))
    area = auroc([p for _, p in asks], [p for _, p in mentions])
    for label, rows in (("ASK", asks), ("MENTION", mentions)):
        ps = sorted(p for _, p in rows)
        if ps:
            print(f"  {label:<8} n {len(ps):>2}  min {ps[0]:.3f}  median {statistics.median(ps):.3f}  max {ps[-1]:.3f}")
    if area is None:
        print(f"\nAUROC undefined: one side is empty")
        return 1
    print(f"\n  AUROC (by rank, ties half) {area:.3f}")
    print("  0.5 is a coin; 1.0 separates every ask from every mention.")
    return 0


# --- the checks, shown to catch what they claim ------------------------------

def selftest() -> int:
    fails = []

    def eq(what, got, want):
        if got != want:
            fails.append(f"{what}: {got!r} != {want!r}")

    def close(what, got, want, tol=1e-9):
        if got is None or abs(got - want) > tol:
            fails.append(f"{what}: {got!r} != {want!r}")

    # forbidden() reads both suites' spellings.
    eq("v1 contrast", forbidden({"kind": "contrast-of:x", "expect": {"question": "intent", "option": "a", "fires": False}}), ("intent", "a"))
    eq("v2 never", forbidden({"kind": "translation-of:x", "never": {"question": "intent", "option": "a"},
                               "expect": {"question": "intent", "option": "t", "fires": True}}), ("intent", "a"))
    eq("a base forbids nothing", forbidden({"kind": "base", "expect": {"question": "q", "option": "o", "fires": True}}), None)
    eq("mode", near_miss_mode({"kind": "past-of:bug-b1"}), "past")
    eq("a paraphrase is not a near miss", near_miss_mode({"kind": "paraphrase-of:b1"}), None)

    # softmax at T=1 is the plain one; a larger T flattens.
    p = softmax([1.0, 0.0], 1.0)
    close("softmax sums", sum(p), 1.0)
    close("softmax value", p[0], math.exp(1) / (math.exp(1) + 1))
    flat = softmax([1.0, 0.0], 10.0)
    if not flat[0] < p[0]:
        fails.append("a larger temperature did not flatten")

    # AUROC: perfect separation, perfect inversion, and a full tie.
    close("auroc perfect", auroc([0.9, 0.8], [0.2, 0.1]), 1.0)
    close("auroc inverted", auroc([0.1, 0.2], [0.8, 0.9]), 0.0)
    close("auroc tied", auroc([0.5, 0.5], [0.5, 0.5]), 0.5)
    eq("auroc of one class is undefined", auroc([0.5], []), None)

    # ECE: a perfectly calibrated set is 0; a confidently wrong one is 1.
    good = [(0.5, True), (0.5, False)] * 6
    e, _ = ece_equal_width(good, 10)
    close("ece calibrated", e, 0.0, 1e-12)
    bad = [(1.0, False)] * 8
    e, _ = ece_equal_width(bad, 10)
    close("ece confidently wrong", e, 1.0, 1e-12)
    # Equal-mass splits what equal-width piles into one bin.
    _, counts = ece_equal_mass([(i / 24.0, True) for i in range(24)], 4)
    eq("equal-mass bins", counts, [6, 6, 6, 6])

    if fails:
        print("SELFTEST FAILED")
        for f in fails:
            print("  " + f)
        return 1
    print("selftest: all checks pass")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="classify-lab.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="run the checks against synthetic cases and exit")
    sub = ap.add_subparsers(dest="cmd")

    v = sub.add_parser("verify", help="re-derive every SUMMARY figure from the cases")
    v.add_argument("dir")
    v.add_argument("--fixture", help="the suite (default: whatever SUMMARY names)")
    v.set_defaults(fn=cmd_verify)

    p = sub.add_parser("parity", help="re-derive every distribution from its own similarities")
    p.add_argument("dir")
    p.add_argument("--tol", type=float, default=1e-4)
    p.set_defaults(fn=cmd_parity)

    e = sub.add_parser("ece", help="ECE three ways, with bin counts")
    e.add_argument("dir")
    e.add_argument("--fixture", required=True)
    e.set_defaults(fn=cmd_ece)

    a = sub.add_parser("auroc", help="separation by rank on one noul")
    a.add_argument("dir")
    a.add_argument("--fixture", required=True)
    a.add_argument("--question", default="is_request")
    a.set_defaults(fn=cmd_auroc)

    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.cmd:
        ap.print_help()
        return 2
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
