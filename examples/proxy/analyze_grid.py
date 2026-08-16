#!/usr/bin/env python3
"""analyze_grid.py - what does the joint configuration space look like?

Reads the GRID CSV written by run_grid.sh and answers, for the measured
(bytes, K, D) workload points:

  1. how much is on the table -- the best config per point against the best
     single global config;
  2. what each knob axis is worth on its own, i.e. how much is lost by
     freezing it at its globally best value;
  3. how the winner moves across the space, and what the worst case of a
     fixed choice is.

Costs are per-point REGRET (policy cost / oracle cost), summarised by the
geometric mean. Totals in microseconds are dominated by the largest points
and say more about which points were measured than about the policy.

Usage:  ./analyze_grid.py docs/experiments/grid/grid.csv
"""
import csv
import math
import sys
from collections import defaultdict

COLS = ["path", "config", "bytes", "K", "D", "B", "P", "L",
        "samples", "median", "min", "max", "p25", "p75"]


def load(path):
    """-> (cells, spin, knobs) with cells[(bytes,K,D)][config] = median us."""
    cells = defaultdict(dict)
    spin = {}
    knobs = {}
    with open(path) as f:
        for row in csv.reader(f):
            if not row or row[0] != "GRID" or row[1] == "path":
                continue
            r = dict(zip(COLS, row[1:]))
            med, D = float(r["median"]), int(r["D"])
            if r["config"] == "spin-only":
                spin[D] = min(spin.get(D, med), med)
                continue
            pt, cfg = (int(r["bytes"]), int(r["K"]), D), r["config"]
            cells[pt][cfg] = min(cells[pt].get(cfg, med), med)
            knobs[cfg] = (r["path"], int(r["B"]), int(r["P"]), int(r["L"]))
    return cells, spin, knobs


def exposed(cells, spin):
    """Communication cost with the phase's independent compute subtracted."""
    out = {}
    for (b, K, D), cfgs in cells.items():
        floor = spin.get(D, 0.0)
        out[(b, K, D)] = {c: max(v - floor, 0.01) for c, v in cfgs.items()}
    return out


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def score(pts, choose):
    """Per-point regret of a policy. choose(pt, cfgs) -> config name."""
    reg, miss = [], 0
    for pt, cfgs in pts.items():
        best = min(cfgs.values())
        pick = choose(pt, cfgs)
        reg.append(cfgs[pick] / best)
        if cfgs[pick] > best * 1.005:
            miss += 1
    reg.sort()
    return {"gmean": gmean(reg), "mean": sum(reg) / len(reg),
            "p95": reg[int(0.95 * (len(reg) - 1))], "max": reg[-1],
            "miss": miss, "n": len(reg)}


def score_against(sub, full, choose):
    """Regret of a policy restricted to `sub`, measured against `full`'s oracle."""
    reg, miss = [], 0
    for pt, cfgs in sub.items():
        best = min(full[pt].values())
        pick = choose(pt, cfgs)
        reg.append(cfgs[pick] / best)
        if cfgs[pick] > best * 1.005:
            miss += 1
    reg.sort()
    return {"gmean": gmean(reg), "mean": sum(reg) / len(reg),
            "p95": reg[int(0.95 * (len(reg) - 1))], "max": reg[-1],
            "miss": miss, "n": len(reg)}


def best_fixed_over(sub):
    """Choose the single config, present everywhere in `sub`, with least regret."""
    univ = [c for c in {c for v in sub.values() for c in v}
            if all(c in v for v in sub.values())]
    if not univ:
        return lambda pt, cf: min(cf, key=cf.get)
    tot = {c: sum(v[c] / min(v.values()) for v in sub.values()) for c in univ}
    win = min(tot, key=tot.get)
    return lambda pt, cf: win


def row(name, s):
    print(f"{name:<46} {s['gmean']:>8.3f}x {s['mean']:>8.3f}x "
          f"{s['p95']:>8.2f}x {s['max']:>8.2f}x {s['miss']:>5d}/{s['n']}")


def header():
    print(f"{'policy':<46} {'gmean':>9} {'mean':>9} {'p95':>9} "
          f"{'worst':>9} {'mispicks':>10}")


def main(path):
    cells, spin, knobs = load(path)
    pts = exposed(cells, spin)
    if not pts:
        print("no GRID rows found")
        return 1

    all_cfgs = sorted({c for cfgs in pts.values() for c in cfgs})
    # A configuration is only a fair candidate for a fixed global policy if it
    # was measured at every point.
    eligible = [c for c in all_cfgs if all(c in cfgs for cfgs in pts.values())]
    print(f"{len(pts)} workload points x {len(all_cfgs)} configurations "
          f"= {sum(len(c) for c in pts.values())} measured cells "
          f"({len(eligible)} configs present at every point)")
    print("spin-only floors: " +
          ", ".join(f"D={d}:{v:.1f}us" for d, v in sorted(spin.items())))
    print()

    def fixed(c):
        return lambda pt, cfgs: c
    best_glob = min(eligible, key=lambda c: score(pts, fixed(c))["gmean"])

    # ---- 1. how much is on the table --------------------------------------
    print("=== 1. how much is on the table (per-point regret vs oracle) ===")
    header()
    row(f"worst single global config", score(pts, fixed(
        max(eligible, key=lambda c: score(pts, fixed(c))["gmean"]))))
    row(f"best single global config ({best_glob})", score(pts, fixed(best_glob)))

    # Best path per point, each path's knobs frozen at its global best. This
    # is exactly the decision the 4-parameter rule already makes.
    def best_frozen(prefix):
        c = [x for x in eligible if x.startswith(prefix)]
        return min(c, key=lambda x: score(pts, fixed(x))["gmean"]) if c else None
    ft, fp = best_frozen("trig-"), best_frozen("proxy-")
    if ft and fp:
        row(f"best path per point, knobs frozen ({ft}/{fp})",
            score(pts, lambda pt, c: ft if c[ft] < c[fp] else fp))
    row("per-point oracle over the joint space",
        score(pts, lambda pt, c: min(c, key=c.get)))
    print()

    # ---- 1b. who can actually set each knob -------------------------------
    # The knobs are not equally reachable:
    #   L  is process-global -- GICC_NUM_PROXY_THREADS is read once in the
    #      Runtime constructor, so one program has ONE fleet size. A runtime
    #      or a deployment script can set it; nobody can vary it per site.
    #   path, B, P are per-call-site CODE GENERATION choices. P in particular
    #      -- how many blocks issue concurrently -- is baked into the kernel's
    #      machine code, so no runtime can change it at any price.
    # So the honest ladder holds L global and asks what per-site choice buys.
    lanes = sorted({knobs[c][3] for c in all_cfgs if knobs[c][0] == "proxy"})

    def under_L(L, cfgs):
        """Configs reachable when the process fleet size is L."""
        return {c: v for c, v in cfgs.items()
                if knobs[c][0] == "trigger" or knobs[c][3] == L}

    def eval_global_L(pick, allow):
        """Best achievable over the choice of one global L."""
        best = None
        for L in lanes:
            sub = {pt: under_L(L, cfgs) for pt, cfgs in pts.items()}
            sub = {pt: {c: v for c, v in cfgs.items() if allow(c)}
                   for pt, cfgs in sub.items()}
            if any(not c for c in sub.values()):
                continue
            s = score_against(sub, pts, pick(sub))
            if best is None or s["gmean"] < best[1]["gmean"]:
                best = (L, s)
        return best

    print("=== 1b. per-site code generation vs one global knob ===")
    print("    (regret is still against the full per-point oracle)")
    header()
    # A: what a context-free lowering emits today -- one lead thread issues
    #    (P<=1), one fixed configuration for the whole program.
    a = eval_global_L(lambda sub: best_fixed_over(sub),
                      lambda c: knobs[c][2] <= 1)
    if a:
        row(f"A. context-free lowering, P=1, one config (L={a[0]})", a[1])
    # B: a runtime free to pick any single fixed configuration, including a
    #    parallel-issue one it has no way to actually emit. Generous.
    b = eval_global_L(lambda sub: best_fixed_over(sub), lambda c: True)
    if b:
        row(f"B. best single fixed config, any P (L={b[0]})", b[1])
    # C: the compiler picks path/B/P per call site; L still global.
    c = eval_global_L(lambda sub: (lambda pt, cf: min(cf, key=cf.get)),
                      lambda c: True)
    if c:
        row(f"C. per-site path/B/P, one global L (L={c[0]})", c[1])
    row("D. per-point oracle incl. L (upper bound only)",
        score(pts, lambda pt, cf: min(cf, key=cf.get)))
    if a and c:
        print(f"\n  per-site code generation is worth "
              f"{a[1]['gmean'] / c[1]['gmean']:.3f}x over what is emitted "
              f"today, and {b[1]['gmean'] / c[1]['gmean']:.3f}x over the best "
              f"fixed configuration any tuner could pick.")
    print()

    # ---- 2. what is each knob axis worth ----------------------------------
    # The space is CONDITIONAL: B exists only on the trigger path, P and L
    # only on the proxy path, and some knob values are illegal at some
    # points (B must divide K, P must not exceed K). So each axis is scored
    # inside the sub-space where it is defined, over whichever of its values
    # are legal at each point -- not against a universally-present subset,
    # which would silently drop the very configurations that only exist
    # where they are legal.
    print("=== 2. what each knob axis is worth (inside its own sub-space) ===")
    for sub, prefix, sub_axes in [
            ("proxy", "proxy-", {"P (concurrent issuing blocks)": 2,
                                 "L (proxy worker lanes)": 3}),
            ("trigger", "trig-", {"B (descriptors per trigger)": 1})]:
        in_sub = {pt: {c: v for c, v in cfgs.items() if c.startswith(prefix)}
                  for pt, cfgs in pts.items()}
        in_sub = {pt: c for pt, c in in_sub.items() if c}
        if not in_sub:
            continue
        print(f"-- {sub} sub-space ({len(in_sub)} points, "
              f"{len({c for v in in_sub.values() for c in v})} configs)")
        header()
        row(f"oracle over all {sub} knobs",
            score(in_sub, lambda pt, c: min(c, key=c.get)))
        for label, idx in sub_axes.items():
            vals = sorted({knobs[c][idx] for v in in_sub.values() for c in v})
            if len(vals) < 2:
                continue

            def frozen_at(v, idx=idx):
                def ch(pt, cfgs):
                    cand = [c for c in cfgs if knobs[c][idx] == v]
                    return min(cand, key=cfgs.get) if cand else min(cfgs, key=cfgs.get)
                return ch
            best_v = min(vals, key=lambda v: score(in_sub, frozen_at(v))["gmean"])
            row(f"  {label} frozen at {best_v}", score(in_sub, frozen_at(best_v)))
        # everything frozen: the best fixed configuration in this sub-space
        univ = [c for c in {c for v in in_sub.values() for c in v}
                if all(c in v for v in in_sub.values())]
        if univ:
            bf = min(univ, key=lambda c: score(in_sub, fixed(c))["gmean"])
            row(f"  all knobs frozen ({bf})", score(in_sub, fixed(bf)))
        print()

    # ---- 3. where the winner moves ----------------------------------------
    wins = defaultdict(int)
    for pt, cfgs in pts.items():
        wins[min(cfgs, key=cfgs.get)] += 1
    print("=== 3. distinct winners across the space ===")
    for c, n in sorted(wins.items(), key=lambda kv: -kv[1]):
        print(f"    {c:<16} wins at {n:>3} / {len(pts)} points")
    print()

    print("=== 4. worst points for the best single global config ===")
    worst = sorted(pts, key=lambda p: -(pts[p][best_glob] / min(pts[p].values())))
    for pt in worst[:8]:
        c = pts[pt]
        print(f"  bytes={pt[0]:>7} K={pt[1]:>3} D={pt[2]:>4}: "
              f"{best_glob} {c[best_glob]:8.1f} us vs "
              f"{min(c, key=c.get)} {min(c.values()):8.1f} us "
              f"= {c[best_glob] / min(c.values()):.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else "docs/experiments/grid/grid.csv"))
