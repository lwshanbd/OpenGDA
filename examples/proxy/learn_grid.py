#!/usr/bin/env python3
"""learn_grid.py - can a model choose configurations for call sites it has
never measured?

This is the question that separates a learned decider from an autotuner. An
autotuner searches the configuration space of the program in front of it, and
pays measurements every time. A model conditioned on compile-time features
can be trained once and then asked about a call site it has never seen, at
zero measurement cost -- but only if the features actually generalise.

Two experiments, both on the GRID CSV from run_grid.sh:

  A. GENERALISATION. Hold out workload points, train on the rest, and score
     the held-out points against their own oracle. Compares the learned
     models against a best-global policy, hand heuristics, and nearest
     neighbour. Splits can be random or grouped by message size, so the test
     points are sizes the model never saw.

  B. MEASUREMENT BUDGET. At a new point, how close does each method get for
     m measurements? Random search and the learned model's top-m shortlist
     are scored on the same axis, which is the only fair way to compare a
     search against a prediction.

The knob taxonomy matters and is enforced: L (proxy lanes) is process-global,
so every policy here picks ONE L for all points and only varies path/B/P per
site. Anything else would credit the model with a choice the runtime cannot
implement.

Usage:  ./learn_grid.py docs/experiments/grid/grid_big.csv [--group-by=size]
"""
import csv
import math
import sys
from collections import defaultdict

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier

COLS = ["path", "config", "bytes", "K", "D", "B", "P", "L",
        "samples", "median", "min", "max", "p25", "p75"]
SEEDS = [0, 1, 2, 3, 4]


# ---------------------------------------------------------------- data ----
def load(path):
    cells, spin, knobs = defaultdict(dict), {}, {}
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
    pts = {}
    for (b, K, D), cfgs in cells.items():
        floor = spin.get(D, 0.0)
        pts[(b, K, D)] = {c: max(v - floor, 0.01) for c, v in cfgs.items()}
    return pts, knobs


def pick_global_L(pts, knobs):
    """One fleet size for the whole program: the one whose per-site oracle is
    best on average. Every policy below is then restricted to it."""
    lanes = sorted({knobs[c][3] for c in knobs if knobs[c][0] == "proxy"})
    best = None
    for L in lanes:
        r = []
        for pt, cfgs in pts.items():
            sub = {c: v for c, v in cfgs.items()
                   if knobs[c][0] == "trigger" or knobs[c][3] == L}
            if sub:
                r.append(min(sub.values()) / min(cfgs.values()))
        g = math.exp(sum(map(math.log, r)) / len(r))
        if best is None or g < best[1]:
            best = (L, g)
    return best[0]


# ------------------------------------------------------------ features ----
def point_feats(pt, which):
    """which: 'all' | 'runtime' (bytes only) | 'nodistance' (bytes, K)."""
    b, K, D = pt
    f = [math.log2(b), math.log2(K), math.log2(b * K)]
    if which == "runtime":
        return f[:1] + [0.0, 0.0, 0.0]
    if which == "nodistance":
        return f + [0.0]
    return f + [math.log1p(D)]


def cfg_feats(cfg, knobs):
    path, B, P, L = knobs[cfg]
    return [1.0 if path == "proxy" else 0.0,
            math.log2(B) if B > 0 else -1.0,
            math.log2(P) if P > 0 else -1.0,
            math.log2(L) if L > 0 else -1.0]


# ------------------------------------------------------------- scoring ----
def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def regret(pts, test, picks):
    return gmean([pts[pt][picks[pt]] / min(pts[pt].values()) for pt in test])


# ------------------------------------------------------------- policies ---
def pol_best_global(train, test, pts, avail, knobs):
    univ = [c for c in {c for pt in train for c in avail[pt]}
            if all(c in avail[pt] for pt in train)]
    tot = {c: sum(pts[pt][c] / min(avail[pt].values()) for pt in train)
           for c in univ}
    win = min(tot, key=tot.get)
    return {pt: win if win in avail[pt] else min(avail[pt], key=avail[pt].get)
            for pt in test}


def pol_hand(train, test, pts, avail, knobs, rule):
    """Expert heuristics over the knobs, no measurements at the test point."""
    out = {}
    for pt in test:
        b, K, D = pt
        want = rule(b, K, D)
        cand = min(avail[pt],
                   key=lambda c: (abs(knobs[c][2] - want)
                                  if knobs[c][0] == "proxy" else 99,
                                  avail[pt][c] * 0))  # tie-break is arbitrary
        out[pt] = cand
    return out


def pol_knn(train, test, pts, avail, knobs, which="all"):
    tr = np.array([point_feats(p, which) for p in train])
    out = {}
    for pt in test:
        d = np.linalg.norm(tr - np.array(point_feats(pt, which)), axis=1)
        for j in np.argsort(d):
            best = min(avail[train[j]], key=avail[train[j]].get)
            if best in avail[pt]:
                out[pt] = best
                break
        else:
            out[pt] = min(avail[pt], key=avail[pt].get)
    return out


def pol_tree(train, test, pts, avail, knobs, which="all", depth=6):
    X = np.array([point_feats(p, which) for p in train])
    y = np.array([min(avail[p], key=avail[p].get) for p in train])
    clf = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X, y)
    out = {}
    for pt in test:
        proba = clf.predict_proba(np.array([point_feats(pt, which)]))[0]
        for j in np.argsort(-proba):
            c = clf.classes_[j]
            if c in avail[pt]:
                out[pt] = c
                break
        else:
            out[pt] = min(avail[pt], key=avail[pt].get)
    return out


def _cost_model(train, avail, knobs, which):
    X, y = [], []
    for pt in train:
        pf = point_feats(pt, which)
        for c, v in avail[pt].items():
            X.append(pf + cfg_feats(c, knobs))
            y.append(math.log(v))
    return GradientBoostingRegressor(random_state=0, n_estimators=200,
                                     max_depth=3).fit(np.array(X), np.array(y))


def pol_gbt(train, test, pts, avail, knobs, which="all", model=None):
    m = model or _cost_model(train, avail, knobs, which)
    out = {}
    for pt in test:
        pf = point_feats(pt, which)
        cands = list(avail[pt])
        pred = m.predict(np.array([pf + cfg_feats(c, knobs) for c in cands]))
        out[pt] = cands[int(np.argmin(pred))]
    return out


def pol_random_search(train, test, pts, avail, knobs, m, rng):
    """m measurements at the test point itself, then keep the best seen."""
    out = {}
    for pt in test:
        cands = list(avail[pt])
        take = rng.choice(len(cands), size=min(m, len(cands)), replace=False)
        out[pt] = min((cands[i] for i in take), key=lambda c: avail[pt][c])
    return out


def pol_gbt_topm(train, test, pts, avail, knobs, m, model, which="all"):
    """Predict, measure only the m most promising, keep the best."""
    out = {}
    for pt in test:
        cands = list(avail[pt])
        pf = point_feats(pt, which)
        pred = model.predict(np.array([pf + cfg_feats(c, knobs) for c in cands]))
        top = [cands[i] for i in np.argsort(pred)[:m]]
        out[pt] = min(top, key=lambda c: avail[pt][c])
    return out


# ------------------------------------------------------------ transfer ---
# A rule and a model are indistinguishable as long as the platform holds
# still. They come apart when it moves. Regime A is one rank pair on the
# wire; regime B is two pairs contending. Same knobs, same features,
# different physics -- so this asks what each decider is worth when it is
# carried to a regime it was not fitted on, and how fast each recovers when
# allowed a few measurements there.

HAND_FAMILY = [("P = min(K, %d)" % m, (lambda m: lambda b, K, D: min(K, m))(m))
               for m in (1, 2, 4, 8)] + \
              [("P = %d" % c, (lambda c: lambda b, K, D: c)(c))
               for c in (1, 2, 4, 8)]


def fit_hand(points, pts, avail, knobs):
    """Pick the family member with least regret on `points` -- the rule
    recalibration that a two-constant analytic model would amount to."""
    best = None
    for name, rule in HAND_FAMILY:
        picks = pol_hand(None, points, pts, avail, knobs, rule)
        r = regret(pts, points, picks)
        if best is None or r < best[0]:
            best = (r, name, rule)
    return best[1], best[2]


def transfer(srcA, srcB, fixed_L=4):
    ptsA, knobs = load(srcA)
    ptsB, knobsB = load(srcB)
    knobs.update(knobsB)

    def restrict(p):
        return {pt: {c: v for c, v in cfgs.items()
                     if knobs[c][0] == "trigger" or knobs[c][3] == fixed_L}
                for pt, cfgs in p.items()}
    avA, avB = restrict(ptsA), restrict(ptsB)
    common = sorted(set(avA) & set(avB))
    print(f"\n=== C. carrying a decider to a regime it was not fitted on ===")
    print(f"    A = {srcA}")
    print(f"    B = {srcB}")
    print(f"    fleet size held at L={fixed_L} in both, so only the per-site "
          f"knobs vary")
    print(f"    {len(avA)} points in A, {len(avB)} in B, {len(common)} shared\n")

    trainA = sorted(avA)
    hand_name, hand_rule = fit_hand(trainA, ptsA, avA, knobs)
    print(f"    best hand rule fitted on A: {hand_name}")
    hbn, _ = fit_hand(sorted(avB), ptsB, avB, knobs)
    print(f"    best hand rule fitted on B: {hbn}"
          f"{'   (SAME -- the rule transfers)' if hbn == hand_name else '   (DIFFERENT -- the rule does not transfer)'}\n")

    modelA = _cost_model(trainA, avA, knobs, "all")
    out = defaultdict(list)
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(common))
        order = [common[i] for i in idx]
        # a fixed evaluation set, so every budget is scored on the same points
        testB = order[32:] if len(order) > 48 else order[len(order) // 2:]
        poolB = [p for p in order if p not in set(testB)]

        out["oracle on B"].append(1.0)
        # The single config that was best across A, carried over to B. Fitted
        # on A's availability, scored on B's -- pol_best_global assumes one
        # table for both, which is exactly what does not hold here.
        univ = [c for c in {c for pt in trainA for c in avA[pt]}
                if all(c in avA[pt] for pt in trainA)]
        tot = {c: sum(avA[pt][c] / min(avA[pt].values()) for pt in trainA)
               for c in univ}
        gwin = min(tot, key=tot.get)
        out["best global config from A"].append(regret(
            ptsB, testB,
            {pt: gwin if gwin in avB[pt] else min(avB[pt], key=avB[pt].get)
             for pt in testB}))
        out["hand rule fitted on A (0 B measurements)"].append(
            regret(ptsB, testB, pol_hand(None, testB, ptsB, avB, knobs, hand_rule)))
        out["GBT trained on A (0 B measurements)"].append(
            regret(ptsB, testB,
                   pol_gbt(trainA, testB, ptsB, avB, knobs, "all", modelA)))

        for n in (4, 8, 16):
            shot = poolB[:n]
            _, rule_n = fit_hand(shot, ptsB, avB, knobs)
            out[f"hand rule refitted on {n} B points"].append(
                regret(ptsB, testB, pol_hand(None, testB, ptsB, avB, knobs, rule_n)))
            out[f"GBT on {n} B points only"].append(
                regret(ptsB, testB,
                       pol_gbt(shot, testB, ptsB, avB, knobs, "all",
                               _cost_model(shot, avB, knobs, "all"))))
            # transfer: A's structure plus a few measurements from B
            both = {**{p: avA[p] for p in trainA}, **{p: avB[p] for p in shot}}
            out[f"GBT on A + {n} B points"].append(
                regret(ptsB, testB,
                       pol_gbt(list(both), testB, ptsB, avB, knobs, "all",
                               _cost_model(list(both), both, knobs, "all"))))

    print(f"{'decider':<44} {'gmean regret on B':>18}")
    order_out = ["oracle on B", "best global config from A",
                 "hand rule fitted on A (0 B measurements)",
                 "GBT trained on A (0 B measurements)"]
    for n in (4, 8, 16):
        order_out += [f"hand rule refitted on {n} B points",
                      f"GBT on {n} B points only",
                      f"GBT on A + {n} B points"]
    for k in order_out:
        if k in out:
            print(f"{k:<44} {np.mean(out[k]):>17.3f}x")


# ----------------------------------------------------------------- main ---
def splits(pts, group_by, seed):
    rng = np.random.default_rng(seed)
    keys = sorted(pts)
    if group_by == "size":
        sizes = sorted({p[0] for p in keys})
        rng.shuffle(sizes)
        held = set(sizes[:max(1, len(sizes) // 3)])
        test = [p for p in keys if p[0] in held]
    elif group_by == "distance":
        ds = sorted({p[2] for p in keys})
        rng.shuffle(ds)
        held = set(ds[:max(1, len(ds) // 3)])
        test = [p for p in keys if p[2] in held]
    else:
        idx = rng.permutation(len(keys))
        test = [keys[i] for i in idx[:len(keys) // 3]]
    tset = set(test)
    return [p for p in keys if p not in tset], test


def main(argv):
    src = next((a for a in argv if not a.startswith("--")),
               "docs/experiments/grid/grid_big.csv")
    group_by = next((a.split("=", 1)[1] for a in argv
                     if a.startswith("--group-by=")), "random")
    xfer = next((a.split("=", 1)[1] for a in argv
                 if a.startswith("--transfer=")), None)
    if xfer:
        a, b = xfer.split(",")
        transfer(a, b)
        return 0

    pts, knobs = load(src)
    L = pick_global_L(pts, knobs)
    avail = {pt: {c: v for c, v in cfgs.items()
                  if knobs[c][0] == "trigger" or knobs[c][3] == L}
             for pt, cfgs in pts.items()}
    print(f"{len(pts)} workload points; global fleet size fixed at L={L} "
          f"(process-wide knob), leaving "
          f"{len({c for v in avail.values() for c in v})} per-site configs")
    print(f"hold-out: {group_by}, {len(SEEDS)} seeds\n")

    hand_rules = {
        "hand: P = 4 (best constant)":        lambda b, K, D: 4,
        "hand: P = min(K, 8)":                lambda b, K, D: min(K, 8),
        "hand: P = 8 if K >= 16 else 4":      lambda b, K, D: 8 if K >= 16 else 4,
        "hand: P = 1 (context-free lowering)": lambda b, K, D: 1,
    }

    rows = defaultdict(list)
    for seed in SEEDS:
        train, test = splits(pts, group_by, seed)
        rng = np.random.default_rng(1000 + seed)

        rows["best global config (from train)"].append(
            regret(pts, test, pol_best_global(train, test, pts, avail, knobs)))
        for name, rule in hand_rules.items():
            rows[name].append(
                regret(pts, test, pol_hand(train, test, pts, avail, knobs, rule)))
        rows["nearest neighbour in feature space"].append(
            regret(pts, test, pol_knn(train, test, pts, avail, knobs)))
        rows["decision tree (argmin classifier)"].append(
            regret(pts, test, pol_tree(train, test, pts, avail, knobs)))

        m_all = _cost_model(train, avail, knobs, "all")
        rows["GBT cost model, all features"].append(
            regret(pts, test, pol_gbt(train, test, pts, avail, knobs, "all", m_all)))
        rows["GBT cost model, no distance feature"].append(
            regret(pts, test, pol_gbt(train, test, pts, avail, knobs, "nodistance",
                                      _cost_model(train, avail, knobs, "nodistance"))))
        rows["GBT cost model, runtime-visible only"].append(
            regret(pts, test, pol_gbt(train, test, pts, avail, knobs, "runtime",
                                      _cost_model(train, avail, knobs, "runtime"))))
        rows["oracle"].append(1.0)

        for m in (1, 2, 4, 8):
            rows[f"random search, {m} measurement(s)"].append(
                regret(pts, test,
                       pol_random_search(train, test, pts, avail, knobs, m, rng)))
            rows[f"GBT + measure top {m}"].append(
                regret(pts, test,
                       pol_gbt_topm(train, test, pts, avail, knobs, m, m_all)))

    order = ["oracle", "GBT cost model, all features",
             "GBT cost model, no distance feature",
             "GBT cost model, runtime-visible only",
             "decision tree (argmin classifier)",
             "nearest neighbour in feature space",
             "best global config (from train)"] + list(hand_rules)
    print("=== A. generalisation to unseen workload points (zero measurements) ===")
    print(f"{'policy':<42} {'gmean regret':>14} {'spread over seeds':>20}")
    for k in order:
        v = rows[k]
        print(f"{k:<42} {np.mean(v):>13.3f}x   "
              f"[{min(v):.3f} .. {max(v):.3f}]")

    print("\n=== B. measurement budget at the new point ===")
    print(f"{'method':<42} {'gmean regret':>14}")
    for m in (1, 2, 4, 8):
        for pre in ("random search", "GBT + measure top"):
            k = (f"{pre}, {m} measurement(s)" if pre.startswith("random")
                 else f"{pre} {m}")
            if k in rows:
                print(f"{k:<42} {np.mean(rows[k]):>13.3f}x")
    print(f"{'GBT, ZERO measurements':<42} "
          f"{np.mean(rows['GBT cost model, all features']):>13.3f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
