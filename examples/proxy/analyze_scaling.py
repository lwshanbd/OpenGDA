#!/usr/bin/env python3
"""analyze_scaling.py - does hand-derivation stop scaling, and where?

Four decision spaces on the cross-node path were separable: their optima
factored into independent per-axis thresholds, so a rule matched every
learned decider. The intra-node copy knobs are not separable. The question
this answers is whether that is a property of those particular knobs or of
how MANY knobs are in play.

For every subset of the measured axes, the remaining axes are pinned at
their globally best value -- that is what a compiler exposing only those
knobs, with the rest hardcoded to a sensible default, would face. Then:

  oracle             best configuration at each size
  separable rules    each axis chosen independently, i.e. what a threshold
                     rule can express
  GBT                a cost model over the joint space
  best global        one configuration for every size

and the gap between `separable rules` and `oracle` is what no set of
independent rules can recover, however carefully they are written.

Deciders are scored leave-one-size-out: fitted on eight sizes, choosing
for a ninth they have never measured.

Usage:  ./analyze_scaling.py docs/experiments/ipc/joint5.csv
"""
import csv
import math
import sys
from itertools import combinations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

AXES = ["vec", "nt", "unroll", "block", "nstream"]


def load(path):
    cost = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = tuple(int(r[a]) for a in AXES)
            cost.setdefault(int(r["bytes"]), {})[key] = float(r["us"])
    return cost


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def global_best(cost):
    """The configuration with least average regret over all sizes -- what
    the pinned axes are held at."""
    keys = set.intersection(*(set(c) for c in cost.values()))
    tot = {k: gmean([cost[s][k] / min(cost[s].values()) for s in cost])
           for k in keys}
    return min(tot, key=tot.get)


def restrict(cost, free, pin):
    """Sub-space where only `free` axis indices vary; the rest sit at `pin`."""
    out = {}
    for s, c in cost.items():
        sub = {k: v for k, v in c.items()
               if all(k[i] == pin[i] for i in range(len(AXES)) if i not in free)}
        if sub:
            out[s] = sub
    return out


def evaluate(sub, free):
    """Leave-one-size-out regret for each decider on this sub-space."""
    sizes = sorted(sub)
    vals = {i: sorted({k[i] for c in sub.values() for k in c}) for i in free}
    res = {"oracle": [], "separable": [], "gbt": [], "global": []}

    def feats(b, k):
        return [math.log2(b)] + [math.log2(k[i]) if k[i] > 0 else 0.0
                                 for i in free]

    for held in sizes:
        train = [s for s in sizes if s != held]
        cands = list(sub[held])
        best = min(sub[held].values())
        res["oracle"].append(1.0)

        X = [feats(s, k) for s in train for k in sub[s]]
        y = [math.log(sub[s][k]) for s in train for k in sub[s]]
        if len(set(map(tuple, X))) > 1:
            m = GradientBoostingRegressor(random_state=0, n_estimators=200,
                                          max_depth=3).fit(np.array(X), np.array(y))
            pred = m.predict(np.array([feats(held, k) for k in cands]))
            res["gbt"].append(sub[held][cands[int(np.argmin(pred))]] / best)
        else:
            res["gbt"].append(1.0)

        # Each axis picked independently on the training sizes: the best
        # this axis looks on average, ignoring what the others are doing.
        pick = {}
        for i in free:
            avg = {v: gmean([sub[s][k] for s in train for k in sub[s] if k[i] == v])
                   for v in vals[i]}
            pick[i] = min(avg, key=avg.get)
        target = tuple(pick.get(i, cands[0][i]) for i in range(len(AXES)))
        if target not in sub[held]:
            target = min(cands, key=lambda k: sum(k[i] != target[i] for i in free))
        res["separable"].append(sub[held][target] / best)

        tot = {k: gmean([sub[s][k] / min(sub[s].values()) for s in train])
               for k in cands}
        res["global"].append(sub[held][min(tot, key=tot.get)] / best)
    return {k: gmean(v) for k, v in res.items()}


def main(path):
    cost = load(path)
    sizes = sorted(cost)
    pin = global_best(cost)
    print(f"{len(sizes)} sizes, {len(next(iter(cost.values())))} configurations")
    print("axes: " + ", ".join(AXES))
    print("pinned axes sit at the globally best configuration "
          + str(dict(zip(AXES, pin))))
    print()
    print(f"{'axes':>4} {'subsets':>8} {'separable':>11} {'GBT':>9} "
          f"{'best global':>12}   {'rules lose by':>13}")
    for k in range(1, len(AXES) + 1):
        rows = []
        for free in combinations(range(len(AXES)), k):
            sub = restrict(cost, set(free), pin)
            if not sub or len(next(iter(sub.values()))) < 2:
                continue
            rows.append(evaluate(sub, set(free)))
        if not rows:
            continue
        sep = gmean([r["separable"] for r in rows])
        gbt = gmean([r["gbt"] for r in rows])
        glo = gmean([r["global"] for r in rows])
        print(f"{k:>4} {len(rows):>8} {sep:>10.3f}x {gbt:>8.3f}x "
              f"{glo:>11.3f}x   {sep / gbt:>12.3f}x")
    print()
    print("'rules lose by' is separable / GBT: what independent per-axis")
    print("thresholds give up to a model that sees the joint space. If it")
    print("rises with the axis count, hand-derivation is what stops scaling,")
    print("not the difficulty of any individual knob.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "docs/experiments/ipc/joint5.csv")
