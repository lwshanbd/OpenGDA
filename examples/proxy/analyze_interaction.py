#!/usr/bin/env python3
"""analyze_interaction.py - do the knobs interact, or decompose?

Four decision spaces on the cross-node path came back worth a few percent,
and in every one a hand-written rule matched every learned decider. The
reason was always the same: the optimum factored into independent per-axis
thresholds, and independent thresholds are exactly what a rule is.

So the question for a new space is not "how large are the effects" but
"does the best value of one knob depend on how another is set". If it does
not, no model can beat a rule, whatever the effect sizes. If it does, no
set of independent rules can express the optimum.

Reports, per transfer size:

  1. the best value of each axis, conditioned on each value of every other
     axis. An axis whose argmin moves as another axis changes is
     interacting; one whose argmin is constant is separable.
  2. what independent per-axis rules cost against the joint oracle -- the
     direct measure of what is left for a model to find.

Usage:  ./analyze_interaction.py docs/experiments/ipc/joint.csv
"""
import csv
import math
import sys
from collections import defaultdict

AXES = ["vec", "nt", "unroll"]


def load(path):
    """-> cost[bytes][(vec, nt, unroll)] = us"""
    cost = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            key = tuple(int(r[a]) for a in AXES)
            cost[int(r["bytes"])][key] = float(r["us"])
    return dict(cost)


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main(path):
    cost = load(path)
    sizes = sorted(cost)
    vals = {i: sorted({k[i] for c in cost.values() for k in c})
            for i in range(len(AXES))}
    print(f"{len(sizes)} sizes, "
          f"{len(next(iter(cost.values())))} configurations each")
    print("axes: " + ", ".join(f"{a}={vals[i]}" for i, a in enumerate(AXES)))
    print()

    # ---- 1. does an axis's best value move when another axis changes? ----
    print("=== 1. conditional optima: does the best value of one axis move? ===")
    interacting = set()
    for i, a in enumerate(AXES):
        for j, b in enumerate(AXES):
            if i == j:
                continue
            moved = []
            for s in sizes:
                best_of_a = {}
                for bv in vals[j]:
                    cand = {k: v for k, v in cost[s].items() if k[j] == bv}
                    if not cand:
                        continue
                    best_of_a[bv] = min(cand, key=cand.get)[i]
                if len(set(best_of_a.values())) > 1:
                    moved.append((s, dict(best_of_a)))
            if moved:
                interacting.add((a, b))
                print(f"  best {a} MOVES with {b}:")
                for s, m in moved[:4]:
                    desc = ", ".join(f"{b}={k} -> {a}={v}" for k, v in sorted(m.items()))
                    print(f"      {s:>9}B: {desc}")
                if len(moved) > 4:
                    print(f"      ... and {len(moved) - 4} more sizes")
    if not interacting:
        print("  none: every axis has the same optimum regardless of the others")
    print()

    # ---- 2. what do independent per-axis rules cost? ----
    # Fit each axis independently: for each size, the value of that axis
    # that is best on average over all settings of the others. Then combine.
    # If the space is separable this reconstructs the oracle exactly.
    print("=== 2. independent per-axis rules vs the joint oracle ===")
    print(f"{'bytes':>10} {'oracle us':>10} {'separable us':>13} "
          f"{'regret':>8}   {'oracle cfg':<18} {'separable cfg':<18}")
    regrets = []
    for s in sizes:
        c = cost[s]
        best = min(c, key=c.get)
        sep = []
        for i, a in enumerate(AXES):
            avg = {}
            for v in vals[i]:
                xs = [t for k, t in c.items() if k[i] == v]
                if xs:
                    avg[v] = gmean(xs)
            sep.append(min(avg, key=avg.get))
        sep = tuple(sep)
        if sep not in c:            # the marginal winners may not co-exist
            sep = min(c, key=lambda k: sum(abs(k[i] - sep[i]) for i in range(3)))
        r = c[sep] / c[best]
        regrets.append(r)
        print(f"{s:>10} {c[best]:>10.3f} {c[sep]:>13.3f} {r:>7.3f}x   "
              f"{str(best):<18} {str(sep):<18}")
    print(f"\n  gmean regret of independent rules: {gmean(regrets):.3f}x"
          f"   worst: {max(regrets):.3f}x")
    print()

    # ---- 3. how much is on the table at all ----
    print("=== 3. spread within each size (worst legal / best) ===")
    for s in sizes:
        c = cost[s]
        print(f"  {s:>9}B  best {min(c.values()):8.3f} us   "
              f"worst {max(c.values()):8.3f} us   "
              f"{max(c.values()) / min(c.values()):.2f}x")

    print("\nverdict:")
    if interacting and gmean(regrets) > 1.02:
        print("  the axes interact AND independent rules leave something on the")
        print("  table -- a model has a job here that a rule cannot do")
    elif interacting:
        print("  the axes interact, but independent rules still land within 2%")
        print("  of the oracle, so the interaction is not worth modelling")
    else:
        print("  separable: independent per-axis rules reconstruct the optimum,")
        print("  so this space is rule-shaped like the previous four")




def model_test(path):
    """Can a model capture what independent rules cannot?

    Leave-one-size-out: fit on every other size's full grid, then pick a
    configuration for the held-out size with no measurement there. The
    independent-rule baseline is fitted on exactly the same data, so the
    comparison is between model classes rather than between information.
    """
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    cost = load(path)
    sizes = sorted(cost)
    vals = {i: sorted({k[i] for c in cost.values() for k in c})
            for i in range(len(AXES))}

    def feats(b, k):
        return [math.log2(b), math.log2(k[0]), k[1], math.log2(k[2] or 1)]

    rows = {"oracle": [], "GBT cost model": [], "independent per-axis rules": [],
            "best single global config": []}
    for held in sizes:
        train = [s for s in sizes if s != held]
        X = [feats(s, k) for s in train for k in cost[s]]
        y = [math.log(cost[s][k]) for s in train for k in cost[s]]
        m = GradientBoostingRegressor(random_state=0, n_estimators=300,
                                      max_depth=3).fit(np.array(X), np.array(y))
        cands = list(cost[held])
        best = min(cost[held].values())

        pred = m.predict(np.array([feats(held, k) for k in cands]))
        rows["GBT cost model"].append(cost[held][cands[int(np.argmin(pred))]] / best)

        # independent rules, fitted on the same training sizes
        sep = []
        for i in range(len(AXES)):
            avg = {v: gmean([cost[s][k] for s in train for k in cost[s] if k[i] == v])
                   for v in vals[i]}
            sep.append(min(avg, key=avg.get))
        sep = tuple(sep)
        if sep not in cost[held]:
            sep = min(cands, key=lambda k: sum(abs(k[i] - sep[i]) for i in range(3)))
        rows["independent per-axis rules"].append(cost[held][sep] / best)

        tot = {k: gmean([cost[s][k] / min(cost[s].values()) for s in train])
               for k in cands}
        gw = min(tot, key=tot.get)
        rows["best single global config"].append(cost[held][gw] / best)
        rows["oracle"].append(1.0)

    print("\n=== 4. leave-one-size-out: model vs rules on identical data ===")
    print(f"{'decider':<32} {'gmean regret':>13} {'worst':>8}")
    for k in ["oracle", "GBT cost model", "independent per-axis rules",
              "best single global config"]:
        v = rows[k]
        print(f"{k:<32} {gmean(v):>12.3f}x {max(v):>7.3f}x")


if __name__ == "__main__":
    src = next((a for a in sys.argv[1:] if not a.startswith("--")),
               "docs/experiments/ipc/joint.csv")
    main(src)
    model_test(src)
