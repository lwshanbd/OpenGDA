#!/usr/bin/env python3
"""transform_plan.py - choose a transformation per call site, and score it.

The decision here is not "which of N configurations", it is "which
transformation to apply", and the set of available transformations differs
per call site because it is bounded by a legality proof the compiler
computes. A site whose transfers are not adjacent cannot be merged at all;
one whose transfers are adjacent can be merged by any factor dividing its
trip count. That is why the action cannot be posed as a fixed-width vector:
its arity follows the program.

  emit    write the prompt: per-site compiler facts, the legal action set
          for each, and the measured cost of every legal action at a few
          TRAINING sites so there is something to reason from.
  score   read a plan back and score it against the per-site oracle, next
          to the controls that could beat it.

Usage:
  ./transform_plan.py emit  docs/experiments/transform/sites.csv > prompt.txt
  ./transform_plan.py score docs/experiments/transform/sites.csv plan.json
"""
import csv
import json
import math
import sys
from collections import defaultdict

COLS = ["site", "bytes", "ops", "stride_mult", "dist_us", "coalescable",
        "merge", "blocks", "msg_bytes", "msgs", "median_us", "min_us",
        "max_us", "verdict"]

# The split is mechanical and is declared in run_transform_sites.sh before
# anything is measured: every third site is held out. Deriving it from the
# name here rather than listing sites keeps the two files from drifting,
# and keeps the split from being quietly adjusted once the results are in.
HELD_OUT_EVERY = 3


def is_test(site):
    digits = "".join(c for c in site if c.isdigit())
    return bool(digits) and int(digits) % HELD_OUT_EVERY == 0


def load(path):
    """-> (facts[site], cost[site][(merge, blocks)])"""
    facts, cost = {}, defaultdict(dict)
    with open(path) as f:
        for row in csv.reader(f):
            if not row or row[0] != "CSV":
                continue
            r = dict(zip(COLS, row[1:]))
            s = r["site"]
            facts.setdefault(s, {
                "site": s,
                "message_bytes": int(r["bytes"]),
                "trip_count": int(r["ops"]),
                "flops_to_first_use_us": int(r["dist_us"]),
                "coalescable": r["coalescable"] == "1",
            })
            if r["verdict"] != "OK" or not r["median_us"]:
                continue          # skipped-illegal rows carry no timing
            cost[s][(int(r["merge"]), int(r["blocks"]))] = float(r["median_us"])
    return facts, dict(cost)


def legal_actions(fact, cost_keys):
    """What the compiler permits here, intersected with what was measured."""
    n = fact["trip_count"]
    merges = [m for m in (1, 2, 4, 8, 16, 32, 64) if m <= n and n % m == 0]
    if not fact["coalescable"]:
        merges = [1]              # not proven adjacent: merging is unsafe
    return sorted({(m, p) for (m, p) in cost_keys if m in merges})


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def emit(path):
    facts, cost = load(path)
    train = [s for s in facts if not is_test(s)]
    test = [s for s in facts if is_test(s)]

    hist = []
    for s in train:
        f = dict(facts[s])
        f["measured_us"] = {f"merge{m}_blocks{p}": round(v, 1)
                            for (m, p), v in sorted(cost[s].items())}
        hist.append(f)
    ask = []
    for s in test:
        f = dict(facts[s])
        f["legal_actions"] = [f"merge{m}_blocks{p}"
                              for (m, p) in legal_actions(f, cost[s])]
        ask.append(f)

    print(f"""You are the transformation stage of a compiler for GPU-initiated
communication. Each call site below is a loop that issues `trip_count`
transfers of `message_bytes`, then does `flops_to_first_use_us`
microseconds of independent work, then waits for them.

You choose two things per site:

  merge<m>   issue the loop's transfers as trip_count/m transfers of
             m*message_bytes each. Fewer, larger messages pay the
             per-message cost fewer times, but leave less to overlap.
             LEGAL ONLY WHERE `coalescable` IS TRUE -- that means the
             compiler proved consecutive transfers land exactly
             message_bytes apart, so a merged transfer covers exactly the
             same bytes. Where it is false the transfers have gaps
             between them and merging would send the wrong bytes; the
             legal_actions list already reflects this.

  blocks<p>  how many GPU thread blocks issue concurrently. This is fixed
             in the kernel's machine code, so it cannot be changed later.

The link is one NIC per rank pair, about 24 GB/s, with a floor around
0.7 microseconds per message however small it is.

MEASUREMENT HISTORY ({len(hist)} sites, every legal action timed)
{json.dumps(hist, indent=1)}

DECIDE THESE ({len(ask)} sites, no timings given)
{json.dumps(ask, indent=1)}

Reply with ONLY a JSON object mapping each site name to one action string
drawn from that site's own legal_actions:
{{"D": "merge4_blocks2", ...}}
No prose, no code fences.""")


def score(path, plan_path):
    facts, cost = load(path)
    test = [s for s in facts if is_test(s)]
    train = [s for s in facts if not is_test(s)]

    raw = open(plan_path).read().strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    plan = json.loads(raw)

    def parse(a):
        m, b = a.split("_")
        return (int(m[5:]), int(b[6:]))

    picks, illegal = {}, []
    for s in test:
        legal = legal_actions(facts[s], cost[s])
        a = plan.get(s)
        try:
            k = parse(a)
        except Exception:
            k = None
        if k is None or k not in legal:
            illegal.append((s, a))
            k = max(legal, key=lambda x: cost[s][x])   # worst legal
        picks[s] = k

    def regret(chooser):
        rs = []
        for s in test:
            legal = legal_actions(facts[s], cost[s])
            best = min(cost[s][k] for k in legal)
            rs.append(cost[s][chooser(s, legal)] / best)
        return gmean(rs)

    import numpy as np
    rng = np.random.default_rng(0)

    def best_global_fixed():
        """One action for the whole program. It must be legal at EVERY
        site, and a site that is not coalescable forces merge=1 -- which
        is the whole argument for choosing per site."""
        common = None
        for s in facts:
            ks = set(legal_actions(facts[s], cost[s]))
            common = ks if common is None else (common & ks)
        if not common:
            return None
        tot = {k: sum(cost[s][k] / min(cost[s][j]
                                       for j in legal_actions(facts[s], cost[s]))
                      for s in train) for k in common}
        return min(tot, key=tot.get)

    def hand_rule(mmax, p):
        def ch(s, legal):
            f = facts[s]
            m = min(mmax, f["trip_count"]) if f["coalescable"] else 1
            cand = [k for k in legal if k[0] <= m]
            return max(cand, key=lambda k: (k[0], -abs(k[1] - p)))
        return ch

    def random_search(k):
        def ch(s, legal):
            idx = rng.choice(len(legal), size=min(k, len(legal)), replace=False)
            return min((legal[i] for i in idx), key=lambda x: cost[s][x])
        return ch

    gf = best_global_fixed()
    # The hand rule's two constants are fitted on the training sites, the
    # same information the model gets.
    best_hand, best_hr = None, None
    for mmax in (1, 2, 4, 8, 16, 32, 64):
        for p in (1, 2, 4, 8):
            r = gmean([cost[s][hand_rule(mmax, p)(s, legal_actions(facts[s], cost[s]))]
                       / min(cost[s][k] for k in legal_actions(facts[s], cost[s]))
                       for s in train])
            if best_hand is None or r < best_hand:
                best_hand, best_hr = r, (mmax, p)

    print(f"scored on {len(test)} held-out call sites: {', '.join(test)}")
    print(f"  (trained on {', '.join(train)})")
    if illegal:
        print(f"  {len(illegal)} plan entries were not legal at their site "
              f"and were scored at the WORST legal action: {illegal}")
    print()
    rows = [
        ("per-site oracle", regret(lambda s, l: min(l, key=lambda k: cost[s][k]))),
        ("LLM plan (compiler facts + legality)", regret(lambda s, l: picks[s])),
        (f"hand rule: merge<=min({best_hr[0]},K) if legal, blocks={best_hr[1]}",
         regret(hand_rule(*best_hr))),
        ("random search, 2 measurements", regret(random_search(2))),
        ("random search, 4 measurements", regret(random_search(4))),
        ("random search, 8 measurements", regret(random_search(8))),
    ]
    if gf:
        rows.insert(2, (f"best single global action, legal everywhere "
                        f"(merge{gf[0]}_blocks{gf[1]})",
                        regret(lambda s, l: gf)))
    rows.append(("context-free lowering (merge1_blocks1)",
                 regret(lambda s, l: (1, 1))))

    print(f"{'decider':<52} {'gmean regret':>13}")
    for name, r in rows:
        print(f"{name:<52} {r:>12.3f}x")

    print("\nper-site detail:")
    for s in test:
        legal = legal_actions(facts[s], cost[s])
        o = min(legal, key=lambda k: cost[s][k])
        k = picks[s]
        f = facts[s]
        print(f"  {s}: {f['message_bytes']:>7}B x{f['trip_count']:<3} "
              f"d={f['flops_to_first_use_us']:<4} "
              f"coalescable={str(f['coalescable']):<5} "
              f"chose merge{k[0]}/blocks{k[1]} {cost[s][k]:8.1f}us   "
              f"oracle merge{o[0]}/blocks{o[1]} {cost[s][o]:8.1f}us   "
              f"{cost[s][k] / cost[s][o]:.2f}x")


def sensitivity(path):
    """How much the choice matters at each site.

    A site whose actions all cost about the same cannot separate one
    decider from another, so a headline averaged over such sites mostly
    reports how many of them there were. Reported next to the results so
    a flat test set is visible rather than inferred.
    """
    facts, cost = load(path)
    print(f"{'site':<6} {'split':<6} {'coal':<5} {'shape':<20} "
          f"{'best':>8} {'median legal':>13} {'sensitivity':>12}")
    for s in sorted(facts):
        f = facts[s]
        legal = legal_actions(f, cost[s])
        vals = sorted(cost[s][k] for k in legal)
        best, med = vals[0], vals[len(vals) // 2]
        shape = (f"{f['message_bytes']}B x{f['trip_count']}"
                 f" d={f['flops_to_first_use_us']}")
        print(f"{s:<6} {'TEST' if is_test(s) else 'train':<6} "
              f"{'yes' if f['coalescable'] else 'NO':<5} {shape:<20} "
              f"{best:>8.1f} {med:>13.1f} {med / best:>11.2f}x")


def dist(path, plan_paths):
    """Several independent plans for the identical prompt."""
    facts, cost = load(path)
    test = [s for s in facts if is_test(s)]
    rs, agree = [], defaultdict(set)
    import numpy as np
    for p in plan_paths:
        raw = open(p).read().strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        plan = json.loads(raw)
        tot = []
        for s in test:
            legal = legal_actions(facts[s], cost[s])
            a = plan.get(s, "")
            try:
                k = (int(a.split("_")[0][5:]), int(a.split("_")[1][6:]))
            except Exception:
                k = None
            if k not in legal:
                k = max(legal, key=lambda x: cost[s][x])
            agree[s].add(k)
            tot.append(cost[s][k] / min(cost[s][x] for x in legal))
        rs.append(gmean(tot))
    print(f"{len(plan_paths)} independent plans, {len(test)} held-out sites")
    print(f"  LLM plan: mean {np.mean(rs):.3f}x  "
          f"[{min(rs):.3f} .. {max(rs):.3f}]")
    split = sum(1 for s in test if len(agree[s]) > 1)
    print(f"  the plans disagreed at {split} of {len(test)} sites")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    if sys.argv[1] == "emit":
        emit(sys.argv[2])
    elif sys.argv[1] == "sensitivity":
        sensitivity(sys.argv[2])
    elif sys.argv[1] == "dist":
        dist(sys.argv[2], sys.argv[3:])
    elif sys.argv[1] == "score":
        score(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        sys.exit(2)
