#!/usr/bin/env python3
"""llm_decider.py - can a language model pick the lowering from compiler facts?

The tabular deciders in learn_grid.py take a fixed feature vector and emit
one of a fixed set of configurations. That framing has to declare, up
front, both what the program's properties are and what the available
actions are. This asks whether a model that reads the facts as text does
any better on the SAME decision, which is the only way to attribute a
difference to the decider rather than to the action space.

Three phases, so the model never sees what it is being scored on:

  emit    write a prompt containing the training points WITH their measured
          times (the measurement history a deployed system would have) and
          the held-out points WITHOUT them.
  score   read the model's answer back and score it against the measured
          per-point oracle, alongside every control from learn_grid.py.

Usage:
  ./llm_decider.py emit  docs/experiments/grid/grid_big.csv > prompt.txt
  # ... hand prompt.txt to a model, save its JSON reply as answer.json ...
  ./llm_decider.py score docs/experiments/grid/grid_big.csv answer.json
"""
import json
import math
import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from learn_grid import (SEEDS, _cost_model, gmean, load, pick_global_L,
                        pol_best_global, pol_gbt, pol_hand, pol_random_search,
                        regret, splits)

# Held out by message size, so the model is asked about sizes it has no
# example of. A random split lets a nearest-neighbour strategy win without
# understanding anything, which is not the question.
GROUP_BY = "size"
SPLIT_SEED = 0


def facts_for(pt, knobs, avail):
    """What the compiler knows about this call site, and what is legal."""
    b, K, D = pt
    return {
        "site": f"b{b}_k{K}_d{D}",
        "message_bytes": b,
        "trip_count": K,               # transfers per phase (ScalarEvolution)
        "batch_size": K,               # all released by one completion point
        "flops_to_first_use_us": D,    # independent work behind the transfer
        "descriptor_reusable": False,  # offsets advance with the iv
        "buffer_reusable": True,
        "legal_configs": sorted(avail[pt]),
    }


def platform_facts():
    """Microbenchmark constants a deployment would measure once."""
    return {
        "link_bandwidth_GB_s": 24.0,
        "note_single_nic": "one CXI NIC per rank pair caps a pair at ~24 GB/s",
        "proxy_per_op_issue_us": 2.87,
        "trigger_per_op_stage_us": 1.033,
        "proxy_fixed_us": 5.37,
        "trigger_fixed_us": 7.88,
        "config_naming": {
            "trig-B<n>": "host stages every descriptor; one MMIO trigger "
                         "releases n of them; issue is a single lead thread",
            "proxy-P<p>-L<l>": "device pushes commands into a ring; p blocks "
                               "push concurrently, l CPU worker threads drain. "
                               "p is fixed in the kernel's machine code.",
        },
    }


def emit(src):
    pts, knobs = load(src)
    L = pick_global_L(pts, knobs)
    avail = {pt: {c: v for c, v in cfgs.items()
                  if knobs[c][0] == "trigger" or knobs[c][3] == L}
             for pt, cfgs in pts.items()}
    train, test = splits(pts, GROUP_BY, SPLIT_SEED)

    hist = []
    for pt in train:
        row = facts_for(pt, knobs, avail)
        row["measured_us"] = {c: round(v, 1) for c, v in sorted(avail[pt].items())}
        hist.append(row)
    ask = [facts_for(pt, knobs, avail) for pt in test]

    print(f"""You are the policy stage of a compiler for GPU-initiated
communication. For each call site below, choose ONE lowering from its
legal_configs. You cannot run anything; decide from the facts.

A call site issues `trip_count` transfers of `message_bytes` each, then
does `flops_to_first_use_us` microseconds of independent work, then waits
for them. The cost you are minimising is the time from issue to the
completion point, minus that independent work -- i.e. the part of the
communication that stays exposed.

The fleet size L is process-global and already fixed at {L}; you are
choosing per call site among what remains.

PLATFORM
{json.dumps(platform_facts(), indent=2)}

MEASUREMENT HISTORY ({len(hist)} call sites, with every configuration timed)
{json.dumps(hist, indent=1)}

DECIDE THESE ({len(ask)} call sites, no timings given; the message sizes
are ones that do not appear above)
{json.dumps(ask, indent=1)}

Reply with ONLY a JSON object mapping each site name to one config string:
{{"b256_k4_d0": "proxy-P4-L{L}", ...}}
No prose, no code fences.""")


def score(src, answer_path):
    pts, knobs = load(src)
    L = pick_global_L(pts, knobs)
    avail = {pt: {c: v for c, v in cfgs.items()
                  if knobs[c][0] == "trigger" or knobs[c][3] == L}
             for pt, cfgs in pts.items()}
    train, test = splits(pts, GROUP_BY, SPLIT_SEED)

    raw = open(answer_path).read().strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    reply = json.loads(raw)

    picks, illegal, missing = {}, 0, 0
    for pt in test:
        name = f"b{pt[0]}_k{pt[1]}_d{pt[2]}"
        c = reply.get(name)
        if c is None:
            missing += 1
            c = min(avail[pt], key=avail[pt].get)      # generous: give it the best
        elif c not in avail[pt]:
            illegal += 1
            c = max(avail[pt], key=avail[pt].get)      # illegal costs the worst
        picks[pt] = c

    print(f"scored on {len(test)} held-out call sites "
          f"(held out by {GROUP_BY}, fleet L={L})")
    if missing:
        print(f"  WARNING {missing} sites had no answer; scored at their oracle")
    if illegal:
        print(f"  {illegal} answers named a configuration that is not legal "
              f"at that site; scored at the worst legal one")
    print()

    import numpy as np
    rng = np.random.default_rng(0)
    m_all = _cost_model(train, avail, knobs, "all")
    rows = [
        ("per-point oracle", 1.0),
        ("LLM one-shot, compiler facts only", regret(pts, test, picks)),
        ("GBT cost model (same facts, trained)",
         regret(pts, test, pol_gbt(train, test, pts, avail, knobs, "all", m_all))),
        ("hand rule: P = min(K, 8)",
         regret(pts, test, pol_hand(None, test, pts, avail, knobs,
                                    lambda b, K, D: min(K, 8)))),
        ("best global config from train",
         regret(pts, test, pol_best_global(train, test, pts, avail, knobs))),
        ("random search, 4 measurements",
         regret(pts, test, pol_random_search(train, test, pts, avail, knobs, 4, rng))),
        ("random search, 8 measurements",
         regret(pts, test, pol_random_search(train, test, pts, avail, knobs, 8, rng))),
        ("context-free lowering: P = 1",
         regret(pts, test, pol_hand(None, test, pts, avail, knobs,
                                    lambda b, K, D: 1))),
    ]
    print(f"{'decider':<40} {'gmean regret':>13}")
    for name, r in rows:
        print(f"{name:<40} {r:>12.3f}x")

    # Where it went wrong, so a tie is not mistaken for agreement.
    print("\nper-site detail (LLM pick vs oracle):")
    bad = sorted(test, key=lambda p: -(pts[p][picks[p]] / min(avail[p].values())))
    for pt in bad[:6]:
        o = min(avail[pt], key=avail[pt].get)
        r = pts[pt][picks[pt]] / avail[pt][o]
        print(f"  b={pt[0]:>7} K={pt[1]:>3} D={pt[2]:>4}  chose {picks[pt]:<14}"
              f" {avail[pt][picks[pt]]:8.1f}us   oracle {o:<14}"
              f" {avail[pt][o]:8.1f}us   {r:.2f}x")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    if sys.argv[1] == "emit":
        emit(sys.argv[2])
    elif sys.argv[1] == "score":
        score(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        sys.exit(2)
