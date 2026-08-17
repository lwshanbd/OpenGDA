# Compiler-guided communication optimization — progress log

Branch `feature/compiler-guided-ml`. GICC is the vehicle here, not the
subject: the question is whether program context a compiler can see opens
communication optimizations a runtime cannot reach, and if so, what should
search that space.

All numbers below are measured on Tioga (MI250X + Slingshot 11), 2 nodes,
1 rank per node unless stated. Medians over the sample counts given.

---

## The claim, and what would falsify it

> The lowering of a communication call has choices. Which choice is best
> differs per call site, the difference is worth ~1.5x, and part of the
> information needed to choose is only available at compile time.

Falsifiable in three places, and each has been checked:

- if a single fixed lowering were near-optimal, per-site specialization
  would be pointless — measured at 1.607x off (§1);
- if the deciding information were visible to a runtime, no compiler would
  be needed — a decider restricted to runtime-visible features mispicks
  1 point in 6 (§2);
- if the winning knob were settable after compilation, a runtime tuner
  could reach it — the dominant one is fixed in machine code (§1).

The baseline that matters is **the best achievable without compiler
information**, not GICC's own default. GICC's default is 1.852x off the
oracle, so beating *it* proves very little. Quote the comparison against
the best fixed configuration.

---

## 1. What per-site lowering is worth

`ctx_bench --exp=grid`, 136 workload points x 23 configurations = 2484
measured cells. A workload point is (message bytes, transfers per phase,
microseconds of independent work before the completion point). Scored as
per-point regret against the per-point oracle; totals in microseconds are
dominated by the largest points and describe the sampling rather than the
policy.

The knobs are not equally reachable, and the ladder is built around that:

| knob | who can set it |
| --- | --- |
| `L` proxy worker lanes | process-global env var — a runtime can |
| `path`, `B` descriptors per trigger | per call site, code generation |
| **`P` concurrent issuing blocks** | **fixed in the kernel's machine code — no runtime can change it at any price** |

Holding `L` global and varying only the per-site knobs:

| policy | cross-node | intra-node |
| --- | --- | --- |
| A. context-free lowering, `P=1`, one configuration everywhere — what GICC emits today | 1.852x | 2.291x |
| B. best single fixed configuration, generously including a `P` no tuner can emit | 1.607x | 1.485x |
| C. per-site `path`/`B`/`P`, one global `L` | **1.031x** | **1.062x** |
| D. per-point oracle including `L` — an upper bound only, one process has one `L` | 1.000x | 1.000x |

**Per-site code generation is worth 1.559x (cross-node) and 1.399x
(intra-node) over the best fixed configuration**, and 1.798x / 2.158x over
what is emitted today. 17 distinct configurations win somewhere across the
136 points; the trigger path wins at 10 of them, so the path axis is not
degenerate even though the proxy path dominates.

`P` is non-monotone: at 4KB x 8 transfers it is 43.2 / 32.6 / **27.2** /
33.4 us for P = 1/2/4/8 — P=8 regresses on ring contention. With two
contending rank pairs the optimum moves *up* to P=8.

Reproduce:

```bash
./examples/proxy/run_grid.sh docs/experiments/grid/grid_big.csv
python3 examples/proxy/analyze_grid.py docs/experiments/grid/grid_big.csv
```

## 2. What only the compiler can see

Same data, deciders restricted to different feature classes, zero
measurements at the test point, 5 seeds, held out three ways:

| decider | random split | held out by size | held out by distance |
| --- | --- | --- | --- |
| GBT cost model, all features | 1.038x | 1.039x | 1.046x |
| GBT, runtime-visible features only | 1.130x | 1.066x | 1.168x |
| best global configuration from train | 1.645x | 1.639x | 1.746x |
| context-free lowering (`P=1`) | 2.577x | 2.568x | 2.812x |

An earlier ablation on the 96-point distance surface put it more sharply:
a decider seeing message size and transfer count mispicks **16 of 96**
points with a worst case of 2.54x; adding the compiler-only
issue-to-first-use distance brings that to **0 of 96**.

Sample efficiency: random search needs **8** measurements at a new call
site (1.060x) to lose to a model that has taken **zero** (1.038x).

## 3. Legality, not preference — O5

`examples/proxy/param_availability.cpp`. Four call sites a runtime cannot
tell apart: same operation, same 64KB message, same 32-transfer trip
count, same peer, same fabric, same independent work behind them. They
differ only in where the descriptor's offsets come from.

The pass's verdict:

| kernel | `hk_capable` | legal paths |
| --- | --- | --- |
| A `k_prelaunch` — `off = base + i*bytes` | true | proxy, trigger, ipc |
| B `k_device_entry` — `off = desc[i].off`, host mirror declared | **true** | proxy, trigger, ipc |
| B' `k_device_entry_unmirrored` — **byte-identical body**, no declaration | **false** | **proxy only** |
| C `k_device_dynamic` — offsets computed on the device | false | proxy only |

What the restriction costs (median of 11):

| issue-to-first-use | A/B best (trigger) | B'/C forced (proxy) | cost |
| --- | --- | --- | --- |
| 0 us | — | — | 1.00x (proxy wins anyway) |
| 100 us | 141.6 us | 216.1 us | **1.53x** |
| 300 us | 343.4 us | 413.2 us | 1.21x |
| 600 us | 658.1 us | 718.7 us | 1.09x |

B and B' are the same kernel body. One annotation moves the legality
boundary, so this is analysis creating an option rather than choosing
among options. At zero distance the restriction costs nothing — the value
of the analysis is regime-dependent, not universal.

Getting legality wrong in the permissive direction does not produce a slow
program. Staging a descriptor for site C reads an offset that does not
exist yet and the transfer carries whatever was in the buffer. Nothing at
runtime checks.

## 4. Transformations, and why they need proof

`coalescable` (new analysis) is true when consecutive iterations of a
transfer loop land exactly `size` apart at both ends, so any run of them
may be issued as one larger transfer. It extracts the induction variable's
coefficient as an *expression* — the stride is normally a kernel formal,
so there is no constant to compare — and tests structural equality against
the size expression.

A runtime cannot establish this: when it sees transfer *i* it does not
know where *i+1* will go, and by then *i* has been issued.

**What merging is worth** (4KB x 64, contiguous, median of 9):

| merge factor | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| best over block counts (us) | 112.4 | 66.6 | 43.3 | 32.4 | 31.2 | 30.9 | 30.8 |

Merging alone is **3.65x**; together with issue width it is **7.10x**
against the context-free `merge1/blocks1` default (218.8 us). The two do
not compose — at merge=1 four blocks beat one by 2.0x, at merge=16 the
block count stops mattering, because there is nothing left to overlap.
(Correction: commit 262033f's message attributes 7.1x to merging alone;
merging alone is 3.65x.)

**What an unproven merge costs.** `--force-illegal` merges a site whose
transfers have gaps between them:

```
merge=1   4096B x 64   110.8 us   OK
merge=8  32768B x  8    32.8 us   WRONG-payload
```

Same bytes on the wire, fewer messages, **3.4x faster, and three quarters
of the payload never arrives**. Only the gap check in the benchmark
notices. A tuner ranking configurations by time picks this every time.

That is the difference between a knob and a transformation: a knob set
badly loses time and the next measurement finds out; a transformation
applied without a legality proof loses data, and the measurement reports a
speedup.

## 5. Per-site transformation planning

`run_transform_sites.sh` measures every legal action at sixteen call sites
spanning message size, trip count, adjacency and available overlap. The
train/test split is mechanical -- every third site -- and is declared in
the script before anything is measured.

| site | shape | coalescable | best action | best | default | headroom | sensitivity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S01 | 256B x 64 | yes | merge64/blocks1 | 21.9 | 213.3 | 9.75x | 2.02x |
| S03 *(test)* | 1KB x 64 | yes | merge64/blocks1 | 23.3 | 218.3 | 9.36x | 1.88x |
| S04 | 4KB x 64 | yes | merge64/blocks1 | 30.7 | 217.1 | 7.07x | 1.51x |
| S06 *(test)* | 16KB x 32 | yes | merge32/blocks1 | 41.5 | 118.4 | 2.85x | 1.12x |
| S08 | 64KB x 32 | yes | **merge4**/blocks1 | 107.2 | 123.3 | 1.15x | 1.01x |
| S09 *(test)* | 64KB x 16, d=100us | yes | merge16/blocks1 | 118.8 | 169.5 | 1.43x | 1.09x |
| S10 | 256KB x 16 | yes | **merge4**/blocks1 | 192.5 | 194.5 | 1.01x | 1.00x |
| S12 *(test)* | 1KB x 64, gaps | no | merge1/blocks4 | 103.1 | 216.5 | 2.10x | 1.16x |
| S15 *(test)* | 16KB x 32, gaps | no | merge1/blocks4 | 69.8 | 119.9 | 1.72x | 1.11x |

(S02, S05, S07, S11, S13, S14, S16 omitted for space; all are training
sites. "Sensitivity" is the median legal action divided by the best, i.e.
how much the choice actually matters there.)

Note S08 and S10: the best merge factor is **interior**, not maximal. Once
a site is bandwidth-bound, further merging only lengthens the last
message's un-overlappable tail. So "merge as much as is legal" is wrong,
which is what makes this a decision rather than a rewrite rule.

Scored on the five held-out sites:

| decider | gmean regret |
| --- | --- |
| per-site oracle | 1.000x |
| hand rule (two constants, fitted on train) | 1.000x |
| LLM plan, 3 independent runs | 1.001x |
| random search, 4 measurements | 1.007x |
| **best single global action, legal everywhere** | **1.550x** |
| **context-free lowering (`merge1/blocks1`)** | **2.677x** |

The best **single global action must be legal at every site**, and one site
that is not provably adjacent forces the whole program to `merge=1`. That
is the argument for per-site specialization stated as a measurement:
**1.550x**, against **2.677x** for what is emitted today.

---

## What is NOT established

Reported here because these are the claims a reviewer will probe.

**Learning does not beat writing the rule down.** On the configuration
space of §1, a one-line hand rule `P = min(K, 8)` matches or beats every
learned decider:

| test | hand rule | GBT |
| --- | --- | --- |
| same regime, unseen points | 1.038x | 1.038x — tie |
| cross-regime, zero-shot | **1.070x** | 1.091x — **rule wins** |
| cross-regime, +16 measurements | 1.070x | 1.069x — tie |

The "thresholds drift across platforms" hypothesis also fails: the same
rule member is optimal in both the cross-node and intra-node regimes. The
per-site *policy* transfers (1.070x); the best fixed *configuration* does
not (1.545x).

**A language model does not beat the rule on the scalar decision either.**
Given identical facts and the same action space, zero measurements, held
out by message size:

| decider | gmean regret |
| --- | --- |
| GBT / hand rule | 1.038x |
| random search, 8 measurements | 1.060x |
| **LLM one-shot, compiler facts only** | **1.069x** |
| best global configuration | 1.783x |
| context-free lowering | 2.952x |

It beat every search-based and fixed baseline but lost to both the rule
and the trained model. Its failure mode was a generalization error on the
held-out sizes: it fitted a cost model correctly and then assigned 4KB and
16KB to the small-message regime, which costs 1.7x at two points.

This negative is load-bearing rather than embarrassing. It means any later
advantage on the transformation space can be attributed to the action
space rather than to the model being generically stronger.

**The transformation space is rule-shaped too.** This was the space that
was supposed to separate them, since merging is an action no fixed-arity
model can express. With sixteen sites and the split fixed in advance,
three independent language-model runs produced *identical* plans scoring
1.001x — and a two-constant hand rule scored 1.000x. Random search reaches
1.007x with four measurements.

So the second attempt to find a decision this project can pose where a
learned or prompted decider beats a written rule has also come back
negative. Both attempts had large compiler-side effects (1.550x and 2.677x
here) and a decider choice that barely mattered. The honest reading is
that **the value measured so far is in the analysis — what is legal and
what is reachable — and not in the sophistication of whatever picks among
the legal options.**

What has not been tried: a transformation whose legality criterion is
*orthogonal* to size and count. Both spaces tested so far are decided by
the same two quantities, which is exactly the situation a threshold rule
handles. If an orthogonal criterion is added and one line still suffices,
this decision problem is rule-shaped and the paper should say so — the
plan anticipated that outcome and keeps compiler-guided autotuning as the
main line with the model layer demoted.

## 6. Cross-site merging — a headroom check, run before any decider

Merging two *different* call sites whose destinations are adjacent. The
structure looked like the one that had been missing: the more profitable
form carries the stricter legality requirement, and the criterion is
dependence rather than size.

  * `hoist-B` — pull the later transfer up to the earlier one. Merged
    message and full overlap, but needs B's payload final that early.
  * `sink-A` — push the earlier one down. Merged message, gives up A's
    overlap, needs only that A's payload may be delayed.
  * `separate` — leave both where they are.

Measured over 20 points of (message size x gap), before involving a
decider, because a decider scoring well on a space with one dominant
answer reports the absence of a problem rather than solving one:

| | result |
| --- | --- |
| hoisting legal | `hoist-B` wins **20 of 20**; always-hoist *is* the oracle |
| hoisting illegal | `sink-A` 12, `separate` 8, flipping with message size — 1.18x for sink at 1KB, 1.23x for separate at 512KB — but **1.026x in aggregate** |

The acceptance bar was fixed before the run: build the decider experiment
only if the oracle clearly beats both fixed strategies. It does not, so it
was not built.

What the run did measure is worth more than what it went looking for:

| | value |
| --- | --- |
| proving the hoist legal (analysis) | **1.055x** mean, **1.33x** at best |
| choosing well among legal options (decision) | 1.026x |

## 7. Where hand-derivation stops scaling — the ML result

Everything above lives on the cross-node proxy path, where one NIC caps
the outcome at 24 GB/s and every knob is really about issue overhead. That
was a scoping accident, not a choice. The intra-node copy path has knobs an
order of magnitude larger, and unlike the cross-node ones **they interact**.

`ipc_copy_sweep --sweep`, 216 configurations x 9 transfer sizes measured in
a single allocation (which also holds node and IPC mapping fixed, so the
configurations are comparable to each other).

**The axes interact.** All six ordered pairs among (vec, nt, unroll) move
each other's optimum:

```
1024B   nt=0 -> best vec=1        nt=1 -> best vec=16
4096B   unroll=1 -> vec=1   unroll=2 -> vec=4   unroll=4 -> vec=16
```

One boolean moves the optimal vector width by a factor of sixteen.

**And the cost of ignoring that grows with how many knobs are exposed.**
Every subset of the five axes, scored leave-one-size-out with the unused
axes pinned at their globally best value:

| knobs exposed | separable rules | GBT | best global | rules lose by |
| --- | --- | --- | --- | --- |
| 1 | 1.023x | 1.017x | 1.023x | **1.006x** |
| 2 | 1.092x | 1.031x | 1.050x | 1.059x |
| 3 | 1.119x | 1.036x | 1.078x | 1.080x |
| 4 | 1.193x | 1.045x | 1.107x | 1.142x |
| 5 | **1.198x** | **1.050x** | 1.137x | **1.141x** |

Independent per-axis thresholds go from 2.3% to 19.8% off the oracle; a
model over the joint space stays between 1.7% and 5.0%. The gap is
monotone: **0.6% at one knob, 14.1% at five.**

**This subsumes the four negatives below rather than contradicting them.**
Those spaces were effectively one or two interacting knobs, and at one knob
this measurement puts rules and model 0.6% apart — so a hand rule matching
every learned decider was the *correct* outcome there, not a failure to
find the right decider. Whether a rule suffices is a property of how many
interacting knobs the compiler exposes, not of the domain.

The legality criterion in this space is layout, provable by the same stride
analysis as `coalescable`, and getting it wrong is not a slowdown:
`vec > 4` on a strided face is a **GPU memory fault**, in every
orientation, and staging through a contiguous gather buffer does not rescue
it because the fault is in the gather kernel's own strided read. Related
measurements on the same path: `gather` is worth **6.46x** on the
worst-stride face and slightly *harmful* on the contiguous one; face
orientation spans **24.5x**; the face/row thread mapping is flat, a dead
knob recorded so nobody measures it twice.

Caveats: one platform, one access pattern, and the five-axis row averages
over a single subset because only one exists. The `separable` control is
the separable model *class*, not "any rule a human could write" — a person
can condition one axis on another, but that is a table whose size grows
with the number of interacting axes, which is precisely what is being
measured.

## The result that keeps reappearing

Four transformation families, measured independently:

| family | analysis is worth | choosing among legal options is worth |
| --- | --- | --- |
| configuration space (`path`/`B`/`P`/`L`) | **1.56x** | ~0% (hand rule ties GBT and LLM) |
| parameter availability (O5) | **1.53x** | — (legality only) |
| in-loop coalescing | **1.55x** | 0.1% |
| cross-site merging | 1.06x (max 1.33x) | 2.6% |

**The compiler's legality and reachability analysis is worth 1.3-2.7x.
Choosing among whatever it leaves legal is worth 0-6%, and a rule with two
constants captures it.** This is not one space that happened to be easy;
it is four, chosen to be different from each other, and the fourth was
specifically constructed to have an orthogonal legality criterion.

The honest reading is that the contribution this project can demonstrate
is the analysis, not the decider. That is the plan's own Gate 3 fallback:
keep compiler-guided communication optimization as the main line and demote
the model layer. The negative results in the section above are what make
that framing credible rather than a retreat — a paper claiming the analysis
matters and the decider does not is stronger when it has measured both.

---

## Compiler analysis available to the decider

`features.json`, schema v4. Version jumps 2 to 4 on purpose: the emitter
had been left at 2 while the schema doc already described a v3, so a file
claiming 2 may or may not carry those fields.

| field | derived or stored | meaning |
| --- | --- | --- |
| `trip_count` | stored | transfers per phase, from ScalarEvolution |
| `flops_to_first_use` | stored | static issue-to-first-use distance |
| `hk_capable` | stored | host can reconstruct the descriptor pre-launch |
| `legal_paths` | derived | proxy always; trigger/ipc need `hk_capable` and a non-degraded loop |
| `descriptor_reusable` | derived | whole descriptor repeats — stage once, trigger N times |
| `buffer_reusable` | derived | buffers fixed even when offsets vary |
| `batch_size` | stored | transfers released by the same completion point |
| `coalescable` | derived | consecutive transfers are exactly adjacent |

The reuse and legality predicates are derived from the argument
expressions on read so there is a single definition. `batch_size` and
`coalescable` need the CFG and the induction variable respectively.

All of them are conservative by construction. An unmodelled value never
counts as invariant or adjacent, two spellings of the same value compare
unequal, and a loop the pass recognised but could not model disqualifies
the trigger path — because trace synthesis would silently drop the
transfer rather than emit wrong code.

---

## Known issues and caveats

- **4-rank proxy hang.** With two contending rank pairs the proxy path
  wedges on cells that run clean at one pair (trigger leg at 4KB/K=2,
  proxy leg at 256B/K=8/P=8), after completing everything before them.
  Same intermittent proxy/quiet hang seen before. `run_grid.sh` wraps
  every srun in `timeout` so a wedge cannot eat a session. The intra-node
  regime was substituted for the transfer experiment.
- **Additivity is assumed** when scoring transformation plans: sites are
  measured independently and summed. It held to 2.6% when checked directly
  on `mixed_callsite --flips`, but the winning plans here have not been
  measured end-to-end.
- **Single LLM runs.** Both language-model results are n=1. Repeat before
  publishing; the variance of a one-shot decision is unmeasured.
- **One machine.** Everything is Tioga. Cross-platform (the plan's H3) is
  untested.
- `GICC_KERNEL_HOST_MIRROR` goes on the **function**, naming the formal.
  On the parameter, Clang emits a local `llvm.var.annotation` that no
  module-level walk sees, and the site silently stays proxy-only with no
  diagnostic.
- Per-kernel metadata is written in `GICC_MODE=feature-extract`, not
  `discover`.
- `param_availability` needs both `enable_host_wait_mode()` and
  `enable_mixed_dispatch()`, and `GICC_SKIP_DWQ_INIT` must be **unset** —
  `VAR= cmd` sets it to empty, which still counts as set, and the trigger
  BAR then never maps.

## Scope decisions

- **Do not benchmark against MPI or NCCL as the load-bearing comparison.**
  GICC is a vehicle; the method could be built into another runtime. The
  controls that decide whether the work has value are all internal:
  context-free lowering, best global, per-site oracle. MPI stays as a
  sanity data point only. (Standing caveat: GICC beats two-sided MPI 1.10x
  on jacobi3d but loses 1.72x to `MPI_Put`, and `bench_pingpong --mode=mpi`
  uses blocking `MPI_Send`, so its numbers are not a fair comparison at
  all.)
- **NCCL is out as an experimental target.** Tioga is AMD, so the relevant
  library is RCCL, and the tuner-plugin prototype is not reachable here.
  NCCL stays in related work as the visibility-boundary argument: what a
  mature runtime tuner can already see, and therefore what does not count
  as novel.

## Next

Items 1-3 of the previous list are done: sixteen sites with the split
fixed in advance, an orthogonal legality criterion (cross-site dependence
rather than size), and four runs of each model experiment.

The question that remains open is not which decider to use — four families
now agree that it barely matters — but how far the *analysis* side goes:

1. **Fence scope.** Still the largest single unexploited effect measured
   anywhere in this project: 8.81 us vs 2.57 us for a 1MB intra-node copy,
   **3.4x**, and currently hardcoded to `__threadfence_system` at every
   site. Its criterion is pure dependence — which consumer reads the data
   and in which address space — so it belongs with the analysis results
   rather than the decider ones.
2. **Real applications.** Everything above is microbenchmarks. ASF and
   Minimod are on disk and already GICC-integrated. Whether the analysis
   proves anything useful on code nobody wrote for it is untested, and it
   is the question a reviewer will ask first.
3. **A second platform**, for the portability claim (the plan's H3).

If the model layer is to be revisited, the honest place is not another
selection task. It is one where the *action cannot be enumerated at all* —
proposing a transformation from IR rather than picking from a legality-
masked list. That experiment has not been run, and this log should not be
read as having ruled it out; what has been ruled out is that a decider
helps on any selection problem this project has been able to construct.
