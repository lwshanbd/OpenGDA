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

`run_transform_sites.sh` measures every legal action at six call sites
chosen to disagree with each other.

| site | shape | coalescable | best action | best | default | headroom |
| --- | --- | --- | --- | --- | --- | --- |
| A | 256B x 64 | yes | merge64/blocks1 | 20.6 | 215.4 | 10.44x |
| B | 256B x 64, gaps | no | merge1/blocks4 | 105.8 | 218.3 | 2.06x |
| C | 4KB x 64 | yes | merge32/blocks1 | 30.8 | 218.8 | 7.10x |
| D | 64KB x 32 | yes | merge8/blocks2 | 107.3 | 121.2 | 1.13x |
| E | 256B x 64, d=400us | yes | merge64/blocks1 | 419.8 | 615.8 | 1.47x |
| F | 4KB x 32, gaps | no | merge1/blocks4 | 67.5 | 118.1 | 1.75x |

The best **single global action must be legal at every site**, and one
site that is not provably adjacent forces the whole program to `merge=1`.
Scored on the held-out sites that is 1.087x, against 1.425x for the
context-free lowering and 1.000x for per-site choice. This is the argument
for per-site specialization as a measurement rather than an assertion.

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

**The transformation-space comparison is not yet discriminating.** The LLM
produced a legal plan matching the oracle at all three held-out sites
(1.000x) — but so did random search at 4 measurements (1.001x) and the
hand rule (1.002x). The test set is too easy: three sites, and at two of
them the top six actions are within 0.6% of each other. The sites with
real headroom (A at 10.44x, C at 7.10x) landed in the training split. Do
not re-split after the fact; add sites and fix the split up front.

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

1. Extend the transformation site set (~16), fixing the train/test split
   before measuring, and report per-site choice sensitivity so flat sites
   do not dilute the result.
2. Add a transformation whose legality criterion is *orthogonal* to the
   present ones — fence scope is the candidate: it is decided by which
   consumer reads the data and in which address space, and is currently
   hardcoded to `__threadfence_system`. Measured at 8.81 us vs 2.57 us for
   a 1MB intra-node copy (3.4x). If a one-line rule still suffices once an
   orthogonal criterion is present, this decision problem is rule-shaped
   and should be reported as such.
3. Repeat both language-model experiments enough times to report a
   distribution.
