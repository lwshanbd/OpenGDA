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

> **RETRACTED. The crossover does not exist.** Half the rows behind this
> section were timing copies that never ran: the sweep sized its stream
> pool by the environment value rather than the per-configuration one, so
> every `nstream=4` point launched three quarters of its copy onto
> uninitialized stack handles and came out about three times faster for
> having done a quarter of the work. Fixed in `44a4943`, which also makes
> each point clear its destination and check that the data arrived.
>
> Re-measured clean (`joint8_fixed.csv`, 23,328 rows, **0 dropped**, 2,592
> configurations at every size; 16 MB best 86.88 us against 87.03 us from
> an independent single-configuration run):
>
> | axes | separable rules | GBT | rules lose by |
> | --- | --- | --- | --- |
> | 1 | 1.005x | 1.014x | 0.991x |
> | 2 | 1.018x | 1.019x | 0.998x |
> | 3 | 1.035x | 1.060x | 0.976x |
> | 4 | 1.063x | 1.064x | 0.999x |
> | 5 | 1.079x | 1.099x | 0.982x |
> | 6 | 1.108x | 1.144x | 0.969x |
> | 7 | 1.169x | 1.187x | 0.985x |
> | **8** | **1.191x** | **1.404x** | **0.848x** |
>
> `rules lose by` is below 1.000 at every axis count. Separable rules
> match or beat the model everywhere, with the widest margin where the
> retracted table claimed the model won hardest. The old numbers below
> remain only so the correction is auditable; do not quote them.

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
Eight axes now — vec, nt, unroll, block, nstream, grid, post-copy fence
scope, direction — giving 2592 configurations x 9 sizes. Every subset of
the axes is scored leave-one-size-out with the unused axes pinned at their
globally best value, which is what a compiler exposing only those knobs,
the rest hardcoded, would face.

| knobs exposed | separable rules | GBT | best global | rules lose by |
| --- | --- | --- | --- | --- |
| 1 | 1.020x | 1.028x | 1.020x | 0.992x |
| 2 | 1.043x | 1.056x | 1.040x | 0.988x |
| **3** | 1.074x | 1.074x | 1.067x | **1.000x** |
| 4 | 1.089x | 1.055x | 1.088x | 1.033x |
| 5 | 1.139x | 1.063x | 1.123x | 1.071x |
| 6 | 1.198x | 1.079x | 1.190x | 1.110x |
| 7 | 1.247x | 1.139x | 1.259x | 1.094x |
| 8 | **1.337x** | **1.207x** | 1.361x | **1.108x** |

**There is a crossover at about three interacting knobs.** Below it a model
is no better than independent thresholds and sometimes slightly worse;
above it the model wins and the margin grows. Separable rules go from 2.0%
to 33.7% off the oracle, the model from 2.8% to 20.7%. Neither is good at
eight knobs — the model is less bad.

**This subsumes the four negatives below rather than contradicting them.**
Those spaces were effectively one or two interacting knobs, which is exactly
the regime where this curve says a rule should win — so a hand rule matching
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

Caveats: one platform, one access pattern, and the eight-axis row averages
over a single subset because only one exists; rows for 2-6 axes sample at
most 20 subsets and are marked as sampled in the tool output. The `separable`
control is
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
with the corrected intra-node copy sweep it is six selection families, and
the model does not win any of them. The defensible contribution is the
compiler analysis that determines reachability and legality, plus an
honest measurement of how little remains for a learned selector.

The honest reading is that the contribution this project can demonstrate
is the analysis, not the decider. That is the plan's own Gate 3 fallback:
keep compiler-guided communication optimization as the main line and demote
the model layer. The negative results in the section above are what make
that framing credible rather than a retreat — a paper claiming the analysis
matters and the decider does not is stronger when it has measured both.

---

## Compiler analysis available to the decider

`features.json`, schema v6. Version jumps 2 to 4 on purpose: the emitter
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
| `launch_grid`, `launch_block` | host LTO callsite | constant `dim3` geometry; dynamic dimensions stay null |
| `grid_blocks`, `threads_per_block` | derived | products of proven launch dimensions |
| `size_bytes`, `launch_contexts` | host LTO callsite | constants bound back to kernel formals; all contexts retained when they differ |

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

## 8. Does the intra-node mechanism reach a real application? — feasibility check

Everything in sections 1-6 lives on the cross-node proxy path and concludes
that the decider barely matters. Section 7 had appeared to supply an
intra-node ML foothold, but the corrected measurements remove that result.
The underlying mechanism gap is still real, so before wiring its knobs into
the pass the cheap feasibility question comes first: does Minimod expose
enough of that path for any selector to matter?

**(a) Minimod's halo shape, from the source.** `halo_size = 4 * (ny+2*ly) *
(nz+2*lz) * sizeof(float)`, and the layout `IDX3_l_D` puts x slowest while
the decomposition splits x, so the face is one **contiguous** slab. At grid
1000 that is **15.5 MB, two transfers per timestep**, no `quiet` in the
kernel (the host `reset()` does the waiting).

Two consequences, both from reading rather than measuring:

- The layout-legality argument does not fire here. `vec > 4` faulting on a
  strided face is the strongest legality result on this path, and Minimod's
  face is contiguous — it sits on the easy end of the 24.5x orientation
  span, so the analysis has nothing to prove.
- No batching or concurrency headroom, which an earlier sweep had already
  found the hard way: two puts per step, every CPU-side runtime knob flat.

**(b) The eight-knob sweep never measured what GICC actually runs.**
`ipc_copy_sweep`'s sweep loop pins `c.mech = "kernel"` at
`examples/proxy/ipc_copy_sweep.cpp:272`, so all 23,329 rows of
`joint8.csv` are tuned copy *kernels*. GICC's same-node IPC path is a host
`hipMemcpyAsync` on a dedicated stream, issued from the synthesized trace
function before the kernel launch — grep for it in `ofi_runtime.hpp`. The
two never met.

So the section 7 crossover is a real effect *inside the kernel-copy
family*, and whether that family is even worth reaching from
`hipMemcpyAsync` is a question the data cannot answer. This is the control
that should have been in the sweep from the start.

**(c) The largest number in section 7 does not reproduce.** Scoring
`joint8.csv` per size against the best single global configuration put 16 MB
— Minimod's working point — at **2.989x**, the largest regret in the table,
against 1.013x at 1 MB and 1.000x at 4 MB. A 3x discontinuity at the one
size that matters for the application is worth re-measuring on its own, and
it does not survive:

| 16 MB | joint8.csv | re-measured |
| --- | --- | --- |
| oracle configuration (`nstream=4`) | 30.78 us | 87.03 us |
| best-global configuration (`nstream=1`) | 92.00 us | 94.18 us |
| **regret** | **2.989x** | **1.08x** |

`nstream` was supposed to be the mechanism; measured on its own it is worth
1.04x (87.03 against 90.94), not 3x. The 30.78 us row is in-process
carryover across 2592 back-to-back configurations — the same noise already
seen on O1. 16 MB anchors the high end of the size axis, so the eight-axis
row of the scaling table is inflated by it; see below.

**(d) What the control found instead.** The `memcpy` rows that had never
been measured say GICC's production intra-node path is a long way off, and
in a strongly size-dependent way (`run_minimod_point.sh`, median of 200):

| transfer | GICC today (`hipMemcpyAsync`, push) | best copy kernel | gap |
| --- | --- | --- | --- |
| 64 KB | 11.24 us | 2.60 | **4.3x** |
| 256 KB | 12.13 | 2.58 | **4.7x** |
| 1 MB | 17.47 | 2.60 | **6.7x** |
| 4 MB | 40.19 | 6.33 | **6.4x** |
| **16 MB** | **129.16** | **87.03** | **1.48x** |

Replacing the host `hipMemcpyAsync` with a copy kernel is worth 4-7x from
64 KB to 4 MB. That is a much larger effect on the real path than anything
the eight knobs choose between, and it was invisible for as long as the
sweep compared tuned kernels only against each other.

Direction matters for the mechanism too: `hipMemcpyAsync` push beats pull
(129 against 332 us at 16 MB), while for the copy kernel pull beats push
(87 against 119). GICC already pushes, so it is on the right side for the
mechanism it uses and the wrong side for the one it would move to.

**(e) Verdict for Minimod: this path does not reach it.** Three facts
compose, and they all point the same way:

| | |
| --- | --- |
| face size | 16 MB — exactly where the memcpy-to-kernel gain collapses, 1.48x not 6.7x |
| face layout | contiguous — the strongest legality result (`vec > 4` faults on a strided face) never fires |
| comm share | **7.4%** of runtime (comm 0.0359 s, comp 0.448 s, 8 ranks on 1 node, grid 800, 100 steps) |

7.4% x (1 - 1/1.48) = **2.4% end to end**, and that overstates it, because
the measured `comm` also contains the barrier and `reset()`, not only the
copy. The eight-knob decision on top of that is worth 1.08x of a 2.4%
slice.

This is a clean negative and it is the answer the feasibility check was
built to get: the intra-node copy space is not where this work should be
anchored for Minimod, and finding that out cost one allocation instead of
two weeks of wiring.

## 9. A learned decision reaches a real LTO lowering

The offline tables did not prove that a learned decision survived the
compiler.  `examples/proxy/ml_path_e2e.cpp` closes that engineering gap with
unchanged source between arms.  LTO extracts 64 legal 4 KiB sites in one
completion group; a deterministic GBT trained after excluding every 4 KiB
row chooses the proxy path, emits `gicc-hint-v1`, and the lowering pass
materializes it.

Across three paired two-node runs, default DWQ-trigger medians are
119.601/122.101/118.426 us and GBT-proxy medians are
80.145/78.908/81.880 us: **1.4948x geometric-mean speedup**.  Route counters
show exactly `6400 staged, 0 pushed` for every default arm and
`0 staged, 7200 pushed` for every GBT arm; all full-buffer hashes are
`f0399b4213db0383`.  This proves learned compiler-fact -> hint -> LTO
materialization, not GBT superiority: a hand rule selects the same path.

Raw logs and the complete gate are under
`docs/experiments/compiler-ml-path/`; reproduction is in
`examples/proxy/ML_PATH_E2E.md`.

## 10. Real-application cost surface and legality evidence

The retained Minimod matrix contains two grids, 1/2/4-node scaling, exact
route counters, full-domain checksums, and O5 legality materialization.  At
grid 800, leave-one-node-count-out GBT is 1.00688x over the preregistered hand
rule (95% 1.00357-1.01063); cross-grid transfer remains 1.00466-1.00741x over
that rule.  The learned advantage is therefore small and problem-size
dependent, but supported by more than the synthetic proxy benchmark.

The scope is deliberately narrow.  LTO materializes the transport choice.
Minimod's serial/overlap schedule is an already hand-written benchmark mode
and is retained only as a measured cost/oracle axis; neither the compiler nor
a model discovered or rewrote it.  O5 is the clean compiler-only result:
host-mirror facts change the legal set and the pass emits different routes in
one binary, although the newly legal trigger route is 1.08-1.6% slower at the
two measured work points.

## 11. The LLM boundary is now compiler-fact -> decision -> LTO hint

`tools/gicc-passes/python/gicc_llm_bridge.py` implements the provider-neutral
part of the correct experiment.  It turns schema-v6 `features.json` plus a
measured platform profile into a content-addressed
`gicc-llm-dossier-v1`; no source text is accepted.  A model response must bind
to that dossier hash, cover every compiler site exactly once, and choose only
from the site's `legal_actions`.  Accepted choices become `gicc-hint-v1` for
the second LTO invocation.

The boundary fails closed.  A malformed, stale, incomplete, or illegal
response applies no model choices: host-capable sites retain `IPC_OR_DWQ`,
and compiler-proven proxy-only sites are pinned to `CPU_PROXY_ENQUEUE`.
Forced IPC is withheld until topology proves the peer is same-node.  The
existing 64-site end-to-end facts produce a deterministic dossier with only
`proxy`, `trigger`, and `default` available.  Schema v6 extracts
`grid=(8,1,1)` and `block=(1,1,1)` from the real host LTO callsite, so a model
can relate eight issuing blocks to the measured eight-worker proxy deployment
without seeing application source.

Schema v6 also fixes the granularity boundary: repeated host calls of one
kernel are aggregated into one device-op decision with explicit
`launch_contexts`.  It also propagates constant launch arguments back to
device formals, so a parameter-sized transfer can become a numeric
`size_bytes` fact at LTO without consulting source. Different/dynamic contexts
remain explicit rather than being collapsed into a false constant.

Two real HIP compilation gates cover both cases.  In `bench_mixed_lto`, the
device metadata calls the size a formal while the host launch binds it to
4096; v6 reports `size_kind=param`, `size_bytes=4096`, and `size_log2=12`.
In `bench_pingpong_lto`, two source-level launch expressions share one device
op and carry runtime size/count values; v6 emits one decision record with
`static_launch_sites=2`, one unknown context of multiplicity two, and the LLM
bridge accepts it instead of rejecting duplicate site IDs.

The unchanged `decider_e2e.cpp` also produces a two-site v6 dossier.  The FAR
site carries compiler-proven `trip_count=64` and a 205-op lower bound on
issue-to-first-use distance, with `proxy/trigger/default` legal.  The DYN
site's offsets come from a device load, so `hk_capable=false` and its legal set
is exactly `proxy`.  This validates heterogeneous compiler facts and legality
in one dossier; it is not yet the larger frozen evaluation workload.

This is infrastructure, not a new performance result.  Ten bridge tests and
all 41 pass tests pass, including pass-side rejection of an invented dispatch;
no LLM trial is counted yet.

## 12. A frozen compiler-only workload now exposes a real decision gap

`examples/proxy/compiler_lto_eval.cpp` freezes seven heterogeneous scenarios
and ten device-operation sites behind the intended boundary. Every arm uses
the identical source (SHA-256
`d707d5b6773a5299b2d8a19a6c832f82b719bc3be08b01f250a481500ecb486a`);
the only prospective model input is the content-addressed compiler dossier
`sha256:2299725094d5d805e6f8732227a2b11c28c57ff3ea194cf32d8e66c65c3a1aa5`.
The source is neither included in the prompt nor writable through the output
schema. Accepted decisions are legal-action labels that the real LTO lowering
materializes.

The workload varies compiler-visible size, batch count, descriptor reuse,
adjacency, issue-to-use distance, launch geometry, and host-knowability. A
device-loaded-offset site is deliberately proxy-only; it tests legality and is
excluded from decision regret. The current batched host placeholder exposed a
real contract bug during this gate: the feature extractor had advertised IPC
and hybrid default although those lowerings cannot materialize a batched loop.
The extractor and bridge now expose only proxy/trigger there, and fail-closed
fallback pins such sites to trigger.

Four allocation-paired, order-rotated control replicates produced 112/112
correct payload/route records. The best action is stable in all four
replicates for every one of the six decision-bearing scenarios: trigger for
the 256 B and 1 MiB single-op cases, proxy for the reusable, adjacent, distant,
and four-static-site cases. Relative to the measured better uniform legal
action in each scenario, the compiler default has **1.1031x** geometric-mean
regret and selects two of six winners; the existing analytic hand rule has
**1.0604x** regret and selects one of six. This is the missing evidence that a
nontrivial compiler-level selection problem exists. It is not evidence that
an LLM solves it.

For the four static sites, all `2^4` proxy/trigger assignments were compiled
into distinct binaries and run in one bounded two-node allocation. Every
payload hash and exact route count passed. The action-space winner is `1111`
(all proxy) at 39.532 us; the next mask is 1.0981x slower and `0000` (all
trigger) is 1.4357x slower. This is an exact assignment-space check with one
allocation-level replicate; repeatability claims come from the four-replicate
uniform controls above. Full hashes, jobs, raw logs, and analyzers are under
`docs/experiments/compiler-lto-eval/`.

## Next

The next model experiment stays entirely on the compiler path:

1. Freeze the scoring protocol around the checked-in dossier and control
   results. A trial is a decision response to `prompt.txt`, not a source patch;
   invalid responses score as the deterministic materializable fallback.
2. Produce repeated LLM and GBT decisions from the same compiler facts, with
   prompt/dossier/model/response hashes retained. No application source is sent
   to a model, and no candidate may generate code.
3. Feed every accepted response through the second LTO build, then retain
   binary hashes, exact routes, full payload hashes, and paired runtime. Compare
   default, hand, GBT, LLM, uniform controls, and measured oracle without using
   the oracle labels as model input.
4. Add same-node IPC only as a separate hash-bound deployment profile. Runtime
   rank placement is not generally knowable at LTO and must never be guessed
   into static compiler features.
5. Only after that dispatch gate is clean, expose another pass-materialized
   action family such as batching/coalescing. The pass must generate and prove
   candidates; the model may rank them but may not write code.

This two-phase compile keeps provider calls out of the linker, makes every
decision cacheable and replayable, and preserves the intended research
question: whether a language model can reason over compiler-level evidence
well enough to improve the decisions made by an LTO pass.
