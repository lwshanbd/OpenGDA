# GICC OFI Runtime Tunable Sweep — Results

**Date:** 2026-04-30
**Cluster:** Tioga (AMD MI250X + HPE Slingshot/CXI, ROCm 6.4)
**Branch:** `feat/window-streams-decider`
**Commit at run time:** `a6ca113`
**Headline:** **Negative result.** Every runtime-side tunable we tested on minimod / jacobi / mm_minimal / barrier_bench is at noise floor. The OFI/CXI runtime layer's tuning surface is saturated for these workloads.

---

## Summary table

| Knob | Range tested | Workload(s) | Spread | Verdict |
|---|---|---|---|---|
| `GICC_WINDOW` (sliding window depth W) | {1, 2, 4, 8, 16, 32} | barrier_bench × 3 rank counts | ≤ 2.2% | **flat** |
| `GICC_STREAMS_MAX` (IPC stream pool size) | {1, 2, 4, 8} | minimod, jacobi, mm_minimal × 2 grids | 0.4–1.5%, monotone *worse* | **flat or slightly hurts** |
| `FI_MR_CACHE_MAX_COUNT` | {0, 1024, 4096} | minimod 32-rank | 13% faster at 1024+, **but produces wrong answers** (min_u/max_u garbage) | **fake win — correctness violation** |
| `FI_CXI_RX_MATCH_MODE`, `FI_CXI_RDZV_THRESHOLD`, `FI_CXI_OPTIMIZED_MRS` | varied | minimod 32-rank | < 0.3% | **flat** |
| `GICC_CQ_SIZE` (libfabric CQ depth) | {64, 128, 256, 512, 1024} | minimod 32-rank | < 0.3% | **flat** |

Total experiments: **126 sweep runs + 6 libfabric env-var configurations + 5 CQ-size configurations**, all on Tioga `pdebug`.

---

## §1 Sliding window W (Sweep-A)

`Barrier::PREFETCH_DEPTH` was promoted from `static constexpr int = 8` to a runtime parameter (`Barrier` ctor argument, threaded from `Runtime` via `GICC_WINDOW` env var). The static cap `BARRIER_SIGNAL_SLOTS` was bumped to 32 so any W ∈ [1, 32] works without rebuild.

| ranks | W=1 | W=2 | W=4 | W=8 | W=16 | W=32 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 21.50 | 21.50 | 21.20 | 21.40 | 21.60 | 21.30 |
| 16 | 63.40 | 63.80 | 63.70 | 62.40 | 63.00 | 63.60 |
| 32 | 80.00 | 81.00 | 80.40 | 80.90 | 80.90 | 80.70 |

Units: µs per barrier (median of 3 reps, `barrier_bench --gicc 1000`).

**Observation:** spread is ≤ 2.2% within each rank-fixed row, well inside Tioga's run-to-run jitter band. The Task 8 smoke result (W=8 vs W=4 → 3.3% gap) did not replicate at scale; it was noise.

**Mechanistic interpretation:** `PREFETCH_DEPTH` controls how many DWQ ops the host queues ahead of the GPU during continuous-mode barriers. The data says: **at any tested rank count, even W=1 keeps up with the GPU's barrier rate.** Setup cost on the host is below the OFI counter wait + DWQ trigger cost, so prefetching does not amortize anything visible. Increasing W only enlarges the slot pool that's never contended.

**Methodological caveat:** `barrier_bench` is the only OFI workload exercising `gicc::barrier()`; minimod / jacobi / mm_minimal use `MPI_Barrier` or `hipDeviceSynchronize`. We did NOT test W under a workload that has substantial compute between barriers (which would make prefetching potentially meaningful). The result is solid for *this* workload class.

---

## §2 IPC stream pool size (Sweep-B)

`Runtime::ipc_stream_` (single `hipStream_t`) was replaced with `std::vector<hipStream_t> ipc_streams_` of size `n_streams_max_` (env `GICC_STREAMS_MAX`, default 8). The dispatch-lowering pass now emits `gicc_runtime_ipc_stream_indexed(rt, idx)` reading `stream_index` from `hint.json`. For this sweep we did NOT alter the decider, so all sites use `stream_index=0`. Varying `n_streams_max_` therefore only varies the pool *capacity*, not which streams the puts use — but every test_gicc / minimod path indeed creates the full pool, exercising the alloc path.

| workload | grid | s=1 | s=2 | s=4 | s=8 |
|---|---:|---:|---:|---:|---:|
| jacobi | 512 | 6.500 | 6.500 | 6.500 | 6.600 |
| jacobi | 1024 | 6.600 | 6.500 | 6.600 | 6.500 |
| minimod | 500 | 28.416 | 28.446 | 28.518 | 28.819 |
| minimod | 1000 | 86.463 | 86.516 | 86.624 | 86.673 |
| mm_minimal | 2048 | 838.601 | 842.571 | 841.450 | 841.244 |
| mm_minimal | 4096 | 3326.93 | 3334.80 | 3335.72 | 3339.79 |

Units: ms (median of 3 reps).

**Observation:** monotone slightly *worse* with more streams on minimod; flat-with-noise everywhere else.

**Mechanistic interpretation:** GICC's same-node IPC fast path uses `hipMemcpyAsync` to peer's IPC mapping. Allocating an 8-stream pool has nontrivial setup cost (hipStreamCreate × 8) and per-call selection overhead with no overlap benefit, because:
1. Minimod has only 2 puts per kernel; the IPC stream pool can't be saturated.
2. Same-node IPC is dominated by HBM-to-HBM bandwidth, which a single stream already saturates.

The "more streams = more concurrency" intuition fails here because we're not bottlenecked on stream-issue rate.

**Methodological caveat:** this experiment varied `n_streams_max_` but always used `stream_index=0` for every put. A future experiment that actually distributes puts across stream indices (round-robin or per-peer-hash) might show different behavior. But given the size and shape of these workloads, the ceiling is low.

---

## §3 libfabric environment variables (Tier B)

Quick check of widely-known CXI tunables, no code changes:

| Configuration | minimod 32-rank median (ms) | Δ vs baseline | Verdict |
|---|---:|---:|---|
| baseline `FI_MR_CACHE_MAX_COUNT=0` | 86.66 | — | reference |
| `FI_MR_CACHE_MAX_COUNT=1024` | 75.23 | **−13.2%** | **fails correctness — see below** |
| `FI_MR_CACHE_MAX_COUNT=4096` | 75.22 | **−13.2%** | **fails correctness** |
| `FI_CXI_RX_MATCH_MODE=hardware` | 86.74 | +0.1% | flat |
| `FI_CXI_RDZV_THRESHOLD=8192` | 86.64 | −0.0% | flat |
| `FI_CXI_OPTIMIZED_MRS=0` | 86.73 | +0.1% | flat |

**The 13% MR-cache "win" is a correctness violation.** With `FI_MR_CACHE_MAX_COUNT=1024`, minimod produces:
- Most ranks: `FINAL min_u = 0.000000, max_u = 0.000000` (no wave propagation observed).
- Rank 16: `FINAL min_u = -1178.444702, max_u = 757.767761` (garbage values).

Expected (per `benchmarks/Minimod_MPI/CLAUDE.md` verification section): `FINAL min_u, max_u ≈ -0.2, 0.14`.

This is the well-known libfabric MR cache + GPU memory hazard: the cache does not invalidate when ROCm IPC mappings are torn down and re-issued, so the NIC writes RDMA payloads to stale physical pages. The convention `FI_MR_CACHE_MAX_COUNT=0` in `CLAUDE.md` exists for this reason; the speedup it gives up is real but unrecoverable without an upstream libfabric/ROCm fix.

The other three CXI tunables made no measurable difference on a 32-rank minimod run.

---

## §4 Completion-queue depth (Tier A)

Made `cq_attr.size` in `fabric_dwq_context.hpp` (line ~223) tunable via `GICC_CQ_SIZE` (default 128, range [16, 16384]).

| `GICC_CQ_SIZE` | minimod 32-rank median (ms) |
|---:|---:|
| 64 | 86.73 |
| 128 (default) | 86.76 |
| 256 | 86.57 |
| 512 | 86.77 |
| 1024 | 86.81 |

Spread: 0.27% across an 8× variation in CQ depth. **No signal.** The default 128 is comfortably oversized for minimod's traffic shape.

---

## §5 Why the runtime tunables are flat

Synthesizing across §1–§4: the OFI runtime path on Tioga has **no host-side queueing or batching bottleneck** for the workloads we ran. Specifically:

- **Host enqueue rate ≫ GPU dispatch rate.** `Runtime::reset()` drains in microseconds; CQ never fills; the IPC stream pool is never saturated.
- **DWQ trigger latency dominates per-message cost.** This is a NIC-side characteristic of CXI's deferred-work-queue mechanism. The host can deliver work faster than the NIC can fire it.
- **Same-node IPC is HBM-bandwidth-bound, not stream-issue-rate-bound.** Splitting across 2/4/8 streams just adds setup overhead; the wire is already saturated by 1 stream for the message sizes here.

The implication for GICC's research framing: **runtime-layer ML decisions cannot extract perf from these workloads.** Any meaningful gain over the GDA reference must come from changes that modify the *shape* of the dispatch — fewer messages (cross-kernel aggregation, batching), earlier-issued work (pre-launch trace prefetching), or smarter peer-affinity reordering that overlaps cross-node DWQ trigger with same-node IPC. These are LTO-pass-side changes, not runtime parameters.

---

## §6 Code artifacts (preserved on this branch even though not perf-relevant)

The plumbing is still useful as **infrastructure**: the env vars and tunability hooks let future workloads (or the CPU-proxy path, if ever implemented) sweep these parameters cheaply.

| Artifact | Purpose | Commit |
|---|---|---|
| `Barrier(comm, window_size)` ctor | W as ctor arg | `736f5cd` |
| `Runtime::window_size_` + `GICC_WINDOW` env | per-Runtime W | `7e000c5` |
| `examples/ofi/barrier_{test,bench}.cpp` reading `GICC_WINDOW` | examples honor W | `2d16940` |
| `Runtime::ipc_streams_` vector + `GICC_STREAMS_MAX` | IPC stream pool | `658ca1f` |
| `gicc_runtime_ipc_stream_indexed` extern "C" | indexed accessor | `658ca1f` |
| `SiteHint::stream_index` + dispatch-lowering routes via index | per-call-site stream selection | `db99700` |
| Lit test for non-default `stream_index` IR lowering | regression guard | `3251cb0` |
| `BARRIER_SIGNAL_SLOTS = 32` (was 8) | static cap for runtime W | `190ae15` |
| `GICC_CQ_SIZE` env var | CQ depth tunable | `a6ca113` |
| `tools/gicc-passes/tests/sweep/{sanity.sh,sweep_W.sh,sweep_streams.sh,extract_latency.py}` | reusable sweep harness | `a0cb424`, `4c3f1e5`, `2f28304`, `c1f5d13` |
| `tools/gicc-passes/tests/sweep/data.csv` | 126-run dataset | `9043548` |

---

## §7 What we did NOT do (deliberate)

- **No decision-tree training.** With four flat knobs, there is no signal to train on; fitting a tree to 126 noise samples is anti-research.
- **No integration into `gicc_decider.py`.** The decider stays at static defaults until a knob with signal is found.
- **No pivot to add new optimization knobs.** Cross-kernel aggregation, pre-launch trace prefetch, peer-affinity reordering, and CPU-proxy multi-channel scheduling are all candidates for *future* work, but require nontrivial implementation effort and are out of scope of this branch.

---

## §8 Reproducing

```bash
# from /p/lustre2/shan4/opengda on branch feat/window-streams-decider

# Sanity gate (~3 min)
./tools/gicc-passes/tests/sweep/sanity.sh

# W sweep (~5 min, 54 runs on barrier_bench)
./tools/gicc-passes/tests/sweep/sweep_W.sh

# Streams sweep (~10 min, 72 runs across 3 workloads)
./tools/gicc-passes/tests/sweep/sweep_streams.sh

# Tier B + Tier A: ad-hoc /tmp/tier_b.sh and /tmp/tier_a.sh referenced in this doc;
# regenerate from the env-var lists in §3 and §4.
```

Data: `tools/gicc-passes/tests/sweep/data.csv`.
All sruns use `-p pdebug` (do **not** use `-p pci` — it has long unpredictable wait times on Tioga).
