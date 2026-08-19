# End-to-end compiler + ML communication experiment

This experiment asks a narrower, executable question than the offline regret
tables in `PROGRESS.md`: can a learned decision, materialized by the compiler,
make the same source run faster than the compiler's no-hint default on two
MI250X nodes without changing its payload?

## Control and treatment

`examples/proxy/ml_path_e2e.cpp` contains 64 distinct, host-knowable 4 KiB
puts released at one completion point.  Both binaries are built from that
exact source and use the same runtime, process-global worker count, launch
geometry, rank placement, warmup, and sample count.

- `ml_path_default`: no hint; cross-node sites take the compiler's staged
  trigger default.
- `ml_path_gbt`: a `GradientBoostingRegressor` emits 64
  `CPU_PROXY_ENQUEUE` site hints, and lowering spreads them over the eight
  launched blocks/rings.

The model is trained on 184 measured trigger/proxy rows from
`docs/experiments/grid/grid_big.csv`.  **Every 4 KiB row is removed before
fitting.**  The query comes from compiler facts, not benchmark constants:
64 legal proxy sites, `size_log2=12`, `batch_size=64`, exact
`flops_to_first_use=0`.  It predicts 98.903 us for trigger and 84.762 us for
proxy.  The held-out measured cell, consulted only as an evaluation, is
98.471 versus 87.179 us (proxy wins by 1.130x).

The payload gate is deliberately stronger than a final min/max.  Each static
site sends a different byte pattern, the receive buffer is cleared after
warmup, and every byte of all 64 regions is checked and hashed.  Runtime
counters independently attribute the operations to staged-trigger versus
proxy-ring execution.  Timings are 100 per-phase samples and report the true
median and IQR.

## Compiler defects exposed by the end-to-end build

The first linked proxy binary exposed two correctness holes that IR-only
tests had missed:

1. Multi-block lowering invented `llvm.amdgcn.grid.size.x` and
   `llvm.amdgcn.workgroup.size.x`; they survived `opt` as unresolved external
   functions and failed at device link.  Lowering now reads the standard HSA
   dispatch packet through `llvm.amdgcn.dispatch.ptr`.
2. Issuing blocks used `slot % gridDim.x`, but ring lanes used the unmodded
   slot and `quiet` drained only ring 0.  The lane now matches the selected
   block, and proxy quiet drains every launched lane touched by preserved
   sites.  Otherwise host `rt.reset()` could hide broken in-kernel quiet
   semantics while making final data appear correct.

`make -C tools/gicc-passes/build -j check-gicc-passes` passes all 38 tests,
including dispatch-packet, lane, and all-ring quiet checks.  Both real HIP
binaries link with no unresolved fake AMDGCN symbols.

## Reproduction

```bash
bash examples/proxy/build_ml_path_e2e.sh
examples/proxy/submit_ml_path_e2e.sh <tag>

# after both bounded jobs complete
python3 examples/proxy/analyze_ml_path_e2e.py \
  docs/experiments/compiler-ml-path/ml-default-<tag>.out \
  docs/experiments/compiler-ml-path/ml-gbt-<tag>.out
```

The submit script requests two nodes and one GPU/rank, applies the model's
eight-worker global choice to both arms, makes the GBT leg depend on a
successful default leg, and gives **each job a two-minute limit**.  It unsets
`GICC_SKIP_DWQ_INIT` and `MPICH_GPU_SUPPORT_ENABLED`; the former is required
for the default trigger BAR, and this binary does not link Cray GTL.

For the 2026-08-17 queued run, the submitted jobs are
`f5qVw8YmBzRV` (default) and `f5qVw8fjvcMd` (GBT).  The artifacts submitted
to Flux were:

| artifact | SHA-256 |
| --- | --- |
| source | `f715ba9c5e34a0e751d8053eb28a4ff08186396574fb587c39e0fa29a56ccc4c` |
| held-out model hint | `6b5ccd58860a9f3acd95b9ab150a95d214b5808f213c8f4cfc0aa893ba505575` |
| measured grid | `c28e3cb69fb78aa4cf9223fb281cf71658a58abd1def661e29351caba0466836` |
| default binary | `06d594c05b58df29402f009b7a62161dddab9d2b4ed9207c44cff7ac486a1d3b` |
| GBT binary | `2089d71ddab3210ac7fde5de6518b479a6ce25c64f964f6a66282938cc5d8afd` |

## Result

All six jobs completed normally on two MI250X nodes.  Every job had a
two-minute allocation limit and actually used 5.9--7.1 seconds.  Run 2
reversed the execution order to expose temporal/node drift.

| run order | default median (IQR), us | GBT median (IQR), us | speedup |
| --- | ---: | ---: | ---: |
| default -> GBT | 119.601 (114.959--122.943) | 80.145 (76.961--83.890) | **1.4923x** |
| GBT -> default | 122.101 (120.232--125.137) | 78.908 (76.308--82.211) | **1.5474x** |
| default -> GBT | 118.426 (113.687--120.806) | 81.880 (78.749--86.100) | **1.4463x** |

The median of the three arm medians is 119.601 versus 80.145 us, or
**1.4923x**.  The geometric mean of paired speedups is **1.4948x**, with
range 1.4463--1.5474x; every pair wins and every pair's IQRs are disjoint.

Correctness and treatment attribution pass independently:

- all six full-payload hashes are `f0399b4213db0383` after clearing the
  receiver following warmup;
- every default timed arm reports exactly `6400 staged, 0 pushed`;
- every GBT timed arm reports exactly `0 staged, 7200 pushed` (64 puts plus
  eight per-block quiet commands in each of 100 phases);
- all 600 timed phases completed and every byte of the 256 KiB region was
  checked after each arm.

Re-run the complete gate with:

```bash
python3 examples/proxy/analyze_ml_path_e2e.py \
  docs/experiments/compiler-ml-path/ml-default-n2-stats.out \
  docs/experiments/compiler-ml-path/ml-gbt-n2-stats.out \
  docs/experiments/compiler-ml-path/ml-default-n2-reverse1.out \
  docs/experiments/compiler-ml-path/ml-gbt-n2-reverse1.out \
  docs/experiments/compiler-ml-path/ml-default-rep3.out \
  docs/experiments/compiler-ml-path/ml-gbt-rep3.out
```

This proves the requested integration statement: compiler facts fed to a
held-out learned decision, materialized as per-site lowering, produce a
correct cross-node communication path that repeatedly beats the same
compiler's no-hint default.  It does **not** show that GBT beats the best hand
rule; the broader negative results in `PROGRESS.md` still say simple rules
match learned selection over this enumerable action space.  It is not yet an
LLM result; the next experiment replaces only the decision producer, while
keeping compiler facts, hint validation, unchanged input source, lowering,
and runtime gates fixed.
