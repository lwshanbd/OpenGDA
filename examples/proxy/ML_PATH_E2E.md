# End-to-end compiler + ML communication result

This is the executable control for the offline decision tables in
`PROGRESS.md`: can a learned hint, actually materialized by the compiler,
make unchanged source run faster than the no-hint default on two MI250X
nodes?

## Design

`ml_path_e2e.cpp` contains 64 distinct, host-knowable 4 KiB puts released by
one completion point. `build_ml_path_e2e.sh` compiles the same source twice:

- `ml_path_default`: no hint, so cross-node puts use staged DWQ trigger;
- `ml_path_gbt`: 64 `CPU_PROXY_ENQUEUE` hints preserve the device puts and
  spread them over eight blocks/rings.

`ml_path_decider.py` fits a deterministic `GradientBoostingRegressor` on 184
trigger/proxy rows from `docs/experiments/grid/grid_big.csv` after removing
**all** 4 KiB rows. The query is read from compiler output: 64 legal proxy
sites, `size_log2=12`, `batch_size=64`, exact
`flops_to_first_use=0`. The model predicts 98.903 us for trigger and 84.762 us
for proxy. The held-out measured cell, used only for evaluation, is 98.471
versus 87.179 us.

Feature schema v5 also recovers `grid=(8,1,1)` and `block=(1,1,1)` directly
from the host LTO launch callsite. Together with the hash-bound platform
profile's eight-worker deployment, this exposes the concurrency fact needed by
an LLM or another compiler decider without providing application source.

Both arms use the same runtime, eight proxy workers, launch geometry, rank
placement, warmup, and sample count. The receiver is cleared after warmup;
each static site has a distinct byte pattern; every byte of the 256 KiB
region is checked and hashed. Runtime counters separately prove which
communication path executed. Each arm records 100 individual phase timings
and reports the true median and IQR.

## Result

All jobs used two MI250X nodes with one rank/GPU per node. Every allocation
was capped at two minutes and actually ran for 5.9--7.1 seconds. Run 2
reversed execution order.

| order | jobs (default / GBT) | default median (IQR), us | GBT median (IQR), us | speedup |
| --- | --- | ---: | ---: | ---: |
| default -> GBT | `f5qVw8YmBzRV` / `f5qVw8fjvcMd` | 119.601 (114.959--122.943) | 80.145 (76.961--83.890) | **1.4923x** |
| GBT -> default | `f5qWEEwgYbVR` / `f5qWEEpojvgf` | 122.101 (120.232--125.137) | 78.908 (76.308--82.211) | **1.5474x** |
| default -> GBT | `f5qWEyAWpJUw` / `f5qWEyHX2uhR` | 118.426 (113.687--120.806) | 81.880 (78.749--86.100) | **1.4463x** |

Median of the three arm medians is 119.601 versus 80.145 us, **1.4923x**.
The geometric mean of paired speedups is **1.4948x**, range
1.4463--1.5474x. Every pair wins and every pair's IQRs are disjoint.

All independent gates pass:

- all six full-payload hashes are `f0399b4213db0383`;
- every default arm is exactly `6400 staged, 0 pushed`;
- every GBT arm is exactly `0 staged, 7200 pushed` (64 puts plus eight quiet
  commands in each of 100 phases);
- all six Flux results are `COMPLETED`, and no experiment job remains active.

The binaries and model inputs submitted to Flux were:

| artifact | SHA-256 |
| --- | --- |
| source | `f715ba9c5e34a0e751d8053eb28a4ff08186396574fb587c39e0fa29a56ccc4c` |
| held-out model hint | `6b5ccd58860a9f3acd95b9ab150a95d214b5808f213c8f4cfc0aa893ba505575` |
| measured grid | `c28e3cb69fb78aa4cf9223fb281cf71658a58abd1def661e29351caba0466836` |
| default binary | `06d594c05b58df29402f009b7a62161dddab9d2b4ed9207c44cff7ac486a1d3b` |
| GBT binary | `2089d71ddab3210ac7fde5de6518b479a6ce25c64f964f6a66282938cc5d8afd` |

Raw generated logs live under `docs/experiments/compiler-ml-path/` (the
repository intentionally ignores `docs/`). Re-run their complete gate with:

```bash
python3 examples/proxy/analyze_ml_path_e2e.py \
  docs/experiments/compiler-ml-path/ml-default-n2-stats.out \
  docs/experiments/compiler-ml-path/ml-gbt-n2-stats.out \
  docs/experiments/compiler-ml-path/ml-default-n2-reverse1.out \
  docs/experiments/compiler-ml-path/ml-gbt-n2-reverse1.out \
  docs/experiments/compiler-ml-path/ml-default-rep3.out \
  docs/experiments/compiler-ml-path/ml-gbt-rep3.out
```

Rebuild and submit another bounded pair with:

```bash
bash examples/proxy/build_ml_path_e2e.sh
examples/proxy/submit_ml_path_e2e.sh <tag>
```

## Compiler defects found by the real link

The experiment exposed two correctness holes not covered by the old IR-only
test:

1. invented AMDGCN grid/workgroup declarations survived `opt` as unresolved
   externals and failed at device link; lowering now reads the HSA dispatch
   packet through `llvm.amdgcn.dispatch.ptr`;
2. issuing blocks used `slot % gridDim.x`, ring lanes used the unmodded slot,
   and quiet drained only ring 0; lanes now match issuing blocks and proxy
   quiet drains every touched lane.

`make -C tools/gicc-passes/build -j check-gicc-passes` passes all 38 tests,
including dispatch-packet, lane, and all-ring quiet coverage. Both linked HIP
binaries have no unresolved fake AMDGCN symbols.

## Scope of the claim

This proves the requested integration statement: compiler facts feed a
held-out learned decision, the compiler emits a different legal cross-node
communication path, and that executable repeatedly beats the no-hint
default. It does **not** prove GBT beats the best hand rule; the broader
negative results in `PROGRESS.md` still show rules matching learned selection
over this enumerable action space. It is also not yet an LLM result; the LLM
arm must use the same compiler facts, validated hint boundary, unchanged
source, and executable gates before it can be compared.
