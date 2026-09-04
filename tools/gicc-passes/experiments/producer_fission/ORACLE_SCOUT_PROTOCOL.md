# Producer-frontier fission oracle scout

Status: frozen execution protocol for the first runtime measurement of the
compiler-only producer-frontier fission oracle.  It is not an LLM comparison
and cannot establish a paper performance claim by itself.

## Boundary

- Both executables come from unchanged `examples/ofi/jacobi.cpp`.
- The only treatment difference is the default-off compiler/LTO switch
  `GICC_PRODUCER_FISSION_ORACLE=1` used while building the fission executable.
- Both builds use the same frozen all-`DWQ_TRIGGER` compiler hint.
- No provider request or model output participates in this scout.

## Allocation and ordering

- Queue: `pdebug` only.
- Exactly one allocation: two nodes, sixteen ranks, eight ranks and GPUs per
  node, eight CPU cores per rank, twenty-minute limit.
- The allocation is submitted only after the preceding N8 collective scout has
  reached Flux's `clean` event.  No job is cancelled or reprioritized.
- Sizes are square command-line requests of 1024 and 4096.  Jacobi rounds the
  global y extent down to an equal sixteen-rank decomposition.
- Four paired blocks use the balanced arm order `AB`, `BA`, `BA`, `AB`, where
  A is the fused baseline and B is compiler fission.  Every executable already
  performs its three source-defined warm-up steps.
- Every run uses `-niter 200 -nccheck 10`.  Norm-check iterations deliberately
  retain the original fused launch; the other iterations exercise the guarded
  compiler schedule.

## Gates

The monitor requires a clean pdebug completion, unchanged hashes for both
binaries and all control scripts, the expected rank/mesh configuration, one
positive runtime, and one finite final norm per run.  Each paired baseline and
fission result must have the same completed iteration count and final norm
within `1e-6` relative or `1e-7` absolute tolerance.

For each size, the analyzer reports all four paired speedups
`baseline_seconds / fission_seconds`, their median, and the number above one.
The exploratory oracle-headroom gate passes when at least one size has at least
three wins and median speedup of at least 1.03.  A pass only justifies a larger
compiler-oracle confirmation; a failure masks this candidate from any future
model experiment.  Neither outcome alone supports the paper's final claim.
