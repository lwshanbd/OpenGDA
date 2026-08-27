# Compiler-only communication-plan capacity experiment

## Question

This experiment asks whether an LLM can exploit a richer compiler-level
communication action space without seeing or modifying program source.  It
does **not** ask whether an LLM is a better tabular classifier than GBT.

The compiler owns discovery, legality, and materialization:

1. HIP/LTO extracts communication facts and compiler proof obligations from
   one unchanged source file.
2. The compiler emits a content-addressed communication opportunity graph.
3. A planner may return only one compiler-generated `candidate_id` per
   `opportunity_id`.
4. The bridge rejects unknown fields, code, IR, invented candidates, stale
   graph IDs, missing selections, and illegal numeric metadata.
5. The LTO pass independently re-proves every requested structural transform.
   Rejection is fail-closed; it cannot become model-authored IR.

The model output therefore affects compiler hints and the final compiler IR,
never the benchmark source.

## V1 structural action space

The first action is deliberately narrow and executable.  For a constant-trip
`put_no_db` loop, the compiler proves that both source and destination offsets
advance by exactly the constant transfer size.  It then exposes three legal
candidates:

- `p` — keep all loop operations on device proxy lanes;
- `t` — reconstruct and stage all descriptors, then trigger the batch;
- `c` — replace the proven-contiguous loop with one larger PUT in host LTO IR.

The unchanged `compiler_lto_eval.cpp` workload contains two such opportunities,
ordered as `eval_adjacent_batch` and `eval_far_batch`.  Their product is the
exact nine-arm set `plan_pp` through `plan_cc`.  `c` reduces 16 x 4096 B to one
65536 B PUT and 64 x 4096 B to one 262144 B PUT.

Sites outside this graph are not silently left to the model.  The compiler
fixes the reusable loop to trigger, the device-loaded dynamic offset to proxy,
and default-capable static sites to the existing `IPC_OR_DWQ` baseline.

## Preregistered gates

The gates are evaluated in this order:

1. **Source invariant:** SHA-256 before and after every build must remain
   `d707d5b6773a5299b2d8a19a6c832f82b719bc3be08b01f250a481500ecb486a`.
2. **Boundary/schema:** graph and response validation tests must pass, including
   rejection of model-authored code/IR and compiler-fixed non-opportunity sites.
3. **Compiler materialization:** all nine exact controls must build through the
   real HIP/LTO pipeline.  IR verification must observe a 16/64-entry batched
   enqueue for `t`, a single 65536/262144-byte enqueue for `c`, and neither host
   materialization for `p`.
4. **Runtime correctness:** every arm must produce the expected data hash and
   exact staged/proxy route counters for all seven workload scenarios.
5. **Structural headroom:** the exact best of the nine executable arms is
   compared with `plan_tt`, the current descriptor-batch behavior.  The primary
   score is the geometric mean of the `adjacent` and `far` scenario medians.
   A paired bootstrap 95% interval is reported.
6. **LLM capacity:** only after gates 1–5 show a non-trivial, reproducible action
   space is a new provider experiment authorized and run.  The frozen prompt
   contains only the graph.  Twenty independent responses are accepted only
   through the candidate-ID schema, compiled by the same pipeline, and mapped
   to an already measured exact arm before any additional runtime job.

If gate 5 finds no measurable headroom, V1 is a compiler/materializer result,
not evidence against LLM reasoning.  The next action must expand the
compiler-generated graph (for example cross-site grouping, descriptor reuse,
or synchronization placement) and preregister a new protocol before invoking a
model.

## Runtime protocol and scheduler discipline

- Platform: Tioga, two nodes, one rank and one MI250X GCD per node, cross-node
  Slingshot/CXI peer.
- Queue: `pdebug` only.
- One scheduler job may be pending or running at a time.  Each replicate is one
  allocation that executes the nine arms sequentially on the same nodes.
- Arm order follows an 18-row odd-order Williams-style rotation; replicates are
  submitted manually one at a time and validated before the next submission.
- Default workload settings are 10 warmups and 100 timed iterations per
  scenario.  A bounded 120-second timeout applies to each arm.
- The exact controls contain no model output.  `model_invoked=false` is recorded
  in their content-addressed manifest.

Commands:

```bash
bash examples/proxy/build_compiler_comm_plan_eval.sh facts
bash examples/proxy/build_compiler_comm_plan_eval.sh controls
python3 examples/proxy/compiler_comm_plan_eval.py verify-ir \
  --graph build_ofi/compiler_comm_plan_eval/generated/opportunity-graph.json \
  --manifest build_ofi/compiler_comm_plan_eval/generated/controls/manifest.json \
  --ir build_ofi/compiler_comm_plan_eval/generated/ir

# Run only when no earlier task is pending/running. This submits one job.
bash examples/proxy/submit_compiler_comm_plan_eval.sh 1

python3 examples/proxy/analyze_compiler_comm_plan_runtime.py \
  --manifest build_ofi/compiler_comm_plan_eval/generated/controls/manifest.json \
  --json docs/experiments/compiler-comm-plan-capacity/runs/\
compiler-comm-plan-controls-v1/summary.json \
  docs/experiments/compiler-comm-plan-capacity/runs/\
compiler-comm-plan-controls-v1/raw/*.log
```

## Current pre-runtime evidence

- 22 Python schema/control/runtime-analysis tests pass.
- 43/43 LLVM lit tests pass, including positive coalescing and negative
  unproved-transform rejection.
- All nine exact controls build from the unchanged source.
- Automated host-LTO IR verification passes for all nine arms.
- No new LLM/provider call has been made for this V1 graph.
- No structural runtime job has been submitted yet; an older `pdebug` job must
  finish first to preserve the one-job-at-a-time rule.

Content IDs and artifact hashes are frozen in `protocol-v1.json`.
