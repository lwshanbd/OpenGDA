# Source-free communication structural controls

Status: frozen offline compiler controls; no provider call, application-source
edit, or evaluation-runtime result was used.

## Scientific role

The communication decision suite needs a transparent non-model comparator in
addition to the semantic compiler anchor and the separately trained GBT.  The
version-1 structural control intentionally covers only the part of the action
space expressible by a simple deployment rule: uniform proxy versus uniform
trigger issue overhead.  It cannot select a mixed route, early-trigger
placement, or producer-frontier fission.  Consequently, a later LLM result is
not credited for reproducing this rule; conversely, any gain from a schedule
candidate remains attributable to ranking a compiler-generated structural
choice rather than to source generation.

The rule reads the verified private compiler graph, not source.  For each
opportunity it requires:

- positive independent platform calibration for the fixed and per-operation
  proxy/trigger issue costs;
- a positive deployment proxy-worker count;
- one uniform compiler `batch_size` equal to the compiler group size;
- one uniform positive compiler-derived launch-grid block count; and
- exact compiler-generated uniform proxy and uniform trigger candidates.

It estimates

```text
proxy  = proxy_fixed_us
       + proxy_per_op_issue_us * operations
         / min(operations, grid_blocks, proxy_worker_lanes)
trigger = trigger_fixed_us + trigger_per_op_stage_us * operations
```

and selects the lower-cost uniform physical route.  The common data-transfer
term cancels, so this is explicitly an issue-overhead rule, not a full latency
predictor.  Any missing input, missing candidate, or numerical tie falls back
atomically to the semantic anchor for that opportunity.  The ordinary strict
decision bridge revalidates every selected opaque candidate ID before it emits
an LTO hint.

## Frozen real-graph outputs

These controls were emitted before the pending Jacobi producer-fission runtime
result.  The profile contains only the pre-existing independent deployment
calibration; none of the five evaluation results or oracle labels is readable
by the tool.

| Entry | Result | Control ID | Decision / hint / control file SHA-256 |
|---|---|---|---|
| Jacobi | semantic default fallback: launch-grid concurrency unknown | `sha256:211e974fb9438b10039c9177a25303331f7d1289b2665acd000ceeaeb41fc769` | `f396ac9d3af1cf0d5ff1341c2e4c41bf54715a7798a2e741d3ae193f822a13f3` / `a28786e1e8a670215140686fd82397e0c6a5ce5baa50c3627461675fceca76a7` / `359a4112260d8fefec55d3a69c14b87afddafec51257fd10b87552d6a04bdbf4` |
| Minimod | uniform proxy; estimated issue cost 8.242 us vs 9.946 us | `sha256:0a88d94a2030ab944f58413dbf2d4ed36443221735d4e273b74a3738edb8b838` | `5044401604a37870d974468b20daf9ac79f51fc1bf0cf9df989698b568a547af` / `28cb7f7c2683ae3ae5687781d05c3548bb49cf2702f0d756937ef981cc5a91c1` / `cfacf26a23520abc0040f29b0fca3d22bfe4eb660d5b66261597d8587abfb3cf` |
| minimal matrix multiply | semantic default fallback: launch-grid concurrency unknown | `sha256:e47831d2cddad5fe849e8c97892d254af49517e735ce72d397597209a6e8e602` | `275b92a8ad64afe2bcc3d7ba54ce6dcafe004038b4f4c477f7060e308eb73cf7` / `eb69135b61aacc5a9ff65f1965604c5b8f6311d0d498c83fe92c85e72995862a` / `bdbd25e18884ed9705534a84985b15abf8325d4cbaaedc51de13628798adeb43` |
| mixed-side-effect LTO | uniform trigger; estimated issue cost 9.946 us vs 11.114 us | `sha256:a4463c417a1f4a55ca6d6feeb577aca832131a14919f7a295928407aa4082a48` | `3ea170d4d1809b01f91e0e2deb4a93e06936da0e46b23546018da84c8c2d02a2` / `de76e4e8fb4ed0a911cb0d6dab792663d99c47ce7c218c28655f1ce615c5e093` / `5ca2013988ed9548d90b187eee22c0053bfd3dacc15fca67358fe39c073455ee` |
| loop-carried LTO | proxy; estimated issue cost 8.242 us vs 8.913 us | `sha256:258a7d203396e1bce050ed19a6b93498d7b2ce12a6ab66f470161b9dd80a3121` | `c3a86264f335092964019c1b6911f20a87a19437cceddd069b8dd6ae980e939b` / `fa8be4e9ac347bdaf7b8a4859e044df05cba119ea2a696951e2c8c6fe4db6e92` / `ea8cba6fb94fb81b49d88d627958bc11101819a2c24ecebdce413ba82fec60d6` |

The mixed-LTO and loop selections are not performance claims.  They are
preregistered compiler baselines whose regret can be measured only in a later
matched runtime experiment.  The two explicit fallbacks are also informative:
an LLM may receive symbolic launch relations that a scalar formula cannot
consume, but it must still choose only an existing compiler candidate and must
beat the unchanged anchor at runtime before that richer input is credited.

## Reproduction

From the repository root, repeat for each of `jacobi`, `minimod`,
`mm_minimal`, `mixed_lto`, and `loop_lto`:

```sh
label=jacobi
python3 tools/gicc-passes/python/gicc_comm_structural_heuristic.py \
  --graph build_ofi/compiler_fact_coverage_20260904/portfolio/$label/group-graph.json \
  --decision build_ofi/compiler_decision_suite_20260904/controls/communication-$label-decision.json \
  --hint build_ofi/compiler_decision_suite_20260904/controls/communication-$label-hint.json \
  --control build_ofi/compiler_decision_suite_20260904/controls/communication-$label-control.json
```

The output control records the graph ID, exact rule, per-opportunity
diagnostic, decision and hint IDs, and the compiler-only boundary.  Re-emission
is deterministic, and the full Python regression suite exercises successful
selection, every fail-closed path, transform exclusion, graph binding, and
insensitivity to an injected evaluation-result field.
