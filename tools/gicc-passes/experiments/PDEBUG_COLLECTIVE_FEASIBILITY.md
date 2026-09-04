# Pdebug collective-topology feasibility

Status: live read-only scheduler audit; no job submission/cancellation, model
call, provider call, or application-source access/modification.

`audit_pdebug_collective_feasibility.py` separates scheduler feasibility from
performance. It aggregates free and allocated `pdebug` nodes into a stable
usable capacity, verifies the exact drained-node record, and content-addresses
both the unexecuted N8 bundle and its N6 replacement. Transient allocation
occupancy is deliberately excluded from the report identity.

The current audit proves:

- nominal `pdebug` capacity: 8 nodes;
- usable capacity: 7 nodes;
- drained capacity: `tioga41`, one node, scheduler reason
  `node falls out consistently.`;
- N8 is currently unschedulable;
- N6 is the largest even topology currently schedulable;
- the cancelled N8 controller state is not performance evidence;
- the N6 offline freeze is not runtime or model evidence.

The audit binds N8 graph
`sha256:ce569e2575cfdc924004631dcf96a2d81077520c3198c8f78edde3a4405cf806`
and bundle
`sha256:4ed11ecb6a6049b07e18ccf4e88cc7f54f7c1247c119ed1d37a133d6b017f361`
to N6 graph
`sha256:a7aa1f681a64574251aa8e2511550fb8d46b2ea08b5a4aae35fd3865b860a0c0`
and bundle
`sha256:dc243d4f358d3c6b64eb9ff26ff5851ef458f9eed2f7be079424657125edc113`.
The normalized audit ID is
`sha256:88d726fcd7b87d5fc83a46881800a99682d49bceac3d81f5c3f2bb5a0a3b0d3c`;
the serialized report SHA-256 is
`485fadad128ca4662882fa1304132d2e7d04008e5c224d16015c2391a9ab2b01`.

This supports only the experiment-design claim that N6 is the defensible
replacement topology under the current partition state. It says nothing about
which compiler policy is fast and does not make N6 model-visible.

## Reproduction

```sh
python3 tools/gicc-passes/experiments/audit_pdebug_collective_feasibility.py verify \
  --n8-bundle build_ofi/compiler_collective_capacity_n8_hierpipe_v1_20260903 \
  --n6-bundle build_ofi/compiler_collective_capacity_n6_hierpipe_v1_20260904 \
  --report build_ofi/pdebug_collective_feasibility_stable_20260904/report.json
```

Use `emit` with the same bundle arguments and report path to create a new
report when the partition's usable/drained topology changes.
