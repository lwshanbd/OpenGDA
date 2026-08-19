# Frozen compiler-level LTO decision evaluation

This experiment evaluates decisions made from compiler facts and materialized
by the LTO passes. It is deliberately not a source-editing experiment.

## Boundary

Every arm is compiled from the unchanged
`examples/proxy/compiler_lto_eval.cpp` source. Its frozen SHA-256 is
`d707d5b6773a5299b2d8a19a6c832f82b719bc3be08b01f250a481500ecb486a`.
The build checks this hash before and after every facts/control/candidate
pipeline.

The only permissible model input is the canonical projection of real LTO
facts plus the hash-bound deployment profile:

- feature schema: 6
- decision sites: 10
- dossier ID:
  `sha256:2299725094d5d805e6f8732227a2b11c28c57ff3ea194cf32d8e66c65c3a1aa5`
- `features.json`:
  `a8991c66736f85cf43e4364904f5bfa73b0810c0845424bd2748b0a1610db190`
- `dossier.json`:
  `c77f4cff5e14552491c7a8cf8129724885c2f9a933898460375f7cadaca32274`
- `prompt.txt`:
  `51cb67f351a8f6e02e64186d1da4d0bff15fc1dcc1000960cb84df12a7b95b94`
- `profile.json`:
  `03c404978aca58b8120e27e73c9c8613f724a6139f2828b3c3019eab776f6105`

The bridge requires a response bound to that dossier ID, exact coverage of
all sites, and an action from each compiler-provided legal set. The LTO pass,
not the model, implements the lowering. Malformed, stale, incomplete, or
illegal responses fail closed. No provider was called and no LLM result is
reported here.

## Frozen workload

Seven separately timed scenarios cover ten real device-operation sites. The
compiler facts distinguish message size, loop batch, descriptor reuse,
coalescing opportunity, issue-to-use distance, launch geometry, and
host-knowability.

| scenario | compiler-visible shape | legal decision |
| --- | --- | --- |
| tiny-k1-grid1 | 256 B, one op, one block | proxy / trigger / default |
| reuse-k32-grid1 | 4 KiB, 32-op reusable descriptor loop | proxy / trigger |
| adjacent-k16-grid1 | 4 KiB, 16 adjacent loop ops | proxy / trigger |
| far-k64-grid8 | 4 KiB, 64 ops, long static distance, eight blocks | proxy / trigger |
| large-k1-grid8 | 1 MiB, one op, eight blocks | proxy / trigger / default |
| dynamic-k1-grid1 | device-loaded offsets, not host-knowable | proxy only |
| static4-k4-grid4 | four static 4 KiB sites, four blocks | proxy / trigger per site |

The dynamic site is a legality control. Since every policy materializes
proxy there, timing variation between separately linked binaries is not
decision regret and is excluded from the primary score.

## Control results

`compiler-lto-controls-v1` contains four allocation-paired replicates, with
the four arm orders rotated. Each arm has 100 timed iterations per scenario.
All 112 result records passed complete-region hashes and exact staged/pushed
route-count checks. Jobs `f5qkHLcrNhBd`, `f5qkHLkNF431`, `f5qkHLsDZjZR`,
and `f5qkHLz1vSX9` completed cleanly.

Geometric means of the four per-replicate medians are:

| scenario | default (us) | proxy (us) | trigger (us) | hand rule (us) | measured best action |
| --- | ---: | ---: | ---: | ---: | --- |
| tiny-k1-grid1 | 25.565 | 28.128 | 25.080 | 28.051 | trigger |
| reuse-k32-grid1 | 350.253 | 324.505 | 347.193 | 346.864 | proxy |
| adjacent-k16-grid1 | 182.239 | 169.512 | 181.874 | 178.910 | proxy |
| far-k64-grid8 | 698.929 | 634.673 | 697.521 | 692.361 | proxy |
| large-k1-grid8 | 75.222 | 78.759 | 74.773 | 78.265 | trigger |
| dynamic-k1-grid1 | 29.127 | 29.366 | 29.000 | 28.728 | forced proxy; excluded |
| static4-k4-grid4 | 56.567 | 41.136 | 56.649 | 40.592 | proxy |

The same winner appears in all four replicates for each of the six
decision-bearing scenarios. Relative to choosing the better uniform legal
action per scenario, the compiler default has 1.1031x geometric-mean regret
and matches two of six winners; the existing hand rule has 1.0604x regret
and matches one of six. This establishes a measurable decision problem. It
does not establish that a learned or language-model policy can solve it.

## Exact static-four action-space check

`compiler-lto-static4-oracle-v1` enumerates all `2^4` proxy/trigger
assignments for the four static sites in one bounded two-node allocation,
alternating order extremes to reduce monotonic drift. All 16 binaries have
distinct hashes. All records passed the complete payload hash and exact
route-count checks. Job `f5qkNmce2Dio` completed on `tioga30,32` with status
zero.

The exact action-space winner in this run is mask `1111` (all four sites use
proxy) at 39.532 us. The next configuration is 1.0981x slower; mask `0000`
(all trigger) is 56.755 us, or 1.4357x slower. This is a complete enumeration
of legal static-four assignments, but only one allocation-level replicate;
the four-replicate uniform controls above carry the stronger repeatability
claim.

## Reproduction and gates

```bash
cmake --build tools/gicc-passes/build --target check-gicc-passes -j2
bash examples/proxy/build_compiler_lto_eval.sh freeze
bash examples/proxy/build_compiler_lto_eval.sh controls
bash examples/proxy/submit_compiler_lto_eval.sh paired compiler-lto-controls-v1
python3 examples/proxy/analyze_compiler_lto_eval.py \
  docs/experiments/compiler-lto-eval/runs/compiler-lto-controls-v1/raw/*.log

bash examples/proxy/build_compiler_lto_eval.sh oracle
bash examples/proxy/submit_compiler_lto_oracle.sh \
  compiler-lto-static4-oracle-v1
python3 examples/proxy/analyze_compiler_lto_oracle.py \
  docs/experiments/compiler-lto-eval/runs/compiler-lto-static4-oracle-v1/raw/*.log
```

Candidate model responses use
`build_compiler_lto_eval.sh candidate RESPONSE.json NAME`. That command
strictly validates the response, lowers the same source through LTO, and
records a distinct binary hash. Provider invocation is intentionally outside
the build and is not part of this evidence set.
