# Minimod paper experiment report

Date: 2026-08-17.  Application scope: **Minimod only**; ASF was not built or
measured.

This experiment closes three paper-facing gaps: it compares the compiler
default with strong fixed, rule, learned, and oracle baselines; measures a
fixed global problem across node/rank layouts; and demonstrates O5 legality
materialization in a real Minimod executable.

Scope boundary: LTO materializes the three transport choices.  The
serial/overlap schedule is a pre-existing, hand-written runtime mode and is
retained only to expose the application's measured cost surface.  Therefore
the six-arm matrix supports transport cost modelling and an upper bound on a
joint policy; it does not establish source transformation, automatic overlap,
or an LLM contribution.

## Experimental contract

- Problem: `--grid 800 --nsteps 100` for the main study.
- Scale/layout cells: 1/2/4/8 nodes and 1/8 ranks per node.
- Candidate actions:
  `{default, DWQ_TRIGGER, CPU_PROXY_ENQUEUE} x {serial, overlap}`.
- Repetition: five independent, exclusive-node allocations per cell.  The
  six arms are run in a deterministic randomized order inside each allocation
  so comparisons are paired on placement.
- Primary time: maximum `Time kernel` over all ranks.  `Time comm` and
  `Time comp` are retained as diagnostics.
- Correctness: every log must contain the exact rank set, every rank's
  full-domain checksum, and exact compiler-route counters.  Checksums must be
  identical across all actions and repetitions of a cell.
- Bounds: three minutes per main allocation, 75 seconds per arm.  O5 uses two
  minutes per allocation.

The 1-node/1-rank cell is retained as a no-communication scaling control but
is excluded from communication-policy selection and scoring.  Policy scores
therefore cannot be won by choosing among equivalent paths using compute
jitter.

## Strong baseline definitions

- **Default:** `default_serial`, which lets the runtime select IPC for local
  peers and staged DWQ for remote peers.
- **Best fixed:** the best single arm among trigger/proxy x serial/overlap,
  selected after measurement and applied to every scored cell.
- **Best global action:** the best single arm among all six candidates.  This
  extra control allows the compiler default itself to win the fixed-action
  comparison.
- **Hand rule (fixed before the full run):** use `trigger_overlap` when every
  neighbor is cross-node (`rpn=1`); use `default_overlap` when local and remote
  peers may coexist (`rpn=8`).
- **GBT over the measured matrix:** gradient-boosted cost prediction over all
  six measured arms, evaluated by leave-one-node-count-out.  Every action cost
  at the held-out node count is absent from training.  Its transport choice is
  compiler-materialized; its schedule choice is the benchmark axis described
  in the scope boundary above.
- **Empirical oracle:** the fastest measured cell median among the six arms.

The final policy table reports geometric-mean regret to the oracle and paired
geometric-mean speedup over `default_serial`, with a deterministic 10,000-draw
bootstrap 95% interval.

## Main scaling and policy results

The strict result below covers **1, 2, and 4 nodes**.  It contains 180 complete
standard samples (six cells x six arms x five allocations), with no ignored
or incomplete logs.  The final 1/2/4-node statistics use PCI exclusively.  An
earlier pair of valid 4-node/pdebug repetitions is retained as auxiliary raw
evidence but excluded from this analysis so partition and node-group effects
cannot confound the scaling curve.

The separately submitted 8-node chain cannot start while two of pdebug's
eight nodes (`tioga39`, `tioga41`) are drained.  A direct `pllm` probe was
rejected because the account is limited to `pdebug,pci`; consequently no
8-node number is inferred or substituted below.

| cell | default serial, s | empirical best | best, s | cell speedup |
| --- | ---: | --- | ---: | ---: |
| 1 node, 1 rank/node (no-comm control) | 3.275660 | trigger serial | 3.275380 | 1.0001x |
| 1 node, 8 ranks/node | 0.494177 | proxy overlap | 0.455222 | 1.0856x |
| 2 nodes, 1 rank/node | 1.712160 | proxy overlap | 1.650280 | 1.0375x |
| 2 nodes, 8 ranks/node | 0.311480 | default overlap | 0.261847 | 1.1895x |
| 4 nodes, 1 rank/node | 1.023250 | proxy overlap | 0.868434 | 1.1783x |
| 4 nodes, 8 ranks/node | 0.211833 | default overlap | 0.159691 | 1.3265x |

The no-communication control is excluded from policy scoring.  Over the other
five completed cells:

| policy | oracle regret (gmean) | speedup over default (gmean, bootstrap 95%) |
| --- | ---: | ---: |
| default serial | 1.159304 | 1.0000x (1.0000-1.0000) |
| best fixed non-default: proxy overlap | 1.119919 | 1.0358x (0.9688-1.0967) |
| best global action: default overlap | 1.007272 | **1.1507x** (1.1121-1.1917) |
| hand rule | 1.007296 | **1.1507x** (1.1121-1.1917) |
| leave-one-node-count-out GBT | 1.000484 | **1.1586x** (1.1209-1.1987) |
| empirical oracle | 1.000000 | **1.1592x** (1.1215-1.1991) |

The direct paired bootstrap supports the small learned advantage rather than
merely comparing two overlapping default-relative intervals.  GBT is
**1.00688x over the hand rule** (95% 1.00357-1.01063) and **1.00687x over the
best global action** (95% 1.00358-1.01059).  It is 1.11850x over the best
fixed non-default arm (95% 1.04457-1.22248), and remains 0.99949x of the
empirical oracle.  GBT selects proxy overlap for the pure cross-node cells and
default overlap for mixed-locality cells; that exactly matches the oracle at
2 and 4 nodes and misses only the 1-node/8-rank cell by 0.24%.

Strong scaling reinforces the route choice.  At one rank per node,
proxy-overlap reaches 3.7733x speedup at four nodes (94.3% efficiency), versus
80.0% efficiency for default-serial.  At eight ranks per node,
default-overlap reaches 2.8575x (71.4% efficiency), versus 58.3% for
default-serial.  Proxy-overlap degrades in this mixed-locality layout at four
nodes, which is why a single forced proxy policy performs poorly despite
winning the pure cross-node cells.

## Problem-size robustness and cross-grid transfer

The complete 1/2/4-node design was repeated at `--grid 400 --nsteps 100`:
another 180 standard samples, all strict-route/checksum clean.  This reduces
volume by 8x and halo bytes by roughly 4x relative to grid 800, making the
application substantially more communication dominated.

| metric | grid 400 | grid 800 |
| --- | ---: | ---: |
| GBT speedup over default | 1.2001x (1.1616-1.2395) | 1.1586x (1.1209-1.1987) |
| GBT oracle regret | 1.004516 | 1.000484 |
| GBT over hand, direct paired | 1.00268x (1.00098-1.00462) | 1.00688x (1.00357-1.01063) |
| GBT over best global action | 1.00255x (1.00060-1.00473) | 1.00687x (1.00358-1.01059) |
| empirical-oracle speedup | 1.2058x | 1.1592x |

The route structure reproduces: proxy overlap wins pure-remote cells while
default overlap wins all mixed-locality cells.  However, within-grid LONO GBT
selects default overlap rather than oracle proxy overlap at grid 400's
4-node/1-rank cell.  The learned advantage is therefore repeatable but its
size and exact decision are problem-size dependent.

Cross-grid evaluation removes the remaining ambiguity by training on one
problem size and scoring only on the other.  Training at grid 800 and testing
at 400 matches the oracle in all 5 policy cells (regret 1.000000) and is
1.00741x over the hand rule (95% 1.00365-1.01158).  Training at grid 400 and
testing at 800 matches 3/5 cells (regret 1.002663) and remains 1.00466x over
the hand rule (95% 1.00182-1.00843).  Thus the data support a **small,
cross-grid learned advantage**, not a broad or large ML claim.

Strong scaling also exposes the regime change.  For one rank per node,
proxy-overlap retains 93.3% four-node efficiency at grid 400 (94.3% at grid
800).  For eight ranks per node, default-overlap falls to 43.1% efficiency at
grid 400 from 71.4% at grid 800; communication and launch overhead dominate
the smaller local workload.

## O5: compiler legality is materialized in a real binary

The O5 executable contains two halo kernels whose pre-lowering source bodies
are byte-identical (255 bytes, SHA-256
`3b12bc3d3b5d8f6295280528ea64682225eb4a438d110bae0ce3b9db1b0894e7`).
Only the compiler-visible host-mirror annotation differs; the lowered machine
paths are intentionally different.  Feature extraction therefore gives the
mirrored loop site `hk_capable=true` and legal paths
`proxy, trigger, ipc`, while the unmirrored site has `hk_capable=false` and
only `proxy`.  One mixed binary materializes `DWQ_TRIGGER` for the former and
`CPU_PROXY_ENQUEUE` for the latter.

Five randomized paired allocations ran on two nodes, one rank/GPU per node,
with `--grid 100 --nsteps 1000`:

| arm | compiler path | median (IQR), s | route count per allocation |
| --- | --- | ---: | ---: |
| mirrored | DWQ trigger | 0.115770 (0.115746-0.115870) | 2000 staged, 0 pushed |
| unmirrored | CPU proxy | 0.114191 (0.113201-0.114602) | 0 staged, 2000 pushed |

The paired geometric mean `unmirrored / mirrored` is **0.984406** (range
0.976911-0.996286), so the proxy-only unmirrored path is about 1.6% faster at
this small, zero-distance, two-transfer Minimod work point.  This is an honest
negative for O5 performance: host-mirror legality makes the trigger path
available and the compiler demonstrably materializes it, but that newly legal
path does not improve this workload.

The same five-allocation design was then repeated at the main paper work point,
`--grid 800 --nsteps 100`:

| arm | compiler path | median (IQR), s | route count per allocation |
| --- | --- | ---: | ---: |
| mirrored | DWQ trigger | 1.713040 (1.713030-1.713430) | 200 staged, 0 pushed |
| unmirrored | CPU proxy | 1.694680 (1.694630-1.694770) | 0 staged, 200 pushed |

Here the paired `unmirrored / mirrored` geometric mean is **0.989190** (range
0.988884-0.989397): proxy remains about 1.08% faster.  The O5 conclusion is
therefore stable across the small legality-focused point and the main
application point; it is compiler legality/materialization evidence, not a
performance win for the newly legal trigger route.

All twenty logs pass checksum validation.  The small-point two-rank checksum
set is `rank0=cb8c08780cf6e9d7, rank1=9ea793e2d4cc5b9e`; the main-point set is
`rank0=b42b67edc1596fdd, rank1=bc43ff4e07a8eea7`.

O5 allocation job IDs:
`f5qWjSrkGYyD`, `f5qWjTHpeQRV`, `f5qWjTudh4Eb`, `f5qWjUNxXpVD`,
`f5qWjV7G9Ksu`; main-point IDs: `f5qYqUxaag1m`, `f5qYqV8AY2h5`,
`f5qYqVGuZo51`, `f5qYqVRn1Vrf`, `f5qYqVaYXFWw`.

## Reproducibility anchors

Source files were copied into an isolated build tree after exact hash checks;
the external Minimod worktree was not edited.

The measurements ran on Tioga MI250X nodes (eight visible GPU GCDs per node).
They predate the branch-history cleanup; exact source, binary, model-input,
and hint hashes below are the durable reproduction anchors.  The preserved
compiler implementation is the learned LTO-lowering commit in this branch.
The build used HIP
6.4.43482 / AMD clang 19.0.0git for `gfx90a`, Cray MPICH 9.0.1 headers and
libraries, and Flux 0.87.0.

| artifact | SHA-256 |
| --- | --- |
| original `target_3d.cpp` | `228b7efe7039fb2a4abb9a4d38009dc24dce7d60722735300d9a7021e6096c1d` |
| original `data_setup.cpp` | `caab6c3f8dde09d2e7bcf883d49ff58b44a8efbaf33845ac30f35c6649e6dfa1` |
| `minimod_default` | `51a643d28fcf466d47053e252e61a212a72ec3961dd5879f224abf7868b08f5c` |
| `minimod_trigger` | `9ae97af07862243529caa4673e5170ab65988c77d02fc363cf2503e2449eddce` |
| `minimod_proxy` | `76f8a32103522a0799afb44f541a0b1ab2e86c7c1a1bde6fbdb2efdc25f354f6` |
| mixed O5 binary | `6da8c52e079246e26eb5342097cda89d3b49be27b4433952bdfae78d70a28d41` |
| O5 extracted features | `cdc4cc7c7416969c4a21827dcba4633658fde51eb838d3f5e934f16a7118c264` |
| O5 hint | `62b01f4ed2311975b43c30c8d8d43f68399a918e190dd329b90b9df55a17c5d8` |
| 1/2/4-node + O5 `summary.json` | `c70972af281d0884edc345f0a2a54ea8ace7e9a474fa5caa66134992264532c3` |
| 1/2/4-node `policies.json` | `15bbc38db48a35b0fe52ea1cfbce9796d976b396785a3d7cf3ede382c102e02c` |
| main-point O5 `summary.json` | `f7593eba05bad96e3bb6ce80c29d545c4460d5dcf2fa998a0e99e11271f92c14` |
| grid-400 `summary.json` | `0ffa7ad0595415ea81eb345426dc30636f00c9ff40188dd2565a5de345de2a68` |
| grid-400 `policies.json` | `1b8d7eab2d5293224976907b85c3f66e6796e33869dea9dc8e50223f80e373aa` |
| cross-grid analysis | `1f564397b9ac6da02827157488f7e86e1a315fdbd127c055b94544c40a63ed7e` |

Raw logs and machine-readable analyses live under
`docs/experiments/minimod-paper/`; the tracked harness and exact reproduction
commands are documented in `examples/minimod_paper/README.md`.
