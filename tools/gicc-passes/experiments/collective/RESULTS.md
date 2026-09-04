# Compiler-collective experiment results

This is a post-run result ledger. Preregistered protocols remain immutable
after a runtime monitor records their hashes; new outcomes are recorded here
instead of mutating those frozen inputs.

## n4 hierarchy-pipeline confirmation (2026-09-03)

The one confirmation job `f5tnzCa7d2qV` completed cleanly on the same
`tioga[36-39]` nodes used by its scout. All three rotated blocks shared that
job and node list; all 81 arm/size rows reported zero errors. The frozen scout
policy slowed the geometric mean by about 9.6%: its paired speedup was
`0.903943x` with exact bootstrap interval `[0.902342, 0.905190]`. Pipe4 beat
the unpipelined tree at 1 KiB in zero of three blocks. Pooled medians put the
unpipelined tree first at all nine sizes, so both best-uniform-to-pointwise
headroom and maximum single-size headroom were exactly `1.0`.

All six preregistered confirmation criteria failed. The original 1 KiB
`3179.296 us` scout measurement is retained as a documented outlier, not
optimization headroom. The audited confirmation result is
`sha256:4c9f0532ec6fdd0957fb1ea4e86510e68b0fd5debe4f31689d312c1af2e70df2`.
No hierarchy-pipeline model prompt or provider call is justified for the n4
profile.

## n8 scheduling infeasibility and n6 replacement (2026-09-04)

The preregistered n8 scout job `f5to6fn64qBM` produced no runtime data. The
`pdebug` partition has eight nominal nodes, but `tioga41` has been drained
since 2026-07-21 with reason `node falls out consistently`; only seven nodes
can therefore be allocated. Flux supplied no start estimate for the eight-node
request. The job was cancelled after this condition was established so that
smaller compiler-only experiments could continue. Its terminal controller
state is an infrastructure failure, not a negative capacity or performance
result.

A separate n6 replacement is preregistered in `HIERPIPE_N6_PROTOCOL.md`. Six
nodes/48 ranks are the largest even topology schedulable on the currently
usable `pdebug` resources. Its frozen compiler graph is
`sha256:a7aa1f681a64574251aa8e2511550fb8d46b2ea08b5a4aae35fd3865b860a0c0`;
its verified offline bundle is
`sha256:dc243d4f358d3c6b64eb9ff26ff5851ef458f9eed2f7be079424657125edc113`.
The bridge accepted all 4096 graph-bound policies and all control builds passed
IR/provenance checks. This is preparation evidence only: the n6 runtime job
has not been submitted and no model or provider was invoked.
