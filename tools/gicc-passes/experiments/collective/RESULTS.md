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
