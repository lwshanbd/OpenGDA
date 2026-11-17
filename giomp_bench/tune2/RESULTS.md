# Tuning campaign 2 (ICS'27): X = GiOMP tuned vs MPI tuned, W = pass knobs vs compiler default

Every number is a confirmation-round median (one allocation per app, >=3 runs each,
all checks passed). Agents: MIMIR + deepseek-flash, budget 5 rounds each side.
Plugin giomp_ib_test/passes-tune2, runtime lib10 (split exclusion per kernel).

| app | MPI default | MPI tuned | GiOMP ref | GiOMP tuned | X | W | pass choices of C_giomp |
|---|---|---|---|---|---|---|---|
| FT C, 2 nodes | 1.245 s (27%) | = default | 1.066 s | 0.957 s | 1.30x | 10.2% (9.5% net) | fftz_send teams 264, chunk 1024 |
| Himeno XL, 3 nodes | 2.956 s | 2.610 s | 2.867 s | 2.645 s | 0.99x | 7.8% | both sites unpipelined |
| MiniMod 1000^3, 3 nodes (mode 3) | 8.320 s | 7.739 s | 7.691 s | 6.965 s | 1.11x | 9.4% | sites unpipelined, split pml+inner, STABLE |
| MiniMod 350^3, 3 nodes (mode 5) | 1.381 s | 1.374 s | 1.447 s | 1.432 s | 0.96x | 1.0% | split pml, chain off, STABLE |
| MoE, 2 nodes (vs Alltoallv) | 20.22 ms | 18.92 ms | 14.92 ms | 13.50 ms | 1.40x | 9.5% | dynamic on gemm, DYN_PIECE_SHIFT 21 |
| Jacobi, 2 nodes | 1.583 s | 1.007 s | 1.709 s | 0.993 s | 1.01x | 41.9% | both sites unpipelined |
| miniWeather, 2 nodes | 2.860 s | 2.824 s | 2.782 s | 2.740 s | 1.03x | 1.5% (noise) | both sites unpipelined |
| CloverLeaf, 2 nodes | 2.448 s | 2.347 s | 1.956 s | - | 1.20x | 0 | (no pipeline site) |

Confirmation outputs: */confirm/confirm.out; mappings: */map*.json.
Manual (not agent) runs: ft/pick.*, ft/xval_* (step 6), minimod/confirm1000_badsetup (script bug).
