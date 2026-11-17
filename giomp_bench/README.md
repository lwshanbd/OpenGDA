# giomp_bench

Build, run and tuning scripts, and the benchmark sources we wrote or ported,
for the GiOMP experiments on Maple (GH200 + InfiniBand). A snapshot of
`/shared/data1/Users/l1065028/giomp_bench`, code only: no binaries, logs or
profiles, and no copies of third-party sources (CloverLeaf, hypre, Himeno's
original, NPB-CPP). Scripts use absolute paths on Maple.

- `tune2/`: the second tuning campaign (MPI compute-only tuning, C_mpi mapped
  onto GiOMP, GiOMP pipeline-only tuning, confirmation rounds); per-app kernel
  mappings in `*/map*.json`, results summary in `tune2/RESULTS.md`.
- `mm/`, `himeno/`, `jacobi/`, `moe/`, `miniweather/`, `npbft/`, `graph/`,
  `cloverleaf/`, `hypre/`: per-program build and run scripts and sources.
