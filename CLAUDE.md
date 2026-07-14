# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GICC (GPU-Initiated Communication and Coordination) is an HPC communication library. Kernels emit a small set of calls — `gicc::put_no_db`, `gicc::flush`, `gicc::quiet` — and the implementation handles RDMA dispatch to same-node IPC peers (via `hipMemcpyAsync` on a dedicated stream) or cross-node peers (via libfabric Deferred Work Queue triggered by an MMIO write at flush). Two backends:

- **OFI / CXI / AMD** (`src/gicc/platform/ofi/`) — primary development on **Tioga** cluster
- **MLX5 / InfiniBand / NVIDIA** (`src/gicc/platform/mlx5/`) — second backend on **MAPLE** cluster

The compilation pipeline is an **LLVM LTO pass plugin** at `tools/gicc-passes/`. Seven cooperating passes (3 device-side + 4 host-side) loaded via `-fpass-plugin=libgicc-passes.so`. They identify GICC API calls, validate the host-knowability invariant, synthesize a host-side trace function before each kernel launch, and lower per-call dispatch decisions (IPC / DWQ / hybrid). The LTO pipeline is the **only** code-generation path — the legacy AST plugin has been deleted.

For a complete narrative description of the system, read `docs/notes/lto_pass_中文总结.md` (gitignored). The design spec lives at `docs/superpowers/specs/2026-04-29-gicc-lto-pass-design.md` (also gitignored — `docs/` is excluded by `.gitignore`).

## IMPORTANT: Tioga Cluster Environment (primary dev)

The OFI backend is the active codebase. Tioga has the ROCm 6.4 toolchain at canonical paths — no module loads needed for builds.

```bash
ROCm:        /opt/rocm-6.4.0/
Clang/LLVM:  /opt/rocm-6.4.0/lib/llvm/bin/clang++
hipcc:       /opt/rocm-6.4.0/bin/hipcc
libfabric:   /opt/cray/libfabric/2.1/
Cray MPI:    /opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/
```

Prefer `srun -p pci -t 2` (the default debug queue is congested). **`pci` and `pdebug` are interchangeable for these runs — if one is full, use the other.** In practice `pci` schedules small/single-node jobs fast but can stall on multi-node (4+ node) requests; `pdebug` is the better fallback for multi-node jobs (watch its ETA — it sometimes shows hours). For 32-rank minimod runs, also export `PMI_MAX_KVS_ENTRIES=512` and `FI_MR_CACHE_MAX_COUNT=0`.

```bash
PMI_MAX_KVS_ENTRIES=512 FI_MR_CACHE_MAX_COUNT=0 \
    srun -p pci -N 4 -n 32 --ntasks-per-node=8 -t 2 ./run.sh \
    ./main_hip_gicc_hipcc_gicc --ngpus 32 --grid 1000 --nsteps 100
```

### MAPLE (NVIDIA + IB)

Reserved for future Phase 4 (NVPTX + MLX5 LTO support). Module loads:

```bash
module use /shared/data1/Projects/CSE_HPC/apps/modules/aarch64/maple
module load gcc/11.4.1 cuda/12.4.131 nvhpc/24.5.cuda_12.4 openmpi-gcc/5.0.5/cuda.12.4 cmake/3.29.6
```

## Directory Structure

```
src/gicc/
├── launch.hpp                   # Top-level gicc::launch + GICC_LAUNCH_SITE annotation
├── gicc.hpp / gicc_types.hpp    # Public API: Runtime, Buffer
├── gicc_device.cuh              # Public device API: put_no_db / flush / quiet
├── bootstrap/                   # MPI / PMI2 init
├── coll.hpp                     # Collective ops
└── platform/
    ├── ofi/                     # AMDGCN + Slingshot/CXI backend
    │   ├── ofi_runtime.hpp      # gicc::Runtime — register_buffer, exchange,
    │   │                        #   put_no_db, prepare, reset
    │   ├── ofi_device.cuh       # struct DeviceCtx (2 fields) + 4 user APIs
    │   ├── runtime_helpers.{h,cpp}  # extern "C" symbols the LTO IR calls
    │   ├── launch.hpp           # gicc::launch wrapper (annotated)
    │   └── internal/            # libfabric / fabric / DwqWorkBuilder bits
    └── mlx5/                    # NVIDIA + IB backend (LTO not yet wired here)

tools/gicc-passes/               # LLVM LTO pass plugin
├── CMakeLists.txt               # builds libgicc-passes.so (-fno-rtti)
├── include/                     # 7 pass headers + 5 data-structure headers
├── src/                         # 7 passes + Plugin entry + helpers
│   ├── PluginEntry.cpp
│   ├── GICCPassConfig.{h,cpp}   # env-var config (GICC_MODE / META_DIR / ...)
│   ├── KernelInventory.h        # GICCKernelInfo / GICCCallSite / GICCOpKind
│   ├── SiteId.{h,cpp}           # site_id formatting + collectGICCSites helper
│   ├── HKAnalysis.{h,cpp}       # isHK utility (recursive)
│   ├── MetadataIO.{h,cpp}       # KernelTemplate JSON read/write
│   ├── TraceTemplateBuilder.{h,cpp}  # IR → KernelTemplate (uses LoopInfo)
│   ├── LaunchSiteInventory.h
│   ├── GICCDeviceDiscovery.{h,cpp}
│   ├── GICCHKAnalysis.{h,cpp}
│   ├── GICCDeviceLowering.{h,cpp}
│   ├── GICCHostDiscovery.{h,cpp}
│   ├── GICCFeatureExtraction.{h,cpp}
│   ├── GICCTraceSynthesis.{h,cpp}
│   └── GICCDispatchLowering.{h,cpp}
├── python/
│   ├── gicc_decider.py          # rule-based decider stub
│   └── schema.md                # features.json + hint.json schemas
└── tests/
    ├── lit/                     # ~22 lit IR snapshot tests
    ├── integration/              # 4 L2 end-to-end cases on Tioga
    ├── run_perf.sh               # 32-rank minimod perf regression
    └── smoke_load.sh             # -fpass-plugin smoke test

examples/ofi/                    # HIP examples using gicc::launch
├── test_gicc.cpp                # 4-put loop sanity test (2 ranks)
├── mm_minimal.cpp               # Ring-topology distributed matmul
├── jacobi.cpp                   # Jacobi solver with halo exchange
├── benchmark.cpp                # DWQ micro-benchmark
└── ...

examples/gicc/                   # CUDA examples (MLX5 backend)
benchmarks/Minimod_MPI/          # gitignored — minimod project (target hip_gicc)
tests/perf_baseline/             # 3-run median JSON for regression checks
docs/                            # gitignored — specs, plans, local notes
prototype/, legacy/, minimal/    # Not built
```

## Build

### OFI backend (Tioga, primary)

```bash
# 1. Build the LTO pass plugin (off by default in top-level; use standalone)
mkdir -p tools/gicc-passes/build && cd tools/gicc-passes/build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm-6.4.0/lib/llvm/bin/clang++ ..
make -j

# 2. Build OFI runtime + examples
cd /p/lustre2/shan4/opengda
mkdir -p build_ofi && cd build_ofi
cmake -DGICC_BACKEND=ofi \
      -Dhip_DIR=/opt/rocm-6.4.0/lib/cmake/hip \
      -DCMAKE_HIP_COMPILER=/opt/rocm-6.4.0/lib/llvm/bin/clang++ ..
make -j
# (the legacy/ofi/test_launch_skeleton.cpp build error at the very end
#  is pre-existing and harmless — all real examples build before it.)
```

### Minimod via LTO pipeline

The `benchmarks/Minimod_MPI/` directory is gitignored (separate project). To build minimod through the LTO pipeline, set `GICC_USE_LTO_PASS=1`:

```bash
cd benchmarks/Minimod_MPI
make TARGET=hip_gicc COMPILER=hipcc_gicc GICC_USE_LTO_PASS=1 clean
GICC_MODE=lower GICC_META_DIR=/tmp/gicc-meta-minimod \
    make TARGET=hip_gicc COMPILER=hipcc_gicc GICC_USE_LTO_PASS=1
```

This bypasses the AST-plugin wrapper script and uses plain hipcc with `-fpass-plugin`. The LTO pipeline emits real DWQ-enqueue calls / IPC memcpy in the host trace function and lowers `flush` to a lead-thread MMIO write on the device side.

The 3-pass build (feature-extract → decider → lower) is at `benchmarks/Minimod_MPI/scripts/gicc-lto-3pass.sh` (also gitignored).

### GICC-from-OpenMP minimod (DiOMP comparison)

A separate effort ports minimod to call `gicc::omp::put` **inside `#pragma omp target`
regions** (no LTO pass), to compare against the SC25 paper's DiOMP. It lives in the
**gitignored** tree `/p/lustre2/shan4/DiOMP/benchmarks/benchmarks/Minimod_DiOMP/`
(target `targets/omp_gicc/`). Sources `#include` GICC headers from this repo via
`GICC_ROOT`, so changes here are picked up on rebuild.

**Build** (`targets/omp_gicc/build_gicc_clang21.sh` — single clang-21 toolchain from
`module load diomp/1.0`; the application compiles its omp-target TUs and links a
matching `libgicc_omp.a`). **You MUST export `LD_LIBRARY_PATH` before building** or the
link fails on `libomptarget.so`'s indirect LLVM deps:

```bash
export LD_LIBRARY_PATH=/p/lustre2/shan4/softwares/diomp/lib:/p/lustre2/shan4/softwares/diomp/lib/x86_64-unknown-linux-gnu:/opt/rocm-6.4.3/lib:$LD_LIBRARY_PATH
export PATH=/p/lustre2/shan4/softwares/diomp/bin:$PATH
cd /p/lustre2/shan4/DiOMP/benchmarks/benchmarks/Minimod_DiOMP/targets/omp_gicc
GICC_ROOT=/p/lustre2/shan4/new-gicc bash ./build_gicc_clang21.sh   # -> ../../main_omp_gicc_clang21
```

**Run / ablation** — canonical driver: `Minimod_DiOMP/run_gicc_diomp_ablation.sh`
(runs N*8 GPUs, grid 1200, `GICC_HALO_OVERLAP` 0 and 1, IPC+proxy hybrid). Four
launch requirements are mandatory — each one is a real failure mode, not optional:

1. **`HSA_XNACK=1`** — without it, any run with >1 rank/node faults with
   `Memory access fault ... virtual address (nil)` in `gicc_halo_issue` (the
   device-side proxy put / in-kernel xGMI store touches host-mapped memory that
   MI250X only reaches with XNACK). 1-rank/node runs happen to work without it.
2. **Top-level `flux run -o mpibind=off`** (run from a login shell, NOT inside
   `flux alloc`; do NOT use `srun`). Multi-node IPC needs every node GPU visible
   so the per-rank device select works. `srun` strong-binds 1 GPU/task (IPC stays
   dark → cross-node still works but same-node hits the NIC); nesting `flux run`
   inside `flux alloc` gives ndev=0.
3. **`ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` + `GICC_HALO_IPC=1`** — enables per-rank
   device select and the same-node xGMI IPC fast path. (Do NOT also set
   `HIP_VISIBLE_DEVICES`; setting both breaks GPU visibility.)
4. `GICC_PROXY_ENABLED=1 GICC_SKIP_DWQ_INIT=1` + the build `LD_LIBRARY_PATH`.

Reference (paper DiOMP, grid 1200, `SC25/Minimod_DiOMP_*_1.log`): 8 GPU 23.93 s,
16 GPU 13.12 s, 24 GPU 9.63 s, 32 GPU 7.73 s (64 GPU OOMs). Fresh full sweep
(2026-06-19, every run bit-exact `field_xor 85c0f82d50171bb9`):

| GPU | GICC overlap-OFF | GICC overlap-ON | DiOMP (paper) |
|----:|----:|----:|----:|
|  8  | 23.09 | 23.07 | 23.93 |
| 16  | 12.22 | 11.58 | 13.12 |
| 24  |  8.57 |  8.29 |  9.63 |
| 32  |  6.86 |  6.60 |  7.73 |

GICC beats DiOMP at every scale in **both** modes; the overlap-OFF column (IPC alone)
already wins, and the gap grows with scale. **GICC wins via two independent levers,
either alone enough**: overlap (hide comm under compute) OR IPC (move same-node
faces to xGMI, ~free comm). The in-kernel xGMI IPC path has a known intermittent
startup race that faults more often the more same-node faces a run has — worst at
8 GPU (1 node, every face is IPC), rare at ≥16 GPU. The driver retries each run up
to `MAX_TRIES` (default 4) to ride through it; fixing the race itself is open work.

### GiOMP single-header library (`gicc/omp.h` + `libgicc_omp`)

A friendly repackaging of the GICC-from-OpenMP path: an app includes ONE header
and links ONE prebuilt library — no application-owned runtime adapter and no
hand-written multi-TU build. GiOMP = DiOMP + GICC: the API reuses DiOMP names
(`omp_get_rank_num`, `omp_get_num_ranks`, `ompx_barrier`) where they map 1:1.

**Toolchain: ROCm 6.4.0 clang** (`/opt/rocm-6.4.0/lib/llvm/bin/clang++`), NOT the
diomp clang-21. This is deliberate: the LTO pass plugin is built against ROCm
6.4.0's LLVM-19 and will NOT load in clang-21 (`undefined symbol:
llvm::DisableABIBreakingChecks`), so the DWQ path needs 6.4.0 — and building the
library + examples with one compiler lets proxy and DWQ share it. Runtime
`LD_LIBRARY_PATH` = `/opt/rocm-6.4.0/lib/llvm/lib:/opt/rocm-6.4.0/lib:/opt/cray/pe/lib64`
(export in the shell; do NOT pass inline to `flux run` — it drops flux's Cray PE
paths and breaks PMI at runtime).

**Build the library once:**
```bash
export LD_LIBRARY_PATH=/opt/rocm-6.4.0/lib/llvm/lib:/opt/rocm-6.4.0/lib:/opt/cray/pe/lib64
GICC_ROOT=/p/lustre2/shan4/new-gicc bash omp/build_libgicc_omp.sh   # -> build_ofi/lib/libgicc_omp.{so,a}
# then regenerate the CMake config + flags script from the templates:
sed -e 's#@PACKAGE_INIT@##' -e 's#@GICC_ROOT@#/p/lustre2/shan4/new-gicc#g' \
  omp/cmake/gicc-omp-config.cmake.in > build_ofi/lib/cmake/gicc-omp/gicc-omp-config.cmake
sed 's#@GICC_ROOT@#/p/lustre2/shan4/new-gicc#g' omp/bin/gicc-omp-config.in > build_ofi/lib/bin/gicc-omp-config
```

**Consume via CMake** (`CMAKE_PREFIX_PATH=<root>/build_ofi/lib/cmake/gicc-omp`):
```cmake
find_package(gicc-omp REQUIRED)
target_link_libraries(app PRIVATE gicc::omp)          # or gicc::omp_dwq for the DWQ path
```
**Or via flags** (non-CMake): `clang++ $(gicc-omp-config --cflags) app.cpp $(gicc-omp-config --libs)` (add `--dwq`).

**API:** `ompx_init/finalize`, `omp_get_rank_num/num_ranks`, `ompx_alloc` (returns
`ompx_buffer{ptr,index,bytes}`, alloc+register) / `ompx_register` / `ompx_free`,
`ompx_exchange`, `ompx_prepare`, `ompx_barrier`, `ompx_quiet_host` (host).
`ompx_put(ctx,peer,dbuf,doff,sbuf,soff,bytes,xport)` is a **smart host-side put**:
IPC (in-kernel xGMI) if the peer is same-node reachable, else the cross-node
transport `xport` (`OMPX_PROXY` default, or `OMPX_DWQ` runtime-triggered); it
issues the `#pragma omp target` region itself. For batching many puts in one
kernel, call the device-side `ompx_put_proxy`/`ompx_get`/`ompx_quiet` inside your
own `#pragma omp target`. `ompx_dwq_put/flush` markers = the pass-based DWQ.
Examples: `examples/omp/{hello_giomp,omp_matmul,omp_pingpong}.cpp`; the minimod
halo (`Minimod_DiOMP/targets/omp_gicc/gicc_halo.cpp`) uses the smart `ompx_put`.

**Multi-rank SAME-node needs the full launch recipe** (per-rank device select),
else the compute kernels fault ("write to read-only page") on a mismatched GPU:
```bash
HSA_XNACK=1 GICC_PROXY_ENABLED=1 GICC_SKIP_DWQ_INIT=1 \
GICC_HALO_IPC=1 ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
flux run -N1 -n8 -g1 -o mpibind=off ./omp_matmul 4096
```
Cross-node (2 ranks / 2 nodes) does not need `GICC_HALO_IPC`/`ROCR_VISIBLE_DEVICES`.

**DWQ is a separate build** (`examples/omp/build_omp_dwq.sh`): only the app TU
needs the pass (2-pass compile, `GICC_MODE=omp-dwq` + `GICC_META_DIR` env,
`-mllvm -openmp-opt-disable=true`), then it links the prebuilt `libgicc_omp`. Run
with `GICC_HALO_DWQ=1` and WITHOUT `GICC_SKIP_DWQ_INIT` (so the trigger BAR maps).

## GICC API

### Host

```cpp
#include "gicc/gicc.hpp"

gicc::Runtime rt;
rt.enable_host_wait_mode();                            // GDA-style fast path
auto buf = rt.register_buffer(d_ptr, size, /*device=*/true);
rt.exchange();                                          // collective IPC + RMA address book

gicc::launch<halo_kernel>(rt, dim3(64), dim3(256),
                          peer, buf.index, dst_off, src_off, bytes,
                          has_left, has_right);
hipDeviceSynchronize();
rt.reset();                                             // poll DWQ completion + drain IPC stream
MPI_Barrier(MPI_COMM_WORLD);
```

### Device

```cpp
#include "gicc/gicc_device.cuh"

__global__ void halo_kernel(gicc::DeviceCtx* ctx,
                            int peer, int my_buf,
                            size_t dst_off, size_t src_off, size_t bytes,
                            int has_left, int has_right) {
    if (has_left)  gicc::put_no_db(ctx, peer, my_buf, dst_off,
                                          my_buf, src_off, bytes);
    if (has_right) gicc::put_no_db(ctx, peer + 1, my_buf, dst_off + bytes,
                                                  my_buf, src_off + bytes, bytes);
    gicc::flush(ctx);
}
```

The kernel is passed as a non-type template parameter to `gicc::launch<...>`. Every `put_no_db` argument must be host-knowable: a kernel formal, a constant, an arithmetic combination of those, or a canonical loop induction variable. HK Analysis rejects with a source-line-precise diagnostic if not.

## LTO Pass Pipeline at a Glance

When `-fpass-plugin=libgicc-passes.so` is set:

| Side | Pass | Action |
|---|---|---|
| Device | `gicc-device-discovery` | Find kernel + `gicc::*` call sites; write per-kernel JSON to `${GICC_META_DIR}` |
| Device | `gicc-hk-analysis` | Validate every put argument is host-knowable |
| Device | `gicc-device-lowering` | Erase `put_no_db` calls; replace `flush` with lead-thread MMIO write |
| Host | `gicc-host-discovery` | Find launch sites via `@llvm.global.annotations`; load matching meta JSON |
| Host | `gicc-feature-extraction` | Emit `features.json` for external decider |
| Host | `gicc-trace-synthesis` | Synthesize `@gicc_trace_<kernel>` function; insert call before each launch site |
| Host | `gicc-dispatch-lowering` | Replace placeholder calls with `IPC_PUSH` / `DWQ_TRIGGER` / `IPC_OR_DWQ` (default) / `DWQ_BATCHED` |

Configuration via env vars (NOT `-mllvm` flags — those don't propagate through `-fpass-plugin` on Clang 19):

```
GICC_MODE          discover | feature-extract | lower | passthrough
GICC_TARGET        ofi-triggered | ib-native | auto
GICC_META_DIR      directory for per-kernel trace template JSON
GICC_FEATURES_OUT  features.json output path
GICC_HINT_IN       hint.json input path
```

## Tests

```bash
# Lit IR tests (fast, ~22 cases)
cd tools/gicc-passes/build && make -j check-gicc-passes

# Integration tests (Tioga, needs allocation)
cd tools/gicc-passes/tests/integration && ./run_all.sh

# Perf regression (Tioga, 4 nodes, ~3 min)
cd /p/lustre2/shan4/opengda
./tools/gicc-passes/tests/run_perf.sh
```

## User Conventions (from project memory)

- **Granular commits**: split work per file or per logical step. Don't bundle.
- **No AI attribution in commits**: never include `Co-Authored-By: Claude` or similar.
- **Code comments in English only** (Chinese comments are rejected, even inside design-doc code blocks).
- **Acronym casing**: `BootstrapMPI` not `BootstrapMpi`, `IPC` not `Ipc`.
- **Planning docs stay local**: `docs/` (specs, plans, notes) is gitignored — never `git add` it.
- **Test timeout cap**: srun tests > 1 min are broken, not slow. Stop and diagnose instead.
- **Partition**: prefer `srun -p pci`; if it's full (esp. multi-node jobs that stall in the queue), fall back to `-p pdebug`. The two are interchangeable for these runs.

## Performance baseline (current)

Minimod 32 ranks, grid 1000, nsteps 100 (3-run median):

| Configuration | comm (ms) |
|---|---|
| GDA reference (`hip_gda_hipcc`) | 88.28 |
| GICC LTO (default `IPC_OR_DWQ`) | 86.44 |

LTO pipeline is 2.1% faster than the GDA reference, well below the v1 acceptance bar of 92.5 ms.
