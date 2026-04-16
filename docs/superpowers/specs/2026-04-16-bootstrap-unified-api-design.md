# Unified Bootstrap API — Design Spec

**Date:** 2026-04-16
**Author:** Baodi Shan (with Claude)
**Status:** Draft (awaiting user review)

## 1. Goal

Eliminate every direct reference to `MPI_*` and `PMI_*` symbols from GICC library
code and examples. Replace them with a single, compile-time-selected
`gicc::Bootstrap` abstraction that provides the minimum set of parallel
primitives GICC and its examples actually need.

Non-goals:
- Runtime switching between MPI and PMI in the same binary.
- A feature-complete MPI replacement — only what the codebase uses.
- Refactoring `legacy/`, `minimal/`, `prototype/`, or top-level `gicc/` and
  `nvidia/` directories (archived / being migrated separately).

## 2. Motivation

Today, initialization and collective metadata exchange are split across two
ad-hoc wrappers (`gicc::MpiBootstrap`, `PmiSession`) plus raw `MPI_*` and `PMI_*`
calls scattered through both backends and every example. Consequences:

- The mlx5 backend is hard-coded to MPI; the OFI backend mixes PMI2 for init
  with raw `MPI_Allgather` inside `gicc_barrier.hpp`.
- `Runtime(MPI_Comm)` constructors leak MPI types through the public API.
- Users can't compile the OFI backend with an MPI-only toolchain, or the mlx5
  backend with a PMI-only toolchain, even though the underlying GICC code is
  agnostic to the bootstrap mechanism.

Unifying behind a single `Bootstrap` type makes the bootstrap choice orthogonal
to the network-backend choice, removes one source of accidental coupling, and
leaves room for a future PMIX implementation.

## 3. User decisions captured during brainstorm

| # | Question | Decision |
|---|---|---|
| 1 | Scope of the abstraction | B — all `.cpp / .cu` files clean of `MPI_/PMI_`, examples included |
| 2 | Selection mechanism | A — compile-time, CMake option |
| 3 | API width | A — minimal core + `gicc::coll::` helpers for reductions |
| 4 | How user code accesses Bootstrap | B — `Runtime` owns it, user calls `rt.boot()` |
| 5a | Standalone `Bootstrap` (no `Runtime`) | Allowed |
| 5b | Migration strategy | One-shot rewrite |
| 6 | PMI naming | `PMI2` explicitly (reserve `PMIX` slot for future) |
| 7 | CMake defaults | Fixed per-option defaults (`BACKEND=ofi`, `BOOTSTRAP=mpi`), no cross-option inference |
| 8 | Acronym casing | All-caps in identifiers: `BootstrapMPI`, `BootstrapPMI2` |

## 4. Architecture

```
src/gicc/
├── bootstrap/
│   ├── bootstrap.hpp                        # Public header
│   └── detail/
│       ├── bootstrap_mpi.hpp                # detail::BootstrapMPI (includes <mpi.h>)
│       └── bootstrap_pmi2.hpp               # detail::BootstrapPMI2 (includes <pmi2.h>)
├── coll.hpp                                 # gicc::coll:: free-function helpers
└── ...                                      # (unchanged)
```

`bootstrap.hpp` picks the implementation via preprocessor switches:

```cpp
// src/gicc/bootstrap/bootstrap.hpp
#pragma once

#if defined(GICC_BOOTSTRAP_MPI)
  #include "detail/bootstrap_mpi.hpp"
  namespace gicc { using Bootstrap = detail::BootstrapMPI; }
#elif defined(GICC_BOOTSTRAP_PMI2)
  #include "detail/bootstrap_pmi2.hpp"
  namespace gicc { using Bootstrap = detail::BootstrapPMI2; }
#else
  #error "Define GICC_BOOTSTRAP_MPI or GICC_BOOTSTRAP_PMI2 (set via -DGICC_BOOTSTRAP=...)"
#endif
```

Only one detail header is ever compiled in. Neither `<mpi.h>` nor `<pmi2.h>`
leaks to translation units that do not include `bootstrap.hpp`, and user code
that does include it never needs to spell `MPI_*` / `PMI_*` itself.

Lifetime: a `Bootstrap` construction calls `MPI_Init` or `PMI2_Init` if nobody
else has; destruction calls the matching `Finalize` only if this instance owns
the session. `Runtime` holds a `Bootstrap boot_` member and exposes
`rt.boot()`; standalone `gicc::Bootstrap bs;` is also valid for code that does
not need a `Runtime`.

## 5. Bootstrap API

```cpp
namespace gicc {

class Bootstrap {
public:
    Bootstrap();                             // MPI_Init(NULL, NULL) / PMI2_Init()
    Bootstrap(int argc, char** argv);        // forwards argc/argv to MPI_Init
    ~Bootstrap();                            // Finalize iff we initialized

    Bootstrap(const Bootstrap&) = delete;
    Bootstrap& operator=(const Bootstrap&) = delete;

    // Identity
    int rank() const noexcept;
    int size() const noexcept;
    int local_rank() const noexcept;
    int local_size() const noexcept;

    // Topology: locality_map[i] == true iff rank i is on the same node as me.
    std::vector<bool> locality_map() const;

    // Sync
    void barrier();

    // Byte-oriented collectives
    std::vector<std::vector<uint8_t>>
        allgather(const void* data, int len);
    template<class T> std::vector<T>
        allgather_fixed(const T& value);
    template<class T> void
        broadcast(T& value, int root = 0);

    // Byte-oriented point-to-point
    void send    (const void* buf, int len, int dest, int tag = 0);
    void recv    (      void* buf, int len, int src,  int tag = 0);
    void sendrecv(const void* sbuf, void* rbuf, int len, int peer, int tag = 0);

    // Monotonic seconds
    double wtime() const noexcept;
};

} // namespace gicc
```

### 5.1 Implementation notes

**BootstrapMPI** — trivial mapping: `MPI_COMM_WORLD` for every collective,
`MPI_BYTE` datatype, `MPI_Comm_split_type(SHARED) + MPI_Allgather(color)` to
build the locality map.

**BootstrapPMI2** — KVS-based. `barrier` is `PMI2_KVS_Fence`. `allgather` /
`broadcast` / `sendrecv` are implemented by hex-encoding the bytes into a KVS
value, fencing, then reading peers' values. Locality map publishes hostname per
rank. Collective throughput in PMI2 mode scales with the KVS server — acceptable
for the experimental node counts used on Tioga but not a long-term
high-performance path.

**Template methods** (`allgather_fixed<T>`, `broadcast<T>`) are header-only and
treat `T` as a POD byte blob (`sizeof(T)`), so both implementations stay simple.

## 6. `gicc::coll::` helpers

```cpp
// src/gicc/coll.hpp
#pragma once
#include "bootstrap/bootstrap.hpp"

namespace gicc::coll {

template<class T> T allreduce_sum(Bootstrap& bs, const T& local);
template<class T> T allreduce_min(Bootstrap& bs, const T& local);
template<class T> T allreduce_max(Bootstrap& bs, const T& local);

} // namespace gicc::coll
```

Implemented inline as `allgather_fixed` + local reduction. Free functions rather
than `Bootstrap` methods so:

1. The Bootstrap surface stays minimal.
2. The same implementation works across both backends (no per-backend allreduce
   code path).
3. Custom reduction ops can be added later without touching `Bootstrap`.

## 7. CMake integration

```cmake
# src/CMakeLists.txt (new options block)

set(GICC_BACKEND   "ofi" CACHE STRING "Network backend: mlx5 | ofi")
set(GICC_BOOTSTRAP "mpi" CACHE STRING "Bootstrap backend: mpi | pmi2")
set_property(CACHE GICC_BACKEND   PROPERTY STRINGS "mlx5" "ofi")
set_property(CACHE GICC_BOOTSTRAP PROPERTY STRINGS "mpi"  "pmi2")

if(GICC_BOOTSTRAP STREQUAL "mpi")
    find_package(MPI REQUIRED)
    target_compile_definitions(gicc PUBLIC GICC_BOOTSTRAP_MPI=1)
    target_link_libraries     (gicc PUBLIC MPI::MPI_CXX)
elseif(GICC_BOOTSTRAP STREQUAL "pmi2")
    find_path   (PMI2_INCLUDE_DIR pmi2.h
                 PATHS ${PMI_ROOT} /opt/cray/pe/pmi/6.1.15
                 PATH_SUFFIXES include)
    find_library(PMI2_LIBRARY pmi2
                 PATHS ${PMI_ROOT} /opt/cray/pe/pmi/6.1.15
                 PATH_SUFFIXES lib lib64)
    if(NOT PMI2_INCLUDE_DIR OR NOT PMI2_LIBRARY)
        message(FATAL_ERROR "PMI2 not found (set PMI_ROOT=/path/to/pmi)")
    endif()
    target_compile_definitions(gicc PUBLIC GICC_BOOTSTRAP_PMI2=1)
    target_include_directories(gicc PUBLIC ${PMI2_INCLUDE_DIR})
    target_link_libraries     (gicc PUBLIC ${PMI2_LIBRARY})
else()
    message(FATAL_ERROR
        "GICC_BOOTSTRAP must be 'mpi' or 'pmi2' (got '${GICC_BOOTSTRAP}').")
endif()
```

Key properties:

- `GICC_BACKEND` and `GICC_BOOTSTRAP` are **independent** knobs. Each has its
  own fixed default (`ofi`, `mpi`). No option ever derives its default from
  another option.
- `libgicc` exposes MPI / PMI2 transitively via `PUBLIC` compile-defs and link
  libraries. User code that includes `gicc/bootstrap/bootstrap.hpp` will pick
  up the right include path and linkage automatically without ever naming
  `<mpi.h>` or `<pmi2.h>`.
- `examples/ofi/CMakeLists.txt` loses its hand-written PMI `find_path` /
  `find_library` block; the dependency is inherited through `libgicc`.

### 7.1 Supported build commands

```bash
# Zero-arg default: ofi + mpi
cmake ..

# MAPLE: mlx5 + mpi
cmake .. -DGICC_BACKEND=mlx5

# Tioga: ofi + pmi2
cmake .. -DGICC_BOOTSTRAP=pmi2 \
         -Dhip_DIR=/opt/rocm-6.4.0/lib/cmake/hip \
         -DCMAKE_HIP_COMPILER=/opt/rocm-6.4.0/lib/llvm/bin/clang++

# Tioga: ofi + mpi (new supported combo)
cmake .. -Dhip_DIR=... -DCMAKE_HIP_COMPILER=...
```

## 8. Migration plan (one-shot rewrite)

### 8.1 New files (4)

| File | Purpose |
|---|---|
| `src/gicc/bootstrap/bootstrap.hpp` | Public header with preprocessor-selected `using` alias |
| `src/gicc/bootstrap/detail/bootstrap_mpi.hpp` | `detail::BootstrapMPI` — wraps `<mpi.h>` |
| `src/gicc/bootstrap/detail/bootstrap_pmi2.hpp` | `detail::BootstrapPMI2` — wraps `<pmi2.h>`, collectives over KVS |
| `src/gicc/coll.hpp` | `gicc::coll::allreduce_{sum,min,max}` |

### 8.2 Deleted files (2)

| File | Replaced by |
|---|---|
| `src/gicc/util/mpi_bootstrap.hpp` | `detail::BootstrapMPI` |
| `src/gicc/platform/ofi/internal/pmi_session.hpp` | `detail::BootstrapPMI2` |

### 8.3 Modified library files (11)

| File | Change |
|---|---|
| `src/CMakeLists.txt` | Add `GICC_BOOTSTRAP` option block; drop the MPI-only hard-coded path |
| `src/gicc/gicc.hpp` | Docstring update: `gicc::Runtime rt;` instead of `gicc::init(MPI_COMM_WORLD)` |
| `src/gicc/platform/mlx5/gicc_api.hpp` | `init(MPI_Comm)` → `init()` (or removed in favor of the `Runtime` path) |
| `src/gicc/platform/mlx5/mlx5_runtime.hpp` | Constructor `Runtime(MPI_Comm)` → `Runtime()`; member `MpiBootstrap*` → `Bootstrap boot_`; line 119 `MPI_Allgather` → `boot_.allgather` |
| `src/gicc/platform/ofi/ofi_runtime.hpp` | Same change; remove `MPI_Comm mpi_comm_` member |
| `src/gicc/platform/ofi/internal/gda_comm.hpp` | `PmiSession pmi` → `Bootstrap& boot` (passed in by Runtime) |
| `src/gicc/platform/ofi/internal/gicc_barrier.hpp` | Lines 363–366 `MPI_Allgather` → `boot.allgather_fixed` |
| `src/gicc/mlx5/am_context.hpp` | Lines 94–233 every `MPI_*` call → `boot.*`; accept `Bootstrap&` via constructor |
| `src/gicc/mlx5/gda_context.hpp` | `MpiBootstrap&` → `Bootstrap&` |
| `src/gicc/mlx5/gpu_comm.hpp` | `MpiBootstrap` → `Bootstrap` |
| `src/gicc/mlx5/comm.hpp` | Same |

### 8.4 Modified examples (9)

| File | Change |
|---|---|
| `examples/gicc/jacobi.cu` | Drop `<mpi.h>`; `MPI_Init/Finalize` → `gicc::Runtime rt;`; `MPI_Wtime` → `rt.boot().wtime()`; `MPI_Allreduce` → `gicc::coll::allreduce_sum`; `MPI_Isend/Irecv/Waitall` → two `rt.boot().sendrecv` calls; `MPI_Bcast/Send/Recv` → `rt.boot().broadcast/send/recv` |
| `examples/gicc/mm.cu` | Same shape of replacements |
| `examples/gicc/pingpong_bench.cu` | Same |
| `examples/ofi/test_gicc.cpp` | Same |
| `examples/ofi/gda_benchmark_gicc.cpp` | Same |
| `examples/ofi/mm_gda_minimal_gicc.cpp` | Same; `PmiSession::build_locality_map` → `rt.boot().locality_map()` |
| `examples/ofi/barrier_test.cpp` | `GdaComm` no longer constructs PMI itself; caller passes `Bootstrap&` |
| `examples/ofi/CMakeLists.txt` | Remove hand-written PMI `find_path` / `find_library` |
| `examples/CMakeLists.txt` | Remove direct MPI dependency (inherited via `libgicc`) |

### 8.5 Untouched

- `legacy/ofi/**` — archived
- `prototype/**` — experimental
- `minimal/**` — pre-migration CXI code
- top-level `gicc/**` and `nvidia/**` — being migrated into `src/` separately

## 9. Error handling

| Condition | Behavior |
|---|---|
| `MPI_Init` / `PMI2_Init` returns non-success | Constructor throws `std::runtime_error` with the return code. Fail-fast; no half-initialized state. |
| Collective returns non-success | Same — `std::runtime_error` with context, no silent fallback. |
| MPI / PMI already initialized by someone else | Bootstrap records `owned_ = false` and skips the matching `Finalize` on destruction. Mirrors today's `MpiBootstrap::initialized_mpi` behavior. |
| Two `Bootstrap` instances in one process | Second instance detects that MPI/PMI is already initialized, sets `owned_ = false`, shares the underlying session. Documented as "one Bootstrap per process is recommended"; an assertion can be added later. |

## 10. Testing

No new unit tests — mocking MPI/PMI has a low payoff for the coverage it would
add. End-to-end verification via existing examples.

| # | Check | Command |
|---|---|---|
| 1 | No raw `MPI_` / `PMI_` outside the two detail headers | `grep -rn 'MPI_\|PMI_' src/ examples/` only matches `src/gicc/bootstrap/detail/bootstrap_{mpi,pmi2}.hpp` |
| 2 | MAPLE mlx5 + mpi builds | `cmake .. -DGICC_BACKEND=mlx5` + `make` |
| 3 | MAPLE examples run | `srun ... ./pingpong_bench / mm / jacobi` produce results within ±5% of baseline; jacobi converges to the same residual |
| 4 | Tioga ofi + pmi2 builds | `cmake .. -DGICC_BOOTSTRAP=pmi2 -Dhip_DIR=... ...` + `make` |
| 5 | Tioga ofi + mpi builds (new combo) | `cmake ..` (defaults) + `make`; compile-only check is enough this round |

## 11. Risks

- **BootstrapPMI2 allgather/broadcast via KVS** scales poorly past ~64 ranks.
  Acceptable for the current Tioga workloads but not a general-purpose high-
  throughput path. Documented, not remediated.
- **The OFI-backend barrier's current `MPI_Allgather`** will, under
  `BOOTSTRAP=pmi2`, go through the KVS allgather — a few extra milliseconds at
  bootstrap time. One-time cost, not in any hot path.

## 12. Out of scope (future work)

- `BootstrapPMIX` implementation (`GICC_BOOTSTRAP=pmix`, `detail::BootstrapPMIX`,
  `bootstrap_pmix.hpp`).
- Single unit-test framework (GoogleTest) if coverage needs grow.
- Cleanup of `legacy/`, `minimal/`, `prototype/`, top-level `gicc/` and
  `nvidia/` directories.
