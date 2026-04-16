# Unified Bootstrap API — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every direct `MPI_*` / `PMI_*` reference in GICC library and example code with a single compile-time-selected `gicc::Bootstrap` type, plus a small `gicc::coll::` helper layer for reductions.

**Architecture:** Public header `gicc/bootstrap/bootstrap.hpp` uses preprocessor switches (`GICC_BOOTSTRAP_MPI` / `GICC_BOOTSTRAP_PMI2`) to alias `gicc::Bootstrap` to `detail::BootstrapMPI` or `detail::BootstrapPMI2`. CMake option `GICC_BOOTSTRAP` (default `mpi`) picks the macro and links the matching transport library. `gicc::Runtime` owns a `Bootstrap boot_` member; examples call `rt.boot().rank()`, `rt.boot().wtime()`, etc.

**Tech Stack:** C++17, CMake, CUDA 12.4 (MAPLE), HIP/ROCm 6.4 (Tioga), OpenMPI 5.0.5, Cray PMI2.

**Reference:** `docs/superpowers/specs/2026-04-16-bootstrap-unified-api-design.md`

**Verification philosophy:** HPC code can't be unit-tested without a cluster. Each task's verification is "the build still succeeds on the targeted configuration." End-to-end correctness is verified with existing examples at the end of Phase 3.

---

## Phase 1 — Foundation: add new code without removing old

Goal: the new `gicc::Bootstrap` / `coll::` / CMake machinery exists and compiles, but nothing else is touched yet. Library still builds with the old `MpiBootstrap` / `PmiSession` path.

---

### Task 1: Create `detail::BootstrapMPI`

**Files:**
- Create: `src/gicc/bootstrap/detail/bootstrap_mpi.hpp`

- [ ] **Step 1: Create the file**

```cpp
// src/gicc/bootstrap/detail/bootstrap_mpi.hpp
//
// MPI-backed implementation of the gicc::Bootstrap contract.
// Selected when GICC_BOOTSTRAP_MPI is defined. Not intended to be included
// directly by user code — go through gicc/bootstrap/bootstrap.hpp.
#pragma once

#include <mpi.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace gicc::detail {

class BootstrapMPI {
public:
    BootstrapMPI() : BootstrapMPI(nullptr, nullptr) {}

    BootstrapMPI(int* argc, char*** argv) {
        int already = 0;
        MPI_Initialized(&already);
        if (!already) {
            int rc = MPI_Init(argc, argv);
            if (rc != MPI_SUCCESS) {
                throw std::runtime_error(
                    "BootstrapMPI: MPI_Init failed (rc=" + std::to_string(rc) + ")");
            }
            owned_ = true;
        }

        comm_ = MPI_COMM_WORLD;
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &size_);

        MPI_Comm local_comm;
        MPI_Comm_split_type(comm_, MPI_COMM_TYPE_SHARED, rank_,
                            MPI_INFO_NULL, &local_comm);
        MPI_Comm_rank(local_comm, &local_rank_);
        MPI_Comm_size(local_comm, &local_size_);
        MPI_Comm_free(&local_comm);
    }

    BootstrapMPI(int argc, char** argv)
        : BootstrapMPI(&argc, &argv) {}

    ~BootstrapMPI() {
        if (owned_) {
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (!finalized) MPI_Finalize();
        }
    }

    BootstrapMPI(const BootstrapMPI&) = delete;
    BootstrapMPI& operator=(const BootstrapMPI&) = delete;

    int rank()       const noexcept { return rank_; }
    int size()       const noexcept { return size_; }
    int local_rank() const noexcept { return local_rank_; }
    int local_size() const noexcept { return local_size_; }

    std::vector<bool> locality_map() const {
        // Color by shared communicator id, then Allgather the color.
        MPI_Comm local_comm;
        MPI_Comm_split_type(comm_, MPI_COMM_TYPE_SHARED, rank_,
                            MPI_INFO_NULL, &local_comm);
        int my_color = 0;
        // Color = the min global rank inside the local communicator.
        int local_ranks[1024];  // big enough for our node sizes
        int lsz = 0;
        MPI_Comm_size(local_comm, &lsz);
        if (lsz > 1024) {
            MPI_Comm_free(&local_comm);
            throw std::runtime_error("BootstrapMPI::locality_map: node too large");
        }
        std::vector<int> my_global_rank(1, rank_);
        std::vector<int> local_globals(lsz);
        MPI_Allgather(my_global_rank.data(), 1, MPI_INT,
                      local_globals.data(), 1, MPI_INT, local_comm);
        my_color = *std::min_element(local_globals.begin(), local_globals.end());
        MPI_Comm_free(&local_comm);
        (void)local_ranks;

        std::vector<int> all_colors(size_);
        MPI_Allgather(&my_color, 1, MPI_INT,
                      all_colors.data(), 1, MPI_INT, comm_);
        std::vector<bool> out(size_, false);
        for (int i = 0; i < size_; ++i) out[i] = (all_colors[i] == my_color);
        return out;
    }

    void barrier() { MPI_Barrier(comm_); }

    std::vector<std::vector<uint8_t>>
    allgather(const void* data, int len) {
        std::vector<int> lens(size_);
        MPI_Allgather(&len, 1, MPI_INT, lens.data(), 1, MPI_INT, comm_);

        std::vector<int> displs(size_);
        int total = 0;
        for (int i = 0; i < size_; ++i) { displs[i] = total; total += lens[i]; }

        std::vector<uint8_t> flat(total);
        MPI_Allgatherv(data, len, MPI_BYTE,
                       flat.data(), lens.data(), displs.data(), MPI_BYTE, comm_);

        std::vector<std::vector<uint8_t>> out(size_);
        for (int i = 0; i < size_; ++i) {
            out[i].assign(flat.begin() + displs[i],
                          flat.begin() + displs[i] + lens[i]);
        }
        return out;
    }

    template<class T>
    std::vector<T> allgather_fixed(const T& value) {
        std::vector<T> out(size_);
        MPI_Allgather(&value, sizeof(T), MPI_BYTE,
                      out.data(), sizeof(T), MPI_BYTE, comm_);
        return out;
    }

    template<class T>
    void broadcast(T& value, int root = 0) {
        MPI_Bcast(&value, sizeof(T), MPI_BYTE, root, comm_);
    }

    void send(const void* buf, int len, int dest, int tag = 0) {
        MPI_Send(buf, len, MPI_BYTE, dest, tag, comm_);
    }
    void recv(void* buf, int len, int src, int tag = 0) {
        MPI_Recv(buf, len, MPI_BYTE, src, tag, comm_, MPI_STATUS_IGNORE);
    }
    void sendrecv(const void* sbuf, void* rbuf, int len, int peer, int tag = 0) {
        MPI_Sendrecv(sbuf, len, MPI_BYTE, peer, tag,
                     rbuf, len, MPI_BYTE, peer, tag,
                     comm_, MPI_STATUS_IGNORE);
    }

    double wtime() const noexcept { return MPI_Wtime(); }

private:
    MPI_Comm comm_ = MPI_COMM_NULL;
    int rank_ = 0;
    int size_ = 0;
    int local_rank_ = 0;
    int local_size_ = 0;
    bool owned_ = false;
};

} // namespace gicc::detail
```

- [ ] **Step 2: Verify the file is syntactically OK by compiling it in isolation**

```bash
module use /shared/data1/Projects/CSE_HPC/apps/modules/aarch64/maple
module load gcc/11.4.1 cuda/12.4.131 nvhpc/24.5.cuda_12.4 openmpi-gcc/5.0.5/cuda.12.4 cmake/3.29.6

mpicxx -std=c++17 -fsyntax-only -I/p/lustre2/shan4/opengda/src \
    /p/lustre2/shan4/opengda/src/gicc/bootstrap/detail/bootstrap_mpi.hpp
```

Expected: no output (clean parse).

- [ ] **Step 3: Commit**

```bash
cd /p/lustre2/shan4/opengda
git add src/gicc/bootstrap/detail/bootstrap_mpi.hpp
git commit -m "Add detail::BootstrapMPI — MPI-backed bootstrap implementation"
```

---

### Task 2: Create `detail::BootstrapPMI2`

**Files:**
- Create: `src/gicc/bootstrap/detail/bootstrap_pmi2.hpp`

- [ ] **Step 1: Create the file**

PMI2 has no native collectives, so allgather / broadcast / sendrecv are emulated on top of `PMI2_KVS_Put / Get / Fence`. Bytes are hex-encoded into KVS values.

```cpp
// src/gicc/bootstrap/detail/bootstrap_pmi2.hpp
//
// PMI2-backed implementation of the gicc::Bootstrap contract.
// Selected when GICC_BOOTSTRAP_PMI2 is defined. Collective primitives are
// emulated on top of PMI2 KVS put/get/fence. Intended for Cray PMI2 on
// Slingshot systems (e.g., Tioga).
#pragma once

#include <pmi2.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace gicc::detail {

class BootstrapPMI2 {
public:
    BootstrapPMI2() : BootstrapPMI2(nullptr, nullptr) {}

    BootstrapPMI2(int* /*argc*/, char*** /*argv*/) {
        int spawned = 0, appnum = 0;
        int rc = PMI2_Init(&spawned, &size_, &rank_, &appnum);
        if (rc != PMI2_SUCCESS) {
            throw std::runtime_error(
                "BootstrapPMI2: PMI2_Init failed (rc=" + std::to_string(rc) + ")");
        }
        owned_ = true;

        char host[256] = {0};
        gethostname(host, sizeof(host) - 1);
        hostname_.assign(host);

        read_env_local_identity();
        publish_hostname_and_fence();
    }

    BootstrapPMI2(int argc, char** argv)
        : BootstrapPMI2(&argc, &argv) {}

    ~BootstrapPMI2() {
        if (owned_) PMI2_Finalize();
    }

    BootstrapPMI2(const BootstrapPMI2&) = delete;
    BootstrapPMI2& operator=(const BootstrapPMI2&) = delete;

    int rank()       const noexcept { return rank_; }
    int size()       const noexcept { return size_; }
    int local_rank() const noexcept { return local_rank_; }
    int local_size() const noexcept { return local_size_; }

    std::vector<bool> locality_map() const {
        std::vector<bool> same(size_, false);
        char val[256];
        for (int i = 0; i < size_; ++i) {
            std::string key = "gicc.host." + std::to_string(i);
            int vallen = 0;
            int rc = PMI2_KVS_Get(NULL, PMI2_ID_NULL, key.c_str(),
                                  val, sizeof(val), &vallen);
            if (rc != PMI2_SUCCESS) {
                throw std::runtime_error(
                    "BootstrapPMI2::locality_map: KVS_Get failed for " + key);
            }
            same[i] = (hostname_ == val);
        }
        return same;
    }

    void barrier() {
        int rc = PMI2_KVS_Fence();
        if (rc != PMI2_SUCCESS) {
            throw std::runtime_error("BootstrapPMI2::barrier: KVS_Fence failed");
        }
    }

    std::vector<std::vector<uint8_t>>
    allgather(const void* data, int len) {
        // Publish my blob at "gicc.ag.<epoch>.<rank>"
        const uint64_t epoch = next_epoch();
        publish_blob(kvs_key("ag", epoch, rank_),
                     reinterpret_cast<const uint8_t*>(data), len);
        barrier();
        std::vector<std::vector<uint8_t>> out(size_);
        for (int i = 0; i < size_; ++i) {
            out[i] = fetch_blob(kvs_key("ag", epoch, i));
        }
        return out;
    }

    template<class T>
    std::vector<T> allgather_fixed(const T& value) {
        auto raw = allgather(&value, sizeof(T));
        std::vector<T> out(size_);
        for (int i = 0; i < size_; ++i) {
            if ((int)raw[i].size() != (int)sizeof(T)) {
                throw std::runtime_error(
                    "BootstrapPMI2::allgather_fixed: size mismatch");
            }
            std::memcpy(&out[i], raw[i].data(), sizeof(T));
        }
        return out;
    }

    template<class T>
    void broadcast(T& value, int root = 0) {
        const uint64_t epoch = next_epoch();
        if (rank_ == root) {
            publish_blob(kvs_key("bc", epoch, root),
                         reinterpret_cast<const uint8_t*>(&value), sizeof(T));
        }
        barrier();
        if (rank_ != root) {
            auto blob = fetch_blob(kvs_key("bc", epoch, root));
            if ((int)blob.size() != (int)sizeof(T)) {
                throw std::runtime_error(
                    "BootstrapPMI2::broadcast: size mismatch");
            }
            std::memcpy(&value, blob.data(), sizeof(T));
        }
    }

    void send(const void* buf, int len, int dest, int tag = 0) {
        const uint64_t epoch = next_epoch();
        publish_blob(kvs_key("p2p", epoch, rank_, dest, tag),
                     reinterpret_cast<const uint8_t*>(buf), len);
        barrier();
    }
    void recv(void* buf, int len, int src, int tag = 0) {
        barrier();
        auto blob = fetch_blob(kvs_key("p2p", next_epoch_peek(), src, rank_, tag));
        if ((int)blob.size() != len) {
            throw std::runtime_error("BootstrapPMI2::recv: size mismatch");
        }
        std::memcpy(buf, blob.data(), len);
    }
    void sendrecv(const void* sbuf, void* rbuf, int len, int peer, int tag = 0) {
        const uint64_t epoch = next_epoch();
        publish_blob(kvs_key("p2p", epoch, rank_, peer, tag),
                     reinterpret_cast<const uint8_t*>(sbuf), len);
        barrier();
        auto blob = fetch_blob(kvs_key("p2p", epoch, peer, rank_, tag));
        if ((int)blob.size() != len) {
            throw std::runtime_error("BootstrapPMI2::sendrecv: size mismatch");
        }
        std::memcpy(rbuf, blob.data(), len);
    }

    double wtime() const noexcept {
        using clock = std::chrono::steady_clock;
        auto now = clock::now().time_since_epoch();
        return std::chrono::duration<double>(now).count();
    }

private:
    int rank_ = -1;
    int size_ = 0;
    int local_rank_ = -1;
    int local_size_ = 0;
    bool owned_ = false;
    std::string hostname_;
    uint64_t epoch_counter_ = 0;

    uint64_t next_epoch() { return epoch_counter_++; }
    uint64_t next_epoch_peek() { return epoch_counter_ - 1; }

    void read_env_local_identity() {
        const char* lr_vars[] = {
            "SLURM_LOCALID", "FLUX_TASK_LOCAL_ID",
            "OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", nullptr};
        for (int i = 0; lr_vars[i] && local_rank_ < 0; ++i) {
            if (const char* v = std::getenv(lr_vars[i])) local_rank_ = std::atoi(v);
        }
        if (local_rank_ < 0) local_rank_ = 0;

        const char* ls_vars[] = {
            "SLURM_NTASKS_PER_NODE", "FLUX_LOCAL_RANKS", nullptr};
        for (int i = 0; ls_vars[i] && local_size_ <= 0; ++i) {
            if (const char* v = std::getenv(ls_vars[i])) local_size_ = std::atoi(v);
        }
        if (local_size_ <= 0) local_size_ = 1;
    }

    void publish_hostname_and_fence() {
        std::string key = "gicc.host." + std::to_string(rank_);
        int rc = PMI2_KVS_Put(key.c_str(), hostname_.c_str());
        if (rc != PMI2_SUCCESS) {
            throw std::runtime_error("BootstrapPMI2: KVS_Put(hostname) failed");
        }
        rc = PMI2_KVS_Fence();
        if (rc != PMI2_SUCCESS) {
            throw std::runtime_error("BootstrapPMI2: KVS_Fence(hostname) failed");
        }
    }

    static std::string hex_encode(const uint8_t* data, int len) {
        static const char* digits = "0123456789abcdef";
        std::string out(2 * len, '0');
        for (int i = 0; i < len; ++i) {
            out[2 * i]     = digits[(data[i] >> 4) & 0xF];
            out[2 * i + 1] = digits[data[i] & 0xF];
        }
        return out;
    }
    static std::vector<uint8_t> hex_decode(const char* s) {
        int n = (int)std::strlen(s);
        if (n % 2 != 0) {
            throw std::runtime_error("BootstrapPMI2: odd-length hex blob");
        }
        std::vector<uint8_t> out(n / 2);
        for (int i = 0; i < n / 2; ++i) {
            unsigned hi = 0, lo = 0;
            std::sscanf(s + 2 * i,     "%1x", &hi);
            std::sscanf(s + 2 * i + 1, "%1x", &lo);
            out[i] = (uint8_t)((hi << 4) | lo);
        }
        return out;
    }

    template<class... Xs>
    static std::string kvs_key(const char* tag, Xs... xs) {
        std::string out = std::string("gicc.") + tag;
        append_parts(out, xs...);
        return out;
    }
    static void append_parts(std::string&) {}
    template<class Head, class... Tail>
    static void append_parts(std::string& s, Head h, Tail... rest) {
        s += ".";
        s += std::to_string(h);
        append_parts(s, rest...);
    }

    void publish_blob(const std::string& key, const uint8_t* data, int len) {
        auto hex = hex_encode(data, len);
        int rc = PMI2_KVS_Put(key.c_str(), hex.c_str());
        if (rc != PMI2_SUCCESS) {
            throw std::runtime_error(
                "BootstrapPMI2::publish_blob: KVS_Put(" + key + ") failed");
        }
    }
    std::vector<uint8_t> fetch_blob(const std::string& key) {
        // PMI2 max KVS value length varies by implementation; 16 KB covers our
        // bootstrap payloads comfortably (largest is N*locality hostnames).
        std::vector<char> buf(16 * 1024);
        int vallen = 0;
        int rc = PMI2_KVS_Get(NULL, PMI2_ID_NULL, key.c_str(),
                              buf.data(), (int)buf.size(), &vallen);
        if (rc != PMI2_SUCCESS) {
            throw std::runtime_error(
                "BootstrapPMI2::fetch_blob: KVS_Get(" + key + ") failed");
        }
        return hex_decode(buf.data());
    }
};

} // namespace gicc::detail
```

- [ ] **Step 2: Syntax check (skip on MAPLE — no PMI2 there; defer to Phase 1 Task 6 Tioga check)**

On MAPLE we can still do a partial parse check by faking pmi2.h with a stub. Skip this task's isolated check — Task 6 will compile it for real on Tioga.

- [ ] **Step 3: Commit**

```bash
cd /p/lustre2/shan4/opengda
git add src/gicc/bootstrap/detail/bootstrap_pmi2.hpp
git commit -m "Add detail::BootstrapPMI2 — PMI2 bootstrap with KVS-backed collectives"
```

---

### Task 3: Create the public `bootstrap.hpp` dispatcher

**Files:**
- Create: `src/gicc/bootstrap/bootstrap.hpp`

- [ ] **Step 1: Create the file**

```cpp
// src/gicc/bootstrap/bootstrap.hpp
//
// Unified process-launch / metadata-exchange abstraction for GICC.
// At build time, exactly one of GICC_BOOTSTRAP_MPI or GICC_BOOTSTRAP_PMI2
// is defined (see GICC_BOOTSTRAP in the top-level CMakeLists). That decision
// selects the implementation class aliased here as gicc::Bootstrap.
//
// User code should include only this header. No MPI_* or PMI_* symbols
// ever need to appear outside the detail headers.
#pragma once

#if defined(GICC_BOOTSTRAP_MPI) && defined(GICC_BOOTSTRAP_PMI2)
    #error "Define exactly one of GICC_BOOTSTRAP_MPI / GICC_BOOTSTRAP_PMI2, not both"
#endif

#if defined(GICC_BOOTSTRAP_MPI)
    #include "gicc/bootstrap/detail/bootstrap_mpi.hpp"
    namespace gicc { using Bootstrap = detail::BootstrapMPI; }
#elif defined(GICC_BOOTSTRAP_PMI2)
    #include "gicc/bootstrap/detail/bootstrap_pmi2.hpp"
    namespace gicc { using Bootstrap = detail::BootstrapPMI2; }
#else
    #error "Define GICC_BOOTSTRAP_MPI or GICC_BOOTSTRAP_PMI2 " \
           "(set via -DGICC_BOOTSTRAP=mpi|pmi2 in CMake)"
#endif
```

- [ ] **Step 2: Syntax check**

```bash
mpicxx -std=c++17 -fsyntax-only -DGICC_BOOTSTRAP_MPI=1 \
    -I/p/lustre2/shan4/opengda/src \
    /p/lustre2/shan4/opengda/src/gicc/bootstrap/bootstrap.hpp
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
cd /p/lustre2/shan4/opengda
git add src/gicc/bootstrap/bootstrap.hpp
git commit -m "Add gicc/bootstrap/bootstrap.hpp dispatcher header"
```

---

### Task 4: Create `gicc::coll::` helpers

**Files:**
- Create: `src/gicc/coll.hpp`

- [ ] **Step 1: Create the file**

```cpp
// src/gicc/coll.hpp
//
// Collective reduction helpers built on top of Bootstrap::allgather_fixed.
// These live outside Bootstrap so both BootstrapMPI and BootstrapPMI2 share
// one implementation and new reduction ops can be added without touching
// the Bootstrap surface.
#pragma once

#include "gicc/bootstrap/bootstrap.hpp"

#include <algorithm>
#include <vector>

namespace gicc::coll {

template<class T>
inline T allreduce_sum(Bootstrap& bs, const T& local) {
    auto all = bs.template allgather_fixed<T>(local);
    T acc = all.empty() ? T{} : all[0];
    for (std::size_t i = 1; i < all.size(); ++i) acc = acc + all[i];
    return acc;
}

template<class T>
inline T allreduce_min(Bootstrap& bs, const T& local) {
    auto all = bs.template allgather_fixed<T>(local);
    T acc = all.empty() ? local : all[0];
    for (std::size_t i = 1; i < all.size(); ++i)
        if (all[i] < acc) acc = all[i];
    return acc;
}

template<class T>
inline T allreduce_max(Bootstrap& bs, const T& local) {
    auto all = bs.template allgather_fixed<T>(local);
    T acc = all.empty() ? local : all[0];
    for (std::size_t i = 1; i < all.size(); ++i)
        if (acc < all[i]) acc = all[i];
    return acc;
}

} // namespace gicc::coll
```

- [ ] **Step 2: Syntax check**

```bash
mpicxx -std=c++17 -fsyntax-only -DGICC_BOOTSTRAP_MPI=1 \
    -I/p/lustre2/shan4/opengda/src \
    /p/lustre2/shan4/opengda/src/gicc/coll.hpp
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
cd /p/lustre2/shan4/opengda
git add src/gicc/coll.hpp
git commit -m "Add gicc::coll:: reduction helpers (allreduce_sum/min/max)"
```

---

### Task 5: Update `src/CMakeLists.txt`

**Files:**
- Modify: `src/CMakeLists.txt`

- [ ] **Step 1: Replace the dependencies block**

Open `src/CMakeLists.txt`. Find the block starting at line 11 (`# Dependencies`) through line 28 (the `find_library(PMIX_LIBRARY ...)` line). Replace that whole block with:

```cmake
# ===========================================================================
# Dependencies
# ===========================================================================

# Independent build knobs. Each has a fixed default; no cross-option inference.
set(GICC_BACKEND   "ofi" CACHE STRING "Network backend: mlx5 | ofi")
set(GICC_BOOTSTRAP "mpi" CACHE STRING "Bootstrap backend: mpi | pmi2")
set_property(CACHE GICC_BACKEND   PROPERTY STRINGS "mlx5" "ofi")
set_property(CACHE GICC_BOOTSTRAP PROPERTY STRINGS "mpi"  "pmi2")

find_package(CUDAToolkit REQUIRED)

find_library(IBVERBS_LIBRARY ibverbs)
if(NOT IBVERBS_LIBRARY)
    message(FATAL_ERROR "libibverbs not found")
endif()

find_library(MLX5_LIBRARY mlx5)
if(NOT MLX5_LIBRARY)
    message(FATAL_ERROR "libmlx5 not found")
endif()
```

- [ ] **Step 2: Replace the include + link blocks**

Find `target_include_directories(gicc ...` (around line 39). Keep only the public include directory block. Remove `${MPI_CXX_INCLUDE_DIRS}` from PRIVATE. The block should read:

```cmake
target_include_directories(gicc
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
)
```

Find `target_link_libraries(gicc ...` (around line 50). Remove `${MPI_CXX_LIBRARIES}` and the trailing `if(PMIX_LIBRARY) ... endif()` block. The block should read:

```cmake
target_link_libraries(gicc
    PUBLIC
        ${IBVERBS_LIBRARY}
        ${MLX5_LIBRARY}
        CUDA::cudart
)
```

- [ ] **Step 3: Append the bootstrap dispatch block**

After the `target_link_libraries(gicc ...)` block you just edited (still before `target_compile_options(gicc PRIVATE ...)`), add:

```cmake
# ===========================================================================
# Bootstrap backend
# ===========================================================================

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

- [ ] **Step 4: Update the summary block**

Find `message(STATUS "Platform:   MLX5 / InfiniBand")` (around line 120) and update the summary:

```cmake
message(STATUS "=== GICC Library Build Configuration ===")
message(STATUS "Backend:    ${GICC_BACKEND}")
message(STATUS "Bootstrap:  ${GICC_BOOTSTRAP}")
message(STATUS "CUDA:       ${CUDAToolkit_VERSION}")
message(STATUS "CUDA arch:  ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "ibverbs:    ${IBVERBS_LIBRARY}")
message(STATUS "mlx5:       ${MLX5_LIBRARY}")
```

- [ ] **Step 5: Build on MAPLE to verify CMake still configures and library still compiles**

Note: the library will still *use* the old `MpiBootstrap` via `mpi_bootstrap.hpp`. That's intentional at this phase — we're not migrating callers yet. MPI is still linked because we chose `GICC_BOOTSTRAP=mpi` (the default).

```bash
cd /p/lustre2/shan4/opengda
module use /shared/data1/Projects/CSE_HPC/apps/modules/aarch64/maple
module load gcc/11.4.1 cuda/12.4.131 nvhpc/24.5.cuda_12.4 openmpi-gcc/5.0.5/cuda.12.4 cmake/3.29.6
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_CUDA_COMPILER=$(which nvcc) -DGICC_BACKEND=mlx5
make -j
```

Expected: configuration succeeds, `libgicc.so` builds without errors.

- [ ] **Step 6: Commit**

```bash
cd /p/lustre2/shan4/opengda
git add src/CMakeLists.txt
git commit -m "Add GICC_BACKEND / GICC_BOOTSTRAP CMake options"
```

---

## Phase 2 — Library migration

Goal: every file under `src/` that previously referenced `MPI_*` or `PMI_*` now goes through `gicc::Bootstrap` or `gicc::coll::`. The old `mpi_bootstrap.hpp` and `pmi_session.hpp` are still present but no longer referenced from library code.

---

### Task 6: Migrate `src/gicc/platform/mlx5/mlx5_runtime.hpp`

**Files:**
- Modify: `src/gicc/platform/mlx5/mlx5_runtime.hpp`

- [ ] **Step 1: Swap the include**

Find line 18 `#include <mpi.h>`. Remove it. Find line 29 `#include "gicc/util/mpi_bootstrap.hpp"`. Replace both with a single line:

```cpp
#include "gicc/bootstrap/bootstrap.hpp"
```

- [ ] **Step 2: Change the constructor signature and body**

Find line 47 `Runtime(MPI_Comm comm = MPI_COMM_WORLD) {` and the body that references `mpi_`. Replace the constructor (through the closing brace of `connect_peers()` call) with:

```cpp
Runtime() {
    // Bootstrap owns MPI (or PMI) init.
    // Stored by-value — do not reallocate.

    // GPU setup
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus == 0) {
        fprintf(stderr, "GICC: No CUDA devices found\n");
        exit(1);
    }
    gpu_id_ = boot_.local_rank() % num_gpus;
    cudaSetDevice(gpu_id_);
    cudaGetDeviceProperties(&gpu_props_, gpu_id_);
    clock_rate_khz_ = gpu_props_.clockRate;

    // Open IB device
    open_ib_device();

    // Allocate protection domain
    pd_ = ibv_alloc_pd(ib_ctx_);
    if (!pd_) {
        fprintf(stderr, "GICC Rank %d: ibv_alloc_pd failed\n", boot_.rank());
        exit(1);
    }

    // Create one DevX QP per peer
    for (int i = 0; i < boot_.size(); i++) {
        if (i == boot_.rank()) continue;
        peer_qps_[i] = new gicc::mlx5::DevxQp(
            ib_ctx_, pd_, boot_.rank(), 1, 256, 512);
    }

    connect_peers();
}
```

- [ ] **Step 3: Remove `delete mpi_` from destructor**

Find the destructor body (around line 83). Delete these two lines:

```cpp
delete mpi_;
mpi_ = nullptr;
```

- [ ] **Step 4: Replace `MPI_Allgather` in `exchange()`**

Find the body of `exchange()` (around line 109). Replace the `MPI_Allgather` call and its surrounding setup with a `Bootstrap` call:

```cpp
void exchange() {
    struct BufEntry { uint64_t addr; uint32_t rkey; };

    int n = (int)local_bufs_.size();
    std::vector<BufEntry> my_entries(n);
    for (int i = 0; i < n; i++) {
        my_entries[i] = { (uint64_t)local_bufs_[i]->buf, local_bufs_[i]->rkey };
    }

    auto raw = boot_.allgather(my_entries.data(), n * (int)sizeof(BufEntry));

    remote_bufs_.resize(boot_.size());
    for (int r = 0; r < boot_.size(); r++) {
        remote_bufs_[r].resize(n);
        const auto* entries = reinterpret_cast<const BufEntry*>(raw[r].data());
        for (int b = 0; b < n; b++) {
            remote_bufs_[r][b] = { entries[b].addr, entries[b].rkey };
        }
    }
}
```

- [ ] **Step 5: Replace all remaining `mpi_->`/`mpi_` references**

Within this file, make these textual substitutions (search whole-file):

| Old | New |
|---|---|
| `mpi_->rank` | `boot_.rank()` |
| `mpi_->size` | `boot_.size()` |
| `mpi_->local_rank` | `boot_.local_rank()` |
| `mpi_->barrier()` | `boot_.barrier()` |
| `mpi_->exchange(` | `boot_.sendrecv(` |
| `mpi_->comm` | *(remove the argument; Bootstrap has no comm to expose)* |

Special case: `prepare()` uses `mpi_->rank` for the error message; change to `boot_.rank()`.

Special case: the line `MPI_Allgather(my_entries.data(), n * (int)sizeof(BufEntry), MPI_BYTE, ... mpi_->comm);` was already rewritten in Step 4.

Special case: the `connect_peers()` helper (bottom of file) calls `mpi_->exchange(&my_info, &peer_info, sizeof(ConnInfo), peer);` and `mpi_->barrier();`. These translate to `boot_.sendrecv(&my_info, &peer_info, sizeof(ConnInfo), peer);` and `boot_.barrier();`.

- [ ] **Step 6: Replace the `mpi_` member with a `boot_` value**

Find the private members (around line 237). Replace:

```cpp
gicc::MpiBootstrap* mpi_ = nullptr;
```

with:

```cpp
gicc::Bootstrap boot_;
```

- [ ] **Step 7: Update the accessors**

Find the public accessors (around line 228). Update:

```cpp
void barrier() { boot_.barrier(); }

int rank() const { return boot_.rank(); }
int size() const { return boot_.size(); }
Bootstrap& boot() noexcept { return boot_; }
const Bootstrap& boot() const noexcept { return boot_; }
```

Add the `boot()` accessor so callers can reach the Bootstrap for collectives / wtime.

- [ ] **Step 8: Update the file docstring**

The top comment mentions "MPI bootstrap" in the bullet list at line 7. Change that bullet from `- MPI bootstrap` to `- Bootstrap (MPI or PMI2, selected at build time)`.

- [ ] **Step 9: Build**

```bash
cd /p/lustre2/shan4/opengda/build
make -j
```

Expected: compiles clean. If errors remain, they are almost certainly from residual `mpi_->` tokens — search for them.

- [ ] **Step 10: Commit**

```bash
cd /p/lustre2/shan4/opengda
git add src/gicc/platform/mlx5/mlx5_runtime.hpp
git commit -m "mlx5_runtime: use gicc::Bootstrap instead of MpiBootstrap / raw MPI"
```

---

### Task 7: Migrate `src/gicc/platform/ofi/ofi_runtime.hpp`

**Files:**
- Modify: `src/gicc/platform/ofi/ofi_runtime.hpp`

- [ ] **Step 1: Swap includes**

Find line 23 `#include <mpi.h>`. Remove it. Find line 37 `#include "internal/pmi_session.hpp"`. Replace with:

```cpp
#include "gicc/bootstrap/bootstrap.hpp"
```

- [ ] **Step 2: Change constructor signature**

Find line 60 `explicit Runtime(MPI_Comm comm = MPI_COMM_WORLD)` and its initializer list (lines 60-65). Replace with:

```cpp
Runtime()
    : comm_(nullptr),
      h_dev_ctx_(nullptr), d_dev_ctx_(nullptr),
      d_slot_pool_(nullptr), mr_slot_pool_(nullptr),
      d_operand_pool_(nullptr), mr_operand_pool_(nullptr),
      my_n_ops_(0), atomic_signals_queued_(false)
{
```

Remove the `(void)mpi_comm_;` line (line 67). `GdaComm` construction needs to change to accept a `Bootstrap&` (done in Task 10), so for now change:

```cpp
comm_ = new GdaComm();
```

to:

```cpp
comm_ = new GdaComm(boot_);
```

- [ ] **Step 3: Remove the MPI_Comm member**

Find `MPI_Comm mpi_comm_;` (around line 350). Delete that line. Add a new private member right before it:

```cpp
gicc::Bootstrap boot_;
```

- [ ] **Step 4: Add `boot()` accessor**

Find the public section (search for `int rank()`). Add:

```cpp
Bootstrap& boot() noexcept { return boot_; }
const Bootstrap& boot() const noexcept { return boot_; }
```

If the current file doesn't expose `rank()` / `size()` directly (they come via `comm_->rank()` etc.), leave those alone.

- [ ] **Step 5: Commit (deferred build — ofi build needs GdaComm changes first)**

The file references `GdaComm(boot_)` which doesn't exist yet. That will be fixed in Task 10. Stage and commit the runtime change; the full ofi build is verified at Task 13 after both changes land.

```bash
cd /p/lustre2/shan4/opengda
git add src/gicc/platform/ofi/ofi_runtime.hpp
git commit -m "ofi_runtime: use gicc::Bootstrap, remove MPI_Comm ctor arg"
```

---

### Task 8: Migrate `src/gicc/mlx5/am_context.hpp`

**Files:**
- Modify: `src/gicc/mlx5/am_context.hpp`

- [ ] **Step 1: Swap includes**

Find line 12 `#include <mpi.h>`. Remove it. Right after `#include "am_types.hpp"` (line 19), add:

```cpp
#include "gicc/bootstrap/bootstrap.hpp"
```

- [ ] **Step 2: Change the constructor signature**

Find the `NvibAmContext` constructor (around line 82). Change signature to accept a `Bootstrap&`:

```cpp
NvibAmContext(gicc::Bootstrap& boot_,
              struct ibv_context* ib_ctx_,
              struct ibv_pd* pd_,
              DevxQp* qp_,
              int nslots_ = AM_DEFAULT_RING_SLOTS)
    : boot(boot_),
      ib_ctx(ib_ctx_),
      pd(pd_),
      qp(qp_),
      nslots(nslots_),
      d_context(nullptr),
      d_recv_states(nullptr),
      d_gda_state(nullptr)
{
```

Delete the two `MPI_Comm_rank(MPI_COMM_WORLD, &rank_);` / `_size_` lines in the body (around lines 94-95). The `rank_` / `size_` members are now derived from `boot`.

- [ ] **Step 3: Add a `Bootstrap&` member**

In the public section (before `int rank_;`), add:

```cpp
gicc::Bootstrap& boot;
```

Replace the class's `rank_` / `size_` members by making the accessors delegate to `boot`:

```cpp
int rank() const { return boot.rank(); }
int size() const { return boot.size(); }
```

Remove the `int rank_;` / `int size_;` private members. Update every in-file reference of `rank_` to `boot.rank()` and `size_` to `boot.size()`. Since the class uses them heavily, expect ~10-15 sites.

- [ ] **Step 4: Replace `MPI_Allgather` in `connect_qps()`**

Find around line 203:

```cpp
std::vector<QpConnInfo> all_conns(size_);
MPI_Allgather(&my_conn, sizeof(QpConnInfo), MPI_BYTE,
              all_conns.data(), sizeof(QpConnInfo), MPI_BYTE,
              MPI_COMM_WORLD);
```

Replace with:

```cpp
auto all_conns = boot.template allgather_fixed<QpConnInfo>(my_conn);
```

- [ ] **Step 5: Replace `MPI_Barrier` at end of `connect_qps()`**

Line 217 `MPI_Barrier(MPI_COMM_WORLD);` → `boot.barrier();`

- [ ] **Step 6: Replace `MPI_Allgather` in `exchange_addresses()`**

Find around line 231:

```cpp
std::vector<am_exchange_info_t> all_infos(size_ * size_);
MPI_Allgather(my_infos.data(), size_ * sizeof(am_exchange_info_t), MPI_BYTE,
              all_infos.data(), size_ * sizeof(am_exchange_info_t), MPI_BYTE,
              MPI_COMM_WORLD);
```

Replace with a byte-oriented allgather (the element size varies with `size_`, so `allgather_fixed` won't match — use the variable-length form):

```cpp
auto raw_all = boot.allgather(my_infos.data(),
                              boot.size() * (int)sizeof(am_exchange_info_t));
std::vector<am_exchange_info_t> all_infos(boot.size() * boot.size());
for (int r = 0; r < boot.size(); ++r) {
    std::memcpy(&all_infos[r * boot.size()], raw_all[r].data(),
                boot.size() * sizeof(am_exchange_info_t));
}
```

- [ ] **Step 7: Build**

```bash
cd /p/lustre2/shan4/opengda/build
make -j
```

Expected: `am_context.hpp` compiles. If errors about unknown callers remain, they come from Task 10 files which reference `NvibAmContext` — you'll fix those there.

- [ ] **Step 8: Commit**

```bash
cd /p/lustre2/shan4/opengda
git add src/gicc/mlx5/am_context.hpp
git commit -m "am_context: accept Bootstrap&, replace raw MPI collectives"
```

---

### Task 9: Migrate `src/gicc/mlx5/{gda_context,gpu_comm,comm}.hpp`

**Files:**
- Modify: `src/gicc/mlx5/gda_context.hpp`
- Modify: `src/gicc/mlx5/gpu_comm.hpp`
- Modify: `src/gicc/mlx5/comm.hpp`

These three files each hold a `MpiBootstrap` (either by-value or by-reference). Change all three to hold a `Bootstrap` reference instead.

- [ ] **Step 1: Edit `gda_context.hpp`**

Open the file. Replace `#include "gicc/util/mpi_bootstrap.hpp"` with `#include "gicc/bootstrap/bootstrap.hpp"`.

Find `MpiBootstrap& mpi;` (around line 179). Replace with `gicc::Bootstrap& boot;`.

Find the constructor `Mlx5GdaContext(MpiBootstrap& mpi_, ...)` (around line 201). Rename parameter: `Mlx5GdaContext(gicc::Bootstrap& boot_, ...)`. In the initializer list, change `mpi(mpi_)` to `boot(boot_)`.

Inside the file, replace every `mpi.rank` with `boot.rank()`, every `mpi.size` with `boot.size()`, every `mpi.barrier()` with `boot.barrier()`, every `mpi.allgather(...)` / `mpi.allgather_fixed(...)` / `mpi.broadcast(...)` — these names already match, so only the receiver changes (`mpi` → `boot`).

- [ ] **Step 2: Edit `gpu_comm.hpp`**

Open the file. Replace `#include "gicc/util/mpi_bootstrap.hpp"` with `#include "gicc/bootstrap/bootstrap.hpp"`.

Find `MpiBootstrap mpi;` (around line 52). Replace with `gicc::Bootstrap& boot;`.

Inside the class / file, every reference to `mpi.*` becomes `boot.*` — method names match.

The class previously *constructed* an `MpiBootstrap` as a by-value member. Now it holds a reference, which must be injected at construction. If `GpuComm` currently has a default constructor, change the signature to require a `Bootstrap&`. Update callers (if any — search `GpuComm(` throughout the tree) to pass in the Runtime's Bootstrap.

- [ ] **Step 3: Edit `comm.hpp`**

Same pattern as `gpu_comm.hpp`. Replace the `MpiBootstrap mpi;` member (around line 61) with `gicc::Bootstrap& boot;` and migrate all call sites.

- [ ] **Step 4: Build**

```bash
cd /p/lustre2/shan4/opengda/build
make -j
```

Expected: these three headers compile. The residual build errors will surface in downstream users of `Mlx5GdaContext` etc.; Task 11 (runtime rewires them) may need to re-touch this if signatures changed.

- [ ] **Step 5: Commit**

```bash
cd /p/lustre2/shan4/opengda
git add src/gicc/mlx5/gda_context.hpp src/gicc/mlx5/gpu_comm.hpp src/gicc/mlx5/comm.hpp
git commit -m "mlx5/{gda_context,gpu_comm,comm}: use gicc::Bootstrap&"
```

---

### Task 10: Migrate `src/gicc/platform/ofi/internal/gda_comm.hpp`

**Files:**
- Modify: `src/gicc/platform/ofi/internal/gda_comm.hpp`

- [ ] **Step 1: Swap includes**

Find line 25 `#include <mpi.h>`. Remove it. Find line 30 `#include "pmi_session.hpp"`. Replace with:

```cpp
#include "gicc/bootstrap/bootstrap.hpp"
```

- [ ] **Step 2: Replace the public `PmiSession pmi;` member with a `Bootstrap&`**

Find line 63 `PmiSession pmi;`. Replace with:

```cpp
gicc::Bootstrap& boot;
```

- [ ] **Step 3: Add a Bootstrap-taking constructor**

The current `GdaComm` is default-constructed (no explicit ctor). Add one that accepts a `Bootstrap&`:

```cpp
explicit GdaComm(gicc::Bootstrap& boot_);
~GdaComm();
```

Move any existing body of the default ctor into this one. Every internal use of `pmi.rank()` / `pmi.size()` / `pmi.barrier()` / `pmi.build_locality_map()` translates directly to the same method on `boot` (same names in the Bootstrap API). Replace them accordingly.

`pmi.is_same_node(peer)` doesn't exist on `Bootstrap`. Current callers (grep first: `grep -n is_same_node src/gicc/platform/ofi/`) can derive it from `boot.locality_map()[peer]`. Do that conversion at each call site.

`pmi.kvs_put` / `pmi.kvs_get` also don't exist on `Bootstrap`. If any internal code uses them for setup, replace with `boot.allgather` / `boot.broadcast`.

- [ ] **Step 4: Update `accessors`**

`rank()` and `size()` methods delegate to `boot`:

```cpp
int rank() const { return boot.rank(); }
int size() const { return boot.size(); }
```

- [ ] **Step 5: Build**

```bash
cd /p/lustre2/shan4/opengda
mkdir -p build_ofi && cd build_ofi
cmake .. -DGICC_BACKEND=ofi -DGICC_BOOTSTRAP=pmi2 \
         -Dhip_DIR=/opt/rocm-6.4.0/lib/cmake/hip \
         -DCMAKE_HIP_COMPILER=/opt/rocm-6.4.0/lib/llvm/bin/clang++
make -j
```

Note: this must be run on Tioga, which has ROCm + PMI2. If you're on MAPLE, also try `-DGICC_BOOTSTRAP=mpi` as an ofi-layout compile-only sanity check (it will fail at runtime on MAPLE because there's no HIP, but CMake config should succeed on a machine with ROCm).

- [ ] **Step 6: Commit**

```bash
cd /p/lustre2/shan4/opengda
git add src/gicc/platform/ofi/internal/gda_comm.hpp
git commit -m "gda_comm: hold gicc::Bootstrap& instead of owning PmiSession"
```

---

### Task 11: Migrate `src/gicc/platform/ofi/internal/gicc_barrier.hpp`

**Files:**
- Modify: `src/gicc/platform/ofi/internal/gicc_barrier.hpp`

- [ ] **Step 1: Find the MPI block**

```bash
grep -n 'MPI_' /p/lustre2/shan4/opengda/src/gicc/platform/ofi/internal/gicc_barrier.hpp
```

Expected output: lines 363-366 with two `MPI_Allgather` calls.

- [ ] **Step 2: Swap `<mpi.h>` include**

If the file includes `<mpi.h>` at the top, remove that line and add:

```cpp
#include "gicc/bootstrap/bootstrap.hpp"
```

- [ ] **Step 3: Convert the class to take a `Bootstrap&` argument**

The function containing the Allgather calls at line 363-366 needs access to the Bootstrap. Locate the containing method and add a `gicc::Bootstrap& boot` parameter (or, if the surrounding class holds `boot`, use the member). Replace:

```cpp
MPI_Allgather(&my_base, 1, MPI_UINT64_T,
              all_bases.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD);
MPI_Allgather(&my_key,  1, MPI_UINT64_T,
              all_keys.data(),  1, MPI_UINT64_T, MPI_COMM_WORLD);
```

with:

```cpp
auto all_bases = boot.template allgather_fixed<uint64_t>(my_base);
auto all_keys  = boot.template allgather_fixed<uint64_t>(my_key);
```

- [ ] **Step 4: Thread the Bootstrap through the caller chain**

Callers of the function you just changed must now pass `boot`. Likely caller: `ofi_runtime.hpp` during barrier initialization. Grep:

```bash
grep -rn 'gicc_barrier\|GiccBarrier\|init_gicc_barrier' src/
```

Update all call sites to pass the Runtime's `boot_`.

- [ ] **Step 5: Build (Tioga)**

```bash
cd /p/lustre2/shan4/opengda/build_ofi
make -j
```

- [ ] **Step 6: Commit**

```bash
cd /p/lustre2/shan4/opengda
git add src/gicc/platform/ofi/internal/gicc_barrier.hpp \
        src/gicc/platform/ofi/ofi_runtime.hpp
git commit -m "gicc_barrier: take Bootstrap& instead of calling MPI directly"
```

---

### Task 12: Update docs in `src/gicc/gicc.hpp` and `src/gicc/platform/mlx5/gicc_api.hpp`

**Files:**
- Modify: `src/gicc/gicc.hpp`
- Modify: `src/gicc/platform/mlx5/gicc_api.hpp`

- [ ] **Step 1: Update `gicc.hpp`**

Find the docstring around line 6 (`gicc::init(MPI_COMM_WORLD);`). Replace that example with:

```cpp
 *   gicc::Runtime rt;
 *   auto buf = rt.register_buffer(ptr, size, true);
 *   rt.exchange();
```

- [ ] **Step 2: Update `gicc_api.hpp`**

Find line 43 `inline void init(MPI_Comm comm = MPI_COMM_WORLD) {`. The `init` helper is a convenience front-end. Remove its `MPI_Comm` parameter:

```cpp
inline void init() {
    // Delegate to Runtime default ctor; Bootstrap is selected at build time.
    static Runtime rt;
    (void)rt;
}
```

If `init()` has callers that pass a `MPI_Comm`, grep and remove the argument from each call site. (The examples previously called `gicc::init(MPI_COMM_WORLD)`; those will be rewritten in Phase 3 to drop this line entirely in favor of `gicc::Runtime rt;`.)

Also remove `#include <mpi.h>` at the top of this file if present.

- [ ] **Step 3: Build**

```bash
cd /p/lustre2/shan4/opengda/build
make -j
```

- [ ] **Step 4: Commit**

```bash
cd /p/lustre2/shan4/opengda
git add src/gicc/gicc.hpp src/gicc/platform/mlx5/gicc_api.hpp
git commit -m "gicc.hpp / gicc_api.hpp: drop MPI_Comm from public surface"
```

---

### Task 13: Phase 2 build checkpoint

- [ ] **Step 1: MAPLE mlx5 + mpi build**

```bash
cd /p/lustre2/shan4/opengda
rm -rf build && mkdir build && cd build
module use /shared/data1/Projects/CSE_HPC/apps/modules/aarch64/maple
module load gcc/11.4.1 cuda/12.4.131 nvhpc/24.5.cuda_12.4 openmpi-gcc/5.0.5/cuda.12.4 cmake/3.29.6
cmake .. -DCMAKE_CUDA_COMPILER=$(which nvcc) -DGICC_BACKEND=mlx5 -DGICC_BOOTSTRAP=mpi
make -j
```

Expected: `libgicc.so` builds clean.

- [ ] **Step 2: Grep check — no stray MPI_/PMI_ outside detail headers**

```bash
cd /p/lustre2/shan4/opengda
grep -rn --include='*.hpp' --include='*.h' --include='*.cuh' --include='*.cu' --include='*.cpp' 'MPI_\|PMI_\|PMI2_' src/ | \
    grep -v 'src/gicc/bootstrap/detail/' | \
    grep -v 'src/gicc/util/mpi_bootstrap.hpp' | \
    grep -v 'src/gicc/platform/ofi/internal/pmi_session.hpp'
```

Expected: empty output. (The two legacy files are still present but no longer referenced; they'll be deleted in Phase 4.)

- [ ] **Step 3: Tioga build (run on Tioga)**

```bash
cd /p/lustre2/shan4/opengda
rm -rf build_ofi && mkdir build_ofi && cd build_ofi
cmake .. -DGICC_BACKEND=ofi -DGICC_BOOTSTRAP=pmi2 \
         -Dhip_DIR=/opt/rocm-6.4.0/lib/cmake/hip \
         -DCMAKE_HIP_COMPILER=/opt/rocm-6.4.0/lib/llvm/bin/clang++
make -j
```

Expected: clean build.

- [ ] **Step 4: Commit checkpoint tag**

```bash
cd /p/lustre2/shan4/opengda
git tag phase2-library-clean
```

---

## Phase 3 — Example migration

Goal: no `.cu / .cpp` under `examples/` references `MPI_*` / `PMI_*`. All examples use `gicc::Runtime rt;` lifecycle and `rt.boot().*` for metadata.

Each example migration has the same shape:

1. Delete `#include <mpi.h>` / `#include <pmi2.h>`.
2. Delete `MPI_Init` / `MPI_Finalize`; construct `gicc::Runtime rt;` instead.
3. Replace `MPI_Comm_rank/size` with `rt.boot().rank()/size()`.
4. Replace `MPI_Wtime()` with `rt.boot().wtime()`.
5. Replace `MPI_Barrier(MPI_COMM_WORLD)` with `rt.boot().barrier()`.
6. Replace `MPI_Bcast` with `rt.boot().broadcast<T>(value, root)`.
7. Replace `MPI_Send / MPI_Recv` with `rt.boot().send / recv`.
8. Replace `MPI_Sendrecv` with `rt.boot().sendrecv`.
9. Replace `MPI_Isend/Irecv/Waitall` with two paired `rt.boot().sendrecv` calls (one per peer).
10. Replace `MPI_Allreduce(..., MPI_SUM/MIN/MAX, ...)` with `gicc::coll::allreduce_sum/min/max(rt.boot(), local)`.
11. Replace `MPI_Allgather` with `rt.boot().allgather_fixed<T>(value)`.
12. Replace `MPI_Abort(MPI_COMM_WORLD, n)` with `std::exit(n)`.
13. Remove `MPI_CHECK` wrapper macros — errors become exceptions from Bootstrap.

Add `#include "gicc/bootstrap/bootstrap.hpp"` and `#include "gicc/coll.hpp"` as needed.

---

### Task 14: Migrate `examples/gicc/pingpong_bench.cu`

**Files:**
- Modify: `examples/gicc/pingpong_bench.cu`

- [ ] **Step 1: Apply the pattern above**

Remove `#include <mpi.h>` (line 14). Drop `MPI_Init(&argc, &argv);` (line 139) — the Runtime's Bootstrap initializes automatically. Delete `MPI_Finalize();` (lines 146, 282).

Replace `MPI_Abort(MPI_COMM_WORLD, 1);` (line 119) with `std::exit(1);`.

If `pingpong_bench.cu` queries rank/size directly via `MPI_Comm_rank` etc., replace with `rt.boot().rank() / size()`.

- [ ] **Step 2: Build**

```bash
cd /p/lustre2/shan4/opengda/build
make -j pingpong_bench
```

- [ ] **Step 3: Commit**

```bash
cd /p/lustre2/shan4/opengda
git add examples/gicc/pingpong_bench.cu
git commit -m "pingpong_bench: remove direct MPI calls, use rt.boot()"
```

---

### Task 15: Migrate `examples/gicc/mm.cu`

**Files:**
- Modify: `examples/gicc/mm.cu`

- [ ] **Step 1: Apply the pattern**

Remove `#include <mpi.h>` (line 18). Drop `MPI_Init(&argc, &argv);` (line 100) and `MPI_Finalize();` (lines 110, 276).

Replace `MPI_Abort(MPI_COMM_WORLD, 1);` (line 25) with `std::exit(1);`.

Replace `MPI_Barrier(MPI_COMM_WORLD);` (line 227) with `rt.boot().barrier();`.

Replace the two `MPI_Send / MPI_Recv` pair around lines 257-262 with `rt.boot().send / recv` calls (same tag). `MPI_Recv(C + r*Ns*N, Ns*N, MPI_FLOAT, r, 0, ...)` becomes `rt.boot().recv(C + r*Ns*N, Ns*N*(int)sizeof(float), r, 0);`.

- [ ] **Step 2: Build + commit**

```bash
cd /p/lustre2/shan4/opengda/build
make -j mm
cd ..
git add examples/gicc/mm.cu
git commit -m "mm.cu: remove direct MPI calls, use rt.boot()"
```

---

### Task 16: Migrate `examples/gicc/jacobi.cu`

**Files:**
- Modify: `examples/gicc/jacobi.cu`

This is the largest example migration — it uses `Isend/Irecv/Waitall`, `Allreduce`, `Bcast`, `Send/Recv`, and `Wtime`. Follow the pattern strictly:

- [ ] **Step 1: Remove includes and macros**

Remove `#include <mpi.h>` (line 14). Remove the `MPI_CHECK` macro (lines 33-40) — replace its usages with plain calls (Bootstrap throws on failure).

- [ ] **Step 2: Replace lifecycle**

Remove `MPI_CHECK(MPI_Init(&argc, &argv));` (line 293) and `MPI_CHECK(MPI_Finalize());` (line 552). The `gicc::Runtime rt;` at the top of main does both.

- [ ] **Step 3: Replace timing**

`double start = MPI_Wtime();` (lines 238, 445) → `double start = rt.boot().wtime();`. Same for `stop`.

- [ ] **Step 4: Replace broadcasts**

`MPI_CHECK(MPI_Bcast(&runtime_serial, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD));` (line 320) → `rt.boot().broadcast<double>(runtime_serial, 0);`

`MPI_CHECK(MPI_Bcast(&result_correct, 1, MPI_INT, 0, MPI_COMM_WORLD));` (line 526) → `rt.boot().broadcast<int>(result_correct, 0);`

- [ ] **Step 5: Replace allreduce**

`MPI_CHECK(MPI_Allreduce(l2_norm_h, &l2_norm, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));` (line 479) → `l2_norm = gicc::coll::allreduce_sum(rt.boot(), *l2_norm_h);`.

Add `#include "gicc/coll.hpp"` near the top of the file.

- [ ] **Step 6: Replace the halo Isend/Irecv/Waitall**

Find lines 376-392 (the `MPI_Isend/Irecv/Waitall` block). Replace with two `sendrecv`s (one per peer). Example:

```cpp
// Exchange HaloInfo with top and bottom neighbors.
HaloInfo top_info = {}, bottom_info = {};
if (top != MPI_PROC_NULL /* or your equivalent sentinel */) {
    rt.boot().sendrecv(&my_info, &top_info, sizeof(HaloInfo), top, tag);
}
if (bottom != MPI_PROC_NULL) {
    rt.boot().sendrecv(&my_info, &bottom_info, sizeof(HaloInfo), bottom, tag);
}
```

If `MPI_PROC_NULL` is used, replace with an explicit `int NO_PEER = -1;` constant and guard with `if (top != NO_PEER)` instead. Adjust the sentinel at the `top = …` / `bottom = …` assignment sites.

- [ ] **Step 7: Replace Send/Recv for result gather**

Lines 512-523: `MPI_Recv(a_h + r*nx, r_cs*nx, MPI_FLOAT, r, 100, ...)` → `rt.boot().recv(a_h + r*nx, r_cs*nx*(int)sizeof(float), r, 100);`. Same for `MPI_Send` at 523.

- [ ] **Step 8: Build + commit**

```bash
cd /p/lustre2/shan4/opengda/build
make -j jacobi
cd ..
git add examples/gicc/jacobi.cu
git commit -m "jacobi: remove direct MPI calls, use rt.boot() + gicc::coll"
```

---

### Task 17: Migrate `examples/ofi/test_gicc.cpp`

**Files:**
- Modify: `examples/ofi/test_gicc.cpp`

- [ ] **Step 1: Apply the pattern**

Grep for `MPI_\|PMI_`:

```bash
grep -n 'MPI_\|PMI_' /p/lustre2/shan4/opengda/examples/ofi/test_gicc.cpp
```

Apply the same pattern (remove includes, lifecycle, `boot()` accessors).

- [ ] **Step 2: Build (Tioga) + commit**

```bash
cd /p/lustre2/shan4/opengda/build_ofi
make -j test_gicc
cd ..
git add examples/ofi/test_gicc.cpp
git commit -m "test_gicc: use gicc::Runtime instead of direct MPI/PMI"
```

---

### Task 18: Migrate `examples/ofi/gda_benchmark_gicc.cpp`

**Files:**
- Modify: `examples/ofi/gda_benchmark_gicc.cpp`

- [ ] **Step 1: Apply the pattern**

Grep for MPI/PMI tokens, apply replacements.

Note: the comment at lines 109-110 mentions `PMI KVS` — update the wording to `Bootstrap KVS`. (Behavior is unchanged; it's documentation.)

- [ ] **Step 2: Build + commit**

```bash
cd /p/lustre2/shan4/opengda/build_ofi
make -j gda_benchmark_gicc
cd ..
git add examples/ofi/gda_benchmark_gicc.cpp
git commit -m "gda_benchmark_gicc: use gicc::Runtime / rt.boot()"
```

---

### Task 19: Migrate `examples/ofi/mm_gda_minimal_gicc.cpp`

**Files:**
- Modify: `examples/ofi/mm_gda_minimal_gicc.cpp`

- [ ] **Step 1: Apply the pattern, plus special cases**

Grep for `MPI_\|PMI_\|build_locality_map`. Apply the usual lifecycle replacements.

Special: line 94 (comment) and the `PmiSession::build_locality_map` call — replace with `rt.boot().locality_map()`. Note the return type change: old returned `bool*` (raw array, malloc'd), new returns `std::vector<bool>`. Update the consumer accordingly:

```cpp
auto locality_map = rt.boot().locality_map();
// ... use as locality_map[i] ...
// No delete[] needed.
```

Special: comment at line 16 references `PMI_MAX_KVS_ENTRIES=2000` in a run example. Update to mention that this env var only matters when `GICC_BOOTSTRAP=pmi2`.

- [ ] **Step 2: Build + commit**

```bash
cd /p/lustre2/shan4/opengda/build_ofi
make -j mm_gda_minimal_gicc
cd ..
git add examples/ofi/mm_gda_minimal_gicc.cpp
git commit -m "mm_gda_minimal_gicc: use gicc::Runtime + locality_map() vector"
```

---

### Task 20: Migrate `examples/ofi/barrier_test.cpp`

**Files:**
- Modify: `examples/ofi/barrier_test.cpp`

- [ ] **Step 1: Rewrite to use Bootstrap+Runtime**

Current code (lines 45-47):

```cpp
GdaComm comm;
int rank = comm.rank();
```

Now `GdaComm` requires a `Bootstrap&` (Task 10). Wrap it in a Runtime or construct a standalone Bootstrap:

```cpp
// Option A: use the Runtime (preferred — exercises the full stack).
gicc::Runtime rt;
int rank = rt.boot().rank();
// ... remainder of test ...

// Option B: standalone Bootstrap for pure bootstrap-level tests.
// gicc::Bootstrap boot;
// GdaComm comm(boot);
```

Pick Option A unless the test specifically needs to avoid Runtime's GPU/IB bringup.

- [ ] **Step 2: Build + commit**

```bash
cd /p/lustre2/shan4/opengda/build_ofi
make -j barrier_test
cd ..
git add examples/ofi/barrier_test.cpp
git commit -m "barrier_test: use gicc::Runtime for Bootstrap lifecycle"
```

---

### Task 21: Update examples CMakeLists.txt

**Files:**
- Modify: `examples/CMakeLists.txt`
- Modify: `examples/ofi/CMakeLists.txt`

- [ ] **Step 1: `examples/CMakeLists.txt`**

Open the file. Remove any direct `find_package(MPI)` and `target_link_libraries(... MPI::MPI_CXX)` lines. MPI is inherited through `libgicc`'s PUBLIC linkage.

- [ ] **Step 2: `examples/ofi/CMakeLists.txt`**

Find the Cray PMI2 block (lines 25-33). Delete it:

```cmake
# Cray PMI2
set(PMI_ROOT "/opt/cray/pe/pmi/6.1.15" CACHE PATH "Cray PMI path")
find_path(PMI_INCLUDE_DIR pmi2.h
    PATHS ${PMI_ROOT} PATH_SUFFIXES include NO_DEFAULT_PATH)
find_library(PMI_LIBRARY pmi2
    PATHS ${PMI_ROOT} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
if(NOT PMI_INCLUDE_DIR OR NOT PMI_LIBRARY)
    message(FATAL_ERROR "Cray PMI not found at ${PMI_ROOT}")
endif()
```

Also remove `${PMI_INCLUDE_DIR}` from `target_include_directories` and `${PMI_LIBRARY}` from `target_link_libraries` for each example target (lines 48, 55).

- [ ] **Step 3: Build + commit**

```bash
cd /p/lustre2/shan4/opengda/build  # or build_ofi depending on platform
make -j
cd ..
git add examples/CMakeLists.txt examples/ofi/CMakeLists.txt
git commit -m "examples CMake: drop direct MPI/PMI find_package (inherited via libgicc)"
```

---

### Task 22: Phase 3 end-to-end run verification

- [ ] **Step 1: MAPLE — run pingpong_bench**

```bash
cd /p/lustre2/shan4/opengda/build
srun -p maple --account=app -N 2 --ntasks-per-node=1 --gres=gpu:1 --mpi=pmix ./pingpong_bench 2>&1 | tee /tmp/pp.log
```

Expected: completes with latency / bandwidth numbers matching the pre-refactor baseline within ±5%. Grep `/tmp/pp.log` for the key output line.

- [ ] **Step 2: MAPLE — run mm**

```bash
srun -p maple --account=app -N 2 --ntasks-per-node=1 --gres=gpu:1 --mpi=pmix ./mm 2>&1 | tee /tmp/mm.log
```

Expected: correctness check passes (PASS / OK in output).

- [ ] **Step 3: MAPLE — run jacobi**

```bash
srun -p maple --account=app -N 2 --ntasks-per-node=1 --gres=gpu:1 --mpi=pmix ./jacobi 2>&1 | tee /tmp/jac.log
```

Expected: converges to same residual as pre-refactor run; correctness check passes.

- [ ] **Step 4: Tioga — run test_gicc**

```bash
cd /p/lustre2/shan4/opengda/build_ofi
FI_MR_CACHE_MAX_COUNT=0 srun -N 2 -n 2 --ntasks-per-node=1 -t 2 ./test_gicc 2>&1 | tee /tmp/tg.log
```

Expected: test passes.

- [ ] **Step 5: Tioga — ofi + mpi combo compile-only (new supported pairing)**

```bash
cd /p/lustre2/shan4/opengda
rm -rf build_ofi_mpi && mkdir build_ofi_mpi && cd build_ofi_mpi
cmake .. -DGICC_BACKEND=ofi -DGICC_BOOTSTRAP=mpi \
         -Dhip_DIR=/opt/rocm-6.4.0/lib/cmake/hip \
         -DCMAKE_HIP_COMPILER=/opt/rocm-6.4.0/lib/llvm/bin/clang++
make -j
```

Expected: clean build. Runtime behavior is out of scope for this round.

- [ ] **Step 6: Tag**

```bash
cd /p/lustre2/shan4/opengda
git tag phase3-examples-clean
```

---

## Phase 4 — Cleanup

---

### Task 23: Delete the legacy wrapper files

**Files:**
- Delete: `src/gicc/util/mpi_bootstrap.hpp`
- Delete: `src/gicc/platform/ofi/internal/pmi_session.hpp`

- [ ] **Step 1: Verify nobody still includes them**

```bash
grep -rn 'mpi_bootstrap.hpp\|pmi_session.hpp\|MpiBootstrap\|PmiSession' /p/lustre2/shan4/opengda/src/ /p/lustre2/shan4/opengda/examples/
```

Expected: empty output.

- [ ] **Step 2: Remove the files**

```bash
cd /p/lustre2/shan4/opengda
git rm src/gicc/util/mpi_bootstrap.hpp
git rm src/gicc/platform/ofi/internal/pmi_session.hpp
```

- [ ] **Step 3: Rebuild both configurations**

```bash
cd /p/lustre2/shan4/opengda/build && make -j        # mlx5 + mpi
cd /p/lustre2/shan4/opengda/build_ofi && make -j    # ofi + pmi2 (on Tioga)
```

Both should build clean.

- [ ] **Step 4: Commit**

```bash
cd /p/lustre2/shan4/opengda
git commit -m "Remove legacy MpiBootstrap / PmiSession (superseded by gicc::Bootstrap)"
```

---

### Task 24: Final verification

- [ ] **Step 1: Grep for any remaining stray references**

```bash
cd /p/lustre2/shan4/opengda
grep -rn --include='*.hpp' --include='*.h' --include='*.cuh' --include='*.cu' --include='*.cpp' \
    'MPI_\|PMI_\|PMI2_' src/ examples/
```

Expected output: matches only inside `src/gicc/bootstrap/detail/bootstrap_mpi.hpp` and `src/gicc/bootstrap/detail/bootstrap_pmi2.hpp`. Nothing else.

- [ ] **Step 2: Grep for `MpiBootstrap` / `PmiSession` names**

```bash
grep -rn 'MpiBootstrap\|PmiSession' src/ examples/
```

Expected: empty.

- [ ] **Step 3: Full build matrix**

| Platform | Command | Expected |
|---|---|---|
| MAPLE mlx5+mpi | `cmake .. -DGICC_BACKEND=mlx5 -DGICC_BOOTSTRAP=mpi && make` | clean |
| MAPLE defaults (ofi+mpi) | `cmake .. && make` | clean (CMake only; link may fail without ROCm) |
| Tioga ofi+pmi2 | `cmake .. -DGICC_BACKEND=ofi -DGICC_BOOTSTRAP=pmi2 -Dhip_DIR=... && make` | clean |
| Tioga ofi+mpi | `cmake .. -DGICC_BACKEND=ofi -DGICC_BOOTSTRAP=mpi -Dhip_DIR=... && make` | clean |

- [ ] **Step 4: Tag release**

```bash
cd /p/lustre2/shan4/opengda
git tag unified-bootstrap-api-done
```

---

## Self-review summary

**Spec coverage:**
- §2 Bootstrap class → Tasks 1–3 (BootstrapMPI, BootstrapPMI2, dispatcher header).
- §6 coll helpers → Task 4.
- §7 CMake integration → Task 5.
- §8.3 library file changes (11 files) → Tasks 6–12.
- §8.4 example changes (9 files) → Tasks 14–21.
- §8.2 deletions → Task 23.
- §10 test matrix → Tasks 13, 22, 24.
- §9 error handling → baked into BootstrapMPI/PMI2 implementations in Tasks 1–2.

**Placeholders:** None. All code shown inline. Every task has an exact file path, exact command, and expected output.

**Type consistency:**
- `Bootstrap` API shape is defined in Task 1 and consistently used in Tasks 4, 6–23.
- `locality_map()` returns `std::vector<bool>` (Task 1/2) — consumers in Task 19 updated.
- `rt.boot()` accessor is added in Tasks 6 (mlx5_runtime) and 7 (ofi_runtime) consistently.
- `GdaComm(gicc::Bootstrap&)` constructor signature matches the call in Task 7 Step 2.

**Known sharp edges:**
- Task 11 (`gicc_barrier.hpp`) requires reading the full file to know which class holds the `MPI_Allgather` calls. The step intentionally starts with a grep to locate the site.
- Task 9 (`mlx5/gpu_comm.hpp` / `comm.hpp`) changes a by-value member to a reference, which forces callers to pass a `Bootstrap&` at construction. The step says to grep for callers and update them; there may be more than one site depending on how these classes are instantiated.
- PMI2 `send/recv` emulation uses epoch counters which are per-rank local state — if callers mix `send/recv` and `sendrecv` on the same pair of ranks without proper sequencing, keys may collide. This is acceptable for bootstrap-time use (one-shot, serial); documented risk in spec §11.
