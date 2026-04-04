/**
 * gicc_api.hpp — NVSHMEM-style simplified GICC API
 *
 * Usage:
 *   gicc::init(MPI_COMM_WORLD);
 *   void* buf = gicc::malloc(size);           // collective: alloc + register + exchange
 *   gicc::GiccContext* ctx = gicc::context();  // get GPU context for kernels
 *   gicc::barrier();
 *   // ... launch kernels using gicc::put(ctx, dst, src, size, peer) ...
 *   gicc::finalize();
 */
#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>

// Forward: Runtime is defined by the platform header included before this file.

namespace gicc {

namespace detail {

struct GlobalState {
    Runtime* rt = nullptr;
    GiccContext* d_ctx = nullptr;
    std::vector<void*> allocs;     // pointers from gicc::malloc
    std::vector<int> buf_indices;  // corresponding buffer indices in Runtime
    bool ctx_dirty = true;
};

inline GlobalState*& state() {
    static GlobalState* s = nullptr;
    return s;
}

} // namespace detail

/// Initialize GICC (MPI, GPU, IB, QP creation). Collective.
inline void init(MPI_Comm comm = MPI_COMM_WORLD) {
    if (detail::state()) return;  // already initialized
    detail::state() = new detail::GlobalState();
    detail::state()->rt = new Runtime(comm);
}

/// Allocate GPU memory, register with IB, and exchange info. Collective.
/// All ranks must call with the same size. Returns a GPU device pointer.
inline void* malloc(size_t size) {
    auto* s = detail::state();
    void* ptr = nullptr;
    cudaMalloc(&ptr, size);
    if (!ptr) {
        fprintf(stderr, "gicc::malloc: cudaMalloc(%zu) failed\n", size);
        return nullptr;
    }
    cudaMemset(ptr, 0, size);
    Buffer buf = s->rt->register_buffer(ptr, size, true);
    s->rt->exchange();
    s->allocs.push_back(ptr);
    s->buf_indices.push_back(buf.index);
    s->ctx_dirty = true;
    return ptr;
}

/// Free a gicc::malloc'd pointer. NOT collective (local only).
inline void free(void* ptr) {
    auto* s = detail::state();
    for (size_t i = 0; i < s->allocs.size(); i++) {
        if (s->allocs[i] == ptr) {
            cudaFree(ptr);
            s->allocs.erase(s->allocs.begin() + i);
            s->buf_indices.erase(s->buf_indices.begin() + i);
            s->ctx_dirty = true;
            return;
        }
    }
}

/// Get (or rebuild) the GPU-accessible context. Pass to kernels.
inline GiccContext* context() {
    auto* s = detail::state();
    if (s->ctx_dirty || !s->d_ctx) {
        if (s->d_ctx) { cudaFree(s->d_ctx); s->d_ctx = nullptr; }
        s->d_ctx = s->rt->build_context();
        s->ctx_dirty = false;
    }
    return s->d_ctx;
}

/// Get the remote address of a gicc::malloc'd buffer on a given PE.
/// Usage: void* remote = gicc::remote_ptr(local_ptr, peer);
inline void* remote_ptr(void* local_ptr, int peer) {
    auto* s = detail::state();
    for (size_t i = 0; i < s->allocs.size(); i++) {
        if (s->allocs[i] == local_ptr) {
            auto info = s->rt->remote_buffer(peer, s->buf_indices[i]);
            return (void*)info.addr;
        }
    }
    return nullptr;
}

/// MPI barrier.
inline void barrier() {
    detail::state()->rt->barrier();
}

/// This rank's PE id.
inline int my_pe() { return detail::state()->rt->rank(); }

/// Total number of PEs.
inline int n_pes() { return detail::state()->rt->size(); }

/// Access the underlying Runtime (escape hatch for advanced use).
inline Runtime& runtime() { return *detail::state()->rt; }

/// Finalize GICC. Collective.
inline void finalize() {
    auto* s = detail::state();
    if (!s) return;
    if (s->d_ctx) cudaFree(s->d_ctx);
    s->rt->reset();
    for (auto* p : s->allocs) cudaFree(p);
    delete s->rt;
    delete s;
    detail::state() = nullptr;
}

} // namespace gicc
