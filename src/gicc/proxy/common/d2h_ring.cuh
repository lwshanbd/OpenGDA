/*
 * d2h_ring.cuh - Host-pinned, device-mapped SPSC ring for TransferCmd.
 * GPU produces; CPU proxy consumes. Supports HIP and CUDA.
 *
 * Layout (cache-line aligned):
 *   head                    GPU CAS / read; CPU read (acquire)
 *   tail                    CPU release-store; GPU read (volatile)
 *   buf[Capacity]           the command slots
 *   ack_mask[ ]             CPU-only bitmap of acked slots (no atomics needed)
 *   proxy_read_cursor       CPU-only read cursor (no atomics needed; SPSC)
 *
 * Two-cursor consumption model (deviates from the plan):
 *   - GPU pushes by CAS-incrementing `head` and writing buf[head & mask].
 *   - CPU `pop()` returns the command at `proxy_read_cursor` and advances
 *     `proxy_read_cursor` (CPU-private — single proxy thread).
 *   - CPU `mark_acked(slot)` flips the bit for `slot & mask` in `ack_mask`.
 *   - CPU `advance_tail_from_mask()` walks contiguous acked bits starting at
 *     `tail` and republishes `tail` with release semantics, which un-blocks
 *     the GPU when the ring was full.
 *
 * Why a separate `proxy_read_cursor`?
 *   The plan's original `pop()` did not advance any cursor and relied on
 *   `tail` advancing only when an ack arrived. That re-reads slot N every
 *   time `pop()` is called between dispatch and ack, which is broken for an
 *   SPSC pattern with in-flight commands. A CPU-private read cursor is
 *   cheap (no atomics, no GPU visibility needed) and fixes the issue.
 */
#pragma once

// When this header is included from a translation unit that is NOT compiled
// in HIP/CUDA device-attribute mode (e.g. a plain `-fopenmp` TU that only
// needs the ring's data layout), `__device__` / `__host__` are not keywords.
// Define them as no-ops so the struct's method declarations still parse. The
// method BODIES are already gated on __CUDA_ARCH__/__HIP_DEVICE_COMPILE__ and
// compile to harmless stubs here; OpenMP code never calls them (it uses the
// gicc::omp::* port instead).
#if !defined(__HIPCC__) && !defined(__CUDACC__)
  #ifndef __device__
    #define __device__
  #endif
  #ifndef __host__
    #define __host__
  #endif
#endif

#include "transfer_cmd.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>

#if defined(GICC_GPU_HIP)
#include <hip/hip_runtime.h>
#define GICC_GPU_HOST_MALLOC(p, sz, flags)  hipHostMalloc((p), (sz), (flags))
#define GICC_GPU_HOST_FREE(p)               hipHostFree((p))
#define GICC_GPU_HOST_GET_DEV_PTR(d, h, fl) hipHostGetDevicePointer((d), (h), (fl))
#define GICC_GPU_HOST_ALLOC_MAPPED          hipHostMallocMapped
#elif defined(GICC_GPU_CUDA)
#include <cuda_runtime.h>
#define GICC_GPU_HOST_MALLOC(p, sz, flags)  cudaHostAlloc((p), (sz), (flags))
#define GICC_GPU_HOST_FREE(p)               cudaFreeHost((p))
#define GICC_GPU_HOST_GET_DEV_PTR(d, h, fl) cudaHostGetDevicePointer((d), (h), (fl))
#define GICC_GPU_HOST_ALLOC_MAPPED          cudaHostAllocMapped
#else
#error "GICC_GPU_HIP or GICC_GPU_CUDA must be defined"
#endif

namespace gicc {
namespace proxy {

template <uint32_t Capacity>
struct alignas(128) D2HRing {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of two");
    static_assert(Capacity > 0, "Capacity must be > 0");

    // GPU-visible cursors -----------------------------------------------------
    uint64_t    head;                              // GPU CAS / read; CPU read
    uint64_t    tail;                              // CPU release-store; GPU read
    TransferCmd buf[Capacity];

    // CPU-only state (GPU never reads these) ----------------------------------
    uint64_t    ack_mask[(Capacity + 63) / 64];    // bitmap of acked slots
    uint64_t    proxy_read_cursor;                 // single-consumer read pos

    __host__ __device__ static constexpr uint32_t mask() { return Capacity - 1; }

    // ---------- Device side ----------
    // These are declared as plain __device__ (not __host__ __device__) so that
    // host-only callers cannot accidentally invoke them. Their bodies use
    // device intrinsics (atomicCAS, __nanosleep / __builtin_amdgcn_s_sleep,
    // inline PTX) that are only resolvable during the device codegen pass.
    // NVCC and HIP-Clang both parse the full body during the host pass; the
    // intrinsics survive parsing but error if instantiated for the host.

    // Atomically reserve a slot, write the command, and publish via cmd_type
    // (the per-slot "ready" flag). Returns the absolute slot index claimed.
    //
    // CRITICAL ordering: the CAS that bumps `head` makes the new head visible
    // to the host BEFORE the slot data is written. If the consumer trusted
    // head alone, it would race and read stale/empty buf[idx]. So the
    // consumer instead waits for cmd_type != EMPTY — cmd_type doubles as
    // the per-slot ready flag, written last, after the payload.
    //
    // Body gated on the device-compile pass: nvcc/hipcc lower it to real
    // atomicAdd / atomicCAS / __threadfence_system; a host-only TU that
    // transitively includes this header (e.g. ProxyThread implementation
    // built by g++) parses an empty stub because those intrinsics aren't
    // declared at namespace scope under -x c++. The function is __device__
    // and never called from host code, so the stub is unreachable.
    __device__ uint64_t atomic_push(const TransferCmd& c) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        unsigned long long h, prev;
        do {
            h = atomicAdd(reinterpret_cast<unsigned long long*>(&head), 0ULL);
            unsigned long long t =
                atomicAdd(reinterpret_cast<unsigned long long*>(&tail), 0ULL);
            // Spin if the ring is full; back off briefly to reduce contention.
            while (h - t == Capacity) {
#if defined(__CUDA_ARCH__)
                __nanosleep(64);
#elif defined(__HIP_DEVICE_COMPILE__)
                // HIP / AMDGCN: s_sleep takes "cycles / 64" — 1 ~= 64 cycles.
                __builtin_amdgcn_s_sleep(1);
#endif
                t = atomicAdd(reinterpret_cast<unsigned long long*>(&tail), 0ULL);
            }
            prev = atomicCAS(reinterpret_cast<unsigned long long*>(&head),
                             h, h + 1);
        } while (prev != h);

        uint32_t idx = static_cast<uint32_t>(h) & mask();

        // Write everything EXCEPT cmd_type first. cmd_type is the per-slot
        // ready flag — the consumer polls it to know the rest of the slot
        // is fully written.
        buf[idx].dst_rank   = c.dst_rank;
        buf[idx].src_buf    = c.src_buf;
        buf[idx].dst_buf    = c.dst_buf;
        buf[idx].bytes      = c.bytes;
        buf[idx].src_offset = c.src_offset;
        buf[idx].dst_offset = c.dst_offset;

        __threadfence_system();

        // Publish via cmd_type. After this store the consumer will see a
        // fully-written slot. The earlier threadfence_system already
        // ordered the field stores; the cmd_type store + the consumer's
        // acquire-load form the release/acquire pair, so a second fence
        // here is redundant.
        buf[idx].cmd_type = c.cmd_type;

        return h;
#else
        (void)c;
        return 0;
#endif
    }

    // Volatile read of the host-published tail (so the GPU sees space free up).
    __device__ uint64_t device_tail_volatile() const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        unsigned long long t;
#if defined(__CUDA_ARCH__)
        asm volatile("ld.volatile.global.u64 %0, [%1];"
                     : "=l"(t) : "l"(&tail) : "memory");
#else
        t = __builtin_nontemporal_load(
                reinterpret_cast<const unsigned long long*>(&tail));
#endif
        return static_cast<uint64_t>(t);
#else
        return 0;
#endif
    }

    // ---------- Host side ----------
    // Acquire-load of head: pairs with the device-side __threadfence_system().
    __host__ uint64_t head_volatile() const {
        return __atomic_load_n(&head, __ATOMIC_ACQUIRE);
    }

    __host__ uint64_t tail_volatile() const {
        return __atomic_load_n(&tail, __ATOMIC_RELAXED);
    }

    // Pop the next command into `out`. Returns false if no new command is
    // available (proxy_read_cursor caught up to head). On success, writes
    // the absolute slot index to *out_slot for later mark_acked() and
    // advances the CPU-private read cursor.
    //
    // Note: this does NOT advance `tail`; that is the host-published
    // back-pressure boundary. Tail moves only when contiguous acks arrive
    // (see advance_tail_from_mask).
    __host__ bool pop(TransferCmd& out, uint64_t* out_slot) {
        uint64_t h = __atomic_load_n(&head, __ATOMIC_ACQUIRE);
        if (proxy_read_cursor >= h) return false;
        uint64_t s = proxy_read_cursor;
        uint32_t idx = static_cast<uint32_t>(s) & mask();
        // The producer (atomic_push) bumps `head` BEFORE writing the slot,
        // so seeing a new head does NOT guarantee the slot is fully written.
        // Wait for cmd_type != EMPTY (the per-slot ready flag the producer
        // writes LAST, after threadfence_system). This pairs with the cmd_type
        // store at the end of atomic_push.
        auto raw = __atomic_load_n(
            reinterpret_cast<const uint8_t*>(&buf[idx].cmd_type),
            __ATOMIC_ACQUIRE);
        if (raw == static_cast<uint8_t>(CmdType::EMPTY)) {
            // Slot reserved by producer but not yet published. Treat as
            // "no command available right now"; caller will retry.
            return false;
        }
        out       = buf[idx];
        *out_slot = s;
        ++proxy_read_cursor;
        return true;
    }

    __host__ void mark_acked(uint64_t slot) {
        uint32_t idx = static_cast<uint32_t>(slot) & mask();
        size_t word = idx >> 6;
        size_t bit  = idx & 63;
        ack_mask[word] |= (1ull << bit);
    }

    // Walk contiguous acked slots starting at `tail`, clearing each bit, and
    // release-store the new tail so the GPU sees space free up.
    __host__ uint64_t advance_tail_from_mask() {
        uint64_t t = __atomic_load_n(&tail, __ATOMIC_RELAXED);
        uint64_t h = __atomic_load_n(&head, __ATOMIC_ACQUIRE);
        while (t < h) {
            uint32_t idx = static_cast<uint32_t>(t) & mask();
            size_t word = idx >> 6;
            size_t bit  = idx & 63;
            if (!((ack_mask[word] >> bit) & 1ull)) break;
            ack_mask[word] &= ~(1ull << bit);
            // Reset slot for reuse: clear cmd_type to EMPTY so when the
            // producer wraps around and writes here again, the consumer
            // correctly waits for the new cmd_type publish (release-store
            // of `tail` below makes this clear visible to the producer
            // before the producer can claim space).
            buf[idx].cmd_type = CmdType::EMPTY;
            ++t;
        }
        __atomic_store_n(&tail, t, __ATOMIC_RELEASE);
        return t;
    }
};

// ---------- Allocator ----------
// Allocate a host-pinned, device-mapped D2HRing<Capacity>. Returns the
// host-side pointer; writes the device-side pointer to *out_dev_ptr.
template <uint32_t Capacity>
inline D2HRing<Capacity>* allocate_d2h_ring_host(D2HRing<Capacity>** out_dev_ptr) {
    D2HRing<Capacity>* host_ptr = nullptr;
    auto err = GICC_GPU_HOST_MALLOC(reinterpret_cast<void**>(&host_ptr),
                                    sizeof(D2HRing<Capacity>),
                                    GICC_GPU_HOST_ALLOC_MAPPED);
    if (err != 0 || !host_ptr) {
        std::fprintf(stderr,
                     "allocate_d2h_ring_host: alloc failed (%d)\n", (int)err);
        std::abort();
    }
    new (host_ptr) D2HRing<Capacity>{};
    // Zero-init the cursors and ack mask explicitly — the placement-new of
    // a struct with uninitialized scalar members would otherwise leave them
    // indeterminate.
    host_ptr->head              = 0;
    host_ptr->tail              = 0;
    host_ptr->proxy_read_cursor = 0;
    for (size_t i = 0; i < sizeof(host_ptr->ack_mask) / sizeof(uint64_t); ++i) {
        host_ptr->ack_mask[i] = 0;
    }

    void* dev = nullptr;
    err = GICC_GPU_HOST_GET_DEV_PTR(&dev, host_ptr, 0);
    if (err != 0) {
        std::fprintf(stderr,
                     "allocate_d2h_ring_host: get_dev_ptr failed (%d)\n", (int)err);
        std::abort();
    }
    *out_dev_ptr = reinterpret_cast<D2HRing<Capacity>*>(dev);
    return host_ptr;
}

template <uint32_t Capacity>
inline void free_d2h_ring_host(D2HRing<Capacity>* host_ptr) {
    if (!host_ptr) return;
    host_ptr->~D2HRing<Capacity>();
    auto err = GICC_GPU_HOST_FREE(host_ptr);
    if (err != 0) {
        std::fprintf(stderr,
                     "free_d2h_ring_host: free failed (%d)\n", (int)err);
    }
}

} // namespace proxy
} // namespace gicc
