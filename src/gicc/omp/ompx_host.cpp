// GiOMP host runtime implementation. This translation unit is compiled as GPU
// source (-x hip / -x cuda) and owns the OFI Runtime, so OpenMP application
// translation units only need the public gicc/omp.h header and libgicc_omp.
//
// Memory is a symmetric heap: one device allocation registered with the NIC at
// ompx_init and exchanged once. ompx_alloc carves from it, so an allocation
// costs a pointer bump, needs no registration, and -- because every rank
// allocates in the same order -- lands at the same heap offset on every rank.
// That is what lets put/get/peer_ptr take ordinary local addresses.
#include <mpi.h>
#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gicc/omp.h"
#include "gicc/platform/ofi/internal/gpu_device_context.hpp"
#include "gicc/platform/ofi/ofi_runtime.hpp"
#include "gicc/platform/ofi/runtime_helpers.h"

// One MMIO store to the NIC trigger counter, which fires every descriptor the
// host staged. A kernel is the only way to reach that mapping from the device.
// The address and value are passed in rather than read from the DeviceCtx:
// Runtime::prepare() and the arm helper BOTH consume the pending-op delta, so
// re-preparing here just to obtain a ctx pointer would zero the value we are
// about to write and nothing would fire.
__global__ void gicc_omp_fire_trigger(volatile uint64_t* addr, uint64_t value) {
    *addr = value;
}

namespace {

gicc::Runtime* g_runtime = nullptr;
bool g_initialized_mpi = false;
bool g_dwq_enabled = false;
bool g_ipc_enabled = false;
int g_local_device = 0;

// ---- symmetric heap ---------------------------------------------------------
gicc::Buffer g_heap{};              // the heap as an address-book entry
char*  g_heap_base  = nullptr;
size_t g_heap_bytes = 0;

// Heap blocks in ascending offset order; adjacent free blocks are coalesced on
// free. Allocation is first-fit, which is enough for the collective
// allocate-once patterns this API targets.
struct HeapBlock { size_t offset; size_t size; bool used; };
std::vector<HeapBlock> g_blocks;

// ompx_bind associations. An OpenMP application names its data by the host
// pointer it passes to `map` clauses, so put/get/peer_ptr accept that pointer
// and translate it here; a heap device address works too.
struct HeapBind { const char* host; const char* dev; size_t bytes; };
std::vector<HeapBind> g_binds;

constexpr size_t kHeapAlign = 256;

size_t align_up(size_t n, size_t a) { return (n + a - 1) / a * a; }

void require_gpu(GpuError error, const char* operation) {
    if (error == GPU_SUCCESS) return;
    std::fprintf(stderr, "[giomp] %s failed: %s\n",
                 operation, gpuGetErrorString(error));
    std::abort();
}

void die(const char* what) {
    std::fprintf(stderr, "[giomp] %s\n", what);
    std::abort();
}

void initialize_mpi_if_needed() {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        int argc = 0;
        char** argv = nullptr;
        MPI_Init(&argc, &argv);
        g_initialized_mpi = true;
    }
}

void select_local_device() {
    const char* ipc_env = std::getenv("GICC_HALO_IPC");
    g_ipc_enabled = ipc_env != nullptr && std::atoi(ipc_env) != 0;
    if (!g_ipc_enabled) return;

    int device_count = 0;
    require_gpu(gpuGetDeviceCount(&device_count), "get device count");

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &node_comm);
    int local_rank = 0;
    int local_size = 0;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_size(node_comm, &local_size);
    MPI_Comm_free(&node_comm);

    if (device_count < local_size) {
        if (local_rank == 0) {
            std::fprintf(stderr,
                "[giomp] IPC disabled: %d GPU(s) visible < %d ranks/node "
                "(launch with all GPUs visible and mpibind=off)\n",
                device_count, local_size);
        }
        g_ipc_enabled = false;
        return;
    }

    g_local_device = local_rank % device_count;
    require_gpu(gpuSetDevice(g_local_device), "select rank-local device");
    for (int device = 0; device < device_count; ++device) {
        if (device == g_local_device) continue;
        GpuError err = gpuDeviceEnablePeerAccess(device, 0);
        if (err != GPU_SUCCESS && err != gpuErrorPeerAccessAlreadyEnabled) {
            (void)gpuGetLastError();
        }
    }
}

// OMPX_HEAP_SIZE accepts plain bytes or a K/M/G suffix. Without it the heap is
// 16 GB, clamped to 70% of free device memory so a default run still leaves
// room for the OpenMP mappings the application makes outside the heap.
size_t configured_heap_bytes() {
    const char* env = std::getenv("OMPX_HEAP_SIZE");
    if (env != nullptr && *env != '\0') {
        char* end = nullptr;
        double value = std::strtod(env, &end);
        size_t scale = 1;
        if (end != nullptr) {
            if (*end == 'k' || *end == 'K') scale = 1ull << 10;
            else if (*end == 'm' || *end == 'M') scale = 1ull << 20;
            else if (*end == 'g' || *end == 'G') scale = 1ull << 30;
        }
        if (value > 0) return align_up((size_t)(value * (double)scale), kHeapAlign);
    }

    size_t free_bytes = 0, total_bytes = 0;
    size_t want = (size_t)16 << 30;
    if (gpuMemGetInfo(&free_bytes, &total_bytes) == GPU_SUCCESS && free_bytes > 0) {
        const size_t cap = (size_t)((double)free_bytes * 0.7);
        if (want > cap) want = cap;
    }
    return align_up(want, kHeapAlign);
}

void create_heap() {
    g_heap_bytes = configured_heap_bytes();
    void* ptr = nullptr;
    require_gpu(gpuMalloc(&ptr, g_heap_bytes), "allocate symmetric heap");
    require_gpu(gpuMemset(ptr, 0, g_heap_bytes), "zero symmetric heap");
    g_heap_base = static_cast<char*>(ptr);

    // One registration, one address-book entry; ompx_alloc never registers.
    g_heap = g_runtime->register_buffer(ptr, g_heap_bytes, /*is_device=*/true);
    g_blocks.clear();
    g_blocks.push_back(HeapBlock{0, g_heap_bytes, false});

    g_runtime->exchange();
    g_runtime->set_symmetric_heap(ptr, g_heap.index);
}

// Heap offset of `addr`, which may be a heap device address or a host address
// bound to the heap by ompx_bind.
size_t offset_of(const void* addr, const char* what) {
    const char* p = static_cast<const char*>(addr);
    if (g_heap_base != nullptr && p >= g_heap_base &&
        p < g_heap_base + g_heap_bytes) {
        return (size_t)(p - g_heap_base);
    }
    for (const HeapBind& b : g_binds) {
        if (p >= b.host && p < b.host + b.bytes) {
            return (size_t)(b.dev - g_heap_base) + (size_t)(p - b.host);
        }
    }
    std::fprintf(stderr,
        "[giomp] %s: %p is not an ompx_alloc/ompx_bind address\n", what, addr);
    std::abort();
}

}  // namespace

extern "C" {

// ---- control ----------------------------------------------------------------

void ompx_init() {
    if (g_runtime != nullptr) return;

    initialize_mpi_if_needed();
    select_local_device();
    g_runtime = new gicc::Runtime();

    const char* dwq_env = std::getenv("GICC_HALO_DWQ");
    g_dwq_enabled = dwq_env != nullptr && std::atoi(dwq_env) != 0;
    if (g_dwq_enabled) g_runtime->enable_host_wait_mode();

    // Keep libomptarget and the GPU runtime on the same rank-local device.
    omp_set_default_device(g_local_device);

    create_heap();
}

void ompx_finalize() {
    if (g_runtime != nullptr) g_runtime->reset();
    if (g_heap_base != nullptr) {
        (void)gpuFree(g_heap_base);
        g_heap_base = nullptr;
        g_heap_bytes = 0;
        g_blocks.clear();
        g_binds.clear();
    }
    delete g_runtime;
    g_runtime = nullptr;
    g_dwq_enabled = false;
    g_ipc_enabled = false;
    g_local_device = 0;

    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized && g_initialized_mpi) MPI_Finalize();
    g_initialized_mpi = false;
}

int ompx_get_rank_num() {
    return g_runtime ? g_runtime->rank() : -1;
}

int ompx_get_num_ranks() {
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

// ---- memory -----------------------------------------------------------------

void* ompx_alloc(size_t bytes) {
    if (g_heap_base == nullptr) die("ompx_alloc before ompx_init");
    const size_t want = align_up(bytes, kHeapAlign);

    for (size_t i = 0; i < g_blocks.size(); ++i) {
        if (g_blocks[i].used || g_blocks[i].size < want) continue;
        // Split the block, keeping no reference across insert(): it reallocates.
        const size_t offset = g_blocks[i].offset;
        if (g_blocks[i].size > want) {
            HeapBlock rest{offset + want, g_blocks[i].size - want, false};
            g_blocks[i].size = want;
            g_blocks.insert(g_blocks.begin() + (long)i + 1, rest);
        }
        g_blocks[i].used = true;
        return g_heap_base + offset;
    }

    std::fprintf(stderr,
        "[giomp] ompx_alloc(%zu) does not fit in the %zu-byte heap; "
        "raise OMPX_HEAP_SIZE\n", bytes, g_heap_bytes);
    std::abort();
}

void* ompx_bind(void* host_ptr, size_t bytes) {
    void* dev = ompx_alloc(bytes);
    // Bind the heap allocation to the application's host pointer so its
    // existing `map` clauses and compute kernels transparently use it.
    if (omp_target_associate_ptr(host_ptr, dev, bytes, 0, g_local_device) != 0) {
        die("ompx_bind: omp_target_associate_ptr failed");
    }
    require_gpu(gpuMemcpy(dev, host_ptr, bytes, gpuMemcpyHostToDevice),
                "copy host buffer into the heap");
    g_binds.push_back(HeapBind{static_cast<const char*>(host_ptr),
                               static_cast<const char*>(dev), bytes});
    return dev;
}

void ompx_free(void* ptr) {
    if (ptr == nullptr || g_heap_base == nullptr) return;
    const size_t off = offset_of(ptr, "ompx_free");
    for (size_t i = 0; i < g_blocks.size(); ++i) {
        if (g_blocks[i].offset != off) continue;
        g_blocks[i].used = false;
        // Coalesce with the neighbours so alloc/free cycles do not fragment.
        if (i + 1 < g_blocks.size() && !g_blocks[i + 1].used) {
            g_blocks[i].size += g_blocks[i + 1].size;
            g_blocks.erase(g_blocks.begin() + (long)i + 1);
        }
        if (i > 0 && !g_blocks[i - 1].used) {
            g_blocks[i - 1].size += g_blocks[i].size;
            g_blocks.erase(g_blocks.begin() + (long)i);
        }
        return;
    }
}

size_t ompx_heap_size() { return g_heap_bytes; }

int ompx_heap_index() { return g_heap.index; }

size_t ompx_heap_offset_of(const void* addr) {
    return offset_of(addr, "ompx_heap_offset_of");
}

// ---- data movement ----------------------------------------------------------

void* ompx_peer_ptr(int peer, const void* addr) {
    if (g_runtime == nullptr || !g_ipc_enabled) return nullptr;
    void* base = g_runtime->peer_mapped(peer, g_heap.index);
    if (base == nullptr) return nullptr;
    return static_cast<char*>(base) + offset_of(addr, "ompx_peer_ptr");
}

void ompx_put(int peer, void* dst, const void* src, size_t bytes) {
    if (g_runtime == nullptr) die("ompx_put before ompx_init");
    const size_t src_off = offset_of(src, "ompx_put src");
    const size_t dst_off = offset_of(dst, "ompx_put dst");

    // Same node: copy straight into the peer's heap over NVLink/xGMI. The copy
    // rides the runtime's IPC stream, which ompx_quiet synchronizes.
    void* peer_base = g_ipc_enabled ? g_runtime->peer_mapped(peer, g_heap.index)
                                    : nullptr;
    if (peer_base != nullptr) {
        require_gpu(gpuMemcpyAsync(static_cast<char*>(peer_base) + dst_off,
                                   g_heap_base + src_off, bytes,
                                   gpuMemcpyDeviceToDevice,
                                   gicc_runtime_ipc_stream(g_runtime)),
                    "same-node put");
        return;
    }

    // Cross node. The DWQ path stages a triggered descriptor (fired by a kernel
    // MMIO write); otherwise hand the descriptor to the CPU proxy fleet.
    if (g_dwq_enabled) {
        (void)g_runtime->put(g_heap, peer, g_heap.index, bytes, src_off, dst_off);
        ++g_dwq_pending;
    } else {
        g_runtime->proxy_push(gicc::proxy::CmdType::WRITE, peer,
                              g_heap.index, dst_off, g_heap.index, src_off, bytes);
    }
}

void ompx_get(int peer, void* dst, const void* src, size_t bytes) {
    if (g_runtime == nullptr) die("ompx_get before ompx_init");
    const size_t dst_off = offset_of(dst, "ompx_get dst");
    const size_t src_off = offset_of(src, "ompx_get src");

    void* peer_base = g_ipc_enabled ? g_runtime->peer_mapped(peer, g_heap.index)
                                    : nullptr;
    if (peer_base != nullptr) {
        require_gpu(gpuMemcpyAsync(g_heap_base + dst_off,
                                   static_cast<char*>(peer_base) + src_off, bytes,
                                   gpuMemcpyDeviceToDevice,
                                   gicc_runtime_ipc_stream(g_runtime)),
                    "same-node get");
        return;
    }

    if (g_dwq_enabled) {
        (void)g_runtime->get(g_heap, peer, g_heap.index, bytes, dst_off, src_off);
        ++g_dwq_pending;
    } else {
        // src_* names the local slice and dst_* the remote one for every cmd type.
        g_runtime->proxy_push(gicc::proxy::CmdType::READ, peer,
                              g_heap.index, src_off, g_heap.index, dst_off, bytes);
    }
}

void ompx_set_transport(ompx_xport xport) {
    if (xport == OMPX_DWQ && !g_dwq_enabled) {
        if (g_runtime != nullptr) g_runtime->enable_host_wait_mode();
        g_dwq_enabled = true;
    }
}

// ---- completion -------------------------------------------------------------

void ompx_quiet() {
    if (g_runtime != nullptr) g_runtime->reset();
}

void ompx_barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
}

void ompx_fence() {
    ompx_quiet();
    ompx_barrier();
}

// ---- device-side ------------------------------------------------------------

ompx_ctx* ompx_prepare() {
    return g_runtime->prepare();
}

// ---- explicit batched DWQ ---------------------------------------------------
// Defined below, next to the other compiler-facing helpers.
void ompx_dwq_arm(void);

int ompx_dwq_enabled() { return g_dwq_enabled ? 1 : 0; }

void ompx_dwq_stage_put(int peer, void* dst, const void* src, size_t bytes) {
    gicc_runtime_dwq_enqueue(g_runtime, peer,
                             g_heap.index, offset_of(dst, "ompx_dwq_stage_put dst"),
                             g_heap.index, offset_of(src, "ompx_dwq_stage_put src"),
                             bytes);
}

void ompx_dwq_stage_get(int peer, void* dst, const void* src, size_t bytes) {
    // In host-wait mode Runtime::get queues a triggered descriptor rather than
    // issuing it, which is exactly the staging this call promises.
    (void)g_runtime->get(g_heap, peer, g_heap.index, bytes,
                         offset_of(dst, "ompx_dwq_stage_get dst"),
                         offset_of(src, "ompx_dwq_stage_get src"));
}

void ompx_dwq_trigger() {
    g_dwq_pending = 0;
    ompx_dwq_arm();
    volatile uint64_t* addr = gicc_runtime_trigger_addr(g_runtime);
    const uint64_t value = gicc_runtime_trigger_val(g_runtime);
    if (addr == nullptr || value == 0) return;
    gicc_omp_fire_trigger<<<1, 1>>>(addr, value);
    require_gpu(gpuDeviceSynchronize(), "fire DWQ trigger");
}

// ---- compiler-facing helpers ------------------------------------------------
// Used by the OMP-DWQ LTO pass and by the synthesized host traces.

void ompx_dwq_stage(int peer, int dst_buffer, size_t dst_offset,
                    int src_buffer, size_t src_offset, size_t bytes) {
    gicc_runtime_dwq_enqueue(g_runtime, peer, dst_buffer, dst_offset,
                             src_buffer, src_offset, bytes);
}

void ompx_dwq_arm() {
    gicc_runtime_arm_dwq_trigger(g_runtime);
}

void* gicc_runtime_current() {
    return g_runtime;
}

}  // extern "C"
