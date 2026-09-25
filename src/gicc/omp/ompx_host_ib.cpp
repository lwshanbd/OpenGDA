// GiOMP host runtime on InfiniBand. Compiled as CUDA source (clang -x cuda)
// and linked into libgicc_omp, so OpenMP application TUs only need the public
// gicc/omp.h header and the library -- the same contract as ompx_host.cpp,
// which implements the API over libfabric.
//
// The difference is who moves the data. Over libfabric a target region hands
// its transfers to a CPU proxy or fires descriptors the host pre-staged. Here
// the GPU thread in the target region writes the work request and rings the
// NIC doorbell itself (gicc/platform/mlx5/mlx5_device.hpp), so there is nothing
// to stage and nothing to trigger: ompx_stage_put_signal and ompx_trigger are
// no-ops. Transfers issued from host code go through ordinary verbs QPs.
//
// Memory is a symmetric heap exactly as on libfabric: one device allocation
// registered once, carved by ompx_alloc, the same offset on every rank. The
// signal inbox is device memory too, so a signal and the payload it announces
// travel the same path into the GPU.
//
// Same-node peers: with GICC_HALO_IPC=1, as on libfabric, every same-node
// peer's heap is mapped through CUDA IPC. ompx_peer_ptr then hands out the
// peer's address of an object, and host-issued ompx_put / ompx_get to such a
// peer copy over NVLink / PCIe instead of the NIC. Device-side puts and every
// put_signal keep using the NIC, whose QP orders a signal behind its payload.
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gicc/omp.h"
#include "gicc/platform/mlx5/mlx5_runtime.hpp"

namespace {

// gicc/omp.h mirrors these two leading fields for C target regions.
static_assert(offsetof(gicc::DeviceCtx, trigger_addr_) == 0, "DeviceCtx layout");
static_assert(offsetof(gicc::DeviceCtx, trigger_val_) == sizeof(void*),
              "DeviceCtx layout");

gicc::Runtime* g_runtime = nullptr;
bool g_initialized_mpi = false;
cudaStream_t g_stream = nullptr;   // small copies of the signal inbox

// Same-node copies through CUDA IPC (GICC_HALO_IPC=1). They ride their own
// stream, which ompx_quiet synchronizes when a copy is outstanding.
bool         g_ipc_enabled = false;
bool         g_ipc_pending = false;
cudaStream_t g_ipc_stream  = nullptr;

// ---- signal inbox -----------------------------------------------------------
// kSigSlots 8-byte slots in device memory, registered, symmetric: slot i means
// the same thing on every rank. A put_signal writes its value with an inline
// RDMA WRITE, so unlike DWQ no source cell is needed.
constexpr int kSigSlots = 64;
gicc::Buffer g_sig{};
uint64_t*    g_sig_dev = nullptr;
uint64_t*    g_sig_host = nullptr;   // pinned bounce for host reads

// ---- symmetric heap ---------------------------------------------------------
gicc::Buffer g_heap{};
char*  g_heap_base  = nullptr;
size_t g_heap_bytes = 0;

// Heap blocks in ascending offset order; adjacent free blocks are coalesced on
// free. First fit, enough for the collective allocate-once patterns this API
// targets.
struct HeapBlock { size_t offset; size_t size; bool used; };
std::vector<HeapBlock> g_blocks;

// ompx_bind associations: an OpenMP application names its data by the host
// pointer it maps, so put/get accept that pointer and translate it here.
struct HeapBind { const char* host; const char* dev; size_t bytes; };
std::vector<HeapBind> g_binds;

constexpr size_t kHeapAlign = 256;

size_t align_up(size_t n, size_t a) { return (n + a - 1) / a * a; }

void require_cuda(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return;
    std::fprintf(stderr, "[giomp] %s failed: %s\n", operation, cudaGetErrorString(error));
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
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess && free_bytes > 0) {
        const size_t cap = (size_t)((double)free_bytes * 0.7);
        if (want > cap) want = cap;
    }
    return align_up(want, kHeapAlign);
}

void create_heap() {
    g_heap_bytes = configured_heap_bytes();
    void* ptr = nullptr;
    require_cuda(cudaMalloc(&ptr, g_heap_bytes), "allocate symmetric heap");
    require_cuda(cudaMemset(ptr, 0, g_heap_bytes), "zero symmetric heap");
    g_heap_base = static_cast<char*>(ptr);
    g_heap = g_runtime->register_buffer(ptr, g_heap_bytes, /*is_device=*/true);
    g_blocks.clear();
    g_blocks.push_back(HeapBlock{0, g_heap_bytes, false});

    const size_t sig_bytes = kSigSlots * sizeof(uint64_t);
    require_cuda(cudaMalloc(&g_sig_dev, sig_bytes), "allocate signal inbox");
    require_cuda(cudaMemset(g_sig_dev, 0, sig_bytes), "zero signal inbox");
    require_cuda(cudaMallocHost(&g_sig_host, sizeof(uint64_t)), "allocate signal bounce");
    g_sig = g_runtime->register_buffer(g_sig_dev, sig_bytes, /*is_device=*/true);

    g_runtime->exchange();
    g_runtime->set_symmetric_heap(ptr, g_heap.index);
    g_runtime->set_signal_table(g_sig_dev, g_sig.index);
}

void check_slot(int sig, const char* what) {
    if (sig >= 0 && sig < kSigSlots) return;
    std::fprintf(stderr, "[giomp] %s: signal slot %d is outside [0, %d)\n",
                 what, sig, kSigSlots);
    std::abort();
}

// The local address of `peer`'s heap, or nullptr when it is not mapped.
char* peer_heap(int peer) {
    if (!g_ipc_enabled) return nullptr;
    return static_cast<char*>(g_runtime->peer_mapped(peer, g_heap.index));
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

uint64_t read_slot(int sig) {
    require_cuda(cudaMemcpyAsync(g_sig_host, g_sig_dev + sig, sizeof(uint64_t),
                                 cudaMemcpyDeviceToHost, g_stream),
                 "read signal slot");
    require_cuda(cudaStreamSynchronize(g_stream), "read signal slot");
    return *g_sig_host;
}

}  // namespace

extern "C" {

// ---- control ----------------------------------------------------------------

void ompx_init() {
    if (g_runtime != nullptr) return;
    initialize_mpi_if_needed();
    g_runtime = new gicc::Runtime();
    require_cuda(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking),
                 "create signal stream");
    const char* ipc_env = std::getenv("GICC_HALO_IPC");
    g_ipc_enabled = ipc_env != nullptr && std::atoi(ipc_env) != 0;
    if (g_ipc_enabled) {
        require_cuda(cudaStreamCreateWithFlags(&g_ipc_stream, cudaStreamNonBlocking),
                     "create IPC stream");
    }
    // Keep libomptarget on the device the runtime selected.
    omp_set_default_device(g_runtime->gpu_id());
    create_heap();
}

void ompx_finalize() {
    if (g_ipc_pending) {
        (void)cudaStreamSynchronize(g_ipc_stream);
        g_ipc_pending = false;
    }
    // The runtime drains, unmaps the peers' heaps and deregisters the heap
    // and the inbox, so it goes before the memory it registered -- and every
    // rank must have unmapped this rank's heap before it is freed.
    delete g_runtime;
    g_runtime = nullptr;
    MPI_Barrier(MPI_COMM_WORLD);
    if (g_heap_base != nullptr) {
        (void)cudaFree(g_heap_base);
        g_heap_base = nullptr;
        g_heap_bytes = 0;
        g_blocks.clear();
        g_binds.clear();
    }
    if (g_sig_dev != nullptr) {
        (void)cudaFree(g_sig_dev);
        (void)cudaFreeHost(g_sig_host);
        g_sig_dev = nullptr;
        g_sig_host = nullptr;
    }
    if (g_stream != nullptr) {
        (void)cudaStreamDestroy(g_stream);
        g_stream = nullptr;
    }
    if (g_ipc_stream != nullptr) {
        (void)cudaStreamDestroy(g_ipc_stream);
        g_ipc_stream = nullptr;
    }
    g_ipc_enabled = false;

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
    if (omp_target_associate_ptr(host_ptr, dev, bytes, 0, g_runtime->gpu_id()) != 0) {
        die("ompx_bind: omp_target_associate_ptr failed");
    }
    require_cuda(cudaMemcpy(dev, host_ptr, bytes, cudaMemcpyHostToDevice),
                 "copy host buffer into the heap");
    g_binds.push_back(HeapBind{static_cast<const char*>(host_ptr),
                               static_cast<const char*>(dev), bytes});
    return dev;
}

void ompx_free(void* ptr) {
    if (ptr == nullptr || g_heap_base == nullptr) return;
    const size_t off = offset_of(ptr, "ompx_free");
    // Drop any ompx_bind record first: a stale one keeps resolving a host
    // pointer whose heap block is about to be handed to the next allocation.
    for (size_t i = 0; i < g_binds.size(); ++i) {
        const char* p = static_cast<const char*>(ptr);
        if ((p >= g_binds[i].host && p < g_binds[i].host + g_binds[i].bytes) ||
            (p >= g_binds[i].dev  && p < g_binds[i].dev  + g_binds[i].bytes)) {
            g_binds.erase(g_binds.begin() + (long)i);
            break;
        }
    }
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

int ompx_heap_index() { return g_heap.index; }

size_t ompx_heap_offset_of(const void* addr) {
    return offset_of(addr, "ompx_heap_offset_of");
}

// ---- data movement ----------------------------------------------------------

// A same-node peer's address for the object at `addr`: its heap is mapped
// through CUDA IPC, and the object sits at the same heap offset there. NULL
// for this rank, for a peer on another node, and without GICC_HALO_IPC.
void* ompx_peer_ptr(int peer, const void* addr) {
    if (g_runtime == nullptr) return nullptr;
    char* base = peer_heap(peer);
    if (base == nullptr) return nullptr;
    return base + offset_of(addr, "ompx_peer_ptr");
}

void ompx_put_host(int peer, void* dst, const void* src, size_t bytes) {
    if (g_runtime == nullptr) die("ompx_put before ompx_init");
    const size_t src_off = offset_of(src, "ompx_put src");
    const size_t dst_off = offset_of(dst, "ompx_put dst");
    if (char* base = peer_heap(peer)) {
        require_cuda(cudaMemcpyAsync(base + dst_off, g_heap_base + src_off, bytes,
                                     cudaMemcpyDeviceToDevice, g_ipc_stream),
                     "same-node put");
        g_ipc_pending = true;
        return;
    }
    g_runtime->put(g_heap, peer, g_heap.index, bytes, src_off, dst_off);
}

void ompx_get_host(int peer, void* dst, const void* src, size_t bytes) {
    if (g_runtime == nullptr) die("ompx_get before ompx_init");
    const size_t dst_off = offset_of(dst, "ompx_get dst");
    const size_t src_off = offset_of(src, "ompx_get src");
    if (char* base = peer_heap(peer)) {
        require_cuda(cudaMemcpyAsync(g_heap_base + dst_off, base + src_off, bytes,
                                     cudaMemcpyDeviceToDevice, g_ipc_stream),
                     "same-node get");
        g_ipc_pending = true;
        return;
    }
    g_runtime->get(g_heap, peer, g_heap.index, bytes, dst_off, src_off);
}

// ---- signals ----------------------------------------------------------------

unsigned long long ompx_signal_read_host(int sig) {
    check_slot(sig, "ompx_signal_read");
    return read_slot(sig);
}

void ompx_signal_reset(int sig) {
    check_slot(sig, "ompx_signal_reset");
    require_cuda(cudaMemsetAsync(g_sig_dev + sig, 0, sizeof(uint64_t), g_stream),
                 "reset signal slot");
    require_cuda(cudaStreamSynchronize(g_stream), "reset signal slot");
}

void ompx_signal_wait_host(int sig, unsigned long long ge) {
    check_slot(sig, "ompx_signal_wait");
    while (read_slot(sig) < ge) g_runtime->progress();
}

// The payload and the signal go out on the same host QP, in that order.
void ompx_put_signal_host(int peer, void* dst, const void* src, size_t bytes,
                          int sig, unsigned long long value) {
    if (g_runtime == nullptr) die("ompx_put_signal before ompx_init");
    check_slot(sig, "ompx_put_signal");
    g_runtime->put(g_heap, peer, g_heap.index, bytes,
                   offset_of(src, "ompx_put_signal src"),
                   offset_of(dst, "ompx_put_signal dst"));
    g_runtime->put_u64(peer, g_sig.index, (size_t)sig * sizeof(uint64_t), value);
}

// The device-side ompx_put_signal carries the whole transfer, so there is
// nothing to stage; the arguments are still checked so a bad call fails here
// the same way it would over libfabric.
void ompx_stage_put_signal(int peer, void* dst, const void* src, size_t bytes,
                           int sig, unsigned long long value) {
    if (g_runtime == nullptr) die("ompx_stage_put_signal before ompx_init");
    check_slot(sig, "ompx_stage_put_signal");
    (void)peer; (void)bytes; (void)value;
    (void)offset_of(dst, "ompx_stage_put_signal dst");
    (void)offset_of(src, "ompx_stage_put_signal src");
}

// ---- completion -------------------------------------------------------------

// Completes host-issued transfers and everything GPU threads have rung, which
// is every device-side put of a target region that has returned.
void ompx_quiet_host() {
    if (g_runtime == nullptr) return;
    if (g_ipc_pending) {
        g_ipc_pending = false;
        require_cuda(cudaStreamSynchronize(g_ipc_stream), "drain the IPC stream");
    }
    g_runtime->drain();
}

void ompx_barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
}

void ompx_fence() {
    ompx_quiet_host();
    ompx_barrier();
}

// ---- device-side ------------------------------------------------------------

ompx_ctx* ompx_prepare_ctx() {
    return g_runtime->prepare();
}

// A GPU-posted transfer starts when its thread rings the doorbell; there is
// no trigger to pull.
void ompx_trigger_host() {}

}  // extern "C"
