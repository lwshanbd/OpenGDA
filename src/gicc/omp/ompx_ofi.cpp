// GiOMP runtime over libfabric (OFI). This translation unit is compiled as GPU
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

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#include "gicc/omp.h"
#include "gicc/platform/ofi/internal/gpu_device_context.hpp"
#include "gicc/platform/ofi/ofi_runtime.hpp"
#include "gicc/platform/ofi/runtime_helpers.h"

namespace {

// gicc/omp.h mirrors these two leading fields for C target regions.
static_assert(offsetof(gicc::DeviceCtx, trigger_addr_) == 0, "DeviceCtx layout");
static_assert(offsetof(gicc::DeviceCtx, trigger_val_) == sizeof(void*),
              "DeviceCtx layout");

gicc::Runtime* g_runtime = nullptr;
bool g_initialized_mpi = false;
bool g_dwq_enabled = false;
// DWQ descriptors staged by ompx_put but not yet fired. A put is non-blocking,
// so the trigger belongs to the completion call: ompx_quiet fires whatever is
// outstanding before it drains.
int g_dwq_pending = 0;
// Set when a same-node put/get was issued on the IPC stream. Only then does
// ompx_quiet have anything to synchronize there.
bool g_ipc_pending = false;
bool g_ipc_enabled = false;
int g_local_device = 0;

// ---- signal table -----------------------------------------------------------
// Host-pinned, mapped and registered, so the NIC writes it, a host thread
// polls it with an ordinary load, and a target region polls the device alias.
// Symmetric like the heap: slot i means the same thing on every rank.
//   [0, kSigSlots)                   this rank's inbox -- peers write here
//   [kSigSlots, kSigSlots+kSigCells) source cells for staged DWQ signals
// A staged descriptor reads its source when it fires, not when it is staged,
// so every staged signal needs a cell of its own until it completes. The
// cells are handed out in order and recycled by ompx_quiet, which waits for
// all of them. The proxy carries the value in its command and needs none.
constexpr int kSigSlots = gicc::Runtime::kSignalSlots;
constexpr int kSigCells = 4096;
gicc::Buffer g_sig{};
uint64_t*    g_sig_host = nullptr;
uint64_t*    g_sig_dev  = nullptr;
int          g_sig_next_cell = 0;

// ---- pipelined puts left to the quiet (ompx_pipe_deferred in gicc/omp.h) ----
ompx_pipe_deferred* g_pipe_deferred_host = nullptr;
ompx_pipe_deferred* g_pipe_deferred_dev  = nullptr;
char**              g_pipe_peer_heap_host = nullptr;   // ompx_pipe_deferred::peer_heap
char**              g_pipe_peer_heap_dev  = nullptr;
// Put i after a launch, queued under DWQ on slot kSigSlots + i when it was
// posted, in this many groups (0: not queued); set until every group has rung.
int g_after_queued[OMPX_PIPE_AFTER_MAX] = {};

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

    const size_t sig_bytes = (size_t)(kSigSlots + kSigCells) * sizeof(uint64_t);
    void* sig = nullptr;
    require_gpu(gpuHostMalloc(&sig, sig_bytes, gpuHostMallocMapped),
                "allocate signal table");
    std::memset(sig, 0, sig_bytes);
    g_sig_host = static_cast<uint64_t*>(sig);
    require_gpu(gpuHostGetDevicePointer((void**)&g_sig_dev, sig, 0),
                "map signal table");
    g_sig = g_runtime->register_buffer(sig, sig_bytes, /*is_device=*/false);

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

// Stage one put_signal on slot `sig` under DWQ: the payload, then the signal,
// both released by the slot's next doorbell increment.
void stage_signal_dwq(int peer, size_t dst_off, size_t src_off, size_t bytes,
                      int sig, unsigned long long value) {
    if (g_sig_next_cell == kSigCells) {
        std::fprintf(stderr, "[giomp] more than %d put_signals staged without an "
                             "ompx_quiet in between\n", kSigCells);
        std::abort();
    }
    const int cell = kSigSlots + g_sig_next_cell++;
    g_sig_host[cell] = value;
    const uint64_t threshold = g_runtime->signal_slot_next(sig);
    g_runtime->signal_slot_write(sig, threshold, g_heap, peer, g_heap.index,
                                 bytes, src_off, dst_off);
    g_runtime->signal_slot_write(sig, threshold, g_sig, peer, g_sig.index,
                                 sizeof(uint64_t), (size_t)cell * sizeof(uint64_t),
                                 (size_t)sig * sizeof(uint64_t));
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

// A pipelined put the device could not make (ompx_pipe_deferred::fail).
void pipe_check() {
    const ompx_pipe_deferred* q = g_pipe_deferred_host;
    if (q == nullptr || q->fail == 0) return;
    char what[256];
    const long long a = q->fail_arg;
    switch (q->fail) {
    case OMPX_PIPE_FAIL_NO_PROXY:
        std::snprintf(what, sizeof(what), "ompx_pipelined_put: peer %lld is not IPC-mapped "
                      "and the CPU proxy is off; nothing could send the data", a);
        break;
    case OMPX_PIPE_FAIL_DIMS:
        std::snprintf(what, sizeof(what), "ompx_pipelined_put: a %lld-dimensional box has "
                      "more than 8 dims", a);
        break;
    case OMPX_PIPE_FAIL_ALIGN:
        std::snprintf(what, sizeof(what), "ompx_pipelined_put: the put range and the written "
                      "box are not aligned to the %lld-byte stores", a);
        break;
    case OMPX_PIPE_FAIL_OVERLAP:
        std::snprintf(what, sizeof(what), "ompx_pipelined_put: the rows of the written box "
                      "overlap; cannot tell which bytes of the put it does not write");
        break;
    case OMPX_PIPE_FAIL_POST:
        std::snprintf(what, sizeof(what), "ompx_pipelined_put: the host posted put %lld with "
                      "other arguments than the kernel computes; were the host and device "
                      "compiled from one source, into one GICC_META_DIR?", a);
        break;
    default:
        std::snprintf(what, sizeof(what), "ompx_pipelined_put: failure %d on the device",
                      q->fail);
    }
    die(what);
}

}  // namespace

extern "C" {

// ---- control ----------------------------------------------------------------

void ompx_init() {
    if (g_runtime != nullptr) return;

    initialize_mpi_if_needed();
    select_local_device();
    g_runtime = new gicc::Runtime();

    // DWQ by default: the GPU triggers its own transfers, which is the point
    // of this library. GICC_HALO_DWQ=0 selects the CPU proxy instead, and
    // GICC_SKIP_DWQ_INIT forces it -- that switch leaves the CXI trigger BAR
    // unmapped, so the DWQ physically cannot fire.
    const char* dwq_env = std::getenv("GICC_HALO_DWQ");
    g_dwq_enabled = (dwq_env == nullptr) || (std::atoi(dwq_env) != 0);
    if (g_dwq_enabled && std::getenv("GICC_SKIP_DWQ_INIT") != nullptr) {
        if (dwq_env != nullptr && g_runtime->rank() == 0) {
            std::fprintf(stderr, "[giomp] GICC_HALO_DWQ ignored: "
                                 "GICC_SKIP_DWQ_INIT leaves the trigger unmapped\n");
        }
        g_dwq_enabled = false;
    }
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
        g_ipc_pending = false;
    }
    if (g_pipe_deferred_host != nullptr) {
        (void)gpuHostFree(g_pipe_deferred_host);
        g_pipe_deferred_host = nullptr;
        g_pipe_deferred_dev  = nullptr;
    }
    if (g_pipe_peer_heap_host != nullptr) {
        (void)gpuHostFree(g_pipe_peer_heap_host);
        g_pipe_peer_heap_host = nullptr;
        g_pipe_peer_heap_dev  = nullptr;
    }
    if (g_sig_host != nullptr) {
        (void)gpuHostFree(g_sig_host);
        g_sig_host = nullptr;
        g_sig_dev  = nullptr;
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

void* ompx_peer_ptr(int peer, const void* addr) {
    if (g_runtime == nullptr || !g_ipc_enabled) return nullptr;
    void* base = g_runtime->peer_mapped(peer, g_heap.index);
    if (base == nullptr) return nullptr;
    return static_cast<char*>(base) + offset_of(addr, "ompx_peer_ptr");
}

void ompx_put_host(int peer, void* dst, const void* src, size_t bytes) {
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
        g_ipc_pending = true;
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

void ompx_get_host(int peer, void* dst, const void* src, size_t bytes) {
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
        g_ipc_pending = true;
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

// ---- signals ----------------------------------------------------------------

unsigned long long ompx_signal_read_host(int sig) {
    check_slot(sig, "ompx_signal_read");
    return __atomic_load_n(&g_sig_host[sig], __ATOMIC_ACQUIRE);
}

void ompx_signal_reset(int sig) {
    check_slot(sig, "ompx_signal_reset");
    __atomic_store_n(&g_sig_host[sig], 0, __ATOMIC_RELEASE);
}

// Spins without waiting for this rank's own transfers: some of them may be
// staged for a kernel that has not run yet, and drain() would wait on those
// forever.
void ompx_signal_wait_host(int sig, unsigned long long ge) {
    check_slot(sig, "ompx_signal_wait");
    while (__atomic_load_n(&g_sig_host[sig], __ATOMIC_ACQUIRE) < ge) {
        ompx_trigger_host();          // release plain puts still staged
        g_runtime->progress();
    }
}

// Sent now. Under DWQ it is staged on the slot and the host rings the slot's
// doorbell at once, so the pair still rides one endpoint in order.
void ompx_put_signal_host(int peer, void* dst, const void* src, size_t bytes,
                          int sig, unsigned long long value) {
    if (g_runtime == nullptr) die("ompx_put_signal before ompx_init");
    check_slot(sig, "ompx_put_signal");
    const size_t src_off = offset_of(src, "ompx_put_signal src");
    const size_t dst_off = offset_of(dst, "ompx_put_signal dst");
    if (g_dwq_enabled) {
        stage_signal_dwq(peer, dst_off, src_off, bytes, sig, value);
        g_runtime->signal_slot_fire(sig);
    } else {
        g_runtime->proxy_push(gicc::proxy::CmdType::WRITE, peer,
                              g_heap.index, dst_off, g_heap.index, src_off, bytes);
        g_runtime->proxy_push(gicc::proxy::CmdType::SIGNAL, peer,
                              g_sig.index, (size_t)sig * sizeof(uint64_t),
                              0, (size_t)value, sizeof(uint64_t));
    }
}

// Staged for a kernel to release: the device-side ompx_put_signal with the
// same slot rings the doorbell. The proxy needs no staging -- its device
// put_signal carries the whole transfer -- so this is a no-op there.
void ompx_stage_put_signal(int peer, void* dst, const void* src, size_t bytes,
                           int sig, unsigned long long value) {
    if (g_runtime == nullptr) die("ompx_stage_put_signal before ompx_init");
    check_slot(sig, "ompx_stage_put_signal");
    if (!g_dwq_enabled) return;
    stage_signal_dwq(peer, offset_of(dst, "ompx_stage_put_signal dst"),
                     offset_of(src, "ompx_stage_put_signal src"), bytes, sig, value);
}

// ---- completion -------------------------------------------------------------

void ompx_trigger_host(void);   // defined below, with the rest of the DWQ calls

static void release_after(int i);   // below, with the rest of ompx_pipe_after

// Put what kernels left in the deferred list. Their kernels have finished
// (a quiet follows them), so the sources hold their final values.
static void send_deferred_pipelined() {
    ompx_pipe_deferred* q = g_pipe_deferred_host;
    if (q == nullptr) return;
    // A put after a launch is released by ompx__after_done; one still queued
    // here never reached it, and would keep the drain below waiting.
    for (int i = 0; i < OMPX_PIPE_AFTER_MAX; ++i) release_after(i);
    const unsigned n = __atomic_load_n(&q->n, __ATOMIC_ACQUIRE);
    if (n == 0) return;
    if (n > OMPX_PIPE_DEFERRED_MAX) die("more pipelined puts deferred than the list holds");
    for (unsigned i = 0; i < n; ++i) {
        const ompx_pipe_deferred_put& e = q->e[i];
        ompx_put_host(e.peer, g_heap_base + e.dst_off, g_heap_base + e.src_off,
                      static_cast<size_t>(e.bytes));
    }
    __atomic_store_n(&q->n, 0u, __ATOMIC_RELEASE);
}

void ompx_quiet_host() {
    if (g_runtime == nullptr) return;
    pipe_check();
    send_deferred_pipelined();
    // Descriptors staged by ompx_put sit in the queue until the NIC is told to
    // go; without this the completion counter below never reaches its threshold.
    ompx_trigger_host();
    // Per-step completion only. The batch teardown in Runtime::reset() -- freeing
    // descriptors, zeroing the NIC counters -- runs once at ompx_finalize.
    if (g_ipc_pending) {
        g_ipc_pending = false;
        require_gpu(gpuStreamSynchronize(gicc_runtime_ipc_stream(g_runtime)),
                    "drain the IPC stream");
    }
    g_runtime->drain(/*sync_ipc=*/false);
    g_sig_next_cell = 0;          // every staged signal has completed
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

// Non-coherent, so the GPU caches it: every wave of a kernel with pipelined
// puts reads it. That is safe because the two sides take turns: the host
// writes it only between launches, and reads what a kernel wrote only after
// the kernel (or at the quiet); a launch's acquire and a kernel's release
// make each side's writes visible to the other.
ompx_pipe_deferred* ompx__pipe_deferred_list() {
    if (g_runtime == nullptr) die("ompx_prepare before ompx_init");
    const int n = ompx_get_num_ranks();
    if (g_pipe_deferred_dev == nullptr) {
        void* h = nullptr;
        require_gpu(gpuHostMalloc(&h, sizeof(ompx_pipe_deferred),
                                  gpuHostMallocMapped | gpuHostMallocNonCoherent),
                    "allocate the deferred pipelined puts");
        std::memset(h, 0, sizeof(ompx_pipe_deferred));
        g_pipe_deferred_host = static_cast<ompx_pipe_deferred*>(h);
        require_gpu(gpuHostGetDevicePointer((void**)&g_pipe_deferred_dev, h, 0),
                    "map the deferred pipelined puts");
        // Each rank's heap as this GPU reaches it, as the device context's
        // IPC table has it.
        require_gpu(gpuHostMalloc(&h, n * sizeof(char*),
                                  gpuHostMallocMapped | gpuHostMallocNonCoherent),
                    "allocate the pipelined puts' peer table");
        g_pipe_peer_heap_host = static_cast<char**>(h);
        for (int r = 0; r < n; ++r)
            g_pipe_peer_heap_host[r] =
                g_runtime->is_local_peer(r)
                    ? static_cast<char*>(g_runtime->peer_mapped(r, g_heap.index))
                    : nullptr;
        require_gpu(gpuHostGetDevicePointer((void**)&g_pipe_peer_heap_dev, h, 0),
                    "map the pipelined puts' peer table");
    }
    ompx_pipe_deferred* q = g_pipe_deferred_host;
    q->heap_base = g_heap_base;
    q->peer_heap = g_pipe_peer_heap_dev;
    q->n_ranks   = n;
#ifdef GICC_CPU_PROXY
    q->proxy_on  = !g_dwq_enabled;   // DWQ mode turns the proxy's device side off
#else
    q->proxy_on  = 0;
#endif
    return g_pipe_deferred_dev;
}

// ---- puts after a kernel launch (ompx_pipe_after in gicc/omp.h) --------------
// The plugin's second compile brackets a launch followed by ompx_put calls:
// each put is posted before the launch, and replaced by ompx__after_done
// after it. The kernel reads the post and sends what it can; the put is
// made here only if it did not -- no list, no lowering for this kernel, a
// launch that fell back to the host, or a post for another put.
//
// Under DWQ the post also queues a put the kernel would otherwise leave to
// the quiet -- its peer is not IPC-mapped -- on a trigger of its own, which
// the kernel rings once the source is written. The kernel's table of mapped
// peers (ompx__peer_base) decides which those are, so the test here is the
// same one.
static bool peer_ipc_mapped(int peer) {
    return g_runtime->is_local_peer(peer) &&
           g_runtime->peer_mapped(peer, g_heap.index) != nullptr;
}

// Rings the groups of put i the kernel did not. Every group queued on a
// slot must ring exactly once: the n-th ring releases the n-th group, and
// the quiet waits for all of them.
static void release_after(int i) {
    const int groups = g_after_queued[i];
    if (groups == 0) return;
    g_after_queued[i] = 0;
    const ompx_pipe_after_put& e = g_pipe_deferred_host->after.e[i];
    const int fired = __atomic_load_n(&e.fired, __ATOMIC_ACQUIRE);
    if (fired < 0 || fired > groups) die("a kernel rang more groups than were queued");
    for (int g = fired; g < groups; ++g) g_runtime->signal_slot_fire(kSigSlots + i);
}

void ompx__after_post(unsigned long long kernel, int i, int src_arg, int armed, int peer,
                      void* dst, const void* src, long long src_rel, size_t bytes) {
    ompx_pipe_deferred* q = g_pipe_deferred_host;
    if (q == nullptr || i < 0 || i >= OMPX_PIPE_AFTER_MAX) return;
    release_after(i);
    ompx_pipe_after_put& e = q->after.e[i];
    e.armed = 0;
    e.handled = 0;
    e.bell = nullptr;
    e.fired = 0;
    e.pieces = 0;
    e.piece = 0;
    if (armed) {
        e.peer    = peer;
        e.src_arg = src_arg;
        e.dst_off = offset_of(dst, "ompx_put dst");
        e.src_rel = src_rel;
        e.bytes   = bytes;
        if (g_dwq_enabled && peer >= 0 && bytes > 0 && !peer_ipc_mapped(peer)) {
            // In pieces, one group each, so the kernel can release each
            // piece as it is written.
            const int slot = kSigSlots + i;
            const size_t src_off = offset_of(src, "ompx_put src");
            const size_t piece = ompx__pipe_piece(bytes, OMPX_PIPE_PIECES_MAX);
            int groups = 0;
            for (size_t off = 0; off < bytes; off += piece, ++groups) {
                const size_t len = bytes - off < piece ? bytes - off : piece;
                g_runtime->signal_slot_write(slot, g_runtime->signal_slot_next(slot), g_heap,
                                             peer, g_heap.index, len, src_off + off,
                                             e.dst_off + off);
            }
            e.bell = reinterpret_cast<volatile unsigned long long*>(
                g_runtime->signal_slot_doorbell(slot));
            e.pieces = groups;
            e.piece  = piece;
            g_after_queued[i] = groups;
        }
        e.armed   = 1;
    }
    q->after.kernel = kernel;
}

void ompx__after_done(unsigned long long kernel, int i, int peer, void* dst,
                      const void* src, size_t bytes) {
    pipe_check();
    ompx_pipe_deferred* q = g_pipe_deferred_host;
    bool sent = false;
    if (q != nullptr && i >= 0 && i < OMPX_PIPE_AFTER_MAX && q->after.kernel == kernel) {
        ompx_pipe_after_put& e = q->after.e[i];
        sent = e.armed && (g_after_queued[i] || __atomic_load_n(&e.handled, __ATOMIC_ACQUIRE));
        release_after(i);
        e.armed = 0;
        e.handled = 0;
    }
    if (!sent) ompx_put_host(peer, dst, src, bytes);
}

// After the launch of a kernel with an in-loop put the host posted as put
// i: the kernel has sent it, or is to be helped to. A post the kernel found
// wrong stops the program here, before any of its groups ring.
void ompx__loop_done(unsigned long long kernel, int i) {
    pipe_check();
    ompx_pipe_deferred* q = g_pipe_deferred_host;
    if (q == nullptr || i < 0 || i >= OMPX_PIPE_AFTER_MAX || q->after.kernel != kernel) return;
    release_after(i);
    q->after.e[i].armed = 0;
}

// ---- explicit batched DWQ ---------------------------------------------------
// Defined below, next to the other compiler-facing helpers.
void ompx_trigger_host() {
    if (g_runtime == nullptr || g_dwq_pending == 0) return;   // no-op for proxy
    g_dwq_pending = 0;
    gicc_runtime_arm_dwq_trigger(g_runtime);
    volatile uint64_t* addr = gicc_runtime_trigger_addr_host(g_runtime);
    const uint64_t value = gicc_runtime_trigger_val(g_runtime);
    if (addr != nullptr && value != 0) *addr = value;
}

// ---- compiler-facing helpers ------------------------------------------------
// Used by the OMP-DWQ LTO pass and by the synthesized host traces.

void ompx_dwq_stage(int peer, int dst_buffer, size_t dst_offset,
                    int src_buffer, size_t src_offset, size_t bytes) {
    gicc_runtime_dwq_enqueue(g_runtime, peer, dst_buffer, dst_offset,
                             src_buffer, src_offset, bytes);
}

void* gicc_runtime_current() {
    return g_runtime;
}

}  // extern "C"
