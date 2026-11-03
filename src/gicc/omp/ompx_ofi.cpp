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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <utility>
#include <vector>

#include "gicc/omp.h"
#include "gicc/omp_box.h"
#include "gicc/platform/ofi/internal/gpu_device_context.hpp"
#include "gicc/platform/ofi/ofi_runtime.hpp"
#include "gicc/platform/ofi/runtime_helpers.h"

extern "C" int __tgt_target_kernel(void* loc, int64_t device, int32_t teams, int32_t threads,
                                   void* host_ptr, void* args);

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

// ---- deferred fences (ompx__fence_defer in gicc/omp.h) ----------------------
// As in ompx_ib.cpp. The gicc-passes plugin defers an ompx_fence past the
// part of the next kernel that touches nothing the epoch's transfers do. For
// that every rank must know what the others did to its heap, which one-sided
// transfers never tell it: so the host records what it sends and fetches,
// and the fence that ends the epoch sends each rank one fixed-size record of
// what this one did to its heap (MPI_Ialltoall, on a communicator of its own)
// before it is deferred. A transfer the record cannot describe (a
// put_signal, more than fits, one a kernel made itself) makes the epoch's
// fence one that no kernel is split around.
//
// Every ompx call makes a deferred fence first, and the plugin makes it
// before anything else that may see what it orders or wait on another rank
// (every MPI call among them); so between the fence and where it is made, a
// rank blocks on no other, and every rank makes its one barrier.
struct EpochOp {
    int peer;
    bool put;            // a put writes the peer's [remote, + bytes); a get reads it
    size_t remote;       // a heap offset at the peer
    const char* local;   // what the put reads here, or the get writes
    size_t bytes;
};
std::vector<EpochOp> g_epoch;
bool g_epoch_untracked = false;
bool g_fence_pending = false;
bool g_split_off = false;          // OMPX_SPLIT=0
MPI_Comm g_fence_comm = MPI_COMM_NULL;
MPI_Request g_manifest_req = MPI_REQUEST_NULL;
// Per pair of ranks: a count, kManifestUnknown when the record cannot say,
// then kManifestMax entries of (heap offset, bytes, written), then which
// of the sender's earlier epochs its records to every rank repeat (bit 0:
// the previous one, bit 1: the one before).
constexpr int kManifestMax = 16;
constexpr long long kManifestUnknown = -1;
constexpr int kManifestWords = 2 + 3 * kManifestMax;
constexpr int kManifestSame = kManifestWords - 1;
std::vector<long long> g_manifest_out, g_manifest_in;
// OMPX_SPLIT_STABLE=1: the program asserts that its epochs transfer the
// same ranges, every epoch or every other one (a time step that swaps two
// fields). Once an exchange shows every rank's records repeating with one
// period, every rank stops exchanging them at the same epoch and reuses,
// in turn, the ones it received over the last period; a rank whose own
// records then break the period stops the program.
bool g_split_stable = false;
bool g_stable_locked = false;
int g_stable_period = 0, g_stable_phase = 0;
std::vector<long long> g_out_hist[2], g_in_hist[2];   // [0]: last epoch, [1]: the one before
std::vector<long long> g_lock_out[2], g_lock_in[2];   // what each phase sends and receives
long long g_stable_reused = 0;
struct Range { const char* lo; size_t bytes; };
// This rank's own part of the epoch: what its gets write and its puts read.
std::vector<Range> g_local_written, g_local_read;
bool g_local_unknown = false;
// ompx__split_ctl, where kernels read it: a device global the host copies
// to before each launch. The CPU cannot store into GPU memory coherently
// here, and host memory every wave reads would cost each launch far more.
ompx_split_ctl* g_split_ctl_dev = nullptr;
// The ranges of the kernel being split, those before the fence first.
struct SplitRange { long long lo, hi; };
std::vector<SplitRange> g_split_ranges;
size_t g_split_before = 0, g_split_at = 0;
long long g_split_lo = 0, g_split_hi = 0;     // its last window, for the stats
long long g_split_rest = 0;
// OMPX_SPLIT_STATS=1: seconds in each part of the split launches, printed
// by ompx_finalize -- before the first launch (the record's wait, the
// window), the first launch, the fence, the second launch.
bool g_split_stats = false;
double g_split_t[4] = {0, 0, 0, 0}, g_split_mark = 0;
long long g_split_n = 0, g_split_whole = 0;
// Launches that found no fence deferred, and those that made it whole: the
// record could not say (unknown) or every iteration touches the epoch's
// transfers (none left).
long long g_split_idle = 0, g_split_unknown = 0, g_split_none = 0;
void split_tick(int part) {
    if (!g_split_stats) return;
    const double now = MPI_Wtime();
    if (part >= 0) g_split_t[part] += now - g_split_mark;
    g_split_mark = now;
}

// A chain of split kernels (ompx__split_begin_launch): what each still has to
// run after the fence, with a copy of its launch to make it again. A launch
// is copied only when nothing of it is read from the host at launch time --
// every argument a literal or a zero-length map of what is already mapped --
// so making it later hands the kernel what it would have been handed now.
struct KernelArgsCopy {   // libomptarget's KernelArgsTy, LLVM 19 to 22
    uint32_t version, num_args;
    void** base_ptrs;
    void** ptrs;
    int64_t* sizes;
    int64_t* types;
    void** names;
    void** mappers;
    uint64_t tripcount;
    uint64_t flags;
    uint32_t num_teams[3], thread_limit[3];
    uint32_t dyn_cgroup_mem;
};
static_assert(sizeof(KernelArgsCopy) == 104 && offsetof(KernelArgsCopy, tripcount) == 56,
              "KernelArgsTy layout (GICCSplitLaunch.cpp)");
struct ChainedLaunch {
    void* loc;
    long long device;
    int teams, threads;
    void* host_ptr;
    KernelArgsCopy args;
    std::vector<void*> base_ptrs, ptrs;
    std::vector<int64_t> sizes, types;
    std::vector<ompx_split_access> acc;
    std::vector<SplitRange> rest;
};
std::vector<ChainedLaunch> g_chain;
bool g_split_chain = true;          // OMPX_SPLIT_CHAIN=0 turns it off
bool g_split_chaining = false;      // the kernel being split ends at the window
long long g_chain_n = 0, g_chain_conflict = 0, g_chain_copyless = 0;
constexpr int64_t kMapLiteral = 0x100;   // OMP_TGT_MAPTYPE_LITERAL
constexpr uint64_t kFlagNoWait = 1;

// The copy of a launch the runtime can make again later; false when the
// launch reads host memory, has a mapper or does not wait for the kernel.
bool copy_launch(ChainedLaunch& c, const void* kargs) {
    std::memcpy(&c.args, kargs, sizeof(c.args));
    const KernelArgsCopy& a = c.args;
    if (a.flags & kFlagNoWait) return false;
    for (uint32_t i = 0; i < a.num_args; ++i) {
        if (a.mappers != nullptr && a.mappers[i] != nullptr) return false;
        if (!(a.types[i] & kMapLiteral) && a.sizes[i] != 0) return false;
    }
    c.base_ptrs.assign(a.base_ptrs, a.base_ptrs + a.num_args);
    c.ptrs.assign(a.ptrs, a.ptrs + a.num_args);
    c.sizes.assign(a.sizes, a.sizes + a.num_args);
    c.types.assign(a.types, a.types + a.num_args);
    return true;
}

static_assert(offsetof(ompx_split_access, stride) == 8 &&
                  offsetof(ompx_split_access, extent) == 8 + 8 * OMPX_SPLIT_DIMS_MAX &&
                  offsetof(ompx_split_access, offset) == 8 + 16 * OMPX_SPLIT_DIMS_MAX &&
                  offsetof(ompx_split_access, dims) == 16 + 16 * OMPX_SPLIT_DIMS_MAX &&
                  offsetof(ompx_split_access, mapped) == 32 + 16 * OMPX_SPLIT_DIMS_MAX &&
                  sizeof(ompx_split_access) == 40 + 16 * OMPX_SPLIT_DIMS_MAX,
              "ompx_split_access layout (GICCSplitLaunch.cpp)");

// An access's box origin in the heap, or null when it is not in it: a device
// address as is; a mapped variable where ompx_bind put it, found by the
// variable's own host address.
const char* heap_box(const ompx_split_access& a) {
    if (!a.mapped) return a.box;
    for (const HeapBind& b : g_binds)
        if (a.box >= b.host && a.box < b.host + b.bytes) return b.dev + (a.box - b.host) + a.offset;
    return nullptr;
}

// Byte span [lo, hi) of everything an access reaches.
void access_span(const ompx_split_access& a, const char* box, const char** lo, const char** hi) {
    int64_t mn = 0, mx = 0;
    for (int d = 0; d < a.dims; ++d) {
        const int64_t span = a.stride[d] * (a.extent[d] - 1);
        (span < 0 ? mn : mx) += span;
    }
    *lo = box + mn;
    *hi = box + mx + a.elem;
}

// An access's box origin as a device address: in the heap (heap_box), or a
// mapped variable outside it through libomptarget's mapping; null if
// neither.
const char* device_box(const ompx_split_access& a) {
    if (const char* b = heap_box(a)) return b;
    if (!a.mapped) return nullptr;
    void* d = omp_get_mapped_ptr(a.box, omp_get_default_device());
    return d == nullptr ? nullptr : static_cast<const char*>(d) + a.offset;
}

// The digits of a byte offset in a layout of strides s[0] > s[1] > ...,
// each a multiple of the next and the last one elem; false when the offset
// is not on an element, or a digit with its box's extent leaves its dim.
bool layout_digits(const ompx_split_access& a, int64_t off, int64_t* q) {
    for (int d = 0; d < a.dims; ++d) {
        q[d] = ompx__floor_div(off, a.stride[d]);
        off -= q[d] * a.stride[d];
        if (d > 0 && q[d] + a.extent[d] > a.stride[d - 1] / a.stride[d]) return false;
    }
    return off == 0;
}

// Whether two accesses may reach a common byte. Exact for two boxes of one
// layout (the same strides, each a multiple of the next, in decreasing
// order, the last one the element): their digits are compared dim by dim --
// from the start of the variable for two boxes of one mapped variable,
// whose offsets into it are known, else from the lower origin. Otherwise
// only their spans are compared.
bool accesses_meet(const ompx_split_access& a, const ompx_split_access& b) {
    const char* ba = device_box(a);
    const char* bb = device_box(b);
    if (ba == nullptr || bb == nullptr) return true;
    const char *alo, *ahi, *blo, *bhi;
    access_span(a, ba, &alo, &ahi);
    access_span(b, bb, &blo, &bhi);
    if (ahi <= blo || bhi <= alo) return false;
    if (a.dims != b.dims || a.elem != b.elem || a.dims <= 0 || a.dims > OMPX_SPLIT_DIMS_MAX)
        return true;
    for (int d = 0; d < a.dims; ++d) {
        if (a.stride[d] != b.stride[d] || a.stride[d] <= 0) return true;
        if (d + 1 < a.dims && (a.stride[d] <= a.stride[d + 1] || a.stride[d] % a.stride[d + 1]))
            return true;
    }
    if (a.stride[a.dims - 1] != a.elem) return true;
    int64_t qa[OMPX_SPLIT_DIMS_MAX], qb[OMPX_SPLIT_DIMS_MAX];
    if (a.mapped && b.mapped && a.box == b.box) {
        if (!layout_digits(a, a.offset, qa) || !layout_digits(b, b.offset, qb)) return true;
    } else {
        const bool a_low = ba <= bb;
        const int64_t diff = a_low ? bb - ba : ba - bb;
        int64_t* lowq = a_low ? qa : qb;
        int64_t* highq = a_low ? qb : qa;
        for (int d = 0; d < a.dims; ++d) lowq[d] = 0;
        if (!layout_digits(a_low ? b : a, diff, highq)) return true;
        for (int d = 1; d < a.dims; ++d)
            if ((a_low ? a : b).extent[d] > a.stride[d - 1] / a.stride[d]) return true;
    }
    for (int d = 0; d < a.dims; ++d)
        if (qa[d] + a.extent[d] <= qb[d] || qb[d] + b.extent[d] <= qa[d]) return false;
    return true;
}

// Whether the kernel's accesses may meet what a chained kernel still has to
// do, where either writes.
bool meets_chain(int n, const ompx_split_access* acc) {
    for (size_t k = 0; k < g_chain.size(); ++k)
        for (size_t j = 0; j < g_chain[k].acc.size(); ++j) {
            const ompx_split_access& b = g_chain[k].acc[j];
            for (int i = 0; i < n; ++i)
                if ((acc[i].store || b.store) && accesses_meet(acc[i], b)) return true;
        }
    return false;
}

// What the next launches of a split kernel run (ompx__split_ctl). A copy
// from pageable memory: done when it returns, before the next launch.
// ompx__split_ctl is a global of the OpenMP device image, which libomptarget
// loaded and the HIP runtime has never seen, so libomptarget copies it.
void set_split(int range, long long lo, long long hi) {
    const ompx_split_ctl ctl{range, 0, lo, hi};
    if (omp_target_memcpy(g_split_ctl_dev, &ctl, sizeof(ctl), 0, 0, g_local_device,
                          omp_get_initial_device()) != 0)
        die("set what a split kernel runs: omp_target_memcpy failed");
}

void record(int peer, bool put, size_t remote, const char* local, size_t bytes) {
    if (bytes == 0 || peer < 0) return;
    g_epoch.push_back(EpochOp{peer, put, remote, local, bytes});
}

// The epoch's records have arrived: lock them when every rank said its own
// were those of its previous epoch (OMPX_SPLIT_STABLE).
void manifest_arrived() {
    if (!g_split_stable || g_stable_locked || g_manifest_in.empty()) return;
    int n = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n);
    long long all = 3;
    for (int r = 0; r < n; ++r) all &= g_manifest_in[(size_t)r * kManifestWords + kManifestSame];
    // This epoch's records, sent and received, in the history.
    std::vector<long long> in = g_manifest_in;
    for (int r = 0; r < n; ++r) in[(size_t)r * kManifestWords + kManifestSame] = 0;
    g_in_hist[1] = std::move(g_in_hist[0]);
    g_in_hist[0] = std::move(in);
    if (all == 0) return;
    // The next epoch repeats the one `period` back: for period 2 the one
    // before this, then this one.
    g_stable_period = (all & 1) ? 1 : 2;
    g_lock_out[0] = g_stable_period == 1 ? g_out_hist[0] : g_out_hist[1];
    g_lock_in[0] = g_stable_period == 1 ? g_in_hist[0] : g_in_hist[1];
    g_lock_out[1] = g_out_hist[0];
    g_lock_in[1] = g_in_hist[0];
    g_stable_phase = 0;
    g_stable_locked = true;
}

void wait_manifest() {
    if (g_manifest_req == MPI_REQUEST_NULL) return;
    MPI_Wait(&g_manifest_req, MPI_STATUS_IGNORE);
    manifest_arrived();
}

// The window of the kernel's iterations [0, iterations) that the epoch's
// transfers leave alone, as split_begin hands it to the kernel: [lo, hi),
// wrapping when lo > hi. Empty (lo == hi, *len 0) when nothing is left.
void quiet_window(int n, const ompx_split_access* acc, long long iterations,
                  const std::vector<Range>& written, const std::vector<Range>& read,
                  long long* lo, long long* hi, long long* len) {
    std::vector<std::pair<long long, long long>> hit;
    auto meet = [&](const ompx_split_access& a, const Range& r) {
        const char* box = heap_box(a);
        if (box == nullptr) return;   // not in the heap: no transfer reaches it
        int64_t l = 0, h = -1;
        ompx__hull(r.lo, r.bytes, box, a.dims, a.loop_dims,
                   reinterpret_cast<const int64_t*>(a.stride),
                   reinterpret_cast<const int64_t*>(a.extent), a.elem, l, h);
        if (h > iterations - 1) h = iterations - 1;
        if (l < 0) l = 0;
        if (l <= h) hit.push_back({l, h});
    };
    for (int i = 0; i < n; ++i) {
        // A load may not see what arrives; a store may not race with it,
        // nor change what a transfer still reads.
        for (const Range& r : written) meet(acc[i], r);
        if (acc[i].store)
            for (const Range& r : read) meet(acc[i], r);
    }
    if (hit.empty()) {
        *lo = 0;
        *hi = iterations;
        *len = iterations;
        return;
    }
    std::sort(hit.begin(), hit.end());
    std::vector<std::pair<long long, long long>> merged;
    for (const auto& x : hit) {
        if (!merged.empty() && x.first <= merged.back().second + 1)
            merged.back().second = std::max(merged.back().second, x.second);
        else
            merged.push_back(x);
    }
    // The widest gap, the one past the last interval wrapping to the first.
    long long best = -1, start = 0;
    for (size_t i = 0; i < merged.size(); ++i) {
        const long long from = merged[i].second + 1;
        const long long to = i + 1 < merged.size() ? merged[i + 1].first
                                                   : merged[0].first + iterations;
        if (to - from > best) {
            best = to - from;
            start = from;
        }
    }
    *len = best;
    *lo = start % iterations;
    *hi = (start + best) % iterations;
}

// A deferred fence, made now; then what the chained kernels left, in order.
// Defined after the quiet, which it makes.
void fence_flush();

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
    MPI_Comm_dup(MPI_COMM_WORLD, &g_fence_comm);
    const char* split_env = std::getenv("OMPX_SPLIT");
    g_split_off = split_env != nullptr && std::atoi(split_env) == 0;
    const char* stable_env = std::getenv("OMPX_SPLIT_STABLE");
    g_split_stable = stable_env != nullptr && std::atoi(stable_env) != 0;
    const char* chain_env = std::getenv("OMPX_SPLIT_CHAIN");
    g_split_chain = chain_env == nullptr || std::atoi(chain_env) != 0;
    const char* stats_env = std::getenv("OMPX_SPLIT_STATS");
    g_split_stats = stats_env != nullptr && std::atoi(stats_env) != 0;
}

void ompx_finalize() {
    fence_flush();
    if (g_split_stats)
        std::fprintf(stderr, "[giomp] rank %d: %lld split launches (%lld run whole before the "
                     "fence; not split: %lld with no fence deferred, %lld unknown, %lld with "
                     "nothing left): before %.4f s, inside %.4f s, fence %.4f s, outside %.4f s; "
                     "last window [%lld, %lld), %lld iterations outside\n", ompx_get_rank_num(),
                     g_split_n, g_split_whole, g_split_idle, g_split_unknown, g_split_none,
                     g_split_t[0], g_split_t[1], g_split_t[2], g_split_t[3], g_split_lo,
                     g_split_hi, g_split_rest);
    if (g_split_stats)
        std::fprintf(stderr, "[giomp] rank %d: chain: %lld kernels chained, %lld stopped by a "
                     "conflict, %lld launches not copyable; %lld epochs reused the locked "
                     "records\n", ompx_get_rank_num(), g_chain_n, g_chain_conflict,
                     g_chain_copyless, g_stable_reused);
    if (g_fence_comm != MPI_COMM_NULL) MPI_Comm_free(&g_fence_comm);
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
    fence_flush();
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
    fence_flush();
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
    fence_flush();
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
    fence_flush();
    const size_t src_off = offset_of(src, "ompx_put src");
    const size_t dst_off = offset_of(dst, "ompx_put dst");
    record(peer, true, dst_off, g_heap_base + src_off, bytes);

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
    fence_flush();
    const size_t dst_off = offset_of(dst, "ompx_get dst");
    const size_t src_off = offset_of(src, "ompx_get src");
    record(peer, false, src_off, g_heap_base + dst_off, bytes);

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
    fence_flush();
    __atomic_store_n(&g_sig_host[sig], 0, __ATOMIC_RELEASE);
}

// Spins without waiting for this rank's own transfers: some of them may be
// staged for a kernel that has not run yet, and drain() would wait on those
// forever.
void ompx_signal_wait_host(int sig, unsigned long long ge) {
    check_slot(sig, "ompx_signal_wait");
    fence_flush();
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
    fence_flush();
    g_epoch_untracked = true;   // the signal lands outside the heap
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
    fence_flush();
    g_epoch_untracked = true;   // a kernel releases it, and the signal lands outside the heap
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

}  // extern "C"

namespace {

// Completes host-issued transfers and everything kernels handed over.
void quiet_now() {
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

void fence_flush() {
    if (!g_fence_pending) return;
    g_fence_pending = false;
    wait_manifest();
    quiet_now();
    MPI_Barrier(g_fence_comm);
    if (g_chain.empty()) return;
    split_tick(-1);
    for (ChainedLaunch& c : g_chain)
        for (const SplitRange& r : c.rest) {
            set_split(1, r.lo, r.hi);
            // libomptarget rewrites the arguments in place: a fresh copy each time.
            KernelArgsCopy a = c.args;
            a.base_ptrs = c.base_ptrs.data();
            a.ptrs = c.ptrs.data();
            a.sizes = c.sizes.data();
            a.types = c.types.data();
            a.tripcount = (uint64_t)(r.hi - r.lo);
            if (__tgt_target_kernel(c.loc, c.device, c.teams, c.threads, c.host_ptr, &a) != 0)
                die("a chained kernel's remaining range failed to launch");
        }
    set_split(0, 0, 0);
    g_chain.clear();
    split_tick(3);
}

}  // namespace

extern "C" {

void ompx_quiet_host() {
    fence_flush();
    quiet_now();
}

void ompx_barrier() {
    fence_flush();
    MPI_Barrier(MPI_COMM_WORLD);
}

void ompx_fence() {
    fence_flush();
    quiet_now();
    MPI_Barrier(MPI_COMM_WORLD);
}

// The fence at the end of an epoch, deferred: what this rank did to each
// other's heap goes out now, and the fence is made where ompx__fence_flush
// or the next ompx call says, or around a kernel split by
// ompx__split_begin.
void ompx__fence_defer() {
    fence_flush();
    if (g_runtime == nullptr) return;
    if (g_split_off) {
        ompx_fence();
        return;
    }
    // Pipelined puts a kernel left for the quiet go now, recorded; and under
    // DWQ the descriptors the epoch's puts staged are released now, so the
    // transfers run while the kernels compute instead of waiting for the fence.
    send_deferred_pipelined();
    ompx_trigger_host();
    const int n = ompx_get_num_ranks(), me = ompx_get_rank_num();
    g_manifest_out.assign((size_t)n * kManifestWords, 0);
    if (!g_stable_locked) g_manifest_in.assign((size_t)n * kManifestWords, 0);
    g_local_written.clear();
    g_local_read.clear();
    ompx_pipe_deferred* q = g_pipe_deferred_host;
    g_local_unknown = g_epoch_untracked || (q != nullptr && q->dev_comm);
    if (q != nullptr) q->dev_comm = 0;
    for (const EpochOp& op : g_epoch) {
        // What it does here.
        (op.put ? g_local_read : g_local_written).push_back(Range{op.local, op.bytes});
        if (op.peer == me) {
            (op.put ? g_local_written : g_local_read)
                .push_back(Range{g_heap_base + op.remote, op.bytes});
            continue;
        }
        long long* m = &g_manifest_out[(size_t)op.peer * kManifestWords];
        if (m[0] == kManifestUnknown) continue;
        if (m[0] == kManifestMax) {
            m[0] = kManifestUnknown;
            continue;
        }
        long long* e = m + 1 + 3 * m[0]++;
        e[0] = (long long)op.remote;
        e[1] = (long long)op.bytes;
        e[2] = op.put ? 1 : 0;
    }
    if (g_local_unknown)
        for (int r = 0; r < n; ++r) g_manifest_out[(size_t)r * kManifestWords] = kManifestUnknown;
    g_epoch.clear();
    g_epoch_untracked = false;
    if (g_split_stable) {
        if (g_stable_locked) {
            // The others go on from what they received for this phase; so
            // must these.
            if (g_manifest_out != g_lock_out[g_stable_phase])
                die("OMPX_SPLIT_STABLE=1, but this epoch transfers other ranges than the "
                    "epoch a period before");
            g_manifest_in = g_lock_in[g_stable_phase];
            g_stable_phase = (g_stable_phase + 1) % g_stable_period;
            ++g_stable_reused;
            g_fence_pending = true;
            return;
        }
        const long long same = (g_manifest_out == g_out_hist[0] ? 1 : 0) |
                               (g_manifest_out == g_out_hist[1] ? 2 : 0);
        g_out_hist[1] = std::move(g_out_hist[0]);
        g_out_hist[0] = g_manifest_out;
        for (int r = 0; r < n; ++r)
            g_manifest_out[(size_t)r * kManifestWords + kManifestSame] = same;
    }
    MPI_Ialltoall(g_manifest_out.data(), kManifestWords, MPI_LONG_LONG, g_manifest_in.data(),
                  kManifestWords, MPI_LONG_LONG, g_fence_comm, &g_manifest_req);
    g_fence_pending = true;
}

void ompx__fence_flush() {
    fence_flush();
}

// Before a launch of a kernel the plugin can split. With a fence deferred,
// works out which of its iterations touch what the epoch's transfers do:
// none -- the kernel runs whole and the fence stays deferred (0); some --
// the kernel is to run in the ranges ompx__split_next hands out, the others
// first, then the fence, then the rest (1); all of them, or the record
// cannot say -- the fence is made now and the kernel runs whole (0).
int ompx__split_begin(unsigned long long kernel, int n, const ompx_split_access* acc,
                      long long iterations) {
    return ompx__split_begin_launch(kernel, n, acc, iterations, nullptr, 0, 0, 0, nullptr,
                                    nullptr);
}

int ompx__split_begin_launch(unsigned long long kernel, int n, const ompx_split_access* acc,
                             long long iterations, void* loc, long long device, int teams,
                             int threads, void* host_ptr, void* kernel_args) {
    (void)kernel;
    g_split_chaining = false;
    if (!g_fence_pending) {
        ++g_split_idle;
        return 0;
    }
    // Whatever runs of this kernel now runs before what the chain left:
    // where they may touch one another, the chain goes first.
    if (!g_chain.empty() && (n < 0 || meets_chain(n, acc))) {
        ++g_chain_conflict;
        fence_flush();
        ++g_split_idle;
        return 0;
    }
    split_tick(-1);
    if (g_split_ctl_dev == nullptr || iterations <= 0 || n < 0) {
        fence_flush();
        return 0;
    }
    wait_manifest();
    const int ranks = ompx_get_num_ranks();
    bool known = !g_local_unknown;
    std::vector<Range> written = g_local_written, read = g_local_read;
    for (int r = 0; r < ranks && known; ++r) {
        const long long* m = &g_manifest_in[(size_t)r * kManifestWords];
        if (m[0] == kManifestUnknown || m[0] < 0 || m[0] > kManifestMax) {
            known = false;
            break;
        }
        for (long long i = 0; i < m[0]; ++i) {
            const long long* e = m + 1 + 3 * i;
            (e[2] ? written : read).push_back(Range{g_heap_base + e[0], (size_t)e[1]});
        }
    }
    if (!known) {
        ++g_split_unknown;
        fence_flush();
        return 0;
    }
    long long lo = 0, hi = 0, len = 0;
    quiet_window(n, acc, iterations, written, read, &lo, &hi, &len);
    if (len == iterations) {   // the kernel touches none of it
        ++g_split_whole;
        return 0;
    }
    if (len == 0) {
        ++g_split_none;
        fence_flush();
        return 0;
    }
    // The window [lo, lo + len) of [0, iterations), wrapping past the end,
    // and the rest: each as one or two ranges.
    g_split_ranges.clear();
    auto add = [](long long a, long long b) {
        if (a < b) g_split_ranges.push_back(SplitRange{a, b});
    };
    const long long end = lo + len;
    add(lo, std::min(end, iterations));
    add(0, end - iterations);
    g_split_before = g_split_ranges.size();
    add(std::max(end - iterations, 0LL), lo);
    add(end, iterations);
    g_split_at = 0;
    g_split_lo = lo;
    g_split_hi = hi;
    g_split_rest = iterations - len;
    ++g_split_n;
    // A chain: the window now, the rest after the one fence.
    if (g_split_chain && kernel_args != nullptr) {
        ChainedLaunch c;
        if (copy_launch(c, kernel_args)) {
            c.loc = loc;
            c.device = device;
            c.teams = teams;
            c.threads = threads;
            c.host_ptr = host_ptr;
            c.acc.assign(acc, acc + n);
            c.rest.assign(g_split_ranges.begin() + g_split_before, g_split_ranges.end());
            g_split_ranges.resize(g_split_before);
            g_chain.push_back(std::move(c));
            g_split_chaining = true;
            ++g_chain_n;
        } else {
            ++g_chain_copyless;
        }
    }
    split_tick(0);
    return 1;
}

// The next range of the kernel being split, set for its next launch, with
// its length for the launch's trip count; 0 when there is none left. The
// fence goes between the ranges before it and the others.
int ompx__split_next(unsigned long long* tripcount) {
    if (g_split_at == g_split_before && !g_split_chaining) {
        split_tick(1);
        fence_flush();
        split_tick(2);
    }
    if (g_split_at >= g_split_ranges.size()) {
        split_tick(3);
        set_split(0, 0, 0);
        g_split_ranges.clear();
        g_split_before = g_split_at = 0;
        return 0;
    }
    const SplitRange& r = g_split_ranges[g_split_at++];
    set_split(1, r.lo, r.hi);
    *tripcount = (unsigned long long)(r.hi - r.lo);
    return 1;
}

void ompx__split_ctl_at(ompx_split_ctl* dev) {
    g_split_ctl_dev = dev;
}

// No control block the CPU stores into in place here: the kernels read the
// device global ompx__split_ctl_at names, which set_split copies to.
ompx_split_ctl* ompx__split_ctl_shared() {
    return nullptr;
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
    e.sig = -1;
    e.value = 0;
    e.dyn_ok = 0;   // run-time counting is wired on InfiniBand only
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

// An ompx_put_signal after the launch stays on the host here: posted
// unarmed, so no kernel sends any of it.
void ompx__after_post_signal(unsigned long long kernel, int i, int src_arg, int armed, int peer,
                             void* dst, const void* src, long long src_rel, size_t bytes,
                             int sig, unsigned long long value) {
    (void)armed; (void)sig; (void)value;
    ompx__after_post(kernel, i, src_arg, 0, peer, dst, src, src_rel, bytes);
}

void ompx__after_done_signal(unsigned long long kernel, int i, int peer, void* dst,
                             const void* src, size_t bytes, int sig, unsigned long long value) {
    pipe_check();
    ompx_pipe_deferred* q = g_pipe_deferred_host;
    if (q != nullptr && i >= 0 && i < OMPX_PIPE_AFTER_MAX && q->after.kernel == kernel) {
        release_after(i);
        q->after.e[i].armed = 0;
        q->after.e[i].handled = 0;
    }
    ompx_put_signal_host(peer, dst, src, bytes, sig, value);
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
