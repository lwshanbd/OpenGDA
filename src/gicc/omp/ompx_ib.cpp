// GiOMP runtime on InfiniBand. Compiled as CUDA source (clang -x cuda)
// and linked into libgicc_omp, so OpenMP application TUs only need the public
// gicc/omp.h header and the library -- the same contract as ompx_ofi.cpp,
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

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "gicc/omp.h"
#include "gicc/omp_box.h"
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
// The inbox is mapped pinned host memory: the host waits on it with plain
// loads instead of a device-to-host copy per poll, and kernels read it
// through its device mapping.
gicc::Buffer g_sig{};
uint64_t*    g_sig_host = nullptr;   // the inbox, as the host addresses it
uint64_t*    g_sig_dev = nullptr;    // the same inbox, as kernels address it

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
    require_cuda(cudaHostAlloc((void**)&g_sig_host, sig_bytes, cudaHostAllocMapped),
                 "allocate signal inbox");
    std::memset(g_sig_host, 0, sig_bytes);
    require_cuda(cudaHostGetDevicePointer((void**)&g_sig_dev, g_sig_host, 0),
                 "map signal inbox");
    g_sig = g_runtime->register_buffer(g_sig_host, sig_bytes, /*is_device=*/false);

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
    return __atomic_load_n(g_sig_host + sig, __ATOMIC_ACQUIRE);
}

// ---- pipelined puts (ompx_pipe_deferred in gicc/omp.h) ----------------------
// A GPU thread posts its own work requests, so a kernel can send a pipelined
// put to any peer at any point: the list says proxy_on, the libfabric
// backend's name for that, and nothing is ever queued behind a bell. What a
// kernel still leaves here -- a put it is not SPMD enough to count -- the
// next quiet sends.
ompx_pipe_deferred* g_pipe_deferred_host = nullptr;
ompx_pipe_deferred* g_pipe_deferred_dev  = nullptr;
char**              g_pipe_peer_heap_host = nullptr;   // ompx_pipe_deferred::peer_heap
char**              g_pipe_peer_heap_dev  = nullptr;

// A pipelined put the device could not make (ompx_pipe_deferred::fail).
void pipe_check() {
    const ompx_pipe_deferred* q = g_pipe_deferred_host;
    if (q == nullptr || q->fail == 0) return;
    char what[256];
    const long long a = q->fail_arg;
    switch (q->fail) {
    case OMPX_PIPE_FAIL_NO_PROXY:
        std::snprintf(what, sizeof(what), "ompx_pipelined_put: peer %lld could not be sent "
                      "to from the device", a);
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
// The gicc-passes plugin defers an ompx_fence past the part of the next
// kernel that touches nothing the epoch's transfers do. For that every rank
// must know what the others did to its heap, which one-sided transfers never
// tell it: so the host records what it sends and fetches, and the fence that
// ends the epoch sends each rank one fixed-size record of what this one did
// to its heap (MPI_Ialltoall, on a communicator of its own) before it is
// deferred. A transfer the record cannot describe (a put_signal, more than
// fits, one a kernel made itself) makes the epoch's fence one that no kernel
// is split around.
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
ompx_split_ctl* g_split_ctl_dev = nullptr;   // ompx__split_ctl, where kernels read it
// The control block the CPU stores into directly (ompx__split_ctl_shared):
// managed memory the GPU holds and the CPU reaches over NVLink-C2C, no copy.
ompx_split_ctl* g_split_ctl_shared = nullptr;
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

const char* heap_box(const ompx_split_access& a);   // below

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
    static const bool debug = [] {
        const char* e = std::getenv("OMPX_SPLIT_CHAIN_DEBUG");
        return e != nullptr && std::atoi(e) != 0;
    }();
    static int reported = 0;
    for (size_t k = 0; k < g_chain.size(); ++k)
        for (size_t j = 0; j < g_chain[k].acc.size(); ++j) {
            const ompx_split_access& b = g_chain[k].acc[j];
            for (int i = 0; i < n; ++i)
                if ((acc[i].store || b.store) && accesses_meet(acc[i], b)) {
                    if (debug && reported++ < 4) {
                        const ompx_split_access& a = acc[i];
                        std::fprintf(stderr,
                                     "[giomp] rank %d: chain conflict: access %d (store %d, mapped "
                                     "%d, box %p -> %p, dims %d) meets chained kernel %zu access "
                                     "%zu (store %d, mapped %d, box %p -> %p, dims %d)\n",
                                     ompx_get_rank_num(), i, a.store, a.mapped, (const void*)a.box,
                                     (const void*)device_box(a), a.dims, k, j, b.store, b.mapped,
                                     (const void*)b.box, (const void*)device_box(b), b.dims);
                        for (const ompx_split_access* x : {&a, &b})
                            for (int d = 0; d < x->dims; ++d)
                                std::fprintf(stderr, "[giomp]     dim %d: stride %lld extent %lld\n",
                                             d, (long long)x->stride[d], (long long)x->extent[d]);
                    }
                    return true;
                }
        }
    return false;
}

// What the next launches of a split kernel run (ompx__split_ctl). A copy
// from pageable memory: done when it returns, before the next launch.
void set_split(int range, long long lo, long long hi) {
    if (g_split_ctl_dev != nullptr && g_split_ctl_dev == g_split_ctl_shared) {
        // Seen by the next launch, which the host makes after these stores.
        __atomic_store_n(&g_split_ctl_shared->lo, lo, __ATOMIC_RELAXED);
        __atomic_store_n(&g_split_ctl_shared->hi, hi, __ATOMIC_RELAXED);
        __atomic_store_n(&g_split_ctl_shared->range, range, __ATOMIC_RELAXED);
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        return;
    }
    const ompx_split_ctl ctl{range, 0, lo, hi};
    require_cuda(cudaMemcpy(g_split_ctl_dev, &ctl, sizeof(ctl), cudaMemcpyHostToDevice),
                 "set what a split kernel runs");
}

// Put what kernels left in the deferred list. Their kernels have finished
// (a quiet follows them), so the sources hold their final values.
void send_deferred_pipelined() {
    ompx_pipe_deferred* q = g_pipe_deferred_host;
    if (q == nullptr) return;
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

void record(int peer, bool put, size_t remote, const char* local, size_t bytes) {
    if (bytes == 0 || peer < 0) return;
    g_epoch.push_back(EpochOp{peer, put, remote, local, bytes});
}

void quiet_now() {
    if (g_runtime == nullptr) return;
    pipe_check();
    send_deferred_pipelined();
    if (g_ipc_pending) {
        g_ipc_pending = false;
        require_cuda(cudaStreamSynchronize(g_ipc_stream), "drain the IPC stream");
    }
    g_runtime->drain();
}

extern "C" int __tgt_target_kernel(void* loc, int64_t device, int32_t teams, int32_t threads,
                                   void* host_ptr, void* args);

// The epoch's records have arrived: lock them when every rank said its own
// were those of its previous epoch (OMPX_SPLIT_STABLE).
void manifest_arrived() {
    if (!g_split_stable || g_stable_locked || g_manifest_in.empty()) return;
    const int n = ompx_get_num_ranks();
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

// A deferred fence, made now; then what the chained kernels left, in order.
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

// The window of the kernel's iterations [0, iterations) that the epoch's
// transfers leave alone, as split_begin hands it to the kernel: [lo, hi),
// wrapping when lo > hi. Empty (lo == hi, *len 0) when nothing is left.
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
    if (g_sig_host != nullptr) {
        (void)cudaFreeHost(g_sig_host);
        g_sig_dev = nullptr;
        g_sig_host = nullptr;
    }
    if (g_pipe_deferred_host != nullptr) {
        (void)cudaFreeHost(g_pipe_deferred_host);
        (void)cudaFreeHost(g_pipe_peer_heap_host);
        g_pipe_deferred_host = nullptr;
        g_pipe_deferred_dev = nullptr;
        g_pipe_peer_heap_host = nullptr;
        g_pipe_peer_heap_dev = nullptr;
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
    fence_flush();
    const size_t src_off = offset_of(src, "ompx_put src");
    const size_t dst_off = offset_of(dst, "ompx_put dst");
    record(peer, true, dst_off, g_heap_base + src_off, bytes);
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
    fence_flush();
    const size_t dst_off = offset_of(dst, "ompx_get dst");
    const size_t src_off = offset_of(src, "ompx_get src");
    record(peer, false, src_off, g_heap_base + dst_off, bytes);
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
    fence_flush();
    __atomic_store_n(g_sig_host + sig, 0, __ATOMIC_RELEASE);
}

void ompx_signal_wait_host(int sig, unsigned long long ge) {
    check_slot(sig, "ompx_signal_wait");
    fence_flush();
    while (read_slot(sig) < ge) g_runtime->progress();
}

// The payload and the signal go out on the same host QP, in that order.
void ompx_put_signal_host(int peer, void* dst, const void* src, size_t bytes,
                          int sig, unsigned long long value) {
    if (g_runtime == nullptr) die("ompx_put_signal before ompx_init");
    check_slot(sig, "ompx_put_signal");
    fence_flush();
    g_epoch_untracked = true;   // the signal lands outside the heap
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
    // Pipelined puts a kernel left for the quiet go now, recorded.
    send_deferred_pipelined();
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

// The GPU's copy of the control block, which the CPU writes in place: only
// where the CPU can store into GPU memory coherently (Grace Hopper), and not
// with OMPX_SPLIT_CTL_COPY=1, which keeps the copy per launch.
ompx_split_ctl* ompx__split_ctl_shared() {
    if (g_split_ctl_shared != nullptr || g_runtime == nullptr) return g_split_ctl_shared;
    const char* e = std::getenv("OMPX_SPLIT_CTL_COPY");
    if (e != nullptr && std::atoi(e) != 0) return nullptr;
    const int gpu = g_runtime->gpu_id();
    int coherent = 0;
    if (cudaDeviceGetAttribute(&coherent, cudaDevAttrDirectManagedMemAccessFromHost, gpu) !=
            cudaSuccess || !coherent)
        return nullptr;
    void* p = nullptr;
    require_cuda(cudaMallocManaged(&p, sizeof(ompx_split_ctl), cudaMemAttachGlobal),
                 "allocate the split control block");
    require_cuda(cudaMemAdvise(p, sizeof(ompx_split_ctl), cudaMemAdviseSetPreferredLocation, gpu),
                 "keep the split control block on the GPU");
    require_cuda(cudaMemAdvise(p, sizeof(ompx_split_ctl), cudaMemAdviseSetAccessedBy,
                               cudaCpuDeviceId),
                 "let the CPU reach the split control block");
    std::memset(p, 0, sizeof(ompx_split_ctl));
    g_split_ctl_shared = static_cast<ompx_split_ctl*>(p);
    return g_split_ctl_shared;
}

// ---- device-side ------------------------------------------------------------

ompx_ctx* ompx_prepare_ctx() {
    return g_runtime->prepare();
}

// Mapped host memory: the host writes it only between launches and reads
// what a kernel wrote only after the kernel (or at the quiet).
ompx_pipe_deferred* ompx__pipe_deferred_list() {
    if (g_runtime == nullptr) die("ompx_prepare before ompx_init");
    const int n = ompx_get_num_ranks();
    if (g_pipe_deferred_dev == nullptr) {
        void* h = nullptr;
        require_cuda(cudaHostAlloc(&h, sizeof(ompx_pipe_deferred), cudaHostAllocMapped),
                     "allocate the deferred pipelined puts");
        std::memset(h, 0, sizeof(ompx_pipe_deferred));
        g_pipe_deferred_host = static_cast<ompx_pipe_deferred*>(h);
        require_cuda(cudaHostGetDevicePointer((void**)&g_pipe_deferred_dev, h, 0),
                     "map the deferred pipelined puts");
        // Each rank's heap as this GPU reaches it through CUDA IPC, or null.
        require_cuda(cudaHostAlloc(&h, n * sizeof(char*), cudaHostAllocMapped),
                     "allocate the pipelined puts' peer table");
        g_pipe_peer_heap_host = static_cast<char**>(h);
        for (int r = 0; r < n; ++r) g_pipe_peer_heap_host[r] = peer_heap(r);
        require_cuda(cudaHostGetDevicePointer((void**)&g_pipe_peer_heap_dev, h, 0),
                     "map the pipelined puts' peer table");
    }
    ompx_pipe_deferred* q = g_pipe_deferred_host;
    q->heap_base = g_heap_base;
    q->peer_heap = g_pipe_peer_heap_dev;
    q->n_ranks   = n;
    q->proxy_on  = 1;   // the GPU posts its own puts
    return g_pipe_deferred_dev;
}

// ---- puts after a kernel launch (ompx_pipe_after in gicc/omp.h) --------------
// The plugin's second compile brackets a launch followed by ompx_put calls:
// each put is posted before the launch, and replaced by ompx__after_done
// after it. The kernel reads the post and sends the put itself; the host
// makes it only if the kernel did not. Nothing is queued: there is no bell.
void ompx__after_post(unsigned long long kernel, int i, int src_arg, int armed, int peer,
                      void* dst, const void* src, long long src_rel, size_t bytes) {
    ompx_pipe_deferred* q = g_pipe_deferred_host;
    if (q == nullptr || i < 0 || i >= OMPX_PIPE_AFTER_MAX) return;
    (void)src;
    ompx_pipe_after_put& e = q->after.e[i];
    e.armed = 0;
    e.handled = 0;
    e.bell = nullptr;
    e.fired = 0;
    e.pieces = 0;
    e.piece = 0;
    e.sig = -1;
    e.value = 0;
    e.dyn_ok = 1;
    if (armed) {
        e.peer    = peer;
        e.src_arg = src_arg;
        e.dst_off = offset_of(dst, "ompx_put dst");
        e.src_rel = src_rel;
        e.bytes   = bytes;
        e.armed   = 1;
    }
    q->after.kernel = kernel;
}

void ompx__after_post_signal(unsigned long long kernel, int i, int src_arg, int armed, int peer,
                             void* dst, const void* src, long long src_rel, size_t bytes,
                             int sig, unsigned long long value) {
    if (armed) check_slot(sig, "ompx_put_signal");
    ompx__after_post(kernel, i, src_arg, armed, peer, dst, src, src_rel, bytes);
    ompx_pipe_deferred* q = g_pipe_deferred_host;
    if (q == nullptr || i < 0 || i >= OMPX_PIPE_AFTER_MAX) return;
    q->after.e[i].sig = sig;
    q->after.e[i].value = value;
}

// The kernel sent the payload and the signal behind it on one QP, or none of
// it reached the peer as a whole: then the host sends all of it again, the
// signal behind it, and clears the counters the kernel left part way.
void ompx__after_done_signal(unsigned long long kernel, int i, int peer, void* dst,
                             const void* src, size_t bytes, int sig, unsigned long long value) {
    pipe_check();
    ompx_pipe_deferred* q = g_pipe_deferred_host;
    bool sent = false;
    if (q != nullptr && i >= 0 && i < OMPX_PIPE_AFTER_MAX && q->after.kernel == kernel) {
        ompx_pipe_after_put& e = q->after.e[i];
        sent = e.armed && __atomic_load_n(&e.handled, __ATOMIC_ACQUIRE);
        if (!sent && e.dyn_cnt != nullptr)
            require_cuda(cudaMemset(e.dyn_cnt, 0, OMPX_DYN_CNT_WORDS * sizeof(unsigned)),
                         "clear the run-time counters");
        e.armed = 0;
        e.handled = 0;
    }
    if (!sent) ompx_put_signal_host(peer, dst, src, bytes, sig, value);
}

void ompx__after_done(unsigned long long kernel, int i, int peer, void* dst,
                      const void* src, size_t bytes) {
    pipe_check();
    ompx_pipe_deferred* q = g_pipe_deferred_host;
    bool sent = false;
    if (q != nullptr && i >= 0 && i < OMPX_PIPE_AFTER_MAX && q->after.kernel == kernel) {
        ompx_pipe_after_put& e = q->after.e[i];
        sent = e.armed && __atomic_load_n(&e.handled, __ATOMIC_ACQUIRE);
        if (!sent && e.dyn_cnt != nullptr)
            require_cuda(cudaMemset(e.dyn_cnt, 0, OMPX_DYN_CNT_WORDS * sizeof(unsigned)),
                         "clear the run-time counters");
        e.armed = 0;
        e.handled = 0;
    }
    if (!sent) ompx_put_host(peer, dst, src, bytes);
}

// After the launch of a kernel with an in-loop put the host posted as put
// i: the kernel has sent it. A post the kernel found wrong stops here.
void ompx__loop_done(unsigned long long kernel, int i) {
    pipe_check();
    ompx_pipe_deferred* q = g_pipe_deferred_host;
    if (q == nullptr || i < 0 || i >= OMPX_PIPE_AFTER_MAX || q->after.kernel != kernel) return;
    q->after.e[i].armed = 0;
}

// A GPU-posted transfer starts when its thread rings the doorbell; there is
// no trigger to pull.
void ompx_trigger_host() {}

}  // extern "C"
