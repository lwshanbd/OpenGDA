// GiOMP host runtime implementation. This translation unit is compiled as HIP
// and owns the OFI Runtime so OpenMP application translation units only need the
// public gicc/omp.h header and libgicc_omp.
#include <mpi.h>
#include <omp.h>

#include <cstdio>
#include <cstdlib>

#include "gicc/omp.h"
#include "gicc/omp_c.h"
#include "gicc/platform/ofi/internal/gpu_device_context.hpp"
#include "gicc/platform/ofi/ofi_runtime.hpp"
#include "gicc/platform/ofi/runtime_helpers.h"

namespace {

gicc::Runtime* g_runtime = nullptr;
bool g_initialized_mpi = false;
bool g_dwq_enabled = false;
bool g_ipc_enabled = false;
int g_local_device = 0;

void require_gpu(GpuError error, const char* operation) {
    if (error == GPU_SUCCESS) return;
    std::fprintf(stderr, "[giomp] %s failed: %s\n",
                 operation, gpuGetErrorString(error));
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

}  // namespace

void ompx_init() {
    if (g_runtime != nullptr) return;

    initialize_mpi_if_needed();
    select_local_device();
    g_runtime = new gicc::Runtime();

    const char* dwq_env = std::getenv("GICC_HALO_DWQ");
    g_dwq_enabled = dwq_env != nullptr && std::atoi(dwq_env) != 0;
    if (g_dwq_enabled) g_runtime->enable_host_wait_mode();

    // Keep libomptarget and HIP on the same rank-local GPU.
    omp_set_default_device(g_local_device);
}

void ompx_finalize() {
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

int omp_get_rank_num() {
    return g_runtime ? g_runtime->rank() : -1;
}

int omp_get_num_ranks() {
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

ompx_buffer ompx_alloc(size_t bytes) {
    void* ptr = nullptr;
    require_gpu(gpuMalloc(&ptr, bytes), "allocate device buffer");
    require_gpu(gpuMemset(ptr, 0, bytes), "initialize device buffer");
    auto buffer = g_runtime->register_buffer(ptr, bytes, /*is_device=*/true);
    return ompx_buffer{ptr, buffer.index, bytes};
}

int ompx_register(void* device_ptr, size_t bytes) {
    return g_runtime->register_buffer(device_ptr, bytes, /*is_device=*/true).index;
}

void ompx_free(ompx_buffer buffer) {
    if (buffer.ptr) require_gpu(gpuFree(buffer.ptr), "free device buffer");
}

void ompx_exchange() {
    g_runtime->exchange();
}

gicc::DeviceCtx* ompx_prepare() {
    return g_runtime->prepare();
}

void ompx_barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
}

void ompx_quiet_host() {
    g_runtime->reset();
}

bool ompx_dwq_enabled() {
    return g_dwq_enabled;
}

// Helpers used by the inline smart ompx_put in gicc/omp.h.
extern "C" int ompx_ipc_reachable(int peer, int buffer) {
    return g_runtime != nullptr && g_ipc_enabled
        && g_runtime->peer_mapped(peer, buffer) != nullptr;
}

extern "C" void* ompx_peer_ipc_base(int peer, int buffer) {
    return g_runtime ? g_runtime->peer_mapped(peer, buffer) : nullptr;
}

extern "C" void* ompx_local_base(int buffer) {
    if (!g_runtime) return nullptr;
    return g_runtime->local_buf_view(buffer).ptr;
}

extern "C" void ompx_dwq_stage(int peer, int dst_buffer, size_t dst_offset,
                                int src_buffer, size_t src_offset, size_t bytes) {
    gicc_runtime_dwq_enqueue(g_runtime, peer, dst_buffer, dst_offset,
                             src_buffer, src_offset, bytes);
}

extern "C" void ompx_dwq_stage_get_impl(int peer, int src_buffer,
                                         size_t src_offset, int dst_buffer,
                                         size_t dst_offset, size_t bytes) {
    if (!g_runtime) return;
    const gicc::Buffer& local_dst = g_runtime->buffer_by_lkey(dst_buffer);
    g_runtime->get(local_dst, peer, src_buffer, bytes,
                   /*local_offset=*/dst_offset,
                   /*remote_offset=*/src_offset);
}

extern "C" void ompx_dwq_arm() {
    gicc_runtime_arm_dwq_trigger(g_runtime);
}

// Used by compiler-synthesized OMP-DWQ host traces.
extern "C" void* gicc_runtime_current() {
    return g_runtime;
}

// C host API for C OpenMP applications such as Minimod.
extern "C" void giomp_init() { ompx_init(); }
extern "C" void giomp_finalize() { ompx_finalize(); }
extern "C" int giomp_rank() { return omp_get_rank_num(); }
extern "C" int giomp_size() { return omp_get_num_ranks(); }

extern "C" giomp_buffer giomp_alloc(size_t bytes) {
    ompx_buffer buffer = ompx_alloc(bytes);
    return giomp_buffer{buffer.ptr, buffer.index, buffer.bytes};
}

extern "C" void giomp_free(giomp_buffer buffer) {
    ompx_free(ompx_buffer{buffer.ptr, buffer.index, buffer.bytes});
}

extern "C" void giomp_memcpy_h2d(void* device_dst, const void* host_src,
                                  size_t bytes) {
    require_gpu(gpuMemcpy(device_dst, host_src, bytes, gpuMemcpyHostToDevice),
                "copy host buffer to device");
}

extern "C" void giomp_exchange() { ompx_exchange(); }
extern "C" void giomp_barrier() { ompx_barrier(); }
