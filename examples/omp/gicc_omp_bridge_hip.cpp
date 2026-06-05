// gicc_omp_bridge_hip.cpp - HIP-compiled implementation of the OpenMP<->GICC
// bridge. Owns a static gicc::Runtime so the OpenMP application TU never has to
// include ofi_runtime.hpp (which is not -fopenmp-includable).
#include <mpi.h>
#include <vector>

#include "gicc/platform/ofi/internal/gpu_device_context.hpp"
#include "gicc/platform/ofi/ofi_runtime.hpp"
#include "examples/omp/gicc_omp_bridge.hpp"

namespace {
gicc::Runtime* g_rt    = nullptr;
void*          g_buf   = nullptr;
size_t         g_bytes = 0;
int            g_bufidx = -1;
bool           g_we_init_mpi = false;
}  // namespace

namespace gicc_omp_bridge {

void init(size_t bytes) {
    int inited = 0; MPI_Initialized(&inited);
    if (!inited) { int argc = 0; char** argv = nullptr; MPI_Init(&argc, &argv); g_we_init_mpi = true; }
    g_rt = new gicc::Runtime();
    g_bytes = bytes;
    gpuMalloc(&g_buf, bytes);
    gpuMemset(g_buf, 0, bytes);
    auto bh = g_rt->register_buffer(g_buf, bytes, /*is_device=*/true);
    g_bufidx = bh.index;
    g_rt->exchange();
}

int rank()      { return g_rt->rank(); }
int nranks()    { int n = 0; MPI_Comm_size(MPI_COMM_WORLD, &n); return n; }
int buf_index() { return g_bufidx; }

gicc::DeviceCtx* prepare() { return g_rt->prepare(); }
void reset()               { g_rt->reset(); }
void barrier()             { MPI_Barrier(MPI_COMM_WORLD); }

void fill_buffer(unsigned char value, size_t bytes) {
    std::vector<unsigned char> h(bytes, value);
    gpuMemcpy(g_buf, h.data(), bytes, gpuMemcpyHostToDevice);
}

size_t count_mismatches(unsigned char expected, size_t bytes) {
    std::vector<unsigned char> h(bytes, 0);
    gpuMemcpy(h.data(), g_buf, bytes, gpuMemcpyDeviceToHost);
    size_t bad = 0;
    for (size_t i = 0; i < bytes; ++i) if (h[i] != expected) ++bad;
    return bad;
}

void finalize() {
    if (g_buf) { gpuFree(g_buf); g_buf = nullptr; }
    delete g_rt; g_rt = nullptr;
    int fin = 0; MPI_Finalized(&fin);
    if (!fin && g_we_init_mpi) MPI_Finalize();
}

}  // namespace gicc_omp_bridge
