// gicc_omp_bridge_hip.cpp - HIP-compiled implementation of the OpenMP<->GICC
// bridge. Owns a static gicc::Runtime so the OpenMP application TU never has to
// include ofi_runtime.hpp (which is not -fopenmp-includable).
#include <mpi.h>
#include <vector>
#include <cstdlib>

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
    // DWQ path (Phase-2 M4): when the CXI trigger BAR is mapped (i.e.
    // GICC_SKIP_DWQ_INIT is NOT set / GICC_PROXY_ENABLED=0), switch the
    // Runtime to host-wait (GDA-style) mode. This (a) disables the device
    // CPU-proxy dispatch so a put marker does not double-deliver, and (b)
    // makes prepare()/reset() use the shared completion counter + delta
    // trigger semantics the DWQ flush MMIO store relies on. The compiler-
    // synthesized host trace pre-stages the DWQ descriptors via
    // gicc_runtime_dwq_enqueue before the omp target region launches.
    const char *skip = std::getenv("GICC_SKIP_DWQ_INIT");
    const char *proxy = std::getenv("GICC_PROXY_ENABLED");
    const bool dwq_mode = (skip == nullptr) &&
                          (proxy == nullptr || std::atoi(proxy) == 0);
    if (dwq_mode) {
        g_rt->enable_host_wait_mode();
    }
    g_bytes = bytes;
    gpuMalloc(&g_buf, bytes);
    gpuMemset(g_buf, 0, bytes);
    auto bh = g_rt->register_buffer(g_buf, bytes, /*is_device=*/true);
    g_bufidx = bh.index;
    g_rt->exchange();
}

// C ABI the compiler-synthesized host trace calls to obtain the live Runtime.
// g_rt is set by init() above before any omp target region runs, so the trace
// function always sees a fully-constructed gicc::Runtime here.
extern "C" void *gicc_runtime_current() { return g_rt; }

int rank()         { return g_rt->rank(); }
int nranks()       { int n = 0; MPI_Comm_size(MPI_COMM_WORLD, &n); return n; }
int buf_index()    { return g_bufidx; }
void* device_buffer() { return g_buf; }

gicc::DeviceCtx* prepare() { return g_rt->prepare(); }
void reset()               { g_rt->reset(); }
void barrier()             { MPI_Barrier(MPI_COMM_WORLD); }
double wtime()             { return MPI_Wtime(); }

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

void fill_region(size_t offset, unsigned char value, size_t len) {
    std::vector<unsigned char> h(len, value);
    gpuMemcpy(static_cast<char*>(g_buf) + offset, h.data(), len, gpuMemcpyHostToDevice);
}

size_t count_region_mismatches(size_t offset, unsigned char expected, size_t len) {
    std::vector<unsigned char> h(len, 0);
    gpuMemcpy(h.data(), static_cast<char*>(g_buf) + offset, len, gpuMemcpyDeviceToHost);
    size_t bad = 0;
    for (size_t i = 0; i < len; ++i) if (h[i] != expected) ++bad;
    return bad;
}

void finalize() {
    if (g_buf) { gpuFree(g_buf); g_buf = nullptr; }
    delete g_rt; g_rt = nullptr;
    int fin = 0; MPI_Finalized(&fin);
    if (!fin && g_we_init_mpi) MPI_Finalize();
}

}  // namespace gicc_omp_bridge
