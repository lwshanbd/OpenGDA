/*
 * jacobi_bench.cpp - Distributed Jacobi solver, GICC DWQ vs Cray MPI.
 *
 * A CUDA/manual-API sibling of examples/ofi/jacobi.cpp. That version drives
 * the halo puts from inside the kernel via gicc::put_no_db, which only does
 * real work once the LTO pass rewrites it. This one uses the host-side
 * Runtime API directly, so it runs on a plain nvcc build:
 *
 *     rt.put(...)              host stages a deferred RMA write per halo face
 *     rt.prepare()             publishes trigger addr + threshold
 *     kernel: compute + flush  one MMIO store fires every staged write
 *     rt.reset()               host waits on the completion counter
 *
 * Domain is decomposed in 1D over rows. Each rank owns `local_ny` interior
 * rows plus a halo row above and below:
 *
 *     row 0              <- top halo    (filled by rank-1's last interior row)
 *     rows 1..local_ny   <- interior    (owned)
 *     row local_ny+1     <- bottom halo (filled by rank+1's first interior row)
 *
 * --mode=mpi runs the identical compute kernel with MPI_Sendrecv halo
 * exchange, so the difference between the two columns is the halo transport
 * and nothing else.
 *
 * Run:
 *   srun -N2 -n2 --ntasks-per-node=1 --cpus-per-task=16 --gpus-per-node=1 \
 *        ./jacobi_bench --nx 4096 --ny 4096 --niter 200 --mode=gicc
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

#include <mpi.h>

#include "gicc/platform/ofi/internal/gpu_device_context.hpp"
#include "gicc/platform/ofi/ofi_runtime.hpp"
#include "gicc/platform/ofi/ofi_device.cuh"

#define GPU_CHECK(cmd)                                                        \
    do {                                                                      \
        GpuError _e = (cmd);                                                  \
        if (_e != GPU_SUCCESS) {                                              \
            fprintf(stderr, "GPU error %s at %s:%d\n",                        \
                    gpuGetErrorString(_e), __FILE__, __LINE__);               \
            MPI_Abort(MPI_COMM_WORLD, 1);                                     \
        }                                                                     \
    } while (0)

// Jacobi stencil over the interior rows. Halo rows are read but never written.
__global__ void jacobi_kernel(const float* __restrict__ u,
                              float* __restrict__ unew,
                              int nx, int local_ny) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;   // column
    int i = blockIdx.y * blockDim.y + threadIdx.y;   // interior row, 0-based
    if (j < 1 || j >= nx - 1 || i >= local_ny) return;
    int r = i + 1;                                   // skip the top halo row
    size_t c = (size_t)r * nx + j;
    unew[c] = 0.25f * (u[c - nx] + u[c + nx] + u[c - 1] + u[c + 1]);
}

// Trigger-only kernel. The halo writes are staged against the buffer the
// stencil is PRODUCING and land in the peer's copy of that same buffer, so
// the grid reads `u` while the NIC writes `unew` and the transfer cannot
// race this iteration's arithmetic -- no "exchange, then compute" ordering
// is needed, unlike the MPI path.
//
// The trigger still must not fire until every halo row is written. Electing
// a last block with a grid-wide atomic counter (as examples/gicc/jacobi.cu
// does) costs one atomic per block on a single address; at 8192^2 that is
// 131072 blocks contending on one cache line and it dominates the iteration.
// Issuing the trigger as its own one-thread kernel on the same stream gets
// the same ordering from stream semantics for the price of one launch, and
// still needs no host round trip between the two.
__global__ void trigger_only_kernel(gicc::DeviceCtx* ctx) {
    gicc::flush(ctx);
}

// Fixed Dirichlet values on the global boundary, seeded interior.
__global__ void init_kernel(float* u, int nx, int local_ny,
                            int row_offset, int global_ny) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= nx || r >= local_ny + 2) return;
    int gr = row_offset + r - 1;                     // global interior row
    float v = 0.0f;
    if (j == 0 || j == nx - 1 || gr <= 0 || gr >= global_ny - 1) v = 1.0f;
    u[(size_t)r * nx + j] = v;
}

// Checksum over interior rows only, so halo state never perturbs the compare.
__global__ void sum_kernel(const float* u, int nx, int local_ny, double* out) {
    __shared__ double s[256];
    int t = threadIdx.x;
    double acc = 0.0;
    size_t n = (size_t)local_ny * nx;
    for (size_t k = t + (size_t)blockIdx.x * blockDim.x; k < n;
         k += (size_t)blockDim.x * gridDim.x) {
        size_t r = k / nx, j = k % nx;
        acc += u[(r + 1) * nx + j];
    }
    s[t] = acc;
    __syncthreads();
    for (int d = blockDim.x / 2; d > 0; d >>= 1) {
        if (t < d) s[t] += s[t + d];
        __syncthreads();
    }
    if (t == 0) atomicAdd(out, s[0]);
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);

    int nx = 4096, global_ny = 4096, niter = 200;
    std::string mode = "gicc";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--nx" && i + 1 < argc)          nx = atoi(argv[++i]);
        else if (a == "--ny" && i + 1 < argc)     global_ny = atoi(argv[++i]);
        else if (a == "--niter" && i + 1 < argc)  niter = atoi(argv[++i]);
        else if (a.rfind("--mode=", 0) == 0)      mode = a.substr(7);
    }
    if (mode != "gicc" && mode != "mpi") {
        fprintf(stderr, "jacobi_bench: --mode must be 'gicc' or 'mpi'\n");
        return 2;
    }

    gicc::Runtime rt;
    if (mode == "gicc") rt.enable_host_wait_mode();
    int rank = rt.rank(), nranks = rt.size();

    if (global_ny % nranks != 0) {
        if (rank == 0)
            fprintf(stderr, "jacobi_bench: --ny (%d) must divide by ranks (%d)\n",
                    global_ny, nranks);
        return 2;
    }
    int local_ny = global_ny / nranks;
    int row_offset = rank * local_ny;
    size_t row_bytes = (size_t)nx * sizeof(float);
    size_t buf_bytes = (size_t)(local_ny + 2) * row_bytes;

    // Two grids, swapped each iteration. Both are registered so a put can
    // name whichever one holds the current iteration's data.
    float *d_u = nullptr, *d_unew = nullptr;
    GPU_CHECK(gpuMalloc((void**)&d_u, buf_bytes));
    GPU_CHECK(gpuMalloc((void**)&d_unew, buf_bytes));

    auto bh_u    = rt.register_buffer(d_u, buf_bytes, /*is_device=*/true);
    auto bh_unew = rt.register_buffer(d_unew, buf_bytes, /*is_device=*/true);
    rt.exchange();

    dim3 tb(32, 8);
    dim3 gi((nx + tb.x - 1) / tb.x, (local_ny + 2 + tb.y - 1) / tb.y);
    gpuLaunchKernel(init_kernel, gi, tb, 0, 0, d_u, nx, local_ny,
                    row_offset, global_ny);
    gpuLaunchKernel(init_kernel, gi, tb, 0, 0, d_unew, nx, local_ny,
                    row_offset, global_ny);
    GPU_CHECK(gpuDeviceSynchronize());

    int up = rank - 1, down = rank + 1;
    bool has_up = up >= 0, has_down = down < nranks;

    // Halo offsets within a grid buffer.
    const size_t off_first_interior = row_bytes;                       // row 1
    const size_t off_last_interior  = (size_t)local_ny * row_bytes;    // row local_ny
    const size_t off_top_halo       = 0;                               // row 0
    const size_t off_bottom_halo    = (size_t)(local_ny + 1) * row_bytes;

    dim3 gc((nx + tb.x - 1) / tb.x, (local_ny + tb.y - 1) / tb.y);

    // Force lazy DWQ init out of the timing window.
    if (mode == "gicc") { (void)rt.prepare(); rt.reset(); }
    rt.barrier();

    // Prime d_u's halos. The fused kernel ships halos AFTER computing, so
    // iteration 0 needs its incoming rows already in place to match what MPI
    // mode sees (it exchanges before every compute, including the first).
    // One-off, so it sits outside the timing window.
    if (mode == "gicc") {
        if (has_up)
            rt.put(bh_u, up, bh_u.index, row_bytes,
                   off_first_interior, off_bottom_halo);
        if (has_down)
            rt.put(bh_u, down, bh_u.index, row_bytes,
                   off_last_interior, off_top_halo);
        gicc::DeviceCtx* ctx = rt.prepare();
        gpuLaunchKernel(trigger_only_kernel, dim3(1), dim3(1), 0, 0, ctx);
        GPU_CHECK(gpuDeviceSynchronize());
        rt.reset();
        rt.barrier();
    }

    double t_comm = 0.0, t_stage = 0.0, t_reset = 0.0, t_bar = 0.0;
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int it = 0; it < niter; ++it) {
        gicc::Buffer& src = (it % 2 == 0) ? bh_u : bh_unew;
        gicc::Buffer& dst = (it % 2 == 0) ? bh_unew : bh_u;
        float* d_src = (it % 2 == 0) ? d_u : d_unew;
        float* d_dst = (it % 2 == 0) ? d_unew : d_u;
        (void)src;

        if (mode == "gicc") {
            // Stage both halo faces against the buffer this iteration is
            // WRITING. My first interior row lands in the up neighbour's
            // bottom halo; my last lands in the down neighbour's top halo.
            // Buffer indices match across ranks (identical registration
            // order), so dst.index names the peer's twin.
            double c0 = MPI_Wtime();
            if (has_up)
                rt.put(dst, up, dst.index, row_bytes,
                       off_first_interior, off_bottom_halo);
            if (has_down)
                rt.put(dst, down, dst.index, row_bytes,
                       off_last_interior, off_top_halo);
            gicc::DeviceCtx* ctx = rt.prepare();
            t_stage += MPI_Wtime() - c0;
            t_comm  += MPI_Wtime() - c0;

            // Both on the default stream: the trigger kernel cannot start
            // until the stencil has fully retired, so the NIC never reads a
            // half-written halo row. One sync covers both.
            gpuLaunchKernel(jacobi_kernel, gc, tb, 0, 0,
                            d_src, d_dst, nx, local_ny);
            gpuLaunchKernel(trigger_only_kernel, dim3(1), dim3(1), 0, 0, ctx);
            GPU_CHECK(gpuDeviceSynchronize());

            // reset() drains my outgoing halos; the barrier is what makes my
            // neighbours' halos visible before the next iteration reads them.
            double c1 = MPI_Wtime();
            rt.reset();
            double c2 = MPI_Wtime();
            rt.barrier();
            double c3 = MPI_Wtime();
            t_reset += c2 - c1;
            t_bar   += c3 - c2;
            t_comm  += c3 - c1;
        } else {
            double c0 = MPI_Wtime();
            MPI_Request req[4];
            int nreq = 0;
            if (has_up) {
                MPI_Irecv((char*)d_src + off_top_halo, (int)row_bytes, MPI_BYTE,
                          up, 0, MPI_COMM_WORLD, &req[nreq++]);
                MPI_Isend((char*)d_src + off_first_interior, (int)row_bytes,
                          MPI_BYTE, up, 1, MPI_COMM_WORLD, &req[nreq++]);
            }
            if (has_down) {
                MPI_Irecv((char*)d_src + off_bottom_halo, (int)row_bytes,
                          MPI_BYTE, down, 1, MPI_COMM_WORLD, &req[nreq++]);
                MPI_Isend((char*)d_src + off_last_interior, (int)row_bytes,
                          MPI_BYTE, down, 0, MPI_COMM_WORLD, &req[nreq++]);
            }
            if (nreq) MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
            t_comm += MPI_Wtime() - c0;

            gpuLaunchKernel(jacobi_kernel, gc, tb, 0, 0,
                            d_src, d_dst, nx, local_ny);
            GPU_CHECK(gpuDeviceSynchronize());
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    // Checksum the final grid so the two modes can be compared numerically.
    float* d_final = (niter % 2 == 0) ? d_u : d_unew;
    double* d_sum = nullptr;
    GPU_CHECK(gpuMalloc((void**)&d_sum, sizeof(double)));
    GPU_CHECK(gpuMemset(d_sum, 0, sizeof(double)));
    gpuLaunchKernel(sum_kernel, dim3(256), dim3(256), 0, 0,
                    d_final, nx, local_ny, d_sum);
    GPU_CHECK(gpuDeviceSynchronize());
    double h_sum = 0.0;
    GPU_CHECK(gpuMemcpy(&h_sum, d_sum, sizeof(double), gpuMemcpyDeviceToHost));
    double total_sum = 0.0;
    MPI_Reduce(&h_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double max_comm = 0.0;
    MPI_Reduce(&t_comm, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double total_ms = (t1 - t0) * 1e3;
        printf("\n=== jacobi_bench (mode=%s) ===\n", mode.c_str());
        printf("ranks=%d  grid=%dx%d  local_ny=%d  niter=%d  halo=%zu B\n",
               nranks, nx, global_ny, local_ny, niter, row_bytes);
        printf("total       %10.3f ms   (%.4f ms/iter)\n",
               total_ms, total_ms / niter);
        printf("halo comm   %10.3f ms   (%.4f ms/iter, %.1f%% of total)\n",
               max_comm * 1e3, max_comm * 1e3 / niter,
               100.0 * max_comm / (t1 - t0));
        printf("checksum    %.6e\n", total_sum);
        if (mode == "gicc")
            printf("  breakdown stage=%.3f ms  reset=%.3f ms  barrier=%.3f ms\n",
                   t_stage * 1e3, t_reset * 1e3, t_bar * 1e3);
    }

    (void)gpuFree(d_sum);
    (void)gpuFree(d_u);
    (void)gpuFree(d_unew);
    return 0;
}
