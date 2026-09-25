// gda_test.cu - GPU-driven InfiniBand transfers from CUDA kernels, checked
// word for word.
//
// Every rank sends to its right neighbour and receives from its left one.
//   put         K blocks each fill a chunk and put it; quiet; the receiver
//               checks after a barrier.
//   put_signal  K sender blocks put a chunk with signal slot k; K receiver
//               blocks wait for their slot and check their chunk on the
//               spot, so a signal that overtook its payload shows up.
//   get         K blocks each fetch a chunk from the left neighbour.
//   host        the same put / put_u64 / get issued from the host.
//   pingpong    device put_signal round trip between ranks 0 and 1.
//
// Run: srun -p maple -N2 -n2 --gres=gpu:1 --mpi=pmix ./gicc_gda_test
//      (GICC_IB_LANES=N gives each chunk's block its own QP when N >= chunks)

#include "gicc/gicc.hpp"
#include "gicc/gicc_device.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    gicc::abort(1, #x); } } while (0)

using gicc::mlx5::GdaCtx;

__device__ __host__ inline uint32_t pattern(int it, int rank, int chunk, size_t i) {
    return (uint32_t)it * 0x9E3779B1u ^ (uint32_t)rank * 0x85EBCA77u ^
           (uint32_t)chunk * 0xC2B2AE3Du ^ (uint32_t)i * 0x27D4EB2Fu;
}

__global__ void fill(uint32_t* buf, size_t words, int it, int rank) {
    const int k = blockIdx.y;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words;
         i += (size_t)gridDim.x * blockDim.x)
        buf[k * words + i] = pattern(it, rank, k, i);
}

__global__ void check(const uint32_t* buf, size_t words, int it, int rank,
                      unsigned long long* bad) {
    const int k = blockIdx.y;
    unsigned long long n = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words;
         i += (size_t)gridDim.x * blockDim.x)
        n += buf[k * words + i] != pattern(it, rank, k, i);
    if (n) atomicAdd(bad, n);
}

__global__ void put_kernel(GdaCtx* ctx, int peer, int dbuf, int sbuf,
                           uint32_t* src, size_t words, int it, int rank) {
    const int k = blockIdx.x;
    for (size_t i = threadIdx.x; i < words; i += blockDim.x)
        src[k * words + i] = pattern(it, rank, k, i);
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        const size_t off = (size_t)k * words * 4;
        gicc::put(ctx, peer, dbuf, off, sbuf, off, words * 4, k);
        gicc::quiet(ctx, k);
    }
}

__global__ void signal_kernel(GdaCtx* ctx, int right, int left, int dbuf, int sbuf,
                              uint32_t* src, const uint32_t* dst, size_t words,
                              int K, int it, int rank, unsigned long long* bad) {
    const uint64_t v = (uint64_t)it + 1;
    if ((int)blockIdx.x < K) {
        const int k = blockIdx.x;
        // Later chunks finish first, so the flags go out of slot order.
        if (threadIdx.x == 0) {
            const long long until = clock64() + (long long)(K - 1 - k) * 20000;
            while (clock64() < until) {}
        }
        __syncthreads();
        for (size_t i = threadIdx.x; i < words; i += blockDim.x)
            src[k * words + i] = pattern(it, rank, k, i);
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
            const size_t off = (size_t)k * words * 4;
            gicc::put_signal(ctx, right, dbuf, off, sbuf, off, words * 4, k, v, k);
        }
    } else {
        const int k = blockIdx.x - K;
        if (threadIdx.x == 0) gicc::signal_wait(ctx, k, v);
        __syncthreads();
        unsigned long long n = 0;
        for (size_t i = threadIdx.x; i < words; i += blockDim.x)
            n += dst[k * words + i] != pattern(it, left, k, i);
        if (n) atomicAdd(bad, n);
    }
    if ((int)blockIdx.x < K && threadIdx.x == 0) gicc::quiet(ctx, blockIdx.x);
}

__global__ void get_kernel(GdaCtx* ctx, int left, int dbuf, int sbuf,
                           const uint32_t* dst, size_t words, int it,
                           unsigned long long* bad) {
    const int k = blockIdx.x;
    if (threadIdx.x == 0) {
        const size_t off = (size_t)k * words * 4;
        gicc::get(ctx, left, sbuf, off, dbuf, off, words * 4, k);
        gicc::quiet(ctx, k);
    }
    __syncthreads();
    unsigned long long n = 0;
    for (size_t i = threadIdx.x; i < words; i += blockDim.x)
        n += dst[k * words + i] != pattern(it, left, k, i);
    if (n) atomicAdd(bad, n);
}

__global__ void pingpong_kernel(GdaCtx* ctx, int me, int peer, int dbuf, int sbuf,
                                size_t bytes, int iters, unsigned long long* ns) {
    const unsigned long long t0 = clock64();
    uint64_t t_start;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_start));
    for (int i = 1; i <= iters; ++i) {
        if (me == 0) {
            gicc::put_signal(ctx, peer, dbuf, 0, sbuf, 0, bytes, 0, i);
            gicc::signal_wait(ctx, 0, i);
        } else {
            gicc::signal_wait(ctx, 0, i);
            gicc::put_signal(ctx, peer, dbuf, 0, sbuf, 0, bytes, 0, i);
        }
    }
    gicc::quiet(ctx);
    uint64_t t_end;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_end));
    (void)t0;
    *ns = t_end - t_start;
}

int main(int argc, char** argv) {
    int iters = 20, K = 8;
    for (int i = 1; i < argc; ++i) {
        if (!std::strncmp(argv[i], "--iters=", 8)) iters = std::atoi(argv[i] + 8);
        if (!std::strncmp(argv[i], "--chunks=", 9)) K = std::atoi(argv[i] + 9);
    }

    gicc::Runtime rt;
    const int rank = rt.rank(), n = rt.size();
    const int right = (rank + 1) % n, left = (rank + n - 1) % n;

    const size_t kMaxChunk = 4u << 20;
    uint32_t *src, *dst;
    uint64_t* sig;
    unsigned long long* bad;
    CK(cudaMalloc(&src, kMaxChunk * K));
    CK(cudaMalloc(&dst, kMaxChunk * K));
    CK(cudaMalloc(&sig, 64 * sizeof(uint64_t)));
    CK(cudaMemset(sig, 0, 64 * sizeof(uint64_t)));
    CK(cudaMallocManaged(&bad, sizeof(*bad)));
    auto bsrc = rt.register_buffer(src, kMaxChunk * K, true);
    auto bdst = rt.register_buffer(dst, kMaxChunk * K, true);
    auto bsig = rt.register_buffer(sig, 64 * sizeof(uint64_t), true);
    rt.exchange();
    rt.set_signal_table(sig, bsig.index);
    GdaCtx* ctx = rt.prepare();

    if (rank == 0)
        std::printf("%d ranks, %d chunks, %d iters, lanes %s\n", n, K, iters,
                    std::getenv("GICC_IB_LANES") ? std::getenv("GICC_IB_LANES") : "1");
    std::printf("%-6s %-10s %10s %12s %10s\n", "rank", "test", "chunk", "bad", "us/iter");

    unsigned long long total = 0;
    int it = 0;
    const size_t sizes[] = {8, 4096, 65536, 1 << 20, 4 << 20};
    for (size_t cb : sizes) {
        const size_t words = cb / 4 ? cb / 4 : 1;
        const size_t bytes = words * 4;

        // --- device put ------------------------------------------------------
        *bad = 0;
        rt.barrier();
        double t0 = MPI_Wtime();
        for (int r = 0; r < iters; ++r, ++it) {
            put_kernel<<<K, 256>>>(ctx, right, bdst.index, bsrc.index, src, words, it, rank);
            CK(cudaDeviceSynchronize());
            rt.barrier();
            check<<<dim3(4, K), 256>>>(dst, words, it, left, bad);
            CK(cudaDeviceSynchronize());
            rt.barrier();
        }
        std::printf("%-6d %-10s %10zu %12llu %10.1f\n", rank, "put", bytes, *bad,
                    (MPI_Wtime() - t0) * 1e6 / iters);
        total += *bad;

        // --- device put_signal ----------------------------------------------
        *bad = 0;
        CK(cudaMemset(sig, 0, 64 * sizeof(uint64_t)));
        rt.barrier();
        t0 = MPI_Wtime();
        for (int r = 0; r < iters; ++r, ++it) {
            signal_kernel<<<2 * K, 256>>>(ctx, right, left, bdst.index, bsrc.index,
                                          src, dst, words, K, it, rank, bad);
            CK(cudaDeviceSynchronize());
            rt.barrier();
        }
        std::printf("%-6d %-10s %10zu %12llu %10.1f\n", rank, "signal", bytes, *bad,
                    (MPI_Wtime() - t0) * 1e6 / iters);
        total += *bad;

        // --- device get -------------------------------------------------------
        *bad = 0;
        t0 = MPI_Wtime();
        for (int r = 0; r < iters; ++r, ++it) {
            fill<<<dim3(4, K), 256>>>(src, words, it, rank);
            CK(cudaDeviceSynchronize());
            rt.barrier();
            get_kernel<<<K, 256>>>(ctx, left, bdst.index, bsrc.index, dst, words, it, bad);
            CK(cudaDeviceSynchronize());
            rt.barrier();
        }
        std::printf("%-6d %-10s %10zu %12llu %10.1f\n", rank, "get", bytes, *bad,
                    (MPI_Wtime() - t0) * 1e6 / iters);
        total += *bad;

        // --- host put + put_u64 + get ----------------------------------------
        *bad = 0;
        t0 = MPI_Wtime();
        for (int r = 0; r < iters; ++r, ++it) {
            fill<<<dim3(4, K), 256>>>(src, words, it, rank);
            CK(cudaDeviceSynchronize());
            for (int k = 0; k < K; ++k)
                rt.put(bsrc, right, bdst.index, bytes, k * bytes, k * bytes);
            rt.put_u64(right, bsig.index, 63 * sizeof(uint64_t), (uint64_t)it + 1);
            rt.drain();
            rt.barrier();
            check<<<dim3(4, K), 256>>>(dst, words, it, left, bad);
            CK(cudaDeviceSynchronize());
            uint64_t s = 0;
            CK(cudaMemcpy(&s, sig + 63, sizeof(s), cudaMemcpyDeviceToHost));
            if (s != (uint64_t)it + 1) ++*bad;
            CK(cudaMemset(dst, 0, bytes * K));
            rt.barrier();
            for (int k = 0; k < K; ++k)
                rt.get(bdst, left, bsrc.index, bytes, k * bytes, k * bytes);
            rt.drain();
            check<<<dim3(4, K), 256>>>(dst, words, it, left, bad);
            CK(cudaDeviceSynchronize());
            rt.barrier();
        }
        std::printf("%-6d %-10s %10zu %12llu %10.1f\n", rank, "host", bytes, *bad,
                    (MPI_Wtime() - t0) * 1e6 / iters);
        total += *bad;
    }

    // --- device ping-pong ----------------------------------------------------
    if (n >= 2) {
        for (size_t bytes : {(size_t)8, (size_t)4096, (size_t)65536}) {
            CK(cudaMemset(sig, 0, 64 * sizeof(uint64_t)));
            rt.barrier();
            unsigned long long* ns;
            CK(cudaMallocManaged(&ns, sizeof(*ns)));
            const int pp = 1000;
            if (rank < 2) {
                pingpong_kernel<<<1, 1>>>(ctx, rank, 1 - rank, bdst.index, bsrc.index,
                                          bytes, pp, ns);
                CK(cudaDeviceSynchronize());
                if (rank == 0)
                    std::printf("pingpong %8zu B: %.2f us half round trip\n", bytes,
                                *ns / 1e3 / pp / 2);
            }
            CK(cudaFree(ns));
            rt.barrier();
        }
    }

    rt.barrier();
    std::printf("rank %d: %s (%llu bad words)\n", rank, total ? "FAIL" : "PASS", total);
    rt.drain();
    return total ? 1 : 0;
}
