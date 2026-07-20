/*
 * fence_cost_micro.cpp - isolate the raw on-GPU cost of __threadfence_system()
 * vs __threadfence() vs none, on gfx90a, using clock64().
 *
 * Motivation: the proxy hot path does ONE __threadfence_system() per put inside
 * D2HRing::atomic_push (d2h_ring.cuh:127), to make the GPU's slot writes visible
 * to the CPU proxy thread reading the host-pinned mapped ring. We measured the
 * proxy is NIC-bound; this isolates whether the per-put device fence is itself a
 * meaningful cost (it would matter for high-rate small puts where push, not NIC,
 * could dominate if the kernel issued many concurrently).
 *
 * Single block, single thread (matches bench_pingpong's proxy_send_kernel which
 * launches <<<1,1>>>), N iterations, each writing a few fields to a host-mapped
 * slot then optionally fencing. clock64() deltas / N = per-iter cost in cycles;
 * gfx90a ~1.7 GHz so cycles/1.7 = ns.
 *
 * Run (single GPU, no MPI needed):
 *   srun -p pci -N 1 -n 1 -t 1 ./fence_cost_micro
 */
#include <cstdio>
#include <cstdint>
#include <hip/hip_runtime.h>

#define ITERS 100000

// fence_mode: 0 = none, 1 = __threadfence() (device/agent), 2 = __threadfence_system()
template <int FENCE>
__global__ void fence_loop(uint8_t* slot, uint64_t* out_cycles) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    // Warm up the path once.
    slot[0] = 1;
    long long t0 = clock64();
    for (int i = 0; i < ITERS; ++i) {
        // Mimic atomic_push's field writes (6 stores) then the publish store.
        slot[1] = (uint8_t)i;
        slot[2] = (uint8_t)(i >> 8);
        slot[3] = (uint8_t)(i >> 16);
        slot[4] = (uint8_t)(i >> 1);
        slot[5] = (uint8_t)(i >> 2);
        slot[6] = (uint8_t)(i >> 3);
        if (FENCE == 1) __threadfence();
        else if (FENCE == 2) __threadfence_system();
        slot[0] = (uint8_t)(i | 1);   // the "ready flag" publish store
    }
    long long t1 = clock64();
    *out_cycles = (uint64_t)(t1 - t0);
}

int main() {
    // Host-pinned, device-mapped slot — same memory class as the real ring
    // (so the system fence has real cross-device work to order).
    uint8_t* h_slot = nullptr;
    if (hipHostMalloc((void**)&h_slot, 64, hipHostMallocMapped) != hipSuccess) {
        fprintf(stderr, "hipHostMalloc failed\n"); return 1;
    }
    uint8_t* d_slot = nullptr;
    hipHostGetDevicePointer((void**)&d_slot, h_slot, 0);

    uint64_t* d_cyc = nullptr;
    hipMalloc((void**)&d_cyc, sizeof(uint64_t));

    // clock rate for ns conversion
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    double khz = prop.clockRate;              // kHz
    double ghz = khz / 1e6;

    const char* names[3] = {"none", "threadfence(device)", "threadfence_system"};
    uint64_t cyc[3] = {0,0,0};

    for (int mode = 0; mode < 3; ++mode) {
        uint64_t h = 0;
        hipMemcpy(d_cyc, &h, sizeof(h), hipMemcpyHostToDevice);
        if (mode == 0) hipLaunchKernelGGL((fence_loop<0>), dim3(1), dim3(1), 0, 0, d_slot, d_cyc);
        else if (mode == 1) hipLaunchKernelGGL((fence_loop<1>), dim3(1), dim3(1), 0, 0, d_slot, d_cyc);
        else hipLaunchKernelGGL((fence_loop<2>), dim3(1), dim3(1), 0, 0, d_slot, d_cyc);
        hipDeviceSynchronize();
        hipMemcpy(&cyc[mode], d_cyc, sizeof(h), hipMemcpyDeviceToHost);
    }

    printf("# gfx90a fence cost, %d iters, clock=%.3f GHz\n", ITERS, ghz);
    printf("# mode,total_cycles,cycles_per_iter,ns_per_iter\n");
    for (int m = 0; m < 3; ++m) {
        double cpi = (double)cyc[m] / ITERS;
        printf("%s,%llu,%.2f,%.3f\n", names[m],
               (unsigned long long)cyc[m], cpi, cpi / ghz);
    }
    // Net fence cost = (mode - none).
    double ns_dev = ((double)cyc[1]-cyc[0])/ITERS/ghz;
    double ns_sys = ((double)cyc[2]-cyc[0])/ITERS/ghz;
    printf("# net __threadfence() ns/op = %.3f\n", ns_dev);
    printf("# net __threadfence_system() ns/op = %.3f\n", ns_sys);

    hipFree(d_cyc);
    hipHostFree(h_slot);
    return 0;
}
