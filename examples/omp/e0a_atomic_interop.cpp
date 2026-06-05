// examples/omp/e0a_atomic_interop.cpp
// E0a: does an omp target region's system-scope atomic into host-pinned,
// GPU-mapped memory become visible to the CPU? Also exercises HIP + libomptarget
// coexistence (hipHostMalloc + omp target in one process, same device).
#include <hip/hip_runtime.h>
#include <omp.h>
#include <cstdint>
#include <cstdio>

#define N_TEAMS   64
#define N_THREADS 64
#define TOTAL     (N_TEAMS * N_THREADS)

int main() {
    // Same device for both runtimes.
    hipSetDevice(0);
    omp_set_default_device(0);

    // Host-pinned, GPU-mapped counter (the GICC ring memory model).
    uint64_t* host_ctr = nullptr;
    if (hipHostMalloc((void**)&host_ctr, sizeof(uint64_t), hipHostMallocMapped) != hipSuccess) {
        printf("FAIL: hipHostMalloc\n"); return 1;
    }
    *host_ctr = 0;
    uint64_t* dev_ctr = nullptr;
    if (hipHostGetDevicePointer((void**)&dev_ctr, host_ctr, 0) != hipSuccess) {
        printf("FAIL: hipHostGetDevicePointer\n"); return 1;
    }

    // TOGGLE for the TDD "red" step: when DO_ATOMIC==0 the region is a no-op
    // and the counter stays 0 (test FAILs). Set to 1 to implement.
#ifndef DO_ATOMIC
#define DO_ATOMIC 1
#endif

    // Note: __hip_atomic_fetch_add / __HIP_MEMORY_SCOPE_SYSTEM are not available
    // in the -fopenmp compilation context (they require HIP device-mode headers).
    // Fallback (a): portable C11 __atomic_fetch_add with __ATOMIC_SEQ_CST.
    // On AMDGCN, seq_cst maps to a system-scope flat atomic, which is visible
    // to the host CPU through the GPU-mapped pointer obtained via
    // hipHostGetDevicePointer on host-pinned memory.
    #pragma omp target teams distribute parallel for is_device_ptr(dev_ctr) \
            num_teams(N_TEAMS) thread_limit(N_THREADS)
    for (int i = 0; i < TOTAL; ++i) {
#if DO_ATOMIC
        __atomic_fetch_add(dev_ctr, (uint64_t)1, __ATOMIC_SEQ_CST);
#else
        (void)dev_ctr; (void)i;
#endif
    }

    uint64_t got = *host_ctr;   // CPU read of the mapped memory
    printf("counter=%llu expected=%d : %s\n",
           (unsigned long long)got, TOTAL, got == TOTAL ? "PASS" : "FAIL");
    hipHostFree(host_ctr);
    return got == TOTAL ? 0 : 2;
}
