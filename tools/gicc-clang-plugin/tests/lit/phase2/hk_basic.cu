// RUN: %clang -std=c++17 -fplugin=%plugin -fplugin-arg-gicc-debug-hk -x hip \
// RUN:   -I%gicc_src/src -DGICC_PLATFORM_OFI -DGICC_BOOTSTRAP_PMI2 \
// RUN:   -c %s -o /dev/null 2>&1 | FileCheck %s
// CHECK: [hk] kernel: k
// CHECK: [hk]   param a: HK
// CHECK: [hk]   param b: HK
// CHECK: [hk]   param dev: HK
// CHECK: [hk]   expr 'a + b': HK
// CHECK: [hk]   expr 'a + threadIdx.x': HK
// CHECK: [hk]   expr '*dev': NOT_HK
#include <hip/hip_runtime.h>
__global__ void k(int a, int b, int* dev) {
    int x = a + b;
    int y = a + threadIdx.x;
    int z = *dev;
    (void)x; (void)y; (void)z;
}
