// RUN: %clang -std=c++17 -fplugin=%plugin -x hip -c %s -o /dev/null 2>&1 | FileCheck %s
// CHECK: [gicc-plugin] kernel: my_kernel_a
// CHECK: [gicc-plugin] kernel: my_kernel_b
#include <hip/hip_runtime.h>
__global__ void my_kernel_a() {}
__global__ void my_kernel_b(int) {}
void not_a_kernel() {}
