// RUN: %clang -std=c++17 -fplugin=%plugin -x hip -I%gicc_src/src \
// RUN:   -DGICC_PLATFORM_OFI -DGICC_BOOTSTRAP_PMI2 -c %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s
// CHECK: [gicc-plugin] launch_site -> kernel: my_kernel
#include "gicc/gicc.hpp"
__global__ void my_kernel(gicc::DeviceCtx*) {}
void caller(gicc::Runtime& rt) {
    gicc::launch<my_kernel>(rt, dim3(1), dim3(1));
}
