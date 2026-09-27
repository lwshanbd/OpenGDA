// Parallel over (k, j) while every k updates the same C[i][j] through a
// sequential i loop: the store moves with the inner loop (and races).
#include <cstddef>
void step(float* C, const float* A, const float* B, int N, int Ns) {
    #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(A, B, C)
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < Ns; ++j)
            for (int i = 0; i < Ns; ++i)
                C[(size_t)i * N + j] += A[(size_t)i * N + k] * B[k * Ns + j];
}

// CHECK: [gicc-write] kernel {{.*}}
// CHECK-NEXT: store to ptr not analysable: store address moves with a sequential loop nested in the worksharing loop
// CHECK-NEXT: store to ptr not analysable: store address moves with a sequential loop nested in the worksharing loop
// CHECK-NEXT: store to ptr not analysable: store address moves with a sequential loop nested in the worksharing loop
// CHECK-NEXT: store to ptr not analysable: store address moves with a sequential loop nested in the worksharing loop
// CHECK-NEXT: store to ptr not analysable: store address moves with a sequential loop nested in the worksharing loop
