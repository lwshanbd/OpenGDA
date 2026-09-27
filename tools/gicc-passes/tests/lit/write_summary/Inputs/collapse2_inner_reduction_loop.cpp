// A sequential k loop accumulates in a register; the store after it does not
// move with k (matmul's shape) and runs on every iteration, even for n = 0.
#include <cstddef>
void step(float* c, const float* a, const float* b, int ns, int n) {
    #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(a, b, c)
    for (int i = 0; i < ns; ++i)
        for (int j = 0; j < ns; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k)
                sum += a[(size_t)i * n + k] * b[(size_t)k * ns + j];
            c[(size_t)i * ns + j] = sum;
        }
}

// CHECK: [gicc-write] kernel {{.*}}
// CHECK-NEXT: when ((trunc i64 %1 to i32) > 0)
// CHECK-NEXT: arg#5: 2-D box extent=[(sext i32 (trunc i64 %1 to i32) to i64), (sext i32 (trunc i64 %1 to i32) to i64)] stride=[(4 * (sext i32 (trunc i64 %1 to i32) to i64))<nsw>, 4] offset=0 (trusts loop-variable trunc)
