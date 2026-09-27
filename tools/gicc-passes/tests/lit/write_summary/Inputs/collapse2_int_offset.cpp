// collapse(2) over int loops starting at 1, rows of C (jacobi's shape): the
// radix is divided sign-extended and multiplied back zero-extended.
#include <cstddef>
void step(float* pb, const float* pa, int R, int C) {
    #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(pa, pb)
    for (int i = 1; i <= R; ++i)
        for (int j = 0; j < C; ++j)
            pb[(size_t)i * C + j] = 0.5f * (pa[(size_t)(i - 1) * C + j] + pa[(size_t)(i + 1) * C + j]);
}

// CHECK: [gicc-write] kernel {{.*}}
// CHECK-NEXT: arg#3: 2-D box extent=[(sext i32 (trunc i64 %1 to i32) to i64), (sext i32 (trunc i64 %2 to i32) to i64)] stride=[(4 * (sext i32 (trunc i64 %2 to i32) to i64))<nsw>, 4] offset=(4 * (sext i32 (trunc i64 %2 to i32) to i64))<nsw> (trusts loop-variable trunc)
