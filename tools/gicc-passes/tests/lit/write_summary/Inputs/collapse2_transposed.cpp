// Stores walk the array column-major: strides swap, still one box.
#include <cstddef>
void step(float* a, int n, int m) {
    #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(a)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            a[(size_t)j * n + i] = 1.0f;
}

// CHECK: [gicc-write] kernel {{.*}}
// CHECK-NEXT: arg#3: 2-D box extent=[(sext i32 (trunc i64 %1 to i32) to i64), (sext i32 (trunc i64 %2 to i32) to i64)] stride=[4, (4 * (sext i32 (trunc i64 %1 to i32) to i64))<nsw>] offset=0 (trusts loop-variable trunc)
