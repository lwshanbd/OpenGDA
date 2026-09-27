// An unsigned inner bound: the loop is guarded by W != 0, not W > 0, and
// that is enough for a trip count (put_signal's shape).
#include <cstddef>
void step(unsigned* sbuf, int K, size_t W) {
    #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(sbuf)
    for (int k = 0; k < K; ++k)
        for (size_t i = 0; i < W; ++i)
            sbuf[k * W + i] = (unsigned)(k + i);
}

// CHECK: [gicc-write] kernel {{.*}}
// CHECK-NEXT: when ((trunc i64 %1 to i32) > 0) (%2 != 0)
// CHECK-NEXT: arg#3: 2-D box extent=[(sext i32 (trunc i64 %1 to i32) to i64), %2] stride=[(4 * %2), 4] offset=0 (trusts loop-variable trunc)
