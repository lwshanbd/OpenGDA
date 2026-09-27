// A 32-bit inner loop with step 2: clang masks the radix it divides by, and
// the analysis cannot prove the mask a no-op.
void step(float* a, int n, int m) {
    #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(a)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; j += 2)
            a[i * m + j] = 1.0f;
}

// CHECK: [gicc-write] kernel {{.*}}
// CHECK-NEXT: when ((trunc i64 %1 to i32) > 0) ((trunc i64 %2 to i32) > 0)
// CHECK-NEXT: store to ptr not analysable: store address: sign extension of index arithmetic that may have wrapped (no nsw)
