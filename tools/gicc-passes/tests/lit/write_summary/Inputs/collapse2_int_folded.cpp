// collapse(2) whose index i*m + j the optimizer folds back into the IV.
void step(float* a, int n, int m) {
    #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(a)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            a[i * m + j] = 1.0f;
}

// CHECK: [gicc-write] kernel {{.*}}
// CHECK-NEXT: arg#3: 1-D box extent=[((sext i32 (trunc i64 %1 to i32) to i64) * (sext i32 (trunc i64 %2 to i32) to i64))] stride=[4] offset=0 (trusts loop-variable trunc)
