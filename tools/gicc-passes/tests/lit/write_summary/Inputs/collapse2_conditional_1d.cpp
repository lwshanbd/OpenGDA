// A store under a data-dependent condition: its 1-D box is only a
// may-write set and must say so.
void step(float* a, const int* flag, int n) {
    #pragma omp target teams distribute parallel for is_device_ptr(a, flag)
    for (int i = 0; i < n; ++i)
        if (flag[i]) a[i] = 1.0f;
}

// CHECK: [gicc-write] kernel {{.*}}
// CHECK-NEXT: when ((trunc i64 %1 to i32) > 0)
// CHECK-NEXT: arg#3: 1-D box extent=[(1 + (sext i32 (-1 + (trunc i64 %1 to i32)) to i64))<nsw>] stride=[4] offset=0 (conditional)
