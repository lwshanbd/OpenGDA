// i % m over i < n: the digits are only a box when n is a multiple of m.
// With n = 3 and m = 10 the loop writes a[0..3), not a[0..10).
void step(float* a, int n, int m) {
    #pragma omp target teams distribute parallel for is_device_ptr(a)
    for (int i = 0; i < n; ++i)
        a[i % m] = 1.0f;
}

// CHECK: [gicc-write] kernel {{.*}}
// CHECK-NEXT: when ((trunc i64 %1 to i32) > 0)
// CHECK-NEXT: store to ptr not analysable: store address: collapse radix (sext i32 (trunc i64 %6 to i32) to i64) cannot be proven positive (a division the source wrote?)
