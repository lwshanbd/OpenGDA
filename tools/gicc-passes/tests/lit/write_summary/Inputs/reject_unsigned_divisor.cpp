// An unsigned divisor of 2^31 or more must not be read as negative.
void step(float* a, int n, unsigned m) {
    #pragma omp target teams distribute parallel for is_device_ptr(a)
    for (int i = 0; i < n; ++i)
        a[i % m] = 1.0f;
}

// CHECK: [gicc-write] kernel {{.*}}
// CHECK-NEXT: when ((trunc i64 %1 to i32) > 0)
// CHECK-NEXT: store to ptr not analysable: store address: collapse radix (zext i32 (trunc i64 %6 to i32) to i64) cannot be proven positive (a division the source wrote?)
