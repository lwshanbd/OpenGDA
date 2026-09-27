// OpenMP 5 non-rectangular collapse (inner bound depends on i): clang walks
// the bounding rectangle, with inner extent max(n, 1) - 1, and skips the
// points outside the triangle. That radix is 0 for n = 1, so it cannot be
// proven positive and the store is not described -- conservative: at best it
// would be a conditional box.
typedef long long llint;
void step(float* a, llint n) {
    #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(a)
    for (llint i = 0; i < n; ++i)
        for (llint j = 0; j < i; ++j)
            a[i * n + j] = 1.0f;
}

// CHECK: [gicc-write] kernel {{.*}}
// CHECK-NEXT: when (%1 > 0)
// CHECK-NEXT: store to ptr not analysable: store address: collapse radix (-1 + (1 smax %4))<nsw> cannot be proven positive (a division the source wrote?)
