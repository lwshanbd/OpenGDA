// A division of the IV the user wrote, not a collapse: i / m walks a[0..k)
// only when m > 0, which nothing here establishes (m = -2, k = -2 passes the
// k*m > 0 precondition and writes a[0] and a[-1]). The radix m must be
// proven positive before a box is claimed.
typedef long long llint;
void step(float* a, llint k, llint m) {
    #pragma omp target teams distribute parallel for is_device_ptr(a)
    for (llint i = 0; i < k * m; ++i)
        a[i / m] = 1.0f;
}

// CHECK: [gicc-write] kernel {{.*}}
// CHECK-NEXT: when ((%1 * %2) > 0)
// CHECK-NEXT: store to ptr not analysable: store address: collapse radix %4 cannot be proven positive (a division the source wrote?)
