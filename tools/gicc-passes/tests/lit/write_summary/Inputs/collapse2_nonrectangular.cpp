// OpenMP 5 non-rectangular collapse (inner bound depends on i): clang walks
// the bounding rectangle and skips the points outside the triangle, so the
// store is a conditional (may-write) box over the rectangle.
typedef long long llint;
void step(float* a, llint n) {
    #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(a)
    for (llint i = 0; i < n; ++i)
        for (llint j = 0; j < i; ++j)
            a[i * n + j] = 1.0f;
}

// CHECK: [gicc-write] kernel {{.*}}
// CHECK-NEXT: arg#2: 2-D box extent=[%1, (-1 + (1 smax %1))<nsw>] stride=[(4 * %1), 4] offset=0 (conditional)
