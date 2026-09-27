// The loop runs only when n > 0 and m > 0; the folded extent n*m alone is
// positive for n = m = -1 too, so the box must carry the precondition.
typedef long long llint;
void step(float* a, llint n, llint m) {
    #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(a)
    for (llint i = 0; i < n; ++i)
        for (llint j = 0; j < m; ++j)
            a[i * m + j] = 1.0f;
}

// CHECK: [gicc-write] kernel {{.*}}
// CHECK-NEXT: when (%1 > 0) (%2 > 0)
// CHECK-NEXT: arg#3: 1-D box extent=[(%1 * %2)] stride=[4] offset=0
