// collapse(4): the decomposition is not limited to three digits.
typedef long long llint;
void step(float* a, llint n0, llint n1, llint n2, llint n3) {
    #pragma omp target teams distribute parallel for collapse(4) is_device_ptr(a)
    for (llint p = 0; p < n0; ++p)
        for (llint q = 0; q < n1; ++q)
            for (llint r = 0; r < n2; ++r)
                for (llint s = 0; s < n3; ++s)
                    a[((p * n1 + q) * n2 + r) * n3 + s] = 1.0f;
}

// CHECK: [gicc-write] kernel {{.*}}
// CHECK-NEXT: when (%1 > 0) (%2 > 0) (%3 > 0) (%4 > 0)
// CHECK-NEXT: arg#5: 4-D box extent=[%1, %2, %3, %4] stride=[(4 * %2 * %3 * %4), (4 * %3 * %4), (4 * %4), 4] offset=0
