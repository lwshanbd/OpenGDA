// The index multiplies two loop indices: not a box.
typedef long long llint;
void step(float* a, llint n) {
    #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(a)
    for (llint i = 0; i < n; ++i)
        for (llint j = 0; j < n; ++j)
            a[i * j] = 1.0f;
}

// CHECK: [gicc-write] kernel {{.*}}
// CHECK-NEXT: store to ptr not analysable: store address: product of two loop-variant values
