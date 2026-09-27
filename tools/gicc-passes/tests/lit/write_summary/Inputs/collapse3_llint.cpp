// collapse(3) over 64-bit indices into a padded 3-D array (minimod's shape).
typedef long long llint;
void step(float* v, const float* u, llint x0, llint x1, llint y0, llint y1, llint z0, llint z1,
          llint ny, llint nz, llint pad) {
    #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(v, u)
    for (llint i = x0; i < x1; ++i)
        for (llint j = y0; j < y1; ++j)
            for (llint k = z0; k < z1; ++k) {
                llint idx = ((i + pad) * (ny + 2 * pad) + (j + pad)) * (nz + 2 * pad) + (k + pad);
                v[idx] = 2.f * u[idx] - v[idx];
            }
}

// CHECK: [gicc-write] kernel {{.*}}
// CHECK-NEXT: arg#10: 3-D box extent=[((-1 * %2) + %1), ((-1 * %4) + %3), ((-1 * %6) + %5)] stride=[(4 * ((2 * %7) + %8) * ((2 * %7) + %9)), ((4 * %9) + (8 * %7)), 4] offset=(4 * ((((2 * %7) + %9) * ((((2 * %7) + %8) * (%2 + %7)) + %4 + %7)) + %6 + %7))
