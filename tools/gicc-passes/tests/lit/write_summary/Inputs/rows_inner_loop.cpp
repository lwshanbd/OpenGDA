// Only the row loop is parallel; each iteration walks its row in a
// sequential loop, which the box takes as a second dimension. (The plugin's
// prep pass keeps such a loop from being unrolled with a remainder; here
// the source says so, since the IR comes from clang -O3.)
void step(const float* u, float* un, int nx, int ny) {
    #pragma omp target teams distribute parallel for is_device_ptr(u, un)
    for (int i = 1; i < nx - 1; ++i)
        #pragma clang loop unroll(disable)
        for (int j = 1; j < ny - 1; ++j)
            un[i * ny + j] = 0.25f * (u[(i - 1) * ny + j] + u[(i + 1) * ny + j] +
                                      u[i * ny + j - 1] + u[i * ny + j + 1]);
}

// CHECK: [gicc-write] kernel {{.*}}
// CHECK-NEXT: when ((-2 + (trunc i64 %nx to i32)) > 0)
// CHECK-NEXT: un: 2-D box extent=[(1 + (sext i32 (-3 + (trunc i64 %nx to i32)) to i64))<nsw>, (0 smax (1 + (sext i32 (-3 + (trunc i64 %ny to i32)) to i64))<nsw>)] stride=[(4 * (sext i32 (trunc i64 %ny to i32) to i64))<nsw>, 4] offset=(4 + (4 * (sext i32 (trunc i64 %ny to i32) to i64))<nsw>)<nsw> (trusts loop-variable trunc)
