// A 1-D int loop: its bounds are unsigned 32-bit (for_static_init_4u), and
// its entry guard compares them zero-extended.
void step(const float* p, float* u, int n, int w) {
    #pragma omp target teams distribute parallel for is_device_ptr(p, u)
    for (int i = w; i < n - w; i++) u[i] = 0.5f * (p[i - 1] + p[i + 1]);
}

// CHECK: [gicc-write] kernel {{.*}}
// CHECK-NEXT: when (((trunc i64 %n to i32) + (-2 * (trunc i64 %w to i32))) > 0)
// CHECK-NEXT: u: 1-D box extent=[(1 + (zext i32 (-1 + (trunc i64 %n to i32) + (-2 * (trunc i64 %w to i32))) to i64))<nuw><nsw>] stride=[4] offset=(4 * (sext i32 (trunc i64 %w to i32) to i64))<nsw> (trusts loop-variable trunc)
