// collapse(3) over int loops whose index (i*ny + j)*nz + k the optimizer
// folds into a truncation of the collapsed IV, its digits divided by a
// 32-bit inner trip count and multiplied back by the 64-bit one.
void step(const float* p, float* u, int nx, int ny, int nz) {
    #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(p, u)
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++) u[(i * ny + j) * nz + k] = p[(i * ny + j) * nz + k] + 1.f;
}

// CHECK: [gicc-write] kernel {{.*}}
// CHECK-NEXT: when ((trunc i64 %nx to i32) > 0) ((trunc i64 %ny to i32) > 0) ((trunc i64 %nz to i32) > 0)
// CHECK-NEXT: u: 2-D box extent=[(sext i32 (trunc i64 %nx to i32) to i64), (zext i32 ((zext i31 (trunc i64 %ny to i31) to i32) * (zext i31 (trunc i64 %nz to i31) to i32)) to i64)] stride=[(4 * (zext i32 ((zext i31 (trunc i64 %ny to i31) to i32) * (zext i31 (trunc i64 %nz to i31) to i32)) to i64))<nuw><nsw>, 4] offset=0 (trusts loop-variable trunc)
