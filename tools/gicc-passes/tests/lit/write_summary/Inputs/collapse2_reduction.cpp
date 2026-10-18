// A Jacobi sweep with a max-residual reduction: the reduction's runtime
// calls and its store to the team's reduction variable write nothing a put
// could send.
#include <cmath>
float step(const float* u, float* un, int nx, int ny) {
    float err = 0.f;
    #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(u, un) \
        reduction(max : err)
    for (int i = 1; i < nx - 1; i++)
        for (int j = 1; j < ny - 1; j++) {
            float v = 0.25f * (u[(i - 1) * ny + j] + u[(i + 1) * ny + j] + u[i * ny + j - 1] +
                               u[i * ny + j + 1]);
            un[i * ny + j] = v;
            err = fmaxf(err, fabsf(v - u[i * ny + j]));
        }
    return err;
}

// CHECK: [gicc-write] kernel {{.*}}
// CHECK-NEXT: when ((-2 + (trunc i64 %nx to i32)) > 0) ((-2 + (trunc i64 %ny to i32)) > 0)
// CHECK-NEXT: un: 2-D box extent=[(sext i32 (-2 + (trunc i64 %nx to i32)) to i64), (-2 + (zext i31 (trunc i64 %ny to i31) to i64))<nsw>] stride=[(4 * (zext i31 (trunc i64 %ny to i31) to i64))<nuw><nsw>, 4] offset=(4 + (4 * (zext i31 (trunc i64 %ny to i31) to i64))<nuw><nsw>)<nuw><nsw> (trusts loop-variable trunc)
