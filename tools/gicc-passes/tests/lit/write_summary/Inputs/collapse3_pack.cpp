// A halo pack as miniWeather writes it: collapse(3) over (var, row, cell)
// with int indices, buf[ll*nz*hs + k*hs + s]. instcombine factors the index
// into ((ll*nz + k) << 1) + s and drops the nsw the source had; every
// value of it is no larger than the index itself, which fits an int.
// CHECK: [gicc-write] kernel {{.*}}pack
// CHECK-NEXT: when ((trunc i64 %1 to i32) > 0)
// CHECK-NEXT: arg#2: 3-D box extent=[4, (zext i31 (trunc i64 %1 to i31) to i64), 2] stride=[(16 * (sext i32 (trunc i64 %1 to i32) to i64))<nsw>, 16, 8] offset=0 (trusts loop-variable trunc)
#include <cstdint>
void pack(double* buf, const double* state, int nz, int nx) {
    constexpr int hs = 2;
    #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(buf, state)
    for (int ll = 0; ll < 4; ll++)
        for (int k = 0; k < nz; k++)
            for (int s = 0; s < hs; s++)
                buf[ll * nz * hs + k * hs + s] = state[ll * (nz + 4) * (nx + 4) + (k + 2) * (nx + 4) + s];
}
