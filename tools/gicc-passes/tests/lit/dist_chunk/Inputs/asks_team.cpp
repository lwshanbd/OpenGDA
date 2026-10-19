// A kernel that asks which team it is in sees the chunks it runs: left alone.
// CHECK: [gicc-dist-chunk] {{.*}}stamp{{.*}}: asks its team, left alone
// IR-LABEL: define {{.*}} @__omp_offloading_{{.*}}stamp
// IR-NOT: gicc.dist.chunk
// IR: call void @__kmpc_distribute_static_init_8(ptr {{.*}}, i32 91, {{.*}}, i64 1, i64 128)
#include <omp.h>
void stamp(int *a, long n) {
    #pragma omp target teams distribute parallel for is_device_ptr(a)
    for (long i = 0; i < n; ++i) a[i] = omp_get_team_num();
}
