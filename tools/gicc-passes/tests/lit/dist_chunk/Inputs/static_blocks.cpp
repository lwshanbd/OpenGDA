// dist_schedule(static) without a chunk asks for one block per team: kept.
// CHECK-NOT: [gicc-dist-chunk]
// IR-LABEL: define {{.*}} @__omp_offloading_{{.*}}blocks
// IR-NOT: gicc.dist.chunk
// IR: call void @__kmpc_distribute_static_init_8(ptr {{.*}}, i32 92,
void blocks(double *a, long n) {
    #pragma omp target teams distribute parallel for dist_schedule(static) is_device_ptr(a)
    for (long i = 0; i < n; ++i) a[i] += 1.0;
}
