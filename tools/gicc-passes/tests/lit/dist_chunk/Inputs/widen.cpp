// The chunk clang deals by default becomes one of up to GICC_DIST_CHUNK
// iterations, a multiple of its own.
// CHECK: [gicc-dist-chunk] {{.*}}scale{{.*}}: chunk i64 128 -> up to 4096
// IR-LABEL: define {{.*}} @__omp_offloading_{{.*}}scale
// IR: call i32 @omp_get_num_teams()
// IR: [[C:%gicc.dist.chunk]] = mul i64
// IR: call void @__kmpc_distribute_static_init_8(ptr {{.*}}, i32 91, {{.*}}, i64 1, i64 [[C]])
void scale(double *a, long n) {
    #pragma omp target teams distribute parallel for is_device_ptr(a)
    for (long i = 0; i < n; ++i) a[i] *= 2.0;
}
