// minimod-mpi's target_3d.c compiled as C++, as the GiOMP build compiles its own.
#define restrict __restrict
#include <mpi.h>
#include <omp.h>
extern "C" {
#include "/shared/data1/Users/l1065028/minimod-mpi/targets/omp_offload_opt-gpu_multi_nowait/target_3d.c"
}
