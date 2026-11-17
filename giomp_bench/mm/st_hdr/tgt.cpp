// minimod-mpi's kernels in a C++ unit that includes the GiOMP headers.
#define restrict __restrict
#include <mpi.h>
#include <omp.h>
#include "gicc/omp.h"
#include "gicc/omp_pipeline.h"
extern "C" {
#include "/shared/data1/Users/l1065028/minimod-mpi/targets/omp_offload_opt-gpu_multi_nowait/target_3d.c"
}
