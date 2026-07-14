// C host API for GiOMP. Device-side operations remain in the C++ gicc/omp.h
// header; this surface lets C applications initialize, allocate, exchange, and
// synchronize by linking libgicc_omp without application-side bridge wrappers.
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct giomp_buffer {
    void* ptr;
    int index;
    size_t bytes;
} giomp_buffer;

void giomp_init(void);
void giomp_finalize(void);
int giomp_rank(void);
int giomp_size(void);

giomp_buffer giomp_alloc(size_t bytes);
void giomp_free(giomp_buffer buffer);
void giomp_memcpy_h2d(void* device_dst, const void* host_src, size_t bytes);

void giomp_exchange(void);
void giomp_barrier(void);

#ifdef __cplusplus
}  // extern "C"
#endif
