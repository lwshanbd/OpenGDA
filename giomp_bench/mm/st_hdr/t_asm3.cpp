#include <cstdint>
extern "C" void probe(uint32_t* p) {
    #pragma omp target is_device_ptr(p)
    {
#ifdef __NVPTX__
        __scoped_atomic_store_n(p, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
#endif
    }
}
