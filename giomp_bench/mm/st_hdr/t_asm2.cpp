#include <cstdint>
[[omp::assume("ompx_no_call_asm")]] static inline void st(uint32_t* p) {
#ifdef __NVPTX__
    asm volatile("st.relaxed.sys.global.b32 [%0], %1;" :: "l"(p), "r"(1u) : "memory");
#endif
}
extern "C" void probe(uint32_t* p) {
    #pragma omp target is_device_ptr(p)
    { st(p); }
}
