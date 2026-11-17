#include <cstdint>
extern "C" void probe(uint32_t* p) {
    #pragma omp target is_device_ptr(p)
    {
#ifdef __NVPTX__
        asm volatile("st.relaxed.sys.global.b32 [%0], %1;" :: "l"(p), "r"(1u) : "memory");
#endif
    }
}
