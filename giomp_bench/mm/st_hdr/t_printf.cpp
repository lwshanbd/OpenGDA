#include <cstdio>
extern "C" void probe(int x) {
    #pragma omp target firstprivate(x)
    { if (x == 12345) printf("x %d\n", x); }
}
