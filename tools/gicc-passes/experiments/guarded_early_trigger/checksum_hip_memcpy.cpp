// Identical post-timing correctness instrumentation for both experiment arms.
// The application source is unchanged; --wrap=hipMemcpy redirects only direct
// executable calls through this harness.  The final D2H result copy is hashed
// after HIP reports success.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <unistd.h>

#include <hip/hip_runtime_api.h>
#include <mpi.h>

extern "C" hipError_t __real_hipMemcpy(
    void *dst, const void *src, std::size_t bytes, hipMemcpyKind kind);

extern "C" hipError_t __wrap_hipMemcpy(
    void *dst, const void *src, std::size_t bytes, hipMemcpyKind kind) {
    const hipError_t status = __real_hipMemcpy(dst, src, bytes, kind);
    if (status != hipSuccess || kind != hipMemcpyDeviceToHost || dst == nullptr ||
        bytes == 0) {
        return status;
    }

    auto *data = static_cast<const unsigned char *>(dst);
    std::uint64_t hash = UINT64_C(1469598103934665603);
    for (std::size_t index = 0; index < bytes; ++index) {
        hash ^= data[index];
        hash *= UINT64_C(1099511628211);
    }

    int initialized = 0;
    int rank = -1;
    if (MPI_Initialized(&initialized) == MPI_SUCCESS && initialized) {
        (void)MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    char line[160];
    const int length = std::snprintf(
        line, sizeof(line),
        "GICC_MM_CHECKSUM rank=%d bytes=%zu fnv64=%016llx\n",
        rank, bytes, static_cast<unsigned long long>(hash));
    if (length > 0) {
        const std::size_t output_bytes =
            static_cast<std::size_t>(length) < sizeof(line)
                ? static_cast<std::size_t>(length)
                : sizeof(line) - 1;
        (void)::write(STDERR_FILENO, line, output_bytes);
    }
    return status;
}
