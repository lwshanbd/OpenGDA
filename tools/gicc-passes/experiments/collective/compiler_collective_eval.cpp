/*
 * Fixed-source compiler collective-policy capacity experiment.
 *
 * The application calls exactly one neutral semantic anchor.  No model sees
 * or edits this file.  During LTO the compiler may preserve that anchor or
 * retarget it to an ABI-identical, compiler-declared catalog entry, including
 * a compiler-owned message-size decision table.  The model output is limited
 * to opaque option IDs and is rejected before this program is built if it
 * contains code, IR, function names, thresholds, or invented legality.
 */

#include "compiler_collective_catalog.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <mpi.h>

namespace {

std::vector<size_t> sizes_from_environment() {
    const char *text = std::getenv("GICC_COLL_SIZES");
    if (!text || !*text)
        return {1024, 4096, 8192, 65536, 262144,
                1048576, 4194304, 8388608, 16777216};
    std::vector<size_t> result;
    const char *cursor = text;
    while (*cursor) {
        char *end = nullptr;
        unsigned long long value = std::strtoull(cursor, &end, 10);
        if (end == cursor || value == 0 || value % sizeof(float) != 0) {
            std::fprintf(stderr, "invalid GICC_COLL_SIZES entry near %s\n", cursor);
            std::exit(2);
        }
        result.push_back(static_cast<size_t>(value));
        if (*end == '\0') break;
        if (*end != ',') {
            std::fprintf(stderr, "invalid GICC_COLL_SIZES separator\n");
            std::exit(2);
        }
        cursor = end + 1;
    }
    return result;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if (values.size() % 2) return values[middle];
    return (values[middle - 1] + values[middle]) * 0.5;
}

// One semantic anchor call serves correctness, warmup, and timing callers.
// This makes the compiler policy a property of the collective operation,
// rather than allowing measurement scaffolding to receive different plans.
__attribute__((noinline)) void run_collective(
    gicc::Runtime &runtime,
    const gicc::Buffer &data_buf, float *d_data,
    const gicc::Buffer &recv_buf, float *d_recv,
    const gicc::Buffer &flag_buf, unsigned int *d_flag,
    const gicc::Buffer &one_buf,
    const gicc::Buffer &nbar_buf, unsigned int *d_nbar,
    int count, int ppn) {
    gicc_compiler_collective::compiler_allreduce_anchor(
        runtime, data_buf, d_data, recv_buf, d_recv,
        flag_buf, d_flag, one_buf, nbar_buf, d_nbar, count, ppn);
}

}  // namespace

int main(int argc, char **argv) {
    gicc::Runtime runtime;
    runtime.set_ipc_fastpath(false);
    const int rank = runtime.rank();
    const int ranks = runtime.size();
    const int ppn = runtime.boot().local_size();
    const int runs = argc > 1 ? std::atoi(argv[1]) : 7;
    const int warmup = argc > 2 ? std::atoi(argv[2]) : 2;
    if (runs < 1 || warmup < 0) return 2;
    const char *label = std::getenv("GICC_COLLECTIVE_PLAN_LABEL");
    if (!label || !*label) label = "unlabeled";

    std::vector<size_t> sizes = sizes_from_environment();
    const size_t max_bytes = *std::max_element(sizes.begin(), sizes.end());
    const size_t max_count = max_bytes / sizeof(float);
    constexpr size_t flag_count = 1024;
    constexpr size_t nbar_count = 128;

    float *d_data = nullptr, *d_recv = nullptr, *d_one = nullptr;
    unsigned int *h_flag = nullptr, *d_flag = nullptr, *d_nbar = nullptr;
    if (hipMalloc(&d_data, max_bytes) != hipSuccess ||
        hipMalloc(&d_recv, max_bytes) != hipSuccess ||
        hipMalloc(&d_one, sizeof(float)) != hipSuccess ||
        hipMalloc(&d_nbar, nbar_count * sizeof(unsigned int)) != hipSuccess ||
        hipHostMalloc(reinterpret_cast<void **>(&h_flag),
                      flag_count * sizeof(unsigned int),
                      hipHostMallocMapped) != hipSuccess ||
        hipHostGetDevicePointer(reinterpret_cast<void **>(&d_flag), h_flag, 0)
            != hipSuccess) {
        std::fprintf(stderr, "rank %d: collective allocation failed\n", rank);
        return 2;
    }
    (void)hipMemset(d_one, 1, sizeof(float));
    (void)hipMemset(d_flag, 0, flag_count * sizeof(unsigned int));
    (void)hipMemset(d_nbar, 0, nbar_count * sizeof(unsigned int));
    (void)hipDeviceSynchronize();

    gicc::Buffer data_buf = runtime.register_buffer(d_data, max_bytes, true);
    gicc::Buffer recv_buf = runtime.register_buffer(d_recv, max_bytes, true);
    gicc::Buffer flag_buf = runtime.register_buffer(
        d_flag, flag_count * sizeof(unsigned int), true);
    gicc::Buffer one_buf = runtime.register_buffer(d_one, sizeof(float), true);
    gicc::Buffer nbar_buf = runtime.register_buffer(
        d_nbar, nbar_count * sizeof(unsigned int), true);
    runtime.exchange();
    runtime.barrier();

    if (rank == 0) {
        std::printf("COLLECTIVE_CONFIG plan=%s ranks=%d ppn=%d runs=%d warmup=%d\n",
                    label, ranks, ppn, runs, warmup);
    }

    int total_errors = 0;
    for (size_t bytes : sizes) {
        const int count = static_cast<int>(bytes / sizeof(float));
        if (static_cast<size_t>(count) != bytes / sizeof(float) ||
            count < ranks || count % ranks != 0 || count % ppn != 0) {
            if (rank == 0)
                std::fprintf(stderr, "size %zu violates fixed catalog contract\n", bytes);
            return 2;
        }
        std::vector<float> host(count);
        for (int index = 0; index < count; ++index)
            host[index] = static_cast<float>((rank + 1) + (index % 7));
        (void)hipMemcpy(d_data, host.data(), bytes, hipMemcpyHostToDevice);
        (void)hipMemset(d_flag, 0, flag_count * sizeof(unsigned int));
        (void)hipMemset(d_nbar, 0, nbar_count * sizeof(unsigned int));
        (void)hipDeviceSynchronize();
        runtime.barrier();

        run_collective(
            runtime, data_buf, d_data, recv_buf, d_recv,
            flag_buf, d_flag, one_buf, nbar_buf, d_nbar, count, ppn);
        (void)hipMemcpy(host.data(), d_data, bytes, hipMemcpyDeviceToHost);
        (void)hipDeviceSynchronize();
        int local_errors = 0;
        for (int index = 0; index < count; ++index) {
            const float expected = static_cast<float>(
                static_cast<double>(ranks) * (ranks + 1) / 2.0 +
                static_cast<double>(ranks) * (index % 7));
            if (host[index] != expected) ++local_errors;
        }
        int errors = 0;
        MPI_Allreduce(&local_errors, &errors, 1, MPI_INT, MPI_SUM,
                      MPI_COMM_WORLD);
        total_errors += errors;

        for (int iteration = 0; iteration < warmup; ++iteration)
            run_collective(
                runtime, data_buf, d_data, recv_buf, d_recv,
                flag_buf, d_flag, one_buf, nbar_buf, d_nbar, count, ppn);

        std::vector<double> rank_maxima;
        if (rank == 0) rank_maxima.reserve(runs);
        for (int iteration = 0; iteration < runs; ++iteration) {
            runtime.barrier();
            const double start = MPI_Wtime();
            run_collective(
                runtime, data_buf, d_data, recv_buf, d_recv,
                flag_buf, d_flag, one_buf, nbar_buf, d_nbar, count, ppn);
            const double local_us = (MPI_Wtime() - start) * 1.0e6;
            double maximum_us = 0.0;
            MPI_Reduce(&local_us, &maximum_us, 1, MPI_DOUBLE, MPI_MAX, 0,
                       MPI_COMM_WORLD);
            if (rank == 0) rank_maxima.push_back(maximum_us);
        }
        if (rank == 0) {
            std::printf(
                "RESULT plan=%s nodes=%d ranks=%d ppn=%d bytes=%zu "
                "median_us=%.3f errors=%d\n",
                label, ranks / ppn, ranks, ppn, bytes,
                median(rank_maxima), errors);
        }
    }

    runtime.barrier();
    (void)hipFree(d_nbar);
    (void)hipFree(d_one);
    (void)hipHostFree(h_flag);
    (void)hipFree(d_recv);
    (void)hipFree(d_data);
    if (rank == 0)
        std::printf("COLLECTIVE_DONE plan=%s total_errors=%d\n",
                    label, total_errors);
    return total_errors == 0 ? 0 : 3;
}
