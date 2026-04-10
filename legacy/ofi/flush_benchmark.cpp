/**
 * flush_benchmark.cpp - Compare different CQ progress strategies
 *
 * Compares three strategies for CQ progress in DWQ operations:
 *   0 = SYNC_FLUSH: Call fi_cq_read in main loop after each communication
 *   1 = BACKGROUND_THREAD: Dedicated thread continuously calls fi_cq_read
 *   2 = NO_FLUSH: No CQ progress at all
 *
 * Run:
 *   FI_MR_CACHE_MAX_COUNT=0 srun -N 2 -n 2 --ntasks-per-node=1 ./flush_benchmark <mode> [iterations] [msg_size]
 *
 * Examples:
 *   ./flush_benchmark 0 100 4096  # SYNC_FLUSH, 100 iters, 4KB
 *   ./flush_benchmark 1 100 4096  # BACKGROUND_THREAD
 *   ./flush_benchmark 2 100 4096  # NO_FLUSH
 */

#include <iostream>
#include <ctime>
#include <cstring>
#include <unistd.h>
#include <vector>
#include <thread>
#include <atomic>

#include <mpi.h>
#include <hip/hip_runtime.h>

#include "hip_device_context.hpp"
#include "pmi_session.hpp"
#include "device_affinity.hpp"
#include "memory_region.hpp"
#include "dwq_work_builder.hpp"

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_cxi_ext.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_trigger.h>

#define HIP_CHECK(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

inline float timediff_us(const timespec& t_start, const timespec& t_end) {
    return (t_end.tv_sec - t_start.tv_sec) * 1.0e6 + (t_end.tv_nsec - t_start.tv_nsec) / 1.0e3;
}

inline void bytes_to_hex(const uint8_t* in, size_t len, char* out) {
    static const char* h = "0123456789abcdef";
    for (size_t i = 0; i < len; i++) {
        out[2 * i] = h[(in[i] >> 4) & 0xF];
        out[2 * i + 1] = h[in[i] & 0xF];
    }
    out[2 * len] = '\0';
}

inline int hex_to_bytes(const char* in, uint8_t* out, size_t outlen) {
    auto hexval = [](char c) -> int {
        if ('0' <= c && c <= '9') return c - '0';
        if ('a' <= c && c <= 'f') return c - 'a' + 10;
        if ('A' <= c && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    size_t n = strlen(in);
    if (n % 2 != 0 || outlen < n / 2) return -1;
    for (size_t i = 0; i < n; i += 2) {
        int hi = hexval(in[i]);
        int lo = hexval(in[i + 1]);
        if (hi < 0 || lo < 0) return -1;
        out[i / 2] = (uint8_t)((hi << 4) | lo);
    }
    return (int)(n / 2);
}

enum class FlushMode { SYNC_FLUSH = 0, BACKGROUND_THREAD = 1, NO_FLUSH = 2 };

const char* flush_mode_name(FlushMode mode) {
    switch (mode) {
        case FlushMode::SYNC_FLUSH: return "SYNC_FLUSH";
        case FlushMode::BACKGROUND_THREAD: return "BACKGROUND_THREAD";
        case FlushMode::NO_FLUSH: return "NO_FLUSH";
    }
    return "UNKNOWN";
}

// Simplified FabricContext
class FabricContext {
public:
    struct fi_info* info = nullptr;
    struct fi_info* cxi_info = nullptr;
    struct fid_fabric* fabric = nullptr;
    struct fid_domain* domain = nullptr;
    struct fid_av* av = nullptr;
    struct fid_cq* cq = nullptr;
    struct fid_ep* ep = nullptr;
    struct fid_cntr* trigger_cntr = nullptr;
    struct fid_cntr* completion_cntr = nullptr;
    struct fi_cxi_cntr_ops* trigger_cntr_ops = nullptr;
    void* trigger_mmio_addr = nullptr;
    size_t trigger_mmio_len = 0;
    volatile uint64_t* dev_trigger_cntr = nullptr;
    void* local_addr = nullptr;
    size_t addrlen = 0;
    int rank;
    DeviceAffinityDetector* affinity_;

    // Background thread
    std::thread cq_thread_;
    std::atomic<bool> cq_stop_{false};

    FabricContext(int rank_, DeviceAffinityDetector* affinity)
        : rank(rank_), affinity_(affinity) {
        init_fabric();
        init_counters();
        init_mmio();
        get_local_address();
    }

    ~FabricContext() {
        stop_background_thread();
        if (trigger_mmio_addr) hipHostUnregister(trigger_mmio_addr);
        if (trigger_cntr) fi_close(&trigger_cntr->fid);
        if (completion_cntr) fi_close(&completion_cntr->fid);
        if (ep) fi_close(&ep->fid);
        if (cq) fi_close(&cq->fid);
        if (av) fi_close(&av->fid);
        if (domain) fi_close(&domain->fid);
        if (fabric) fi_close(&fabric->fid);
        if (info) fi_freeinfo(info);
        if (local_addr) free(local_addr);
    }

    void start_background_thread() {
        cq_stop_.store(false, std::memory_order_release);
        cq_thread_ = std::thread([this]() {
            while (!cq_stop_.load(std::memory_order_acquire)) {
                fi_cq_read(cq, NULL, 0);
                // Sleep 1us between calls to reduce lock contention
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
        });
    }

    void stop_background_thread() {
        cq_stop_.store(true, std::memory_order_release);
        if (cq_thread_.joinable()) {
            cq_thread_.join();
        }
    }

    // Real fast_flush: spin on CQ until completion counter reaches expected value
    void fast_flush(uint64_t expected_completions) {
        while (fi_cntr_read(completion_cntr) < expected_completions) {
            fi_cq_read(cq, NULL, 0);
        }
    }

    bool is_virt_addr_mode() const {
        return (cxi_info->domain_attr->mr_mode & FI_MR_VIRT_ADDR) != 0;
    }

private:
    void check(int ret, const char* msg) {
        if (ret) {
            fprintf(stderr, "Rank %d: %s failed: %s (%d)\n", rank, msg, fi_strerror(-ret), ret);
            exit(1);
        }
    }

    void init_fabric() {
        struct fi_info* hints = fi_allocinfo();
        hints->caps = FI_RMA | FI_MSG | FI_HMEM;
        hints->mode = FI_CONTEXT2;
        hints->ep_attr->type = FI_EP_RDM;
        hints->domain_attr->mr_mode = FI_MR_VIRT_ADDR | FI_MR_ALLOCATED |
                                      FI_MR_PROV_KEY | FI_MR_LOCAL | FI_MR_ENDPOINT;
        hints->domain_attr->threading = FI_THREAD_SAFE;
        hints->domain_attr->control_progress = FI_PROGRESS_MANUAL;
        hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;

        check(fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION),
                         NULL, NULL, 0, hints, &info), "fi_getinfo");
        fi_freeinfo(hints);

        if (affinity_) {
            cxi_info = affinity_->select_cxi_provider(info);
        } else {
            for (struct fi_info* cur = info; cur; cur = cur->next) {
                if (cur->fabric_attr && cur->fabric_attr->prov_name &&
                    strcmp(cur->fabric_attr->prov_name, "cxi") == 0) {
                    cxi_info = cur;
                    break;
                }
            }
        }

        check(fi_fabric(cxi_info->fabric_attr, &fabric, NULL), "fi_fabric");
        check(fi_domain(fabric, cxi_info, &domain, NULL), "fi_domain");

        struct fi_av_attr av_attr = {};
        av_attr.type = FI_AV_TABLE;
        check(fi_av_open(domain, &av_attr, &av, NULL), "fi_av_open");

        struct fi_cq_attr cq_attr = {};
        cq_attr.size = 128;
        cq_attr.format = FI_CQ_FORMAT_CONTEXT;
        check(fi_cq_open(domain, &cq_attr, &cq, NULL), "fi_cq_open");

        check(fi_endpoint(domain, cxi_info, &ep, NULL), "fi_endpoint");
        check(fi_ep_bind(ep, &av->fid, 0), "fi_ep_bind(av)");
        check(fi_ep_bind(ep, &cq->fid, FI_TRANSMIT | FI_RECV), "fi_ep_bind(cq)");
        check(fi_enable(ep), "fi_enable");
    }

    void init_counters() {
        struct fi_cntr_attr cntr_attr = {};
        cntr_attr.events = FI_CNTR_EVENTS_COMP;
        check(fi_cntr_open(domain, &cntr_attr, &trigger_cntr, NULL), "fi_cntr_open(trigger)");

        struct fi_cntr_attr comp_attr = {};
        comp_attr.events = FI_CNTR_EVENTS_COMP;
        comp_attr.wait_obj = FI_WAIT_UNSPEC;
        check(fi_cntr_open(domain, &comp_attr, &completion_cntr, NULL), "fi_cntr_open(completion)");

        check(fi_open_ops(&trigger_cntr->fid, FI_CXI_COUNTER_OPS, 0,
                          (void**)&trigger_cntr_ops, NULL), "fi_open_ops(trigger)");
    }

    void init_mmio() {
        check(trigger_cntr_ops->get_mmio_addr(&trigger_cntr->fid,
                                               &trigger_mmio_addr, &trigger_mmio_len),
              "get_mmio_addr(trigger)");
        HIP_CHECK(hipHostRegister(trigger_mmio_addr, trigger_mmio_len, hipHostRegisterMapped));
        HIP_CHECK(hipHostGetDevicePointer((void**)&dev_trigger_cntr, trigger_mmio_addr, 0));
    }

    void get_local_address() {
        addrlen = 0;
        fi_getname(&ep->fid, NULL, &addrlen);
        local_addr = malloc(addrlen);
        check(fi_getname(&ep->fid, local_addr, &addrlen), "fi_getname");
    }
};

__global__ void trigger_kernel(volatile uint64_t* trigger_addr, uint64_t threshold) {
    *trigger_addr = threshold;
}

int main(int argc, char** argv)
{
    unset_rocr_visible_devices();
    MPI_Init(&argc, &argv);

    int mype, npes;
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);

    if (argc < 2) {
        if (mype == 0) {
            std::cerr << "Usage: " << argv[0] << " <mode> [iterations] [msg_size]\n";
            std::cerr << "  mode: 0=SYNC_FLUSH, 1=BACKGROUND_THREAD, 2=NO_FLUSH\n";
        }
        MPI_Finalize();
        return 1;
    }

    FlushMode mode = static_cast<FlushMode>(atoi(argv[1]));
    int iterations = (argc > 2) ? atoi(argv[2]) : 100;
    size_t msg_size = (argc > 3) ? atol(argv[3]) : 4096;

    if (mype == 0) {
        std::cout << "=== Flush Benchmark ===" << std::endl;
        std::cout << "Mode: " << flush_mode_name(mode) << std::endl;
        std::cout << "Iterations: " << iterations << std::endl;
        std::cout << "Message size: " << msg_size << " bytes" << std::endl;
    }

    // Initialize
    PmiSession pmi;
    int local_rank = pmi.local_rank >= 0 ? pmi.local_rank : mype;
    DeviceAffinityDetector affinity(local_rank);
    HipDeviceContext hip(affinity.selected_gpu_id);

    FabricContext fabric(mype, &affinity);

    // Allocate GPU buffers
    void* send_buf;
    void* recv_buf;
    HIP_CHECK(hipMalloc(&send_buf, msg_size));
    HIP_CHECK(hipMalloc(&recv_buf, msg_size));
    HIP_CHECK(hipMemset(send_buf, mype + 1, msg_size));
    HIP_CHECK(hipMemset(recv_buf, 0, msg_size));

    // Register memory
    MemoryRegion mr_send(fabric.domain, fabric.ep, fabric.cxi_info,
                         send_buf, msg_size, true, hip.gpu_id, mype);
    MemoryRegion mr_recv(fabric.domain, fabric.ep, fabric.cxi_info,
                         recv_buf, msg_size, true, hip.gpu_id, mype);

    // Exchange fabric addresses
    std::vector<fi_addr_t> av_addrs(npes);
    {
        char* my_hex = (char*)malloc(2 * fabric.addrlen + 1);
        bytes_to_hex((uint8_t*)fabric.local_addr, fabric.addrlen, my_hex);

        char key[PMI2_MAX_KEYLEN];
        snprintf(key, sizeof(key), "fabaddr-%d", mype);
        pmi.kvs_put(key, my_hex);
        pmi.barrier();

        uint8_t* peer_bin = (uint8_t*)malloc(fabric.addrlen);
        for (int r = 0; r < npes; r++) {
            snprintf(key, sizeof(key), "fabaddr-%d", r);
            char peer_hex[PMI2_MAX_VALLEN];
            pmi.kvs_get(key, peer_hex, sizeof(peer_hex));
            hex_to_bytes(peer_hex, peer_bin, fabric.addrlen);
            fi_av_insert(fabric.av, peer_bin, 1, &av_addrs[r], 0, NULL);
        }
        free(my_hex);
        free(peer_bin);
    }

    // Exchange RMA info
    int dest_rank = (npes + mype - 1) % npes;
    uint64_t remote_addr, remote_key;
    {
        struct { uint64_t addr; uint64_t key; } my_info, peer_info;
        my_info.addr = (uint64_t)recv_buf;
        my_info.key = mr_recv.key;

        char info_hex[64];
        bytes_to_hex((uint8_t*)&my_info, sizeof(my_info), info_hex);

        char key[PMI2_MAX_KEYLEN];
        snprintf(key, sizeof(key), "rma-%d", mype);
        pmi.kvs_put(key, info_hex);
        pmi.barrier();

        snprintf(key, sizeof(key), "rma-%d", dest_rank);
        char peer_hex[PMI2_MAX_VALLEN];
        pmi.kvs_get(key, peer_hex, sizeof(peer_hex));
        hex_to_bytes(peer_hex, (uint8_t*)&peer_info, sizeof(peer_info));

        remote_addr = peer_info.addr;
        remote_key = peer_info.key;
    }

    // Compute RMA address based on MR mode
    uint64_t rma_remote_addr = fabric.is_virt_addr_mode() ? remote_addr : 0;

    // Start background thread if needed
    if (mode == FlushMode::BACKGROUND_THREAD) {
        fabric.start_background_thread();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Warmup
    {
        DwqWorkBuilder dwq(mype);
        dwq.queue_rma_write(fabric.domain, fabric.ep,
                           send_buf, mr_send.desc, msg_size,
                           av_addrs[dest_rank], rma_remote_addr, remote_key,
                           fabric.trigger_cntr, fabric.completion_cntr, 1);

        hipLaunchKernelGGL(trigger_kernel, dim3(1), dim3(1), 0, 0,
                           fabric.dev_trigger_cntr, 1ULL);
        HIP_CHECK(hipDeviceSynchronize());

        while (fi_cntr_read(fabric.completion_cntr) < 1) {
            fi_cq_read(fabric.cq, NULL, 0);
        }
        fi_cntr_set(fabric.trigger_cntr, 0);
        fi_cntr_set(fabric.completion_cntr, 0);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Timed run
    std::vector<DwqWorkBuilder*> pending_ops;
    timespec t0, t1;

    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

    for (int i = 0; i < iterations; i++) {
        uint64_t threshold = i + 1;

        auto* dwq = new DwqWorkBuilder(mype);
        dwq->queue_rma_write(fabric.domain, fabric.ep,
                            send_buf, mr_send.desc, msg_size,
                            av_addrs[dest_rank], rma_remote_addr, remote_key,
                            fabric.trigger_cntr, fabric.completion_cntr, threshold);
        pending_ops.push_back(dwq);

        hipLaunchKernelGGL(trigger_kernel, dim3(1), dim3(1), 0, 0,
                           fabric.dev_trigger_cntr, threshold);
        HIP_CHECK(hipDeviceSynchronize());

        // Wait for completion based on mode
        if (mode == FlushMode::SYNC_FLUSH) {
            // Call fast_flush after each operation
            fabric.fast_flush(threshold);
        } else {
            // BACKGROUND_THREAD: thread handles CQ progress
            // NO_FLUSH: just spin on counter without CQ progress
            while (fi_cntr_read(fabric.completion_cntr) < threshold) {
                // spin
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

    // Stop background thread
    if (mode == FlushMode::BACKGROUND_THREAD) {
        fabric.stop_background_thread();
    }

    double total_us = timediff_us(t0, t1);
    double avg_us = total_us / iterations;

    // Gather and print results
    double max_avg_us = 0;
    MPI_Reduce(&avg_us, &max_avg_us, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (mype == 0) {
        std::cout << "Total time: " << total_us << " us" << std::endl;
        std::cout << "Avg latency: " << max_avg_us << " us/iter" << std::endl;
    }

    // Cleanup
    for (auto* op : pending_ops) delete op;
    HIP_CHECK(hipFree(send_buf));
    HIP_CHECK(hipFree(recv_buf));

    MPI_Finalize();
    return 0;
}
