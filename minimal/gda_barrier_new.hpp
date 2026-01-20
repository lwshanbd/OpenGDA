/**
 * gda_barrier_new.hpp - GPU-triggered Dissemination Barrier
 *
 * Implements a dissemination barrier using DWQ (Deferred Work Queue).
 * The barrier executes entirely from the GPU via RDMA puts.
 *
 * Algorithm:
 *   For N ranks, requires ceil(log2(N)) rounds.
 *   In round k (0 to n_rounds-1):
 *     - Rank i sends signal to rank (i + 2^k) mod N
 *     - Rank i receives signal from rank (i - 2^k + N) mod N
 *
 * Usage Mode 1: Single barrier per kernel launch
 *   GdaComm comm;
 *   GdaBarrierNew barrier(comm);
 *   barrier.init();  // Spawn monitor thread
 *
 *   for (int iter = 0; iter < N; iter++) {
 *       barrier.setup();
 *       my_kernel<<<...>>>(args, barrier.get_device_context());
 *       hipDeviceSynchronize();
 *       barrier.reset();
 *   }
 *
 *   barrier.finalize();  // Cleanup
 *
 * Usage Mode 2: Multiple barriers in single kernel (continuous mode)
 *   GdaComm comm;
 *   GdaBarrierNew barrier(comm);
 *   barrier.init();
 *
 *   barrier.start_continuous(100);  // Prepare for 100 barriers
 *   my_kernel<<<...>>>(args, barrier.get_device_context());  // Kernel does 100 barriers
 *   barrier.wait_continuous();  // Wait for all barriers to complete
 *
 *   barrier.finalize();
 */
#pragma once

#include <hip/hip_runtime.h>
#include <mpi.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <thread>
#include <atomic>
#include <chrono>

#include "gda_comm.hpp"
#include "dwq_work_builder.hpp"

namespace opengda {

// =============================================================================
// Device context for barrier kernel
// =============================================================================

// Number of signal slots to avoid overwrites between consecutive barriers
constexpr int N_SIGNAL_SLOTS = 2;

struct barrier_new_context_t {
    int n_rounds;
    int rank;
    int size;
    int n_signal_slots;                   // = N_SIGNAL_SLOTS
    volatile uint64_t* signals;           // My signal buffers [n_rounds * n_signal_slots]
    volatile uint64_t** trigger_addrs;    // Trigger addresses for each round
    uint64_t expected_signal;             // Expected signal value for this barrier
    volatile uint64_t* done_counter;      // Host-visible counter for CPU notification
    volatile uint64_t* ready_counter;     // Host-visible counter: CPU signals DWQ is ready
    volatile uint64_t* debug_state;       // Host-visible debug: 0=idle, 1=wait_ready, 2=round_k (k in high bits)
};

// =============================================================================
// Device function - call this from your GPU kernel to perform barrier
// =============================================================================

/**
 * GPU device function that executes the dissemination barrier.
 * Call this from your kernel when you need a global barrier.
 *
 * IMPORTANT: Only one thread should call this function (e.g., thread 0).
 *
 * Usage in your kernel:
 *   __global__ void my_kernel(..., barrier_new_context_t* barrier_ctx) {
 *       // ... do work ...
 *       if (threadIdx.x == 0 && blockIdx.x == 0) {
 *           barrier_new(barrier_ctx);
 *       }
 *       __syncthreads();  // Sync other threads after barrier
 *       // ... continue ...
 *   }
 */
__device__ void barrier_new(barrier_new_context_t* ctx) {
    uint64_t expected = ctx->expected_signal;

    // Debug: waiting for ready_counter
    *ctx->debug_state = (expected << 32) | 1;  // state=1: wait_ready
    __threadfence_system();

    // Wait for CPU to signal that DWQ operations are ready
    while (*ctx->ready_counter < expected) {
        // Busy wait for CPU to setup DWQ
    }
    __threadfence_system();

    // Calculate signal slot for this barrier to avoid overwrites
    int slot = expected % ctx->n_signal_slots;

    for (int k = 0; k < ctx->n_rounds; k++) {
        // Debug: about to trigger round k
        *ctx->debug_state = (expected << 32) | (2 + k * 2);  // state=2,4,6...: trigger round k
        __threadfence_system();

        // Trigger send for round k (write threshold to trigger DWQ put)
        *ctx->trigger_addrs[k] = expected;
        __threadfence_system();

        // Debug: waiting for signal round k
        *ctx->debug_state = (expected << 32) | (3 + k * 2);  // state=3,5,7...: wait signal k
        __threadfence_system();

        // Wait for signal from peer (poll until expected value arrives)
        // Use signal slot based on expected to avoid overwrites from later barriers
        int sig_idx = k * ctx->n_signal_slots + slot;
        while (ctx->signals[sig_idx] != expected) {
            // Busy wait
        }
        __threadfence_system();
    }

    // Debug: done
    *ctx->debug_state = (expected << 32) | 0xFF;
    __threadfence_system();

    // Notify CPU that this barrier is done
    atomicAdd((unsigned long long*)ctx->done_counter, 1ULL);
    __threadfence_system();
}

/**
 * Simple test kernel that just calls barrier_new().
 * Used for testing the barrier implementation.
 */
__global__ void barrier_new_kernel(barrier_new_context_t* ctx) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    barrier_new(ctx);
}

// =============================================================================
// GdaBarrierNew - Dissemination barrier using DWQ
// =============================================================================

class GdaBarrierNew {
public:
    static constexpr int MAX_ROUNDS = 12;  // Supports up to 4096 ranks

    // Reference to comm
    GdaComm& comm;
    int n_rounds;

    // Barrier count (incremented after each barrier)
    // Threshold and signal value = barrier_count + 1
    uint64_t barrier_count;

    // Signal buffers (device memory) - one uint64_t per round
    uint64_t* d_signals;
    MemoryRegion* mr_signals;

    // Remote signal addresses for each round
    // In round k, I write to peer's signal buffer at index k
    std::vector<uint64_t> remote_signal_addrs;
    std::vector<uint64_t> remote_signal_keys;
    std::vector<int> send_peers;  // Peer to send to in each round

    // Counter pairs for each round (reused across barriers)
    std::vector<FabricDwqContext::CounterPair> counter_pairs;

    // Trigger address array for kernel (device memory)
    volatile uint64_t** d_trigger_addrs;

    // Device context
    barrier_new_context_t h_context;
    barrier_new_context_t* d_context;

    // Signal source buffers (host, for DWQ put source) - double buffered to avoid race
    static constexpr int N_SIGNAL_BUFFERS = 2;
    uint64_t* h_signal_values[N_SIGNAL_BUFFERS];
    MemoryRegion* mr_signal_values[N_SIGNAL_BUFFERS];

    // DWQ work builders (one per round, recreated each barrier)
    std::vector<DwqWorkBuilder*> dwq_ops;

    // Host-visible counter for GPU->CPU notification
    volatile uint64_t* h_done_counter;

    // Host-visible counter for CPU->GPU notification (DWQ ready)
    volatile uint64_t* h_ready_counter;

    // Host-visible debug state from GPU
    volatile uint64_t* h_debug_state;

    // =========================================================================
    // CPU Proxy Thread Management
    // =========================================================================
    std::thread monitor_thread;
    std::atomic<bool> thread_running{false};      // Thread should keep running
    std::atomic<bool> ever_active{false};         // Has any barrier happened? (for sleep->busy transition)
    std::atomic<bool> continuous_mode{false};     // Are we in continuous mode?
    std::atomic<uint64_t> target_barrier_count{0}; // Target count for continuous mode

    /**
     * Initialize dissemination barrier
     * @param comm Reference to initialized GdaComm
     */
    explicit GdaBarrierNew(GdaComm& comm_)
        : comm(comm_),
          barrier_count(0),
          d_signals(nullptr), mr_signals(nullptr),
          d_trigger_addrs(nullptr), d_context(nullptr),
          h_signal_values{nullptr, nullptr}, mr_signal_values{nullptr, nullptr},
          h_done_counter(nullptr),
          h_ready_counter(nullptr),
          h_debug_state(nullptr)
    {
        // Calculate number of rounds: ceil(log2(size))
        int size = comm.size();
        n_rounds = 0;
        while ((1 << n_rounds) < size) {
            n_rounds++;
        }

        if (n_rounds > MAX_ROUNDS) {
            fprintf(stderr, "Rank %d: Too many ranks (%d) for barrier, max %d\n",
                    comm.rank(), size, 1 << MAX_ROUNDS);
            exit(1);
        }

        // Allocate and initialize signal buffers
        allocate_signals();

        // Exchange signal buffer addresses with peers
        exchange_addresses();

        // Create counter pairs for each round (reused across barriers)
        create_counters();

        // Setup device context (static parts)
        setup_device_context();
    }

    ~GdaBarrierNew() {
        // Ensure finalize was called
        if (thread_running.load()) {
            finalize();
        }
    }

    // No copy
    GdaBarrierNew(const GdaBarrierNew&) = delete;
    GdaBarrierNew& operator=(const GdaBarrierNew&) = delete;

    // =========================================================================
    // Lifecycle: init() and finalize()
    // =========================================================================

    /**
     * Initialize the barrier system and spawn the monitor thread.
     * Call this once after construction, before using the barrier.
     */
    void init() {
        if (thread_running.load()) {
            return;  // Already initialized
        }

        thread_running = true;
        ever_active = false;
        continuous_mode = false;
        target_barrier_count = 0;

        monitor_thread = std::thread(&GdaBarrierNew::monitor_loop, this);

        if (comm.rank() == 0) {
            printf("GdaBarrierNew: Monitor thread started\n");
            fflush(stdout);
        }
    }

    /**
     * Finalize the barrier system.
     * Stops the monitor thread and releases all resources.
     * Call this once when done with the barrier.
     */
    void finalize() {
        if (!thread_running.load()) {
            return;  // Not initialized or already finalized
        }

        // Stop the monitor thread
        thread_running = false;
        continuous_mode = false;

        if (monitor_thread.joinable()) {
            monitor_thread.join();
        }

        if (comm.rank() == 0) {
            printf("GdaBarrierNew: Monitor thread stopped\n");
            fflush(stdout);
        }

        // Cleanup DWQ ops
        for (auto* op : dwq_ops) delete op;
        dwq_ops.clear();

        // Cleanup counters
        for (auto& cp : counter_pairs) {
            comm.fabric->destroy_counter_pair(cp);
        }
        counter_pairs.clear();

        // Cleanup device context
        if (d_context) hipFree(d_context);
        if (d_trigger_addrs) hipFree(d_trigger_addrs);
        d_context = nullptr;
        d_trigger_addrs = nullptr;

        // Cleanup signal source buffers (double buffered)
        for (int i = 0; i < N_SIGNAL_BUFFERS; i++) {
            delete mr_signal_values[i];
            mr_signal_values[i] = nullptr;
            if (h_signal_values[i]) hipHostFree(h_signal_values[i]);
            h_signal_values[i] = nullptr;
        }
        delete mr_signals;
        mr_signals = nullptr;
        if (d_signals) hipHostFree(d_signals);
        d_signals = nullptr;

        // Cleanup host-visible counters
        if (h_done_counter) hipHostFree((void*)h_done_counter);
        h_done_counter = nullptr;
        if (h_ready_counter) hipHostFree((void*)h_ready_counter);
        h_ready_counter = nullptr;
        if (h_debug_state) hipHostFree((void*)h_debug_state);
        h_debug_state = nullptr;
    }

    // =========================================================================
    // Continuous Mode: Multiple barriers in single kernel
    // =========================================================================

    /**
     * Start continuous barrier mode.
     * The CPU monitor thread will automatically setup/reset barriers.
     *
     * @param num_barriers Number of barriers the kernel will execute
     */
    void start_continuous(uint64_t num_barriers) {
        // Mark as active (switch to busy polling)
        ever_active = true;

        // Set target
        target_barrier_count = barrier_count + num_barriers;

        // Setup first barrier
        setup();

        // Enable continuous mode (monitor thread will take over)
        continuous_mode = true;
    }

    /**
     * Wait for all continuous barriers to complete.
     * Call this after the kernel finishes.
     */
    void wait_continuous() {
        // Wait until monitor thread has processed all barriers
        while (barrier_count < target_barrier_count.load()) {
            std::this_thread::yield();
        }

        // Disable continuous mode
        continuous_mode = false;
    }

    /**
     * Get the target barrier count for continuous mode.
     * Useful for debugging.
     */
    uint64_t get_target_barrier_count() const {
        return target_barrier_count.load();
    }

    /**
     * Queue DWQ operations for barrier.
     * Call this before launching the barrier kernel (single mode)
     * or called by monitor thread (continuous mode).
     * Uses current barrier_count to determine threshold.
     *
     * Note: In continuous mode, we don't update expected_signal via hipMemcpy
     * because the GPU tracks it independently.
     */
    void setup() {
        int rank = comm.rank();
        int size = comm.size();

        // Calculate threshold for this barrier
        uint64_t threshold = barrier_count + 1;

        // Use double buffer to avoid race with in-flight DWQ operations
        int buf_idx = barrier_count % N_SIGNAL_BUFFERS;
        uint64_t* h_signal_value = h_signal_values[buf_idx];
        MemoryRegion* mr_signal_value = mr_signal_values[buf_idx];

        // Update signal value to be sent
        *h_signal_value = threshold;

        // Update device context with expected signal
        // Only do this in single mode (not continuous mode)
        // In continuous mode, the GPU tracks expected_signal independently
        if (!continuous_mode.load(std::memory_order_relaxed)) {
            h_context.expected_signal = threshold;
            hipMemcpy(&d_context->expected_signal, &h_context.expected_signal,
                      sizeof(uint64_t), hipMemcpyHostToDevice);
        }

        // Calculate signal slot to avoid overwrites
        int slot = threshold % N_SIGNAL_SLOTS;

        for (int k = 0; k < n_rounds; k++) {
            // Calculate peer for this round
            int peer = (rank + (1 << k)) % size;

            // Create DWQ work builder
            auto* dwq = new DwqWorkBuilder(rank);

            // Calculate remote address for signal buffer
            // Layout: signals[k * N_SIGNAL_SLOTS + slot]
            int sig_idx = k * N_SIGNAL_SLOTS + slot;
            uint64_t remote_offset = sig_idx * sizeof(uint64_t);
            uint64_t remote_addr = comm.is_virt_addr_mode()
                ? (remote_signal_addrs[k] + remote_offset)
                : remote_offset;

            // Queue the put operation with current threshold
            dwq->queue_rma_write(
                comm.fabric->domain,
                comm.fabric->ep,
                h_signal_value,            // Source: signal value (double buffered)
                mr_signal_value->desc,     // Source descriptor
                sizeof(uint64_t),          // Size
                comm.av_addrs[peer],       // Destination address (fi_addr_t)
                remote_addr,               // Remote buffer address
                remote_signal_keys[k],     // Remote buffer key
                counter_pairs[k].trigger_cntr,     // Trigger counter
                counter_pairs[k].completion_cntr,  // Completion counter
                threshold);                // Threshold = barrier_count + 1

            dwq_ops.push_back(dwq);
        }

        // Signal GPU that DWQ operations are ready for this barrier
        // Use atomic increment with memory fence to ensure visibility
        __atomic_add_fetch(h_ready_counter, 1, __ATOMIC_SEQ_CST);
    }

    /**
     * Get device context for use in kernel
     */
    barrier_new_context_t* get_device_context() const {
        return d_context;
    }

    /**
     * Wait for barrier completion on CPU side
     */
    void wait_completion() {
        uint64_t expected = barrier_count + 1;
        for (int k = 0; k < n_rounds; k++) {
            while (fi_cntr_read(counter_pairs[k].completion_cntr) < expected) {
                fi_cq_read(comm.fabric->cq, NULL, 0);
            }
        }
    }

    /**
     * Reset for next barrier.
     * Cleans up old DWQ ops and increments barrier_count.
     * Call this after wait_completion() and before next setup().
     */
    void reset() {
        // Cleanup old DWQ work builders
        for (auto* op : dwq_ops) delete op;
        dwq_ops.clear();

        // Increment barrier count for next barrier
        barrier_count++;

        // Fast flush: wait for each round's completion counter and progress CQ
        uint64_t expected = barrier_count;  // After increment, this is the completed barrier's threshold
        for (int k = 0; k < n_rounds; k++) {
            while (fi_cntr_read(counter_pairs[k].completion_cntr) < expected) {
                fi_cq_read(comm.fabric->cq, NULL, 0);
            }
        }

        // Extra CQ progress to release DWQ resources
        for (int i = 0; i < 100; i++) {
            fi_cq_read(comm.fabric->cq, NULL, 0);
        }
    }

    /**
     * Get current barrier count (number of completed barriers)
     */
    uint64_t get_barrier_count() const {
        return barrier_count;
    }

private:
    /**
     * Monitor thread loop.
     * Polls completion counters and handles setup/reset in continuous mode.
     *
     * Behavior:
     * - Initially: sleep 1ms between polls
     * - After first barrier: switch to busy polling (no sleep)
     */
    void monitor_loop() {
        int rank = comm.rank();
        uint64_t last_printed = 0;
        uint64_t stuck_count = 0;
        uint64_t last_gpu_done = 0;

        while (thread_running.load(std::memory_order_relaxed)) {
            // If not in continuous mode, just wait
            if (!continuous_mode.load(std::memory_order_relaxed)) {
                if (!ever_active.load(std::memory_order_relaxed)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                } else {
                    std::this_thread::yield();
                }
                stuck_count = 0;
                continue;
            }

            // Progress CQ to help DWQ and release resources
            fi_cq_read(comm.fabric->cq, NULL, 0);

            // Check if GPU has completed a barrier by reading host-visible counter
            uint64_t gpu_done = *h_done_counter;

            // Debug: print progress every 10 barriers
            if (gpu_done > last_printed && gpu_done % 10 == 0) {
                printf("Rank %d: GPU done=%lu, CPU barrier_count=%lu\n",
                       rank, gpu_done, barrier_count);
                fflush(stdout);
                last_printed = gpu_done;
            }

            // Detect if GPU is stuck
            if (gpu_done == last_gpu_done) {
                stuck_count++;
                if (stuck_count > 1000000) {  // ~1 second of spinning
                    uint64_t debug = *h_debug_state;
                    uint64_t barrier_num = debug >> 32;
                    uint64_t state = debug & 0xFF;
                    const char* state_name = "unknown";
                    if (state == 1) state_name = "wait_ready";
                    else if (state == 0xFF) state_name = "done";
                    else if (state % 2 == 0) state_name = "trigger_round";
                    else state_name = "wait_signal";
                    printf("Rank %d: STUCK! debug_state=0x%lx (barrier=%lu, state=%s, round=%lu)\n",
                           rank, debug, barrier_num, state_name, (state > 1) ? (state - 2) / 2 : 0);
                    printf("Rank %d:   gpu_done=%lu, cpu_barrier_count=%lu, ready_counter=%lu\n",
                           rank, gpu_done, barrier_count, *h_ready_counter);
                    // d_signals is host-visible, can read directly
                    int slot = barrier_num % N_SIGNAL_SLOTS;
                    int sig_idx = 0 * N_SIGNAL_SLOTS + slot;  // round 0
                    printf("Rank %d:   signals[%d]=%lu (expected=%lu, slot=%d)\n",
                           rank, sig_idx, d_signals[sig_idx], barrier_num, slot);
                    fflush(stdout);
                    stuck_count = 0;
                }
            } else {
                stuck_count = 0;
                last_gpu_done = gpu_done;
            }

            if (gpu_done > barrier_count) {
                // GPU has completed barrier(s), do reset/setup
                reset();

                // Check if we need more barriers
                if (barrier_count < target_barrier_count.load(std::memory_order_relaxed)) {
                    setup();
                } else {
                    // All barriers done, exit continuous mode
                    printf("Rank %d: All done! barrier_count=%lu\n", rank, barrier_count);
                    fflush(stdout);
                    continuous_mode = false;
                }
            }
        }
    }

    void allocate_signals() {
        // Allocate host-visible signal buffers (accessible from both CPU and GPU)
        // Use n_rounds * N_SIGNAL_SLOTS to avoid overwrites between consecutive barriers
        size_t signals_size = n_rounds * N_SIGNAL_SLOTS * sizeof(uint64_t);
        hipHostMalloc(&d_signals, signals_size, hipHostMallocDefault);
        memset((void*)d_signals, 0, signals_size);

        // Register for RDMA (as host memory, not device)
        mr_signals = new MemoryRegion(
            comm.fabric->domain, comm.fabric->ep, comm.fabric->cxi_info,
            d_signals, signals_size, false, comm.gpu_id(), comm.rank());

        // Allocate host signal value buffers (double buffered to avoid race)
        for (int i = 0; i < N_SIGNAL_BUFFERS; i++) {
            hipHostMalloc(&h_signal_values[i], sizeof(uint64_t), hipHostMallocDefault);
            *h_signal_values[i] = 0;

            // Register for RDMA
            mr_signal_values[i] = new MemoryRegion(
                comm.fabric->domain, comm.fabric->ep, comm.fabric->cxi_info,
                h_signal_values[i], sizeof(uint64_t), false, comm.gpu_id(), comm.rank());
        }

        // Allocate host-visible counter for GPU->CPU notification
        hipHostMalloc((void**)&h_done_counter, sizeof(uint64_t), hipHostMallocDefault);
        *h_done_counter = 0;

        // Allocate host-visible counter for CPU->GPU notification (DWQ ready)
        hipHostMalloc((void**)&h_ready_counter, sizeof(uint64_t), hipHostMallocDefault);
        *h_ready_counter = 0;

        // Allocate host-visible debug state
        hipHostMalloc((void**)&h_debug_state, sizeof(uint64_t), hipHostMallocDefault);
        *h_debug_state = 0;
    }

    void exchange_addresses() {
        int rank = comm.rank();
        int size = comm.size();

        // My signal buffer info
        uint64_t my_base = (uint64_t)d_signals;
        uint64_t my_key = mr_signals->key;

        // Exchange with all ranks
        std::vector<uint64_t> all_bases(size);
        std::vector<uint64_t> all_keys(size);

        MPI_Allgather(&my_base, 1, MPI_UINT64_T,
                      all_bases.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD);
        MPI_Allgather(&my_key, 1, MPI_UINT64_T,
                      all_keys.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD);

        // Calculate peers and store their info
        remote_signal_addrs.resize(n_rounds);
        remote_signal_keys.resize(n_rounds);
        send_peers.resize(n_rounds);

        for (int k = 0; k < n_rounds; k++) {
            int peer = (rank + (1 << k)) % size;
            send_peers[k] = peer;
            remote_signal_addrs[k] = all_bases[peer];
            remote_signal_keys[k] = all_keys[peer];
        }
    }

    void create_counters() {
        counter_pairs.resize(n_rounds);
        for (int k = 0; k < n_rounds; k++) {
            counter_pairs[k] = comm.fabric->create_counter_pair();
        }
    }

    void setup_device_context() {
        // Allocate trigger address array on device
        hipMalloc(&d_trigger_addrs, n_rounds * sizeof(uint64_t*));

        // Copy trigger addresses to device
        std::vector<volatile uint64_t*> h_trigger_addrs(n_rounds);
        for (int k = 0; k < n_rounds; k++) {
            h_trigger_addrs[k] = counter_pairs[k].dev_trigger_cntr;
        }
        hipMemcpy(d_trigger_addrs, h_trigger_addrs.data(),
                  n_rounds * sizeof(uint64_t*), hipMemcpyHostToDevice);

        // Setup host context (static parts)
        h_context.n_rounds = n_rounds;
        h_context.rank = comm.rank();
        h_context.size = comm.size();
        h_context.n_signal_slots = N_SIGNAL_SLOTS;
        h_context.signals = d_signals;
        h_context.trigger_addrs = d_trigger_addrs;
        h_context.expected_signal = 0;  // Will be set in setup()
        h_context.done_counter = h_done_counter;    // Host-visible counter for GPU->CPU
        h_context.ready_counter = h_ready_counter;  // Host-visible counter for CPU->GPU
        h_context.debug_state = h_debug_state;      // Host-visible debug state

        // Allocate and copy context to device
        hipMalloc(&d_context, sizeof(barrier_new_context_t));
        hipMemcpy(d_context, &h_context, sizeof(barrier_new_context_t),
                  hipMemcpyHostToDevice);
    }
};

}  // namespace opengda
