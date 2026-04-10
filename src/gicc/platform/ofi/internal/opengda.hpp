/**
 * opengda.hpp - GPU-Direct Async Runtime
 *
 * Single-header encapsulation of the OpenGDA library.
 * Provides a clean API for GPU-triggered RDMA and Active Messages
 * without exposing DWQ (Deferred Work Queue) internals.
 *
 * Usage (RMA Put):
 *   gda::Runtime rt;
 *   auto src = rt.register_buffer(d_src, size, true);
 *   auto dst = rt.register_buffer(d_dst, size, true);
 *   rt.exchange();
 *
 *   rt.put(src, dest_rank, dst.index, size);
 *   auto* ctx = rt.prepare();
 *
 *   // In your kernel:
 *   //   gda::trigger(ctx);
 *   //   gda::wait(ctx);
 *
 *   rt.reset();
 *
 * NOTE: RDMA writes from the NIC go directly to GPU HBM. GPU kernel reads
 * see the correct data immediately (kernel launch invalidates L2). However,
 * hipMemcpy(DeviceToHost) may read stale L2 cache for small transfers (<16KB).
 * For verification/debug, use a GPU kernel to copy data instead of hipMemcpy.
 *
 * Usage (Active Message):
 *   gda::Am am(rt);
 *   gda::Args args; args[0] = val;
 *   am.send(dest, handler_id, args);
 *   am.flush();
 *
 *   // On receiver (in kernel):
 *   //   gda::am_poll_all(am.device_context(), 16);
 */
#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Internal implementation headers
#include "hip_device_context.hpp"
#include "pmi_session.hpp"
#include "device_affinity.hpp"
#include "fabric_dwq_context.hpp"
#include "memory_region.hpp"
#include "dwq_work_builder.hpp"
#include "ofi_barrier.hpp"
#include "am_types.hpp"
#include "gda_am_context.hpp"
#include "gda_am_device.hpp"
#include "gda_comm.hpp"
#include "gda_am.hpp"

namespace gda {

// =============================================================================
// Device context - passed to GPU kernels for trigger/wait
// =============================================================================

struct DeviceCtx {
    volatile uint64_t* trigger_addr_;   // MMIO address (GPU writes here)
    volatile uint64_t* completion_;     // atomic_result (GPU polls here)
    uint64_t trigger_val_;              // value to write to trigger all ops
    uint64_t n_ops_;                    // number of completions to wait for
};

// =============================================================================
// Device functions - zero overhead, compile to single memory ops
// =============================================================================

/**
 * Trigger all queued operations.
 * Call from exactly ONE thread (e.g., threadIdx.x == 0).
 * The MMIO write causes the NIC to execute all queued RMA operations.
 */
__device__ __forceinline__ void trigger(DeviceCtx* ctx) {
    *ctx->trigger_addr_ = ctx->trigger_val_;
}

/**
 * Wait for all queued operations to complete.
 * Can be called from any/all threads - they all poll the same location.
 * Returns when all RDMA transfers are done.
 */
__device__ __forceinline__ void wait(DeviceCtx* ctx) {
    while (*ctx->completion_ < ctx->n_ops_) {}
}

/**
 * Trigger then wait. Convenience for single-thread usage.
 */
__device__ __forceinline__ void trigger_and_wait(DeviceCtx* ctx) {
    trigger(ctx);
    __threadfence_system();
    wait(ctx);
}

// =============================================================================
// Buffer - handle to a registered memory region
// =============================================================================

struct Buffer {
    void* ptr;        // User's buffer pointer
    size_t size;      // Buffer size in bytes
    int index;        // Registration index (use in put() as dest_buf_index)

    // Internal - users should not access these directly
    void* desc_;
    uint64_t key_;
    uint64_t addr_;
};

// =============================================================================
// Args - active message arguments (48 bytes = 6 x uint64_t)
// =============================================================================

using Args = opengda::am::am_args_t;

// =============================================================================
// Runtime - main API for GPU-triggered RDMA
// =============================================================================

class Am;  // forward declaration

class Runtime {
public:
    /**
     * Initialize the GDA runtime.
     * Performs: GPU init, PMI bootstrap, libfabric setup, address exchange.
     * This is a collective operation - all ranks must call it.
     */
    explicit Runtime(int local_rank = -1)
        : comm_(nullptr), h_dev_ctx_(nullptr), d_dev_ctx_(nullptr),
          n_ops_(0), exchanged_(false)
    {
        unset_rocr_visible_devices();
        comm_ = new GdaComm(local_rank);

        // Allocate device context in pinned mapped memory (zero-copy)
        (void)hipHostMalloc(&h_dev_ctx_, sizeof(DeviceCtx), hipHostMallocMapped);
        (void)hipHostGetDevicePointer((void**)&d_dev_ctx_, h_dev_ctx_, 0);

        h_dev_ctx_->trigger_addr_ = comm_->get_trigger_addr();
        h_dev_ctx_->completion_ = comm_->get_atomic_result();
        h_dev_ctx_->trigger_val_ = 0;
        h_dev_ctx_->n_ops_ = 0;
    }

    ~Runtime() {
        if (h_dev_ctx_) (void)hipHostFree(h_dev_ctx_);
        delete comm_;
    }

    // No copy/move
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    /**
     * Register a buffer for RDMA operations.
     * All ranks must register buffers in the same order (same number of
     * register_buffer calls) before calling exchange().
     *
     * @param buf   Pointer to buffer (host or device memory)
     * @param size  Buffer size in bytes
     * @param is_device  true if buf is GPU device memory
     * @return Buffer handle (use .index as dest_buf_index in put())
     */
    Buffer register_buffer(void* buf, size_t size, bool is_device) {
        GdaHandle h = comm_->register_buffer(buf, size, is_device);

        Buffer b;
        b.ptr = buf;
        b.size = size;
        b.index = (int)local_bufs_.size();
        b.desc_ = h.local_desc;
        b.key_ = h.rma_key;
        b.addr_ = h.rma_addr;
        local_bufs_.push_back(b);
        return b;
    }

    /**
     * Exchange registered buffer metadata with all ranks.
     * Collective operation - all ranks must call after registering their buffers.
     * After this call, put() can target any (rank, buffer_index) pair.
     */
    void exchange() {
        int nbuf = (int)local_bufs_.size();

        struct BufMeta { uint64_t addr; uint64_t key; };
        std::vector<BufMeta> my_metas(nbuf);
        for (int i = 0; i < nbuf; i++) {
            my_metas[i].addr = local_bufs_[i].addr_;
            my_metas[i].key = local_bufs_[i].key_;
        }

        // Publish my buffer info
        char key[PMI2_MAX_KEYLEN];
        char hex[PMI2_MAX_VALLEN];
        snprintf(key, sizeof(key), "gda-bufs-%d", comm_->rank());
        buf_to_hex((uint8_t*)my_metas.data(), nbuf * sizeof(BufMeta), hex);
        comm_->pmi.kvs_put(key, hex);
        comm_->pmi.barrier();

        // Collect all peers' buffer info
        for (int r = 0; r < comm_->size(); r++) {
            snprintf(key, sizeof(key), "gda-bufs-%d", r);
            char peer_hex[PMI2_MAX_VALLEN];
            comm_->pmi.kvs_get(key, peer_hex, sizeof(peer_hex));

            std::vector<BufMeta> peer_metas(nbuf);
            hex_to_buf(peer_hex, (uint8_t*)peer_metas.data(), nbuf * sizeof(BufMeta));

            for (int i = 0; i < nbuf; i++) {
                comm_->set_remote_info_by_index(r, i,
                    peer_metas[i].addr, peer_metas[i].key);
            }
        }

        exchanged_ = true;
    }

    /**
     * Queue a put (RDMA write) operation.
     * The operation is not executed until trigger() is called from the GPU.
     * When the transfer completes, the GPU-side completion counter increments.
     *
     * @param src            Local source buffer
     * @param dest_rank      Destination rank
     * @param dest_buf_index Index of destination buffer on remote rank
     * @param size           Transfer size in bytes
     * @param src_offset     Byte offset within source buffer
     * @param dst_offset     Byte offset within destination buffer
     */
    void put(const Buffer& src, int dest_rank, int dest_buf_index,
             size_t size, size_t src_offset = 0, size_t dst_offset = 0) {
        comm_->current_threshold++;
        uint64_t threshold = comm_->current_threshold;
        n_ops_++;

        // Look up remote buffer info
        GdaRemoteInfo ri = comm_->get_remote_info(dest_rank, dest_buf_index);
        if (ri.rma_key == 0 && ri.rma_addr == 0) {
            fprintf(stderr, "Rank %d: remote info not set for rank %d buf %d "
                    "(call exchange() first)\n",
                    comm_->rank(), dest_rank, dest_buf_index);
            exit(1);
        }

        // Compute remote address with offset
        uint64_t remote_addr;
        if (comm_->is_virt_addr_mode()) {
            remote_addr = ri.rma_addr + dst_offset;
        } else {
            remote_addr = (ri.rma_addr - ri.base_addr) + dst_offset;
        }

        // Queue RMA write (triggered by trigger_cntr >= threshold)
        auto* dwq = new DwqWorkBuilder(comm_->rank());
        dwq->queue_rma_write(
            comm_->fabric->domain, comm_->fabric->ep,
            (char*)src.ptr + src_offset, src.desc_, size,
            comm_->av_addrs[dest_rank], remote_addr, ri.rma_key,
            comm_->fabric->trigger_cntr, comm_->fabric->completion_cntr,
            threshold);

        // Queue atomic signal (fires when RMA completes, increments completion_)
        uint64_t atomic_result_addr = comm_->is_virt_addr_mode()
            ? (uint64_t)comm_->atomic_result : 0;
        dwq->queue_atomic_signal(
            comm_->fabric->domain, comm_->fabric->ep,
            comm_->atomic_operand, comm_->mr_atomic_operand->desc,
            comm_->atomic_result, comm_->mr_atomic_result->key,
            atomic_result_addr,
            comm_->fabric->local_addr_in_av,
            comm_->fabric->completion_cntr,
            comm_->atomic_completion_cntr, threshold);

        comm_->pending_ops.push_back(dwq);
    }

    /**
     * Prepare the device context for GPU-side trigger/wait.
     * Call after queueing all put() operations for this batch.
     * Returns a device-accessible pointer to pass to your GPU kernel.
     *
     * The returned pointer is valid until reset() is called.
     */
    DeviceCtx* prepare() {
        // Reset atomic result to 0 before the batch
        uint64_t zero = 0;
        (void)hipMemcpy(comm_->atomic_result, &zero, sizeof(uint64_t),
                        hipMemcpyHostToDevice);
        (void)hipDeviceSynchronize();

        // Update device context (zero-copy: host write visible to GPU)
        h_dev_ctx_->trigger_val_ = comm_->current_threshold;
        h_dev_ctx_->n_ops_ = n_ops_;

        return d_dev_ctx_;
    }

    /**
     * Reset for the next batch of operations.
     * Call after the GPU kernel completes and all transfers are done.
     * Frees internal resources and resets counters.
     */
    void reset() {
        // Wait for all completions on CPU side to ensure DWQ resources are free
        while (fi_cntr_read(comm_->fabric->completion_cntr)
               < comm_->current_threshold) {
            // CQ progress is handled by background thread
        }

        // Clean up
        for (auto* op : comm_->pending_ops) delete op;
        comm_->pending_ops.clear();
        for (auto* wb : comm_->pending_wb_ops) delete wb;
        comm_->pending_wb_ops.clear();

        fi_cntr_set(comm_->fabric->trigger_cntr, 0);
        fi_cntr_set(comm_->fabric->completion_cntr, 0);
        if (comm_->atomic_completion_cntr)
            fi_cntr_set(comm_->atomic_completion_cntr, 0);
        comm_->current_threshold = 0;
        n_ops_ = 0;
    }

    /** Global barrier across all ranks. */
    void barrier() { comm_->barrier(); }

    /** This rank's ID. */
    int rank() const { return comm_->rank(); }

    /** Total number of ranks. */
    int size() const { return comm_->size(); }

    /** GPU device ID. */
    int gpu_id() const { return comm_->gpu_id(); }

    /** Access underlying GdaComm (for advanced/debug usage). */
    GdaComm& comm() { return *comm_; }

private:
    friend class Am;

    GdaComm* comm_;
    DeviceCtx* h_dev_ctx_;   // Host-mapped (pinned)
    DeviceCtx* d_dev_ctx_;   // Device pointer (zero-copy)
    uint64_t n_ops_;
    bool exchanged_;
    std::vector<Buffer> local_bufs_;

    // Hex utilities for PMI exchange
    static void buf_to_hex(const uint8_t* in, size_t len, char* out) {
        static const char* h = "0123456789abcdef";
        for (size_t i = 0; i < len; i++) {
            out[2 * i] = h[(in[i] >> 4) & 0xF];
            out[2 * i + 1] = h[in[i] & 0xF];
        }
        out[2 * len] = '\0';
    }

    static int hexval(char c) {
        if ('0' <= c && c <= '9') return c - '0';
        if ('a' <= c && c <= 'f') return c - 'a' + 10;
        if ('A' <= c && c <= 'F') return c - 'A' + 10;
        return -1;
    }

    static int hex_to_buf(const char* in, uint8_t* out, size_t outlen) {
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
};

// =============================================================================
// Am - Active Message API
// =============================================================================

class Am {
public:
    /**
     * Initialize the Active Message subsystem.
     * Collective - all ranks must call.
     *
     * @param rt           Initialized Runtime (must outlive this Am object)
     * @param ring_slots   Slots per peer inbox ring (power of 2, default 128)
     * @param staging_size Number of staging buffers for queued sends
     */
    Am(Runtime& rt, int ring_slots = 128, int staging_size = 16)
        : rt_(rt),
          am_(new opengda::am::GdaAm(*rt.comm_, ring_slots, staging_size))
    {}

    ~Am() { delete am_; }

    // No copy/move
    Am(const Am&) = delete;
    Am& operator=(const Am&) = delete;

    /**
     * Queue a short active message (header + args only, 64 bytes on wire).
     *
     * @param dest       Destination rank
     * @param handler_id Handler to invoke on receiver
     * @param args       Arguments (6 x uint64_t)
     * @return 0 on success, -1 on error (staging pool exhausted)
     */
    int send(int dest, uint16_t handler_id, const Args& args) {
        return am_->send_handle(dest, handler_id, args);
    }

    /**
     * Queue a long active message (header + args + payload).
     *
     * @param dest        Destination rank
     * @param handler_id  Handler to invoke on receiver
     * @param args        Arguments (6 x uint64_t)
     * @param payload     Payload data (host memory, max 8192 bytes)
     * @param payload_len Payload length
     * @return 0 on success, -1 on error
     */
    int send(int dest, uint16_t handler_id, const Args& args,
             const void* payload, size_t payload_len) {
        return am_->send_payload(dest, handler_id, args, payload, payload_len);
    }

    /**
     * Send a lightweight 8-byte reply (ack) to a peer.
     * More efficient than a full AM for simple acknowledgments.
     *
     * @param dest      Destination rank
     * @param ack_value 8-byte value to deliver
     * @return 0 on success, -1 on error
     */
    int reply(int dest, uint64_t ack_value) {
        return am_->reply(dest, ack_value);
    }

    /**
     * Trigger all queued sends and wait for completion.
     * Resets staging pools for the next batch.
     */
    void flush() {
        am_->trigger_and_wait();
        auto& c = am_->am_ctx.comm;
        c.fast_flush(c.get_current_threshold());
        c.cleanup_pending_ops();
        c.reset_counters();
    }

    /**
     * Poll for incoming active messages (host-side, launches a GPU kernel).
     *
     * @param max_per_peer Max messages to process per peer
     * @return Total messages processed
     */
    int poll(int max_per_peer = 16) {
        return am_->poll_once(max_per_peer);
    }

    /**
     * Get device-accessible AM context for GPU-side polling.
     * Pass this pointer to a GPU kernel that calls gda::am_poll_all().
     */
    opengda::am::am_context_t* device_context() {
        return am_->am_ctx.get_device_context();
    }

    int rank() const { return am_->rank(); }
    int size() const { return am_->size(); }

private:
    Runtime& rt_;
    opengda::am::GdaAm* am_;
};

// =============================================================================
// Device-side AM functions
// Re-export from opengda::am for convenience in gda:: namespace
// =============================================================================

/**
 * Poll all peers for incoming active messages (device function).
 * Call from a GPU kernel. Dispatches received messages to handlers.
 *
 * @param ctx            Device AM context from Am::device_context()
 * @param max_per_peer   Max messages to process per peer
 * @return Total messages processed
 */
__device__ inline int am_poll_all(opengda::am::am_context_t* ctx,
                                  int max_per_peer = 16) {
    return opengda::am::am_poll_all(ctx, max_per_peer);
}

/**
 * Poll a single peer for incoming active messages (device function).
 *
 * @param ctx          Device AM context
 * @param recv_state   Receiver state for the specific peer
 * @param max_poll     Max messages to process
 * @return Number of messages processed
 */
__device__ inline int am_poll_peer(opengda::am::am_context_t* ctx,
                                   opengda::am::am_recv_state_t* recv_state,
                                   int max_poll) {
    return opengda::am::am_poll_peer(ctx, recv_state, max_poll);
}

// Re-export handler IDs for user convenience
using opengda::am::AM_HANDLER_NOOP;
using opengda::am::AM_HANDLER_COUNTER_ADD;
using opengda::am::AM_HANDLER_CHECKSUM;
using opengda::am::AM_HANDLER_ECHO;
using opengda::am::AM_HANDLER_USER_BASE;

}  // namespace gda
