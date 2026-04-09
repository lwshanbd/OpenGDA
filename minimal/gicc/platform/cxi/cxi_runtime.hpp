/**
 * cxi_runtime.hpp - libfabric/CXI implementation of gicc::Runtime
 *
 * Wraps the existing minimal/ gda::Runtime (which manages GdaComm,
 * FabricDwqContext, MemoryRegion and DwqWorkBuilder) and re-exposes it under
 * the unified gicc:: API. The translation is mechanical:
 *
 *   gicc::Runtime::register_buffer  -> gda::Runtime::register_buffer
 *   gicc::Runtime::exchange         -> gda::Runtime::exchange
 *   gicc::Runtime::put_no_db        -> gda::Runtime::put     (renamed)
 *   gicc::Runtime::prepare(p, i)    -> gda::Runtime::prepare (args ignored;
 *                                       the CXI DeviceCtx is global, one
 *                                       trigger MMIO drives every queued op)
 *   gicc::Runtime::reset            -> gda::Runtime::reset
 *   gicc::Runtime::barrier          -> gda::Runtime::barrier
 *   gicc::DeviceCtx layout matches gda::DeviceCtx 1:1 (reinterpret_cast).
 *
 * The MPI_Comm constructor argument is accepted for source compatibility
 * with the mlx5 backend but is currently unused: bootstrap is performed via
 * Cray PMI2 inside GdaComm. A future revision can route address exchange
 * through MPI to fully match the mlx5 path.
 */
#pragma once

#include <mpi.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

#include "gicc/gicc_types.hpp"
#include "gicc/platform/cxi/cxi_device.cuh"

// Pull in the existing libfabric/CXI implementation. opengda.hpp defines
// gda::Runtime, gda::Buffer, gda::DeviceCtx and the underlying GdaComm.
#include "opengda.hpp"

namespace gicc {

// Sanity: gicc::DeviceCtx and gda::DeviceCtx must be layout-compatible so
// that reinterpret_cast in prepare() is well-defined.
static_assert(sizeof(DeviceCtx) == sizeof(gda::DeviceCtx),
              "gicc::DeviceCtx and gda::DeviceCtx must have identical layout");

/**
 * Token returned by put_no_db. Identifies a specific queued operation by its
 * monotonically-increasing trigger threshold. Used by:
 *   - prepare_trigger(Token) to build a DeviceCtx whose flush() fires exactly
 *     up to that threshold (overlap-style: separate trigger kernel),
 *   - wait(Token) for host-side completion of that op.
 * The simple "queue many → prepare() → kernel does flush+quiet" pattern can
 * ignore the return value entirely.
 */
struct Token {
    uint64_t threshold;
};

class Runtime {
public:
    explicit Runtime(MPI_Comm comm = MPI_COMM_WORLD)
        : impl_(), mpi_comm_(comm),
          h_trigger_ctx_(nullptr), d_trigger_ctx_(nullptr)
    {
        (void)mpi_comm_;  // currently unused; PMI2 drives bootstrap inside GdaComm

        // Allocate a second device context (zero-copy pinned) used by the
        // "trigger-only" pattern: a kernel that just calls gicc::flush(ctx)
        // to fire a specific threshold while the host waits separately.
        // The completion_/n_ops_ fields are unused on this code path.
        (void)hipHostMalloc(&h_trigger_ctx_, sizeof(DeviceCtx),
                            hipHostMallocMapped);
        (void)hipHostGetDevicePointer((void**)&d_trigger_ctx_, h_trigger_ctx_, 0);
        h_trigger_ctx_->trigger_addr_ = impl_.comm().get_trigger_addr();
        h_trigger_ctx_->completion_   = nullptr;
        h_trigger_ctx_->trigger_val_  = 0;
        h_trigger_ctx_->n_ops_        = 0;
    }

    ~Runtime() {
        if (h_trigger_ctx_) (void)hipHostFree(h_trigger_ctx_);
    }

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    //--------------------------------------------------------------------------
    // Buffer registration
    //--------------------------------------------------------------------------
    Buffer register_buffer(void* buf, size_t size, bool is_device) {
        gda::Buffer gb = impl_.register_buffer(buf, size, is_device);

        Buffer b;
        b.ptr   = gb.ptr;
        b.size  = gb.size;
        b.addr  = (uint64_t)gb.ptr;
        // CXI: lkey/rkey have no direct ibv equivalent. We surface the
        // libfabric remote key as rkey, and use the buffer index as a stand-in
        // for lkey (the actual local descriptor is held inside GdaComm and
        // looked up by index when put_no_db is called).
        b.lkey  = (uint32_t)gb.index;
        b.rkey  = (uint32_t)(gb.key_ & 0xFFFFFFFFu);
        b.index = gb.index;

        // Stash the gda::Buffer so we can hand it back to gda::Runtime::put
        // without re-registering.
        if ((int)gda_bufs_.size() <= gb.index) gda_bufs_.resize(gb.index + 1);
        gda_bufs_[gb.index] = gb;
        return b;
    }

    //--------------------------------------------------------------------------
    // Collective metadata exchange
    //--------------------------------------------------------------------------
    void exchange() { impl_.exchange(); }

    RemoteBufferInfo remote_buffer(int rank, int buf_index) const {
        // The libfabric remote info lives inside GdaComm; we expose only
        // what fits the unified type.
        auto& c = const_cast<gda::Runtime&>(impl_).comm();
        auto ri = c.get_remote_info(rank, buf_index);
        RemoteBufferInfo r;
        r.addr = ri.rma_addr;
        r.rkey = (uint32_t)(ri.rma_key & 0xFFFFFFFFu);
        return r;
    }

    //--------------------------------------------------------------------------
    // Host-side put_no_db: queue an RDMA write into the libfabric Deferred
    // Work Queue. The NIC will not execute it until the GPU later calls
    // gicc::flush(ctx) from a kernel.
    //
    // This is the libfabric counterpart to mlx5's __device__ gicc::put_no_db.
    // The contract is identical: "queue, don't ring the doorbell". The only
    // difference is the call site (host vs device) — see cxi_device.cuh.
    //--------------------------------------------------------------------------
    Token put_no_db(const Buffer& src, int dest_rank, int dest_buf_index,
                    size_t size, size_t src_offset = 0, size_t dst_offset = 0)
    {
        const gda::Buffer& gb = gda_bufs_.at(src.index);
        impl_.put(gb, dest_rank, dest_buf_index, size, src_offset, dst_offset);
        // gda::Runtime::put bumps comm_->current_threshold to the just-queued
        // op's threshold. Surface it to the caller as a Token.
        return Token{ impl_.comm().current_threshold };
    }

    //--------------------------------------------------------------------------
    // prepare_trigger — build a DeviceCtx whose flush() fires exactly up to
    // `tok.threshold`. Unlike prepare(), this does NOT touch atomic_result,
    // does NOT set n_ops_ (so a kernel calling gicc::quiet(ctx) returns
    // immediately), and is intended to be used together with host-side
    // wait(tok). Pattern:
    //
    //     auto tok = rt.put_no_db(...);
    //     auto* ctx = rt.prepare_trigger(tok);
    //     trigger_kernel<<<>>>(ctx);   // gicc::flush(ctx) inside
    //     compute_kernel<<<>>>(...);   // overlapping work
    //     hipDeviceSynchronize();
    //     rt.wait(tok);                // host wait
    //--------------------------------------------------------------------------
    DeviceCtx* prepare_trigger(Token tok) {
        h_trigger_ctx_->trigger_val_ = tok.threshold;
        return d_trigger_ctx_;
    }

    //--------------------------------------------------------------------------
    // Host-side wait for a specific token (poll the libfabric completion
    // counter). The background CQ progress thread inside FabricDwqContext
    // drives provider progress.
    //--------------------------------------------------------------------------
    void wait(Token tok) {
        auto& c = impl_.comm();
        while (fi_cntr_read(c.fabric->completion_cntr) < tok.threshold) {
            // background thread handles progress
        }
    }

    //--------------------------------------------------------------------------
    // Prepare a GPU-side context for the queued batch.
    //
    // The peer_rank / remote_buf_index parameters are accepted for API
    // symmetry with mlx5 but ignored: the CXI DeviceCtx is global to the
    // process — a single trigger counter MMIO drives every operation that
    // was queued via put_no_db().
    //--------------------------------------------------------------------------
    DeviceCtx* prepare(int peer_rank = -1, int remote_buf_index = -1) {
        (void)peer_rank;
        (void)remote_buf_index;
        gda::DeviceCtx* d = impl_.prepare();
        return reinterpret_cast<DeviceCtx*>(d);
    }

    void reset()   { impl_.reset(); }
    void barrier() { impl_.barrier(); }

    int rank()   const { return impl_.rank(); }
    int size()   const { return impl_.size(); }
    int gpu_id() const { return impl_.gpu_id(); }

    // Escape hatch for advanced users that need the underlying gda::Runtime.
    gda::Runtime& gda_runtime() { return impl_; }

private:
    gda::Runtime             impl_;
    MPI_Comm                 mpi_comm_;
    DeviceCtx*               h_trigger_ctx_;  // pinned host (mapped)
    DeviceCtx*               d_trigger_ctx_;  // device pointer (zero-copy)
    std::vector<gda::Buffer> gda_bufs_;
};

} // namespace gicc
