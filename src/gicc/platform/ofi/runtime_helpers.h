/**
 * runtime_helpers.h - C ABI helpers exposed to LTO-emitted IR.
 *
 * The LTO host pass (GICCDispatchLowering) emits IR that calls these
 * symbols by their C name. Each helper is a thin wrapper around
 * gicc::Runtime internals: the dispatch lowering pass synthesizes
 * trace functions, and those functions only ever interact with the
 * runtime through this fixed C ABI. The implementation file is built
 * alongside the user binary so the symbols resolve at link time.
 */
#pragma once

#include <cstddef>
#include <cstdint>

#include "gicc/platform/ofi/internal/gpu_device_context.hpp"

namespace gicc { class Runtime; }

extern "C" {

// IPC dispatch helpers (used by the IPC_PUSH lowering).
void* gicc_runtime_peer_ipc_base(gicc::Runtime *rt, int peer, int buf_idx);
void* gicc_runtime_local_buf_base(gicc::Runtime *rt, int buf_idx);
GpuStream_t gicc_runtime_ipc_stream(gicc::Runtime *rt);

// Returns ipc_streams_[idx], or ipc_streams_[0] if idx is out of range.
GpuStream_t gicc_runtime_ipc_stream_indexed(gicc::Runtime *rt, int idx);

// DWQ dispatch helper (used by the DWQ_TRIGGER lowering). Mirrors the
// Runtime::put_no_db DWQ host-stage path: queues an RMA WRITE descriptor
// against the shared completion counter at threshold mono_total_ops_.
void gicc_runtime_dwq_enqueue(gicc::Runtime *rt,
                               int          peer,
                               int          dst_buf,
                               std::size_t  dst_off,
                               int          src_buf,
                               std::size_t  src_off,
                               std::size_t  size);

// DWQ batched dispatch helper (used by the DWQ_BATCHED lowering).
// Queues N RMA WRITE descriptors in one host call. All N descriptors
// share the same trigger threshold (mono_total_ops_ after += n_ops),
// so the NIC fires a single trigger after the kernel terminates and
// the libfabric per-call overhead is amortized across the batch.
//
// The arrays are caller-owned, all of length n_ops. Element i
// corresponds to the i'th queued op:
//   peers[i], dst_bufs[i], dst_offs[i], src_bufs[i], src_offs[i], sizes[i]
void gicc_runtime_dwq_enqueue_batched(gicc::Runtime    *rt,
                                       int               n_ops,
                                       const int        *peers,
                                       const int        *dst_bufs,
                                       const std::size_t *dst_offs,
                                       const int        *src_bufs,
                                       const std::size_t *src_offs,
                                       const std::size_t *sizes);

// Device-side helpers (called from device IR emitted by GICCDeviceLowering).
volatile std::uint64_t *gicc_runtime_trigger_addr(gicc::Runtime *rt);
std::uint64_t           gicc_runtime_trigger_val (gicc::Runtime *rt);

#ifdef GICC_CPU_PROXY
// Returns the device-mapped pointer to the lazily-started proxy ring.
// Reserved for future LTO-pass plumbing; the MVP path writes the value
// directly into DeviceCtx::proxy_ring inside Runtime::prepare(), so this
// helper is currently uncalled.
void* gicc_runtime_proxy_ring_device_ptr(gicc::Runtime *rt);
#endif

}  // extern "C"
