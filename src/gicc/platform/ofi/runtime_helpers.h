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

// OpenMP-DWQ trigger arming (used by GICCOmpHostDiscovery). Called AFTER the
// synthesized trace has enqueued the region's DWQ ops and BEFORE the
// __tgt_target_kernel launch. Re-arms DeviceCtx::trigger_val_ to the post-trace
// delta, because the OpenMP flow calls Runtime::prepare() before the region
// (when no ops are staged yet) and so would otherwise leave trigger_val_=0 and
// the kernel's flush would never fire the descriptors. No-op-safe on null rt.
void gicc_runtime_arm_dwq_trigger(gicc::Runtime *rt);

// Device-side helpers (called from device IR emitted by GICCDeviceLowering).
volatile std::uint64_t *gicc_runtime_trigger_addr(gicc::Runtime *rt);
std::uint64_t           gicc_runtime_trigger_val (gicc::Runtime *rt);

// Host-mirror lookup (used by the pass-synthesized DWQ trace function when
// a kernel formal carries a "host-mirrored" annotation: at trace time the
// pass needs to read the array's contents to pre-stage one DWQ descriptor
// per element, and that data lives in a host-side mirror registered via
// Runtime::register_host_mirror).  Returns nullptr if no mirror is known
// for `dev_ptr` (= unregistered or invalid pointer).
const void* gicc_runtime_host_mirror_of(gicc::Runtime *rt, const void* dev_ptr);

#ifdef GICC_CPU_PROXY
// Returns the device-mapped pointer to the lazily-started proxy ring.
// Reserved for future LTO-pass plumbing; the MVP path writes the value
// directly into DeviceCtx::proxy_ring inside Runtime::prepare(), so this
// helper is currently uncalled.
void* gicc_runtime_proxy_ring_device_ptr(gicc::Runtime *rt);
#endif

}  // extern "C"
