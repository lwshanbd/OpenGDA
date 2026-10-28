#pragma once
// A kernel split around a deferred fence (GICC_MODE=chunk-lower).
//
// A program that fences after its puts and then launches a kernel waits for
// the transfers before the kernel starts, although most of the kernel's
// iterations usually touch none of what was transferred (a stencil sweep
// reads the received halo only near its edges). This pass lets those
// iterations run while the transfers are in flight:
//
//     ompx_fence();               ompx__fence_defer();
//     ...                   ->    ...
//     #pragma omp target ...      if (ompx__split_begin(kernel, n, accesses, N))
//     for (...) ...                   while (ompx__split_next(&tripcount))
//                                         launch;   // one range of the loop
//                                 else
//                                     launch;
//
// The ranges come first those that touch none of the epoch's transfers,
// then, after the fence ompx__split_next makes between them, the others.
//
// Off unless asked for: GICC_SPLIT=1 splits every kernel that can be, and a
// tuning file's "split": true or false per kernel overrides that.
// GICC_SPLIT_DUMP=1 prints each function rewritten and every flush point; at
// run time OMPX_SPLIT=0 makes every deferred fence at once and
// OMPX_SPLIT_STATS=1 has ompx_finalize print where the split launches spent
// their time.
//
// Device side: every offload kernel gets a record in GICC_META_DIR
// (split-<hash>.json) saying whether it may communicate itself (any read of
// the GiOMP device context, an atomic, a call the pass cannot see into).
// A kernel that may not, is SPMD, has a chunked distribute loop from 0 and
// no lastprivate, does not ask its team number, and whose every load and
// store of what it is handed is a box of its loop's iterations
// (collectAccesses) is split: its record holds the boxes and the iteration
// count as expressions of its formals, and the kernel reads at its start the
// range of iterations the launch is to run (ompx__split_window), which
// replaces its distribute loop's bounds.
//
// Host side: when every kernel the unit launches has a record and none of
// them communicates, the launches of the split kernels that wait for them
// and hand over as is what their boxes are computed from are rewritten as
// above; ompx_fence, and ompx_quiet with an ompx_barrier after it, become
// ompx__fence_defer; and ompx__fence_flush goes before every call that may
// see what the fence orders or wait on another rank -- anything not
// defined in the unit, not an ompx_ entry point (the runtime makes the
// fence first itself) and not known to do neither -- and before each
// return from a function code outside the unit may have called. The
// runtime decides each time which iterations the epoch's transfers leave
// alone (gicc/omp.h, ompx__split_begin).
//
// Trusted: host code reaches device memory only through the OpenMP and
// ompx runtimes (no requires unified_shared_memory, which turns this off),
// and a unit's kernels are launched by one host thread at a time.

#include "llvm/IR/PassManager.h"

namespace gicc::pass {

struct GICCSplitLaunchPass : llvm::PassInfoMixin<GICCSplitLaunchPass> {
    llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);
    static llvm::StringRef name() { return "GICCSplitLaunchPass"; }
};

}  // namespace gicc::pass
