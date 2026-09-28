#pragma once
// Puts after a kernel launch: an ompx_put the host makes right after a
// synchronous target region, from an object the kernel was handed, becomes a
// pipelined put of that kernel -- with no marker in the source.
//
// The device side of a translation unit is compiled before its host side,
// so the build runs twice:
//
//   GICC_MODE=put-discover (host)   finds each launch followed by such puts
//                                   and records, per kernel, which kernel
//                                   argument each put's source is an offset
//                                   from, in GICC_META_DIR/after-*.json.
//   GICC_MODE=chunk-lower           host: posts each put before the launch
//                                   (ompx__after_post) and makes it after the
//                                   launch only if the kernel did not
//                                   (ompx__after_done). Device: the kernel
//                                   reads the post (ompx__after_put) and
//                                   sends it as an ompx_pipelined_put after
//                                   its loop would, when that can be proven.
//
// The two sides meet at run time only, through the post, so a kernel the
// device side could not rewrite simply leaves the put to the host. One it
// cannot read at all: ROCm clang emits a combined construct whose loop
// makes no call as a specialized kernel (big-jump-loop / no-loop), with no
// distribute loop; -fno-openmp-target-big-jump-loop and
// -fno-openmp-target-no-loop keep the ordinary form.
//
// Why moving the put into the kernel is sound: the put may land at the peer
// at any moment of the synchronization epoch it is made in, so a race-free
// program has the peer leave dst alone for all of it. Writes made earlier in
// the same epoch keep that guarantee, provided nothing between the launch and
// the put orders this rank with another -- which the host side checks (no
// call with side effects, no fence, atomic or volatile access, no store but
// to the stack), along with there being nothing that could change the
// source after the kernel.

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/PassManager.h"

#include <cstdint>
#include <string>

namespace gicc::pass {

// FNV-1a of a kernel's name, the key both sides post and look up by.
uint64_t afterPutHash(llvm::StringRef kernel);

// What the first compile found for one kernel: per put, in program order,
// the kernel argument its source is an offset from (0 is the first captured
// value, the kernel's formal 1).
struct AfterPutSite {
    llvm::SmallVector<unsigned, 2> srcArgs;
};

// Every site recorded in GICC_META_DIR, by kernel name; empty when the
// build did not name a directory.
llvm::StringMap<AfterPutSite> readAfterPutSites();

struct GICCAfterPutPass : llvm::PassInfoMixin<GICCAfterPutPass> {
    llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);
    static llvm::StringRef name() { return "GICCAfterPutPass"; }
    static bool isRequired() { return true; }
};

}  // namespace gicc::pass
