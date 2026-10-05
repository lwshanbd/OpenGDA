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
// device side could not rewrite simply leaves the put to the host. So does a
// region with an if clause that runs on the host: that path posts nothing,
// and the put counts as after the kernel when every path to it runs the
// region first, on either side. Under
// DWQ the post also queues a put to a peer that is not IPC-mapped, on a
// trigger of its own that the kernel rings once the source is written; a
// kernel that does not leaves the host to ring it after the launch. One it
// cannot read at all: ROCm clang emits a combined construct whose loop
// makes no call as a specialized kernel (big-jump-loop / no-loop), with no
// distribute loop; -fno-openmp-target-big-jump-loop and
// -fno-openmp-target-no-loop keep the ordinary form.
//
// An ompx_pipelined_put in a kernel's loop is the other way round: the
// device side knows it and the host side does not, and the device side is
// compiled first. So GICC_MODE=chunk-lower records each one the kernel
// lowers in GICC_META_DIR/loop-<kernel hash>.json, its arguments as
// expressions of the kernel's formals, and the host side of the same
// compile posts it before every launch of the kernel whose arguments those
// formals are passed as they are (a scalar or an is_device_ptr pointer,
// not a mapped one) and that waits for the kernel (no nowait), after the
// puts that follow the launch (ompx__after_post), and calls ompx__loop_done
// right after it. The kernel checks the post against what it computes
// (ompx__loop_after) and uses it only when it agrees. Under DWQ that is what
// lets a kernel release such a put as it is written to a peer that is not
// IPC-mapped; without the post, the put waits for the next quiet.
//
// Why moving the put into the kernel is sound: the put may land at the peer
// at any moment of the synchronization epoch it is made in, so a race-free
// program has the peer leave dst alone for all of it. Writes made earlier in
// the same epoch keep that guarantee, provided nothing between the launch and
// the put orders this rank with another -- which the host side checks (no
// call with side effects, no fence, atomic or volatile access, no store but
// to the stack), along with there being nothing that could change the
// source after the kernel.

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <optional>
#include <string>

namespace llvm {
class SCEV;
class Function;
}  // namespace llvm

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

// The puts one launch posts (OMPX_PIPE_AFTER_MAX): those after it first,
// then the kernel's in-loop ones.
constexpr unsigned kLaunchPostsMax = 8;

// An in-loop put as the device side records it: its index among the
// launch's posts, the kernel argument its source is an offset from (as in
// AfterPutSite), and the put's peer, dst, source offset and length. Each
// is a SCEV of the kernel's formals as JSON: {"op", "ty", ...} with op
// const ("v"), arg ("n": formal n, launch argument n - 1), trunc, zext,
// sext, ptrtoint, add, mul, udiv, umax, smax, umin, smin ("ops"), and ty
// "ptr" or "i<bits>".
struct LoopPutSite {
    unsigned index = 0;
    unsigned srcArg = 0;
    llvm::json::Value peer = nullptr, dst = nullptr, rel = nullptr, bytes = nullptr;
};

// The SCEV S of kernel K's formals as such an expression; nullopt when it
// has anything else in it.
std::optional<llvm::json::Value> loopPutExpr(const llvm::SCEV *S, const llvm::Function &K);

// Records kernel's in-loop puts, replacing what an earlier compile left;
// removes the record when there are none.
void writeLoopPutSites(llvm::StringRef kernel, llvm::ArrayRef<LoopPutSite> sites);

// Every kernel's in-loop puts recorded in GICC_META_DIR.
llvm::StringMap<llvm::SmallVector<LoopPutSite, 1>> readLoopPutSites();

struct GICCAfterPutPass : llvm::PassInfoMixin<GICCAfterPutPass> {
    llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);
    static llvm::StringRef name() { return "GICCAfterPutPass"; }
    static bool isRequired() { return true; }
};

}  // namespace gicc::pass
