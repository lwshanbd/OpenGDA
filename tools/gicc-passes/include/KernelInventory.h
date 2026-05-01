#pragma once

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

#include <string>
#include <vector>

namespace gicc::pass {

enum class GICCOpKind { PutNoDb, GetNoDb, Flush, Quiet };

const char *opKindName(GICCOpKind k);

struct GICCCallSite {
    llvm::CallInst *CI = nullptr;
    GICCOpKind      kind;
    std::string     siteId;             // <TU_basename>:<line>:<kernel>::<idx>
    unsigned        callIndexInKernel = 0;
    // Capability bit computed by HK Analysis. True when every non-ctx
    // argument of this site is host-knowable; false when any arg
    // depends on per-thread state (threadIdx, device load, ...) and
    // therefore must be routed to CPU_PROXY_ENQUEUE rather than
    // IPC_PUSH / DWQ_TRIGGER / IPC_OR_DWQ / DWQ_BATCHED.
    bool            hk_capable = true;
    // Empty when hk_capable=true; populated for diagnostics with the
    // first failing-argument reason ("arg N not host-knowable: ...").
    std::string     hk_fail_reason;
};

struct GICCKernelInfo {
    llvm::Function           *kernel = nullptr;
    std::string               mangledName;
    std::string               simpleName;
    std::vector<GICCCallSite> sites;
};

struct GICCKernelInventory {
    std::vector<GICCKernelInfo> kernels;
    bool valid = false;
};

// True if F has the AMDGPU_KERNEL or PTX_Kernel calling convention.
bool isGPUKernel(const llvm::Function &F);

// Classify a CallInst by looking up the callee's mangled name. Returns
// true if it's a recognized gicc:: API call and writes the kind to
// `out`. Substring matching against the Itanium-mangled identifier:
//
//   gicc::put_no_db   → contains "10put_no_db"
//   gicc::get_no_db   → contains "10get_no_db"
//   gicc::flush       → contains "5flushE"
//   gicc::quiet       → contains "5quietE"
bool classifyGICCCall(const llvm::CallInst &CI, GICCOpKind &out);

// Walk F's instructions; populate `info.sites` with every GICC call,
// indexed in source order. `info.kernel` / `info.mangledName` /
// `info.simpleName` are also filled in.
void collectGICCSites(llvm::Function &F, GICCKernelInfo &info);

}  // namespace gicc::pass
