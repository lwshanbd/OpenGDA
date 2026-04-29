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

}  // namespace gicc::pass
