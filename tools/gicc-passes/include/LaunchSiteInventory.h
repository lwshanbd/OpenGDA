#pragma once

#include "MetadataIO.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

#include <string>
#include <vector>

namespace gicc::pass {

struct GICCLaunchSite {
    // CallBase covers both CallInst (regular call) and InvokeInst
    // (exception-aware invoke). Real C++ source compiles to invoke for
    // any function the optimizer can't prove nothrow.
    llvm::CallBase   *callsite       = nullptr;
    llvm::Function   *launchWrapper  = nullptr;   // gicc::launch<K> instantiation
    std::string       kernelMangled;              // _Z11halo_kernel...
    KernelTemplate    kernelTemplate;             // loaded from meta JSON
    bool              haveTemplate   = false;
};

struct GICCLaunchInventory {
    std::vector<GICCLaunchSite> sites;
    bool valid = false;
};

}  // namespace gicc::pass
