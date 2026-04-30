#pragma once

#include "MetadataIO.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

#include <string>
#include <vector>

namespace gicc::pass {

struct GICCLaunchSite {
    llvm::CallInst   *callsite       = nullptr;
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
