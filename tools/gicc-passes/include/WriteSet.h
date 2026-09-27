#pragma once
// What the worksharing loop of an OpenMP offload kernel writes, per kernel
// object: dense 1-D ranges (base + c*i + d) and collapsed boxes
// (AccessDecomp), plus the stores it could not describe.

#include "AccessDecomposition.h"
#include "OmpKernel.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/PassManager.h"

#include <optional>
#include <string>
#include <utility>

namespace gicc::pass {

// base + c*i + d over the worksharing IV i.
struct DenseWrite {
    const llvm::SCEV *c = nullptr;   // bytes per iteration
    const llvm::SCEV *d = nullptr;   // constant byte offset
    uint64_t storeSize = 0;
};

struct BoxWrite {
    llvm::StoreInst *st;
    llvm::Value     *obj;             // the kernel formal written
    AccessDecomp     box;             // in outlined-function values
    bool             unconditional;   // runs on every iteration
};

struct WriteSet {
    llvm::DenseMap<llvm::Value *, DenseWrite> dense;
    llvm::SmallVector<BoxWrite, 4> boxes;
    // Every store of the outlined region, by the kernel object it writes.
    llvm::DenseMap<llvm::Value *, llvm::SmallVector<llvm::StoreInst *, 2>> storesTo;
    // Stores it could not describe (UnknownStore::Record only).
    llvm::SmallVector<std::pair<llvm::StoreInst *, std::string>, 2> unknown;
    // Some loop exits on "i.next < UB + 1": equivalent to "i <= UB" only
    // while UB + 1 does not wrap, which the caller must prove.
    bool ubPlusOne = false;
};

// What to do with a store whose address cannot be described: give up on
// the whole region, or list it in WriteSet::unknown and go on.
enum class UnknownStore { Fail, Record };

std::optional<WriteSet> collectWrites(const OmpKernel &KI,
                                      llvm::FunctionAnalysisManager &FAM,
                                      UnknownStore policy, std::string &why);

}  // namespace gicc::pass
