#pragma once
// What the worksharing loop of an OpenMP offload kernel writes: per store, a
// D-dimensional box (a dense 1-D range is the D = 1 case) in kernel values,
// plus the stores it could not describe.

#include "AccessDecomposition.h"
#include "OmpKernel.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/PassManager.h"

#include <optional>
#include <string>
#include <utility>

namespace gicc::pass {

// x > 0 (signed), or x != 0 (an unsigned bound); under the trust that trip
// counts stay below 2^63 either makes a trip count positive.
struct EntryFact {
    const llvm::SCEV *x;
    bool nonzeroOnly;
};

// A store writes base + offset + sum(stride_d * q_d), q_d in [0, extent_d),
// every point once as the loop runs -- in kernel values (i64), and only
// when the kernel's loop runs at all (see WriteSet::preconditions).
struct BoxWrite {
    llvm::StoreInst *st;
    llvm::Value     *obj;           // the kernel formal written (`base`)
    llvm::SmallVector<const llvm::SCEV *, 4> extent, stride;   // outermost first
    const llvm::SCEV *offset;
    uint64_t storeSize;
    bool unconditional;             // runs on every iteration
    bool trustedTrunc;              // relied on the loop-variable trunc rule
};

struct WriteSet {
    llvm::SmallVector<BoxWrite, 4> boxes;
    // Every described store of the outlined region, by the object it writes.
    llvm::DenseMap<llvm::Value *, llvm::SmallVector<llvm::StoreInst *, 2>> storesTo;
    // Stores it could not describe (UnknownStore::Record only).
    llvm::SmallVector<std::pair<llvm::StoreInst *, std::string>, 2> unknown;
    // What holds whenever the loop runs: the conditions of the branches that
    // lead to it, in kernel values. Box extents mean nothing otherwise (n * m
    // is positive for n = m = -1 too).
    llvm::SmallVector<EntryFact, 4> preconditions;
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

// What holds on every path into `target`, read off the comparisons of the
// branches whose taken edge dominates it: a < b gives b - a > 0, a <= b
// gives b - a + 1 > 0, x != 0 and x >u 0 give x != 0.
llvm::SmallVector<EntryFact, 4>
factsOnEntry(llvm::BasicBlock *target, llvm::DominatorTree &DT,
                llvm::ScalarEvolution &SE);

}  // namespace gicc::pass
