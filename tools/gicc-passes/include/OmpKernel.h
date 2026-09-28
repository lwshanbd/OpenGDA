#pragma once
// The structure of an OpenMP offload kernel as clang emits it for a teams
// region with a `distribute parallel for`, and the value provenance needed
// to see through generic-mode globalization.
//
//   kernel K:   __kmpc_distribute_static_init(..., &lb, &ub, &stride, ...)
//               loop over the team's blocks:
//                   __kmpc_parallel_51(..., outlined, wrapper, args[LB, UB, captures...])
//               __kmpc_distribute_static_fini
//   outlined O: __kmpc_for_static_init(...)  thread share of [LB, UB]
//
// Trusted, not proven: __kmpc_distribute_static_init partitions [lb0, ub0]
// among teams without gaps or overlap, __kmpc_for_static_init partitions a
// block [LB, UB] among a team's threads the same way, and clang passes the
// current block's LB / UB in slots 0 and 1 of __kmpc_parallel_51's argument
// array, captured value k in slot k.

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include <optional>
#include <string>

namespace gicc::pass {

// Operand positions in the device runtime calls clang emits.
namespace kmpc {
// __kmpc_parallel_51
constexpr unsigned ParallelFn = 5, ParallelWrapper = 6, ParallelArgs = 7;
// __kmpc_{distribute,for}_static_init_*
constexpr unsigned InitSched = 2, InitLower = 4, InitUpper = 5, InitStride = 6,
                   InitChunk = 8;
// Outlined region formals: gtid*, btid*, then the argument-array slots.
constexpr unsigned OutlinedFirstSlot = 2, OutlinedLB = 2, OutlinedUB = 3;
// __kmpc_distribute_static_init schedule with a dist_schedule chunk.
constexpr int SchedDistChunked = 91;
// NVPTX / AMDGPU team-shared memory, and global memory.
constexpr unsigned SharedAddrSpace = 3;
constexpr unsigned GlobalAddrSpace = 1;
}  // namespace kmpc

// GICC entry points the analyses recognise.
constexpr llvm::StringLiteral kPipelinedPut = "ompx_pipelined_put";
constexpr llvm::StringLiteral kPlainPut     = "ompx_put";

llvm::StringRef calleeName(const llvm::CallBase &CB);
// Calls that cannot write user memory, or whose writes the trusted runtime
// contract covers (they only touch their bound slots).
bool isBenignCall(const llvm::CallBase &CB);
bool isCallTo(const llvm::Instruction &I, llvm::StringRef prefix);
// An offload kernel entry (not one of its outlined helpers).
bool isOffloadKernel(const llvm::Function &F);
// A readable name for a kernel object: its IR name or arg#N.
std::string objName(const llvm::Value *V);
std::string scevStr(const llvm::SCEV *S);
// V without pointer casts, an inttoptr, and one integer cast.
llvm::Value *stripIntCasts(llvm::Value *V);
// The unique store to Obj (after pointer-cast stripping) in F, or null.
llvm::StoreInst *uniqueStoreTo(llvm::Function &F, const llvm::Value *Obj);

class OmpKernel {
public:
    // The kernel's worksharing structure, or nullopt with `why` set.
    static std::optional<OmpKernel> locate(llvm::Function &K, std::string &why);

    llvm::Function   &K;
    llvm::CallInst   *distInit = nullptr, *distFini = nullptr, *parallel = nullptr;
    llvm::Function   *outlined = nullptr;
    llvm::AllocaInst *argsArray = nullptr;
    const llvm::DataLayout &DL;
    bool distSigned = true;   // __kmpc_distribute_static_init_{4,8} vs _{4,8}u

    // A kernel value followed back through phis whose inputs agree and loads
    // of team-shared globals with a single store; V itself when nothing
    // simpler is known.
    llvm::Value *resolve(llvm::Value *V, unsigned depth = 0) const;
    // The kernel value stored in slot `idx` of the parallel argument array.
    llvm::Value *slotValue(unsigned idx) const;
    // The kernel value an outlined-function capture holds (by value, or
    // loaded through the captured variable's address); null if unknown.
    llvm::Value *resolveOutlined(llvm::Value *V) const;
    // S with every unknown replaced by what it resolves to.
    const llvm::SCEV *normalize(llvm::ScalarEvolution &SE, const llvm::SCEV *S) const;
    // An outlined-function SCEV in terms of kernel values; null when some
    // part has no kernel counterpart.
    const llvm::SCEV *toKernel(const llvm::SCEV *S, llvm::ScalarEvolution &SK) const;
    // S extended to T the way the distribute loop reads its bounds.
    const llvm::SCEV *extDist(llvm::ScalarEvolution &SE, const llvm::SCEV *S,
                              llvm::Type *T) const;
    // The distribute bound slot for operand `opnd` of the init call.
    const llvm::Value *distSlot(unsigned opnd) const;
    // The stores of lb0 and ub0 that dominate the init call.
    std::optional<std::pair<llvm::StoreInst *, llvm::StoreInst *>>
    distBounds(llvm::DominatorTree &DT) const;
    // Every thread of a team runs the kernel's code outside its parallel
    // regions: the exec mode in the kernel environment has the SPMD bit.
    // False for a generic kernel, even one the device link may yet make
    // SPMD.
    bool spmd() const;

private:
    explicit OmpKernel(llvm::Function &K)
        : K(K), DL(K.getParent()->getDataLayout()) {}
};

}  // namespace gicc::pass
