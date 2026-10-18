// Puts after a kernel launch, host side (see include/GICCAfterPut.h).
#include "GICCAfterPut.h"
#include "GICCPassConfig.h"
#include "OmpKernel.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

#include <algorithm>
#include <cstdio>

using namespace llvm;

namespace gicc::pass {

uint64_t afterPutHash(StringRef kernel) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : kernel.bytes()) {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h;
}

StringMap<AfterPutSite> readAfterPutSites() {
    StringMap<AfterPutSite> sites;
    const Config &cfg = getConfig();
    if (!cfg.metaDirSet) return sites;
    std::error_code ec;
    for (sys::fs::directory_iterator it(cfg.metaDir, ec), end; it != end && !ec;
         it.increment(ec)) {
        StringRef file = sys::path::filename(it->path());
        if (!file.starts_with("after-") || !file.ends_with(".json")) continue;
        auto buf = MemoryBuffer::getFile(it->path());
        if (!buf) continue;
        Expected<json::Value> parsed = json::parse((*buf)->getBuffer());
        if (!parsed) {
            consumeError(parsed.takeError());
            continue;
        }
        const json::Object *root = parsed->getAsObject();
        const json::Array *list = root ? root->getArray("sites") : nullptr;
        if (!list) continue;
        for (const json::Value &v : *list) {
            const json::Object *o = v.getAsObject();
            auto kernel = o ? o->getString("kernel") : std::nullopt;
            const json::Array *args = o ? o->getArray("src_args") : nullptr;
            if (!kernel || !args) continue;
            AfterPutSite s;
            for (const json::Value &a : *args)
                if (auto j = a.getAsInteger()) s.srcArgs.push_back(static_cast<unsigned>(*j));
            sites[*kernel] = std::move(s);
        }
    }
    return sites;
}

std::optional<json::Value> loopPutExpr(const SCEV *S, const Function &K) {
    Type *T = S->getType();
    if (T->isIntegerTy() && T->getIntegerBitWidth() > 64) return std::nullopt;
    const std::string ty =
        T->isPointerTy() ? std::string("ptr") : "i" + std::to_string(T->getIntegerBitWidth());
    StringRef op;
    switch (S->getSCEVType()) {
    case scConstant:
        return json::Object{{"op", "const"}, {"ty", ty},
                            {"v", cast<SCEVConstant>(S)->getAPInt().getSExtValue()}};
    case scUnknown: {
        // Formal 0 is the launch environment, which the host does not pass.
        auto *A = dyn_cast<Argument>(cast<SCEVUnknown>(S)->getValue());
        if (!A || A->getParent() != &K || A->getArgNo() == 0) return std::nullopt;
        return json::Object{{"op", "arg"}, {"ty", ty}, {"n", static_cast<int64_t>(A->getArgNo())}};
    }
    case scTruncate:           op = "trunc"; break;
    case scZeroExtend:         op = "zext"; break;
    case scSignExtend:         op = "sext"; break;
    case scPtrToInt:           op = "ptrtoint"; break;
    case scAddExpr:            op = "add"; break;
    case scMulExpr:            op = "mul"; break;
    case scUDivExpr:           op = "udiv"; break;
    case scUMaxExpr:           op = "umax"; break;
    case scSMaxExpr:           op = "smax"; break;
    case scUMinExpr:           op = "umin"; break;
    case scSMinExpr:           op = "smin"; break;
    // Differs from umin only in which operand's poison it passes on.
    case scSequentialUMinExpr: op = "umin"; break;
    default:
        return std::nullopt;
    }
    json::Array ops;
    for (const SCEV *O : S->operands()) {
        auto v = loopPutExpr(O, K);
        if (!v) return std::nullopt;
        ops.push_back(std::move(*v));
    }
    return json::Object{{"op", op}, {"ty", ty}, {"ops", std::move(ops)}};
}

namespace {
SmallString<256> loopPutFile(StringRef kernel) {
    char name[64];
    snprintf(name, sizeof(name), "loop-%016llx.json",
             static_cast<unsigned long long>(afterPutHash(kernel)));
    SmallString<256> path(getConfig().metaDir);
    sys::path::append(path, name);
    return path;
}
}  // namespace

void writeLoopPutSites(StringRef kernel, ArrayRef<LoopPutSite> sites) {
    const Config &cfg = getConfig();
    if (!cfg.metaDirSet) return;
    SmallString<256> path = loopPutFile(kernel);
    if (sites.empty()) {
        sys::fs::remove(path);
        return;
    }
    json::Array puts;
    for (const LoopPutSite &s : sites)
        puts.push_back(json::Object{{"index", static_cast<int64_t>(s.index)},
                                    {"src_arg", static_cast<int64_t>(s.srcArg)},
                                    {"peer", s.peer}, {"dst", s.dst}, {"rel", s.rel},
                                    {"bytes", s.bytes}});
    sys::fs::create_directories(cfg.metaDir);
    std::error_code ec;
    raw_fd_ostream os(path, ec);
    if (ec) {
        errs() << "[gicc-chunk] cannot write " << path << ": " << ec.message() << "\n";
        return;
    }
    os << json::Value(json::Object{{"kernel", kernel}, {"puts", std::move(puts)}}) << "\n";
}

StringMap<SmallVector<LoopPutSite, 1>> readLoopPutSites() {
    StringMap<SmallVector<LoopPutSite, 1>> sites;
    const Config &cfg = getConfig();
    if (!cfg.metaDirSet) return sites;
    std::error_code ec;
    for (sys::fs::directory_iterator it(cfg.metaDir, ec), end; it != end && !ec;
         it.increment(ec)) {
        StringRef file = sys::path::filename(it->path());
        if (!file.starts_with("loop-") || !file.ends_with(".json")) continue;
        auto buf = MemoryBuffer::getFile(it->path());
        if (!buf) continue;
        Expected<json::Value> parsed = json::parse((*buf)->getBuffer());
        if (!parsed) {
            consumeError(parsed.takeError());
            continue;
        }
        const json::Object *root = parsed->getAsObject();
        auto kernel = root ? root->getString("kernel") : std::nullopt;
        const json::Array *puts = root ? root->getArray("puts") : nullptr;
        if (!kernel || !puts) continue;
        SmallVector<LoopPutSite, 1> list;
        for (const json::Value &v : *puts) {
            const json::Object *o = v.getAsObject();
            auto index = o ? o->getInteger("index") : std::nullopt;
            auto srcArg = o ? o->getInteger("src_arg") : std::nullopt;
            const json::Value *peer = o ? o->get("peer") : nullptr;
            const json::Value *dst = o ? o->get("dst") : nullptr;
            const json::Value *rel = o ? o->get("rel") : nullptr;
            const json::Value *bytes = o ? o->get("bytes") : nullptr;
            if (!index || !srcArg || *index < 0 || *index >= kLaunchPostsMax || *srcArg < 0 ||
                !peer || !dst || !rel || !bytes)
                continue;
            LoopPutSite s;
            s.index = static_cast<unsigned>(*index);
            s.srcArg = static_cast<unsigned>(*srcArg);
            s.peer = *peer;
            s.dst = *dst;
            s.rel = *rel;
            s.bytes = *bytes;
            list.push_back(std::move(s));
        }
        sites[*kernel] = std::move(list);
    }
    return sites;
}

namespace {

constexpr StringLiteral kLaunch  = "__tgt_target_kernel";
constexpr StringLiteral kHostPut = "ompx_put_host";
constexpr StringLiteral kLoopDone = "ompx__loop_done";
constexpr StringLiteral kTask = "__kmpc_omp_task";
constexpr StringLiteral kTaskDeps = "__kmpc_omp_task_with_deps";   // with depend clauses
constexpr StringLiteral kTargetTask = "__kmpc_omp_target_task_alloc";
constexpr StringLiteral kTaskwait = "__kmpc_omp_taskwait";

raw_ostream &out() { return errs(); }

struct FoundPut {
    CallInst *put;
    unsigned srcArg;
    Value *base;   // what the launch hands the kernel as argument srcArg
    // The branch conditions that lead from code every launch reaches to the
    // put, each with the value it must have.
    SmallVector<std::pair<Value *, bool>, 2> conds;
    // What the code between the launch and the put may write.
    SmallVector<Instruction *, 4> writes;
    // A put in a loop after the launch posts one put per iteration, for its
    // first `slots` iterations; the host makes the others as before.
    Loop *loop = nullptr;
    PHINode *iv = nullptr;              // an integer induction variable of it
    const SCEV *backedges = nullptr;    // how many times it repeats
    Value *trips = nullptr;             // that, computed in its preheader
    Value *iter = nullptr;              // the iteration the put is in, at the put
    unsigned first = 0;                 // the post of its first iteration
    unsigned slots = 1;
};

// The posts a loop of puts takes at most.
constexpr unsigned kLoopSlots = 6;

struct FoundSite {
    CallInst *launch;
    std::string kernel;
    SmallVector<FoundPut, 2> puts;
    AAResults *AA = nullptr;   // set when loads may move before the launch
    ScalarEvolution *SE = nullptr;
};

struct Analyses {
    DominatorTree &DT;
    PostDominatorTree &PDT;
    LoopInfo &LI;
    ScalarEvolution &SE;
    AAResults &AA;
};

// A launch the puts after which may go from its kernel: a target region
// that waits for the kernel (post and start both the launch), or a nowait
// one whose task the host then waits for (post its task's creation, start
// the taskwait).
struct Launch {
    CallInst *post = nullptr;       // the puts are posted before it
    Instruction *start = nullptr;   // the kernel is over after it
    std::string kernel;
    SmallVector<Value *, 16> args;   // as launchArgs
    const ConstantDataArray *maps = nullptr;
    SmallVector<BasicBlock *, 2> hosts;   // as hostRuns
    SmallVector<Instruction *, 2> writes;   // what runs from post to start writes
};

// The kernel a launch starts: its region id is the global
// ".<kernel>.region_id".
std::string launchedKernel(CallInst *L) {
    auto *G = dyn_cast<GlobalVariable>(L->getArgOperand(4)->stripPointerCasts());
    if (!G) return "";
    StringRef n = G->getName();
    if (!n.consume_front(".") || !n.consume_back(".region_id")) return "";
    return n.str();
}

// Whether call C runs region `kernel` on the host: it calls the region's host
// function, directly or as the microtask of __kmpc_fork_teams / _call.
bool runsRegionOnHost(const CallBase &C, StringRef kernel) {
    auto names = [&](const Value *V) {
        auto *F = dyn_cast<Function>(V->stripPointerCasts());
        if (!F) return false;
        StringRef n = F->getName();
        return n.consume_front(kernel) && (n.empty() || n.starts_with("."));
    };
    return names(C.getCalledOperand()) ||
           any_of(C.args(), [&](const Use &U) { return names(U.get()); });
}

// The blocks that run L's region on the host because its if clause was
// false. Clang branches on the clause, at D, to the launch or to that run,
// which may be one block with the launch's own fallback (the run on the host
// when the launch fails, which lies under L).
SmallVector<BasicBlock *, 2> hostRuns(CallInst *L, StringRef kernel, DominatorTree &DT) {
    SmallVector<BasicBlock *, 2> runs;
    BasicBlock *LB = L->getParent();
    for (BasicBlock &B : *L->getFunction()) {
        if (DT.dominates(LB, &B)) continue;
        if (none_of(B, [&](Instruction &I) {
                auto *C = dyn_cast<CallBase>(&I);
                return C && runsRegionOnHost(*C, kernel);
            }))
            continue;
        BasicBlock *D = DT.findNearestCommonDominator(LB, &B);
        auto *Br = D ? dyn_cast<BranchInst>(D->getTerminator()) : nullptr;
        if (!Br || !Br->isConditional() || Br->getSuccessor(0) == Br->getSuccessor(1)) continue;
        for (unsigned k = 0; k < 2; ++k) {
            BasicBlockEdge toL(D, Br->getSuccessor(k)), other(D, Br->getSuccessor(1 - k));
            if (!DT.dominates(toL, LB)) continue;
            // Every way into B is the other arm or the launch's fallback.
            if (all_of(predecessors(&B), [&](BasicBlock *P) {
                    return DT.dominates(LB, P) || DT.dominates(other, P) ||
                           (P == D && other.getEnd() == &B);
                }))
                runs.push_back(&B);
        }
    }
    return runs;
}

// Whether every path from the entry to I runs L's region first: on the
// device through L, or on the host through one of `hosts`.
bool afterRegion(Instruction *L, ArrayRef<BasicBlock *> hosts, Instruction *I) {
    BasicBlock *LB = L->getParent(), *IB = I->getParent();
    if (IB == LB) return L->comesBefore(I);
    SmallPtrSet<BasicBlock *, 16> seen;
    SmallVector<BasicBlock *, 16> work{&IB->getParent()->getEntryBlock()};
    while (!work.empty()) {
        BasicBlock *B = work.pop_back_val();
        if (B == IB) return false;
        if (B == LB || is_contained(hosts, B) || !seen.insert(B).second) continue;
        append_range(work, successors(B));
    }
    return true;
}

// The one store to `base` + `off` that dominates `at`, or null.
// Of the stores to one slot, the one `at` reads: it dominates `at`, and no
// other can run between them -- a slot stored once, or in each of the
// copies of a loop the optimizer unswitched. Null otherwise.
StoreInst *reachingStore(ArrayRef<StoreInst *> stores, Instruction *at, DominatorTree &DT) {
    StoreInst *found = nullptr;
    for (StoreInst *S : stores)
        if (DT.dominates(S, at) && (!found || DT.dominates(found, S))) found = S;
    if (!found) return nullptr;
    for (StoreInst *S : stores)
        if (S != found && !DT.dominates(S, found) && isPotentiallyReachable(found, S, nullptr, &DT) &&
            isPotentiallyReachable(S, at, nullptr, &DT))
            return nullptr;
    return found;
}

StoreInst *storeTo(Function &F, const Value *base, int64_t off, Instruction *at,
                   DominatorTree &DT, const DataLayout &DL) {
    SmallVector<StoreInst *, 2> stores;
    for (Instruction &I : instructions(F)) {
        auto *S = dyn_cast<StoreInst>(&I);
        if (!S) continue;
        APInt o(DL.getIndexTypeSizeInBits(S->getPointerOperand()->getType()), 0);
        const Value *b =
            S->getPointerOperand()->stripAndAccumulateConstantOffsets(DL, o, true);
        if (b != base || o.getSExtValue() != off) continue;
        stores.push_back(S);
    }
    return reachingStore(stores, at, DT);
}

// The map-type flag of an entry the kernel receives as an argument; the
// others (a mapped pointer's attachment, a struct's members) it does not.
constexpr uint64_t kMapTargetParam = 0x20;
constexpr int64_t kArgsTypes = 32;   // the map types, in __tgt_kernel_arguments

// The map types of launch L's entries, or null.
const ConstantDataArray *launchMapTypes(CallInst *L, DominatorTree &DT, const DataLayout &DL) {
    const Value *kargs = L->getArgOperand(5)->stripPointerCasts();
    if (StoreInst *mt = storeTo(*L->getFunction(), kargs, kArgsTypes, L, DT, DL))
        if (auto *G = dyn_cast<GlobalVariable>(mt->getValueOperand()->stripPointerCasts());
            G && G->isConstant() && G->hasInitializer())
            return dyn_cast<ConstantDataArray>(G->getInitializer());
    return nullptr;
}

// The kernel arguments in a base-pointer array at `arr` + `at0` in F, as
// stored before `at`, by the map types `maps`; see launchArgs.
SmallVector<Value *, 16> entryArgs(Function &F, const Value *arr, int64_t at0,
                                   const ConstantDataArray *maps, Instruction *at,
                                   DominatorTree &DT, const DataLayout &DL,
                                   SmallVectorImpl<uint64_t> *types) {
    SmallVector<Value *, 16> args;
    const int64_t slot = static_cast<int64_t>(DL.getPointerSize());
    const int64_t end = at0 + slot * static_cast<int64_t>(maps->getNumElements());
    DenseMap<int64_t, SmallVector<StoreInst *, 2>> slots;
    for (Instruction &I : instructions(F)) {
        auto *S = dyn_cast<StoreInst>(&I);
        if (!S) continue;
        APInt o(DL.getIndexTypeSizeInBits(S->getPointerOperand()->getType()), 0);
        if (S->getPointerOperand()->stripAndAccumulateConstantOffsets(DL, o, true) != arr)
            continue;
        const int64_t off = o.getSExtValue();
        if (off < at0 || (at0 > 0 && off >= end)) continue;   // the rest of a task
        if (off >= end || (off - at0) % slot) return args;
        slots[off - at0].push_back(S);
    }
    int64_t last = -1;
    for (auto &[off, S] : slots) last = std::max(last, off);
    for (int64_t off = 0; off <= last; off += slot) {
        const uint64_t k = static_cast<uint64_t>(off / slot);
        const uint64_t type = maps->getElementAsInteger(k);
        if (!(type & kMapTargetParam)) continue;
        if (types) types->push_back(type);
        auto it = slots.find(off);
        StoreInst *S = it == slots.end() ? nullptr : reachingStore(it->second, at, DT);
        args.push_back(S ? S->getValueOperand() : nullptr);
    }
    return args;
}

// The values a launch hands its kernel, by argument: the entries of the
// base-pointer array its __tgt_kernel_arguments points at (the third field,
// after two i32s) that are kernel arguments by their map type (the sixth
// field). Null where an entry is stored more than once or not before the
// launch. With `types`, also each argument's map type.
SmallVector<Value *, 16> launchArgs(CallInst *L, DominatorTree &DT, const DataLayout &DL,
                                    SmallVectorImpl<uint64_t> *types = nullptr) {
    SmallVector<Value *, 16> args;
    Function &F = *L->getFunction();
    const Value *kargs = L->getArgOperand(5)->stripPointerCasts();
    StoreInst *bp = storeTo(F, kargs, 8, L, DT, DL);
    if (!bp) return args;
    const ConstantDataArray *maps = launchMapTypes(L, DT, DL);
    if (!maps) return args;   // which entries are arguments is unknown
    return entryArgs(F, bp->getValueOperand()->stripPointerCasts(), 0, maps, L, DT, DL, types);
}

// What a load of a pointer variable reads at `at` when nothing but the
// offload runtime can have written it since: a stack slot stored once,
// before `at`, whose address goes only into other stack slots -- the
// launch's base-pointer array, which the runtime reads (a mapped pointer's
// attachment) but whose host variables it never writes. Null otherwise.
Value *forwarded(LoadInst *Ld, Instruction *at, DominatorTree &DT) {
    auto *A = dyn_cast<AllocaInst>(Ld->getPointerOperand()->stripPointerCasts());
    if (!A || Ld->isVolatile() || Ld->isAtomic()) return nullptr;
    StoreInst *only = nullptr;
    for (User *U : A->users()) {
        if (isa<LoadInst>(U)) continue;
        if (auto *II = dyn_cast<IntrinsicInst>(U); II && II->isLifetimeStartOrEnd()) continue;
        auto *S = dyn_cast<StoreInst>(U);
        if (!S) return nullptr;
        if (S->getPointerOperand() == A) {
            if (only) return nullptr;
            only = S;
        } else if (S->getValueOperand() != A ||
                   !isa<AllocaInst>(getUnderlyingObject(S->getPointerOperand()))) {
            return nullptr;
        }
    }
    if (!only || only->getValueOperand()->getType() != Ld->getType() || !DT.dominates(only, at))
        return nullptr;
    return only->getValueOperand();
}

// getUnderlyingObject, looking through forwarded loads.
const Value *sourceObject(Value *V, Instruction *at, DominatorTree &DT) {
    for (unsigned n = 0; n < 4; ++n) {
        auto *Ld = dyn_cast<LoadInst>(getUnderlyingObject(V));
        Value *f = Ld ? forwarded(Ld, at, DT) : nullptr;
        if (!f) break;
        V = f;
    }
    return getUnderlyingObject(V);
}

constexpr uint64_t kMapFrom = 0x2;   // OMP_MAP_FROM: copied back at the end
// OMP_REQ_UNIFIED_ADDRESS and OMP_REQ_UNIFIED_SHARED_MEMORY.
constexpr uint64_t kReqUnified = 0x002 | 0x008;

// Whether a requires directive lets host and device share memory
// (unified_address, unified_shared_memory). Clang records the directive in
// an offload entry named ".requires" (LLVM 20 on) or registers it with
// __tgt_register_requires (before).
bool unifiedMemory(const Module &M) {
    for (const GlobalVariable &G : M.globals()) {
        if (G.getName() != ".offloading.entry..requires") continue;
        auto *C = G.hasInitializer() ? dyn_cast<ConstantStruct>(G.getInitializer()) : nullptr;
        auto *flags = C && C->getNumOperands() > 7 ? dyn_cast<ConstantInt>(C->getOperand(7)) : nullptr;
        if (!flags || (flags->getZExtValue() & kReqUnified)) return true;
    }
    if (const Function *R = M.getFunction("__tgt_register_requires"))
        for (const User *U : R->users()) {
            auto *C = dyn_cast<CallBase>(U);
            auto *flags = C && C->arg_size() == 1 ? dyn_cast<ConstantInt>(C->getArgOperand(0))
                                                  : nullptr;
            if (!flags || (flags->getZExtValue() & kReqUnified)) return true;
        }
    return false;
}

// Whether a load of host memory reads the same before a launch with map
// types `maps` as after it, as far as the kernel goes. A kernel writes device memory -- its
// copies of what it maps, what it is handed with is_device_ptr -- and host
// memory changes only as the launch copies a map(from) back, or where a
// requires directive has the two share memory. (Memory a program shares
// behind OpenMP's back, pinned host memory handed over as a device
// pointer, it is taken not to read a put's arguments from.)
bool loadsSurvive(const Module &M, const ConstantDataArray *maps) {
    if (unifiedMemory(M) || !maps) return false;
    for (unsigned k = 0; k < maps->getNumElements(); ++k)
        if (maps->getElementAsInteger(k) & kMapFrom) return false;
    return true;
}

// Computes before launch L what the code after it computes: values
// available there already and pure computations of them; with `AA`, loads
// whose memory neither the kernel (loadsSurvive) nor the code between
// (`writes`) changes; and, inside loop `loop`, its induction variables at
// a given iteration. No other loads: the kernel may change memory.
struct Hoister {
    Instruction *L;
    DominatorTree &DT;
    AAResults *AA = nullptr;
    ArrayRef<Instruction *> writes;
    Loop *loop = nullptr;
    ScalarEvolution *SE = nullptr;

    // P's constant step, when it is an induction variable of `loop`.
    std::optional<int64_t> step(PHINode *P) const {
        if (!loop || !SE || P->getParent() != loop->getHeader() || !loop->getLoopPreheader() ||
            !SE->isSCEVable(P->getType()))
            return std::nullopt;
        auto *AR = dyn_cast<SCEVAddRecExpr>(SE->getSCEV(P));
        if (!AR || AR->getLoop() != loop || !AR->isAffine()) return std::nullopt;
        auto *C = dyn_cast<SCEVConstant>(AR->getStepRecurrence(*SE));
        if (!C || C->getAPInt().getSignificantBits() > 64) return std::nullopt;
        return C->getAPInt().getSExtValue();
    }

    bool survives(LoadInst *Ld) const {
        if (!AA || !Ld->isSimple()) return false;
        const MemoryLocation loc = MemoryLocation::get(Ld);
        return none_of(writes, [&](Instruction *W) { return isModSet(AA->getModRefInfo(W, loc)); });
    }

    bool can(Value *V, unsigned depth = 0) const {
        auto *I = dyn_cast<Instruction>(V);
        if (!I || DT.dominates(I, L)) return true;
        if (depth > 32) return false;
        if (auto *Ld = dyn_cast<LoadInst>(I)) {
            if (Value *f = forwarded(Ld, L, DT)) return can(f, depth + 1);
            return survives(Ld) && can(Ld->getPointerOperand(), depth + 1);
        }
        if (auto *P = dyn_cast<PHINode>(I))
            return step(P) && can(P->getIncomingValueForBlock(loop->getLoopPreheader()), depth + 1);
        if (isa<AllocaInst>(I) || I->mayHaveSideEffects() || !isSafeToSpeculativelyExecute(I))
            return false;
        for (Value *op : I->operands())
            if (!can(op, depth + 1)) return false;
        return true;
    }

    // Whether V computes before L without loading memory (but what
    // forwarded() reads): safe to compute whether or not the put runs.
    bool speculable(Value *V, unsigned depth = 0) const {
        auto *I = dyn_cast<Instruction>(V);
        if (!I || DT.dominates(I, L)) return true;
        if (depth > 32) return false;
        if (auto *Ld = dyn_cast<LoadInst>(I)) {
            Value *f = forwarded(Ld, L, DT);
            return f && speculable(f, depth + 1);
        }
        if (isa<PHINode>(I)) return false;
        for (Value *op : I->operands())
            if (!speculable(op, depth + 1)) return false;
        return true;
    }

    // V at B's insertion point, at iteration `iter` (an i64) of `loop`.
    Value *emit(Value *V, IRBuilder<> &B, DenseMap<Value *, Value *> &done, Value *iter) const {
        auto *I = dyn_cast<Instruction>(V);
        if (!I || DT.dominates(I, L)) return V;
        if (Value *c = done.lookup(I)) return c;
        Value *r;
        auto *Ld = dyn_cast<LoadInst>(I);
        if (Value *f = Ld ? forwarded(Ld, L, DT) : nullptr) {
            r = emit(f, B, done, iter);
        } else if (auto *P = dyn_cast<PHINode>(I)) {
            Value *start = emit(P->getIncomingValueForBlock(loop->getLoopPreheader()), B, done, iter);
            Value *by = B.CreateMul(iter, B.getInt64(*step(P)));
            r = P->getType()->isPointerTy()
                    ? B.CreateGEP(B.getInt8Ty(), start, by, P->getName() + ".post")
                    : B.CreateAdd(start, B.CreateSExtOrTrunc(by, P->getType()), P->getName() + ".post");
        } else {
            Instruction *C = I->clone();
            for (unsigned k = 0; k < C->getNumOperands(); ++k)
                C->setOperand(k, emit(I->getOperand(k), B, done, iter));
            B.Insert(C, I->getName() + ".post");
            r = C;
        }
        done[I] = r;
        return r;
    }
};

// Whether loads `before` (ahead of the launch, at `post`) and `after` (past
// it) read one value: of one address, which nothing between them may write
// -- the code in before's block up to the launch, but the OpenMP runtime
// calls that set the launch up (a target task's allocation writes only the
// task), and what H checks for the code past it and the kernel
// (Hoister::survives).
bool sameLoad(const Value *before, const Value *after, const Hoister &H, Instruction *post) {
    auto *La = dyn_cast<LoadInst>(before);
    auto *Lp = dyn_cast<LoadInst>(after);
    if (!La || !Lp || !H.AA || !La->isSimple() || La->getType() != Lp->getType() ||
        La->getPointerOperand() != Lp->getPointerOperand() ||
        !H.survives(const_cast<LoadInst *>(Lp)))
        return false;
    if (La->getParent() != post->getParent() || !La->comesBefore(post)) return false;
    const MemoryLocation loc = MemoryLocation::get(La);
    for (const Instruction *I = La->getNextNode(); I != post; I = I->getNextNode()) {
        if (auto *CB = dyn_cast<CallBase>(I); CB && calleeName(*CB).starts_with("__kmpc_"))
            continue;
        if (I->mayWriteToMemory() && isModSet(H.AA->getModRefInfo(I, loc))) return false;
    }
    return true;
}

// Code that neither orders this rank with another nor can change what a
// put after it reads.
bool benign(Instruction &I) {
    if (isa<FenceInst>(I) || isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I)) return false;
    if (auto *LdI = dyn_cast<LoadInst>(&I)) return !LdI->isVolatile() && !LdI->isAtomic();
    if (auto *S = dyn_cast<StoreInst>(&I))
        return !S->isVolatile() && !S->isAtomic() &&
               isa<AllocaInst>(getUnderlyingObject(S->getPointerOperand()));
    if (auto *CB = dyn_cast<CallBase>(&I)) {
        StringRef n = calleeName(*CB);
        // Another put: it only writes the peer's memory. So does the end of
        // a launch with in-loop puts, which may ring their groups.
        if (n == kHostPut || n == kLoopDone) return true;
        // The region's host fallback: the kernel itself, run on the host.
        if (n.starts_with("__omp_offloading_")) return true;
        if (isa<DbgInfoIntrinsic>(CB) || CB->isLifetimeStartOrEnd()) return true;
        return !CB->mayHaveSideEffects();
    }
    return !I.mayHaveSideEffects();
}

// Whether benign code I writes memory of this rank's host.
bool writesHost(Instruction &I) {
    if (auto *CB = dyn_cast<CallBase>(&I)) {
        StringRef n = calleeName(*CB);
        if (n == kHostPut || n == kLoopDone || CB->isLifetimeStartOrEnd() ||
            isa<DbgInfoIntrinsic>(CB))
            return false;
    }
    return I.mayWriteToMemory();
}

// Checks one put after launch L; fills P when it can be posted.
bool checkPut(const Launch &Ln, CallInst *put, Analyses &A, AAResults *AA, FoundPut &P,
              std::string &why) {
    DominatorTree &DT = A.DT;
    LoopInfo &LI = A.LI;
    Instruction *L = Ln.start;
    ArrayRef<BasicBlock *> hosts = Ln.hosts;
    ArrayRef<Value *> args = Ln.args;
    BasicBlock *LB = L->getParent(), *PB = put->getParent();
    Loop *home = LI.getLoopFor(LB), *in = LI.getLoopFor(PB);
    P.loop = nullptr;
    if (in != home) {
        // A loop of puts: directly inside the launch's loop, with the put on
        // every iteration (it dominates the latch, the one way out).
        BasicBlock *latch = in ? in->getLoopLatch() : nullptr;
        if (!in || in->getParentLoop() != home || !in->getLoopPreheader() || !latch ||
            in->getExitingBlock() != latch || !DT.dominates(PB, latch)) {
            why = "the put is in a loop the launch is not";
            return false;
        }
        P.loop = in;
    }
    auto here = [&](BasicBlock *B) {
        Loop *l = LI.getLoopFor(B);
        return l == home || (P.loop && l == P.loop);
    };
    // Every block on a path from the region to the put: from the launch, or
    // from the region run on the host.
    SmallPtrSet<BasicBlock *, 8> region{PB};
    SmallVector<BasicBlock *, 8> work{PB};
    while (!work.empty()) {
        BasicBlock *B = work.pop_back_val();
        if (B == LB || is_contained(hosts, B)) continue;
        for (BasicBlock *pred : predecessors(B)) {
            bool after = is_contained(hosts, pred) || afterRegion(L, hosts, pred->getTerminator());
            if (!after || !here(pred)) {
                why = "a path reaches the put without the launch";
                return false;
            }
            if (region.insert(pred).second) work.push_back(pred);
        }
    }
    // The launch's host fallback (the region run on the host when the device
    // cannot take it: the branch on a nonzero return) is the kernel itself;
    // the device side then marks nothing handled and the host puts.
    BasicBlock *fallback = nullptr;
    if (auto *Br = dyn_cast<BranchInst>(LB->getTerminator()); Br && Br->isConditional())
        if (auto *Cmp = dyn_cast<ICmpInst>(Br->getCondition()); Cmp && Cmp->getOperand(0) == L)
            if (auto *Z = dyn_cast<ConstantInt>(Cmp->getOperand(1)); Z && Z->isZero())
                fallback = Br->getSuccessor(Cmp->getPredicate() == ICmpInst::ICMP_NE ? 0 : 1);
    // So is a run on the host when the if clause is false, which posts
    // nothing: after_done finds no post and the host puts.
    P.writes.assign(Ln.writes.begin(), Ln.writes.end());
    for (BasicBlock *B : region) {
        if (fallback && B != PB && DT.dominates(fallback, B)) continue;
        if (B != PB && any_of(hosts, [&](BasicBlock *H) { return DT.dominates(H, B); })) continue;
        for (Instruction &I : *B) {
            if (B == LB && (&I == L || !L->comesBefore(&I))) continue;
            // In a loop of puts, all of the put's block runs before later
            // iterations' puts.
            if (B == PB && (&I == put || (!P.loop && !I.comesBefore(put)))) continue;
            if (!benign(I)) {
                why = "code between the launch and the put may synchronize or write the "
                      "source: " + (isa<CallBase>(I) ? calleeName(cast<CallBase>(I)).str()
                                                    : std::string(I.getOpcodeName()));
                return false;
            }
            if (writesHost(I)) P.writes.push_back(&I);
        }
    }
    const Hoister H{Ln.post, DT, AA, P.writes, P.loop, &A.SE};
    // The branches that decide whether the put runs: whether its loop does.
    P.conds.clear();
    BasicBlock *B = P.loop ? P.loop->getLoopPreheader() : PB;
    for (unsigned n = 0; !A.PDT.dominates(B, LB); ++n) {
        BasicBlock *pred = B->getSinglePredecessor();
        auto *Br = pred ? dyn_cast<BranchInst>(pred->getTerminator()) : nullptr;
        // A launch's own branch is to its host fallback; a taskwait's is
        // the program's.
        if (!Br || (pred == LB && Ln.post == Ln.start) || n > 16) {
            why = "cannot tell before the launch whether the put runs";
            return false;
        }
        if (Br->isConditional()) P.conds.push_back({Br->getCondition(), Br->getSuccessor(0) == B});
        B = pred;
    }
    for (auto &[c, v] : P.conds)
        if (!H.can(c)) {
            why = "whether the put runs is decided by something the kernel may change";
            return false;
        }
    for (unsigned k = 0; k < 4; ++k)
        if (!H.can(put->getArgOperand(k))) {
            why = "argument " + std::to_string(k) + " of the put cannot be computed before "
                  "the launch";
            return false;
        }
    if (P.loop) {
        // How many times the loop runs, and which iteration a put is in.
        const SCEV *n = A.SE.getBackedgeTakenCount(P.loop);
        if (isa<SCEVCouldNotCompute>(n) || !n->getType()->isIntegerTy() ||
            SCEVExprContains(n, [&](const SCEV *S) {
                if (auto *D = dyn_cast<SCEVUDivExpr>(S)) return !isa<SCEVConstant>(D->getRHS());
                if (auto *U = dyn_cast<SCEVUnknown>(S)) return !H.can(U->getValue());
                return isa<SCEVAddRecExpr>(S);
            })) {
            why = "how many times the loop of puts runs cannot be computed before the launch";
            return false;
        }
        P.backedges = n;
        P.iv = nullptr;
        for (PHINode &Ph : P.loop->getHeader()->phis())
            if (Ph.getType()->isIntegerTy() && H.step(&Ph).value_or(0) != 0) {
                P.iv = &Ph;
                break;
            }
        if (!P.iv) {
            why = "the loop of puts has no induction variable";
            return false;
        }
    }
    // The source: an offset from something the kernel is handed.
    const Value *obj = sourceObject(put->getArgOperand(2), Ln.post, DT);
    for (unsigned j = 0; j < args.size(); ++j) {
        if (!args[j] || !args[j]->getType()->isPointerTy()) continue;
        const Value *argObj = sourceObject(args[j], Ln.post, DT);
        if (argObj != obj && !sameLoad(argObj, obj, H, Ln.post)) continue;
        P.put = put;
        P.srcArg = j;
        P.base = args[j];
        return true;
    }
    why = "the put's source is not an offset from anything the kernel is handed";
    return false;
}

Launch plainLaunch(CallInst *L, DominatorTree &DT, const DataLayout &DL) {
    Launch Ln;
    Ln.post = L;
    Ln.start = L;
    Ln.kernel = launchedKernel(L);
    if (Ln.kernel.empty()) return Ln;
    Ln.args = launchArgs(L, DT, DL);
    Ln.maps = launchMapTypes(L, DT, DL);
    Ln.hosts = hostRuns(L, Ln.kernel, DT);
    return Ln;
}

// A nowait target region is a task (__kmpc_omp_target_task_alloc, then
// __kmpc_omp_task at TC, or __kmpc_omp_task_with_deps with depend clauses)
// whose entry function launches the kernel from a base-pointer array in the
// task, which the creating function fills. Its kernel is over after a
// taskwait that follows TC with only benign code between them. Tasks it
// depends on may still run after the posts; none of them has puts after
// it, which would have needed a taskwait of their own, before TC.
std::optional<Launch> taskLaunch(CallInst *TC, FunctionAnalysisManager &FAM,
                                 const DataLayout &DL) {
    auto *alloc = dyn_cast<CallInst>(TC->getArgOperand(2)->stripPointerCasts());
    if (!alloc || calleeName(*alloc) != kTargetTask || alloc->arg_size() < 6) return std::nullopt;
    auto *E = dyn_cast<Function>(alloc->getArgOperand(5)->stripPointerCasts());
    if (!E || E->isDeclaration() || E->arg_size() < 2) return std::nullopt;
    CallInst *TL = nullptr;
    for (Instruction &I : instructions(*E))
        if (auto *C = dyn_cast<CallInst>(&I); C && calleeName(*C) == kLaunch) {
            if (TL) return std::nullopt;
            TL = C;
        }
    if (!TL) return std::nullopt;
    auto &EDT = FAM.getResult<DominatorTreeAnalysis>(*E);
    Launch Ln;
    Ln.post = TC;
    Ln.kernel = launchedKernel(TL);
    Ln.maps = launchMapTypes(TL, EDT, DL);
    StoreInst *bp = storeTo(*E, TL->getArgOperand(5)->stripPointerCasts(), 8, TL, EDT, DL);
    if (Ln.kernel.empty() || !Ln.maps || !bp) return std::nullopt;
    // The array, at an offset in the task (the entry's second argument).
    APInt at0(DL.getIndexTypeSizeInBits(bp->getValueOperand()->getType()), 0);
    if (bp->getValueOperand()->stripAndAccumulateConstantOffsets(DL, at0, true) != E->getArg(1) ||
        at0.getSExtValue() <= 0)
        return std::nullopt;
    // Said only when a put follows the task.
    Function &F = *TC->getFunction();
    auto stays = [&](const Twine &why) {
        if (any_of(instructions(F), [&](Instruction &I) {
                auto *C = dyn_cast<CallInst>(&I);
                return C && calleeName(*C) == kHostPut && afterRegion(TC, {}, C);
            }))
            out() << "[gicc-after] " << Ln.kernel << ": the puts after this nowait region stay "
                     "after it: " << why << "\n";
        return std::nullopt;
    };
    for (Instruction *I = TC->getNextNode(); I && !Ln.start; I = I->getNextNode()) {
        if (auto *C = dyn_cast<CallInst>(I); C && calleeName(*C) == kTaskwait) {
            Ln.start = C;
        } else if (!benign(*I)) {
            return stays("code between its task and the taskwait may synchronize: " +
                         (isa<CallBase>(I) ? calleeName(*cast<CallBase>(I))
                                           : StringRef(I->getOpcodeName())));
        } else if (writesHost(*I)) {
            Ln.writes.push_back(I);
        }
    }
    if (!Ln.start) return stays("no taskwait follows its task in the same block");
    Ln.args = entryArgs(F, alloc, at0.getSExtValue(), Ln.maps, TC,
                        FAM.getResult<DominatorTreeAnalysis>(F), DL, nullptr);
    return Ln;
}

bool findSite(const Launch &Ln, ArrayRef<Launch> all, Analyses &A, FoundSite &S) {
    S.launch = Ln.post;
    S.kernel = Ln.kernel;
    if (Ln.args.empty()) return false;
    S.AA = loadsSurvive(*Ln.post->getModule(), Ln.maps) ? &A.AA : nullptr;
    S.SE = &A.SE;
    Function &F = *Ln.post->getFunction();
    unsigned used = 0;   // posts
    for (Instruction &I : instructions(F)) {
        auto *put = dyn_cast<CallInst>(&I);
        if (!put || calleeName(*put) != kHostPut || !afterRegion(Ln.start, Ln.hosts, put)) continue;
        // The launch nearest before the put only.
        if (any_of(all, [&](const Launch &L2) {
                return L2.start != Ln.start && afterRegion(Ln.start, Ln.hosts, L2.start) &&
                       afterRegion(L2.start, L2.hosts, put);
            }))
            continue;
        if (used == kLaunchPostsMax) break;
        FoundPut P;
        std::string why;
        if (checkPut(Ln, put, A, S.AA, P, why)) {
            P.first = used;
            P.slots = P.loop ? std::min(kLoopSlots, kLaunchPostsMax - used) : 1;
            used += P.slots;
            S.puts.push_back(std::move(P));
        } else {
            out() << "[gicc-after] " << S.kernel << ": a put stays after the kernel: " << why
                  << "\n";
        }
    }
    return !S.puts.empty();
}

// What bracket needs computed in a loop of puts, done before it changes
// the control flow (which leaves the loop and scalar evolution analyses
// behind): the loop's trip count, in its preheader, and the iteration
// number at the put. False if the trip count cannot be computed before the
// launch after all.
bool prepareLoops(FoundSite &S, DominatorTree &DT, const DataLayout &DL) {
    for (FoundPut &P : S.puts) {
        if (!P.loop) continue;
#if LLVM_VERSION_MAJOR >= 22   // which dropped the DataLayout argument
        SCEVExpander X(*S.SE, "gicc.trips");
        (void)DL;
#else
        SCEVExpander X(*S.SE, DL, "gicc.trips");
#endif
        P.trips = X.expandCodeFor(P.backedges, P.backedges->getType(),
                                  P.loop->getLoopPreheader()->getTerminator());
        if (!Hoister{S.launch, DT, S.AA, P.writes, P.loop, S.SE}.can(P.trips)) return false;
        IRBuilder<> B(P.put);
        auto *AR = cast<SCEVAddRecExpr>(S.SE->getSCEV(P.iv));
        const APInt step = cast<SCEVConstant>(AR->getStepRecurrence(*S.SE))->getAPInt();
        Value *d = B.CreateSub(P.iv, P.iv->getIncomingValueForBlock(P.loop->getLoopPreheader()));
        if (step.isNegative()) d = B.CreateNeg(d);
        const APInt by = step.abs();
        if (!by.isOne()) d = B.CreateUDiv(d, ConstantInt::get(d->getType(), by));
        P.iter = B.CreateZExtOrTrunc(d, B.getInt64Ty(), "gicc.iter");
    }
    return true;
}

// Posts every put before the launch and makes each one after it only if the
// kernel did not.
void bracket(Module &M, FoundSite &S, DominatorTree &DT) {
    LLVMContext &Ctx = M.getContext();
    Type *I64 = Type::getInt64Ty(Ctx), *I32 = Type::getInt32Ty(Ctx);
    auto *Ptr = PointerType::get(Ctx, 0);
    FunctionCallee post = M.getOrInsertFunction("ompx__after_post", Type::getVoidTy(Ctx), I64,
                                                I32, I32, I32, I32, Ptr, Ptr, I64, I64);
    FunctionCallee done = M.getOrInsertFunction("ompx__after_done", Type::getVoidTy(Ctx), I64,
                                                I32, I32, Ptr, Ptr, I64);
    Constant *hash = ConstantInt::get(I64, afterPutHash(S.kernel));
    DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Eager);
    for (FoundPut &P : S.puts) {
        const Hoister H{S.launch, DT, S.AA, P.writes, P.loop, S.SE};
        Constant *srcArg = ConstantInt::get(I32, P.srcArg);
        // One put that loads nothing: posted armed by its conditions.
        const bool plain =
            !P.loop && all_of(P.conds, [&](auto &c) { return H.speculable(c.first); }) &&
            all_of(seq(0u, 4u), [&](unsigned k) { return H.speculable(P.put->getArgOperand(k)); });
        if (plain) {
            DenseMap<Value *, Value *> hoisted;
            IRBuilder<> B(S.launch);
            Value *armed = B.getTrue();
            for (auto &[c, v] : P.conds) {
                Value *h = H.emit(c, B, hoisted, nullptr);
                armed = B.CreateAnd(armed, v ? h : B.CreateNot(h));
            }
            auto arg = [&](unsigned k) { return H.emit(P.put->getArgOperand(k), B, hoisted, nullptr); };
            Value *src = arg(2);
            Value *rel = B.CreateSub(B.CreatePtrToInt(src, I64), B.CreatePtrToInt(P.base, I64));
            B.CreateCall(post, {hash, ConstantInt::get(I32, P.first), srcArg,
                                B.CreateZExt(armed, I32), arg(0), arg(1), src, rel,
                                B.CreateZExtOrTrunc(arg(3), I64)});
        }
        // Otherwise each post unarmed, then armed once each condition in
        // turn holds, from the outermost: what one loads is safe to load
        // once the ones around it hold.
        for (unsigned j = 0; !plain && j < P.slots; ++j) {
            Constant *slot = ConstantInt::get(I32, P.first + j);
            IRBuilder<>(S.launch).CreateCall(
                post, {hash, slot, srcArg, ConstantInt::get(I32, 0), ConstantInt::get(I32, 0),
                       ConstantPointerNull::get(Ptr), ConstantPointerNull::get(Ptr),
                       ConstantInt::get(I64, 0), ConstantInt::get(I64, 0)});
            DenseMap<Value *, Value *> hoisted;
            Value *iter = P.loop ? ConstantInt::get(I64, j) : nullptr;
            Instruction *at = S.launch;
            auto guard = [&](Value *c) {
                at = SplitBlockAndInsertIfThen(c, at, false, nullptr, &DTU);
            };
            for (auto it = P.conds.rbegin(); it != P.conds.rend(); ++it) {
                IRBuilder<> B(at);
                Value *c = H.emit(it->first, B, hoisted, iter);
                guard(it->second ? c : B.CreateNot(c));
            }
            if (P.loop) {
                IRBuilder<> B(at);
                Value *n = H.emit(P.trips, B, hoisted, iter);
                guard(B.CreateICmpULE(ConstantInt::get(n->getType(), j), n));
            }
            IRBuilder<> B(at);
            auto arg = [&](unsigned k) { return H.emit(P.put->getArgOperand(k), B, hoisted, iter); };
            Value *peer = arg(0), *dst = arg(1), *src = arg(2), *bytes = arg(3);
            Value *rel = B.CreateSub(B.CreatePtrToInt(src, I64), B.CreatePtrToInt(P.base, I64));
            B.CreateCall(post, {hash, slot, srcArg, ConstantInt::get(I32, 1), peer, dst, src, rel,
                                B.CreateZExtOrTrunc(bytes, I64)});
        }
        CallInst *put = P.put;
        Value *slot = ConstantInt::get(I32, P.first);
        if (P.loop) {
            // The first iterations' puts were posted; the others the host
            // makes as before.
            IRBuilder<> A(put);
            Value *posted = A.CreateICmpULT(P.iter, ConstantInt::get(I64, P.slots));
            Instruction *then = nullptr, *other = nullptr;
            SplitBlockAndInsertIfThenElse(posted, put, &then, &other, nullptr, &DTU);
            put->moveBefore(*other->getParent(), other->getIterator());
            IRBuilder<> T(then);
            slot = T.CreateAdd(slot, T.CreateTrunc(P.iter, I32));
            T.CreateCall(done, {hash, slot, put->getArgOperand(0), put->getArgOperand(1),
                                put->getArgOperand(2), T.CreateZExtOrTrunc(put->getArgOperand(3), I64)});
            continue;
        }
        IRBuilder<> A(put);
        A.CreateCall(done, {hash, slot, put->getArgOperand(0), put->getArgOperand(1),
                            put->getArgOperand(2), A.CreateZExtOrTrunc(put->getArgOperand(3), I64)});
        put->eraseFromParent();
    }
}


// ---- in-loop puts (LoopPutSite) ---------------------------------------------

constexpr uint64_t kMapLiteral = 0x100;   // OMP_MAP_LITERAL: handed over as is
constexpr int64_t kArgsFlags = 64;   // in __tgt_kernel_arguments

// What launch L leaves at offset `off` of its __tgt_kernel_arguments: the
// one store there before it, or a zeroing memset over it; null if neither.
Value *launchField(CallInst *L, int64_t off, DominatorTree &DT, const DataLayout &DL) {
    Function &F = *L->getFunction();
    const Value *kargs = L->getArgOperand(5)->stripPointerCasts();
    if (StoreInst *S = storeTo(F, kargs, off, L, DT, DL)) return S->getValueOperand();
    for (Instruction &I : instructions(F)) {
        auto *MS = dyn_cast<MemSetInst>(&I);
        if (!MS || !DT.dominates(MS, L)) continue;
        APInt o(DL.getIndexTypeSizeInBits(MS->getDest()->getType()), 0);
        if (MS->getDest()->stripAndAccumulateConstantOffsets(DL, o, true) != kargs) continue;
        auto *len = dyn_cast<ConstantInt>(MS->getLength());
        auto *val = dyn_cast<ConstantInt>(MS->getValue());
        if (len && val && val->isZero() && o.getSExtValue() <= off &&
            off + 8 <= o.getSExtValue() + static_cast<int64_t>(len->getZExtValue()))
            return ConstantInt::get(Type::getInt64Ty(F.getContext()), 0);
    }
    return nullptr;
}

Value *coerce(IRBuilder<> &B, Value *V, Type *T) {
    Type *S = V->getType();
    if (S == T) return V;
    if (S->isPointerTy() && T->isIntegerTy()) return B.CreatePtrToInt(V, T);
    if (S->isIntegerTy() && T->isPointerTy())
        return B.CreateIntToPtr(B.CreateZExtOrTrunc(V, B.getInt64Ty()), T);
    if (S->isIntegerTy() && T->isIntegerTy()) return B.CreateZExtOrTrunc(V, T);
    return nullptr;
}

// What the launch hands the kernel, by argument, with the map types that
// say which of them reach it unchanged.
struct LaunchView {
    SmallVector<Value *, 16> args;
    SmallVector<uint64_t, 16> types;
};

// Launch argument j as the kernel receives it, as a T; null unless the
// launch passes it as is (a scalar, an is_device_ptr pointer): a mapped
// pointer reaches the kernel translated.
Value *launchArg(IRBuilder<> &B, const LaunchView &LV, uint64_t j, Type *T) {
    if (j >= LV.args.size() || !LV.args[j] || j >= LV.types.size() ||
        !(LV.types[j] & kMapLiteral))
        return nullptr;
    return coerce(B, LV.args[j], T);
}

Type *exprType(LLVMContext &C, StringRef ty) {
    if (ty == "ptr") return PointerType::get(C, 0);
    unsigned bits = 0;
    if (ty.consume_front("i") && !ty.getAsInteger(10, bits) && bits >= 1 && bits <= 64)
        return IntegerType::get(C, bits);
    return nullptr;
}

// An expression of loopPutExpr's, computed before the launch; null if it
// does not make sense or names an argument launchArg will not give.
Value *buildExpr(const json::Value &V, IRBuilder<> &B, const LaunchView &LV, unsigned depth = 0) {
    const json::Object *o = V.getAsObject();
    auto op = o ? o->getString("op") : std::nullopt;
    auto tyName = o ? o->getString("ty") : std::nullopt;
    Type *T = tyName ? exprType(B.getContext(), *tyName) : nullptr;
    if (!op || !T || depth > 64) return nullptr;
    if (*op == "const") {
        auto v = o->getInteger("v");
        return v && T->isIntegerTy() ? ConstantInt::get(T, *v, true) : nullptr;
    }
    if (*op == "arg") {
        auto n = o->getInteger("n");
        return n && *n >= 1 ? launchArg(B, LV, static_cast<uint64_t>(*n - 1), T) : nullptr;
    }
    const json::Array *list = o->getArray("ops");
    if (!list || list->empty()) return nullptr;
    SmallVector<Value *, 4> xs;
    for (const json::Value &e : *list) {
        Value *x = buildExpr(e, B, LV, depth + 1);
        if (!x) return nullptr;
        xs.push_back(x);
    }
    if (*op == "trunc" || *op == "zext" || *op == "sext") {
        if (xs.size() != 1 || !T->isIntegerTy() || !xs[0]->getType()->isIntegerTy()) return nullptr;
        const unsigned from = xs[0]->getType()->getIntegerBitWidth(), to = T->getIntegerBitWidth();
        if (*op == "trunc" ? from < to : from > to) return nullptr;
        return *op == "trunc" ? B.CreateTrunc(xs[0], T)
             : *op == "zext"  ? B.CreateZExt(xs[0], T)
                              : B.CreateSExt(xs[0], T);
    }
    if (*op == "ptrtoint") {
        if (xs.size() != 1 || !xs[0]->getType()->isPointerTy() || !T->isIntegerTy()) return nullptr;
        return B.CreatePtrToInt(xs[0], T);
    }
    if (*op == "add" && T->isPointerTy()) {
        // One pointer, and offsets of its index width.
        Value *base = nullptr, *off = nullptr;
        for (Value *x : xs) {
            if (x->getType()->isPointerTy()) {
                if (base) return nullptr;
                base = x;
            } else if (x->getType()->isIntegerTy()) {
                Value *w = B.CreateSExtOrTrunc(x, B.getInt64Ty());
                off = off ? B.CreateAdd(off, w) : w;
            } else {
                return nullptr;
            }
        }
        if (!base) return nullptr;
        return off ? B.CreateGEP(B.getInt8Ty(), base, off) : base;
    }
    if (!T->isIntegerTy() || any_of(xs, [&](Value *x) { return x->getType() != T; }))
        return nullptr;
    if (*op == "udiv") {
        if (xs.size() != 2) return nullptr;
        // Never a division by zero on the host; the kernel would compute
        // something else there, and say so.
        return B.CreateUDiv(xs[0], B.CreateBinaryIntrinsic(Intrinsic::umax, xs[1],
                                                           ConstantInt::get(T, 1)));
    }
    Intrinsic::ID minmax = *op == "umax" ? Intrinsic::umax
                         : *op == "smax" ? Intrinsic::smax
                         : *op == "umin" ? Intrinsic::umin
                         : *op == "smin" ? Intrinsic::smin
                                         : Intrinsic::not_intrinsic;
    if (*op != "add" && *op != "mul" && minmax == Intrinsic::not_intrinsic) return nullptr;
    Value *r = xs[0];
    for (size_t k = 1; k < xs.size(); ++k)
        r = *op == "add"   ? B.CreateAdd(r, xs[k])
          : *op == "mul"   ? B.CreateMul(r, xs[k])
                           : B.CreateBinaryIntrinsic(minmax, r, xs[k]);
    return r;
}

// Posts the kernel's in-loop puts before launch L, from what L hands it,
// and has ompx__loop_done follow L for each one posted. Returns how many
// were; `why` says why the rest were not.
unsigned postLoopPuts(Module &M, CallInst *L, StringRef kernel, ArrayRef<LoopPutSite> sites,
                      DominatorTree &DT, std::string &why) {
    const DataLayout &DL = M.getDataLayout();
    // A nowait launch returns before the kernel ends: its puts are not
    // over when the call after it would ring them.
    auto *flags = dyn_cast_or_null<ConstantInt>(launchField(L, kArgsFlags, DT, DL));
    if (!flags || (flags->getZExtValue() & 1)) {
        why = "the launch may not wait for the kernel (nowait)";
        return 0;
    }
    LaunchView LV;
    LV.args = launchArgs(L, DT, DL, &LV.types);
    if (LV.args.empty()) {
        why = "cannot read what the launch hands the kernel";
        return 0;
    }
    LLVMContext &Ctx = M.getContext();
    Type *I64 = Type::getInt64Ty(Ctx), *I32 = Type::getInt32Ty(Ctx);
    auto *Ptr = PointerType::get(Ctx, 0);
    FunctionCallee post = M.getOrInsertFunction("ompx__after_post", Type::getVoidTy(Ctx), I64,
                                                I32, I32, I32, I32, Ptr, Ptr, I64, I64);
    FunctionCallee done = M.getOrInsertFunction(kLoopDone, Type::getVoidTy(Ctx), I64, I32);
    Constant *hash = ConstantInt::get(I64, afterPutHash(kernel));
    unsigned posted = 0;
    for (const LoopPutSite &s : sites) {
        IRBuilder<> B(L);
        Instruction *before = L->getPrevNode();
        Value *peer = buildExpr(s.peer, B, LV), *dst = buildExpr(s.dst, B, LV);
        Value *rel = buildExpr(s.rel, B, LV), *bytes = buildExpr(s.bytes, B, LV);
        Value *base = launchArg(B, LV, s.srcArg, Ptr);
        if (peer) peer = coerce(B, peer, I32);
        if (dst) dst = coerce(B, dst, Ptr);
        if (rel) rel = coerce(B, rel, I64);
        if (bytes) bytes = coerce(B, bytes, I64);
        if (!peer || !dst || !rel || !bytes || !base) {
            // Take back what was built for it.
            while (L->getPrevNode() != before) L->getPrevNode()->eraseFromParent();
            why = "put " + std::to_string(s.index) + ": an argument it is computed from is not "
                  "handed to the kernel as is (a mapped pointer?)";
            continue;
        }
        // A negative peer sends nothing, and its dst may be anything.
        Value *armed = B.CreateZExt(B.CreateICmpSGE(peer, ConstantInt::get(I32, 0)), I32);
        Value *src = B.CreateGEP(Type::getInt8Ty(Ctx), base, rel);
        B.CreateCall(post, {hash, ConstantInt::get(I32, s.index), ConstantInt::get(I32, s.srcArg),
                            armed, peer, dst, src, rel, bytes});
        IRBuilder<>(L->getNextNode()).CreateCall(done, {hash, ConstantInt::get(I32, s.index)});
        ++posted;
    }
    return posted;
}

}  // namespace

PreservedAnalyses GICCAfterPutPass::run(Module &M, ModuleAnalysisManager &MAM) {
    const Config &cfg = getConfig();
    const bool discover = cfg.mode == Mode::PutDiscover;
    const bool rewrite = cfg.mode == Mode::ChunkLower && cfg.metaDirSet;
    if (!discover && !rewrite) return PreservedAnalyses::all();
    Triple T(M.getTargetTriple());
    if (T.isNVPTX() || T.isAMDGPU()) return PreservedAnalyses::all();
    if (!M.getFunction(kLaunch)) return PreservedAnalyses::all();
    const bool hostPuts = M.getFunction(kHostPut) != nullptr;
    auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
    StringMap<AfterPutSite> known;
    StringMap<SmallVector<LoopPutSite, 1>> loops;
    if (rewrite) {
        known = readAfterPutSites();
        loops = readLoopPutSites();
    }
    if (!hostPuts && loops.empty()) return PreservedAnalyses::all();

    json::Array recorded;
    bool changed = false;
    // Records (put-discover) or brackets (chunk-lower) the puts after one
    // launch.
    auto afterPuts = [&](FoundSite &S, DominatorTree &DT) {
        json::Array srcArgs;
        SmallVector<unsigned, 8> slots;
        for (const FoundPut &P : S.puts)
            for (unsigned j = 0; j < P.slots; ++j) slots.push_back(P.srcArg);
        for (unsigned a : slots) srcArgs.push_back(static_cast<int64_t>(a));
        out() << "[gicc-after] " << S.kernel << ": " << S.puts.size()
              << " put(s) after the launch can go from the kernel";
        if (slots.size() > S.puts.size())
            out() << ", in " << slots.size() << " posts (a loop's first iterations)";
        out() << "\n";
        if (discover) {
            recorded.push_back(json::Object{{"kernel", S.kernel}, {"src_args", std::move(srcArgs)}});
            return;
        }
        // Bracket only what the first compile recorded the same way: the
        // device side of this unit was built from that record.
        auto it = known.find(S.kernel);
        bool same = it != known.end() && it->second.srcArgs.size() == slots.size();
        for (unsigned i = 0; same && i < slots.size(); ++i)
            same = it->second.srcArgs[i] == slots[i];
        if (!same) {
            out() << "[gicc-after] " << S.kernel << ": not in GICC_META_DIR as found; "
                     "rebuild with GICC_MODE=put-discover first\n";
            return;
        }
        bracket(M, S, DT);
        changed = true;
    };
    for (Function &F : M) {
        if (F.isDeclaration()) continue;
        SmallVector<CallInst *, 4> launches, tasks;
        for (Instruction &I : instructions(F))
            if (auto *CI = dyn_cast<CallInst>(&I); CI && calleeName(*CI) == kLaunch)
                launches.push_back(CI);
            else if (CI && (calleeName(*CI) == kTask || calleeName(*CI) == kTaskDeps))
                tasks.push_back(CI);
        if (launches.empty() && tasks.empty()) continue;
        auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);
        Analyses A{DT, FAM.getResult<PostDominatorTreeAnalysis>(F),
                   FAM.getResult<LoopAnalysis>(F), FAM.getResult<ScalarEvolutionAnalysis>(F),
                   FAM.getResult<AAManager>(F)};
        // The puts after each launch first, all found before any is
        // bracketed: their check reads the code between the launch and
        // them, which bracketing and the in-loop puts' end call change.
        SmallVector<FoundSite, 2> sites;
        if (hostPuts) {
            SmallVector<Launch, 4> all;
            for (CallInst *L : launches) all.push_back(plainLaunch(L, DT, M.getDataLayout()));
            for (CallInst *TC : tasks)
                if (auto Ln = taskLaunch(TC, FAM, M.getDataLayout())) all.push_back(std::move(*Ln));
            for (const Launch &Ln : all) {
                FoundSite S;
                if (findSite(Ln, all, A, S)) sites.push_back(std::move(S));
            }
        }
        if (rewrite)
            for (FoundSite &S : sites) {
                changed |= any_of(S.puts, [](const FoundPut &P) { return P.loop; });
                if (prepareLoops(S, DT, M.getDataLayout())) continue;
                    out() << "[gicc-after] " << S.kernel << ": the puts after the launch stay "
                         "there: a loop's trip count cannot be computed before it\n";
                S.puts.clear();
            }
        for (FoundSite &S : sites)
            if (!S.puts.empty()) afterPuts(S, DT);
        for (CallInst *L : launches) {
            if (!rewrite) continue;
            const std::string kernel = launchedKernel(L);
            auto it = loops.find(kernel);
            if (it == loops.end()) continue;
            std::string why;
            const unsigned posted = postLoopPuts(M, L, kernel, it->second, DT, why);
            if (posted) {
                out() << "[gicc-after] " << kernel << ": " << posted
                      << " in-loop put(s) posted before the launch\n";
                changed = true;
            }
            if (posted < it->second.size())
                out() << "[gicc-after] " << kernel << ": an in-loop put is not posted (under "
                         "DWQ it waits for the next quiet): " << why << "\n";
        }
    }
    if (discover && !recorded.empty()) {
        char name[64];
        snprintf(name, sizeof(name), "after-%016llx.json",
                 static_cast<unsigned long long>(afterPutHash(M.getModuleIdentifier())));
        SmallString<256> path(cfg.metaDir);
        sys::fs::create_directories(path);
        sys::path::append(path, name);
        std::error_code ec;
        raw_fd_ostream os(path, ec);
        if (ec) {
            out() << "[gicc-after] cannot write " << path << ": " << ec.message() << "\n";
        } else {
            os << json::Value(json::Object{{"module", M.getModuleIdentifier()},
                                           {"sites", std::move(recorded)}});
        }
    }
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

}  // namespace gicc::pass
