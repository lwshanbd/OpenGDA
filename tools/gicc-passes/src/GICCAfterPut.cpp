// Puts after a kernel launch, host side (see include/GICCAfterPut.h).
#include "GICCAfterPut.h"
#include "GICCPassConfig.h"
#include "OmpKernel.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
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

raw_ostream &out() { return errs(); }

struct FoundPut {
    CallInst *put;
    unsigned srcArg;
    Value *base;   // what the launch hands the kernel as argument srcArg
    // The branch conditions that lead from code every launch reaches to the
    // put, each with the value it must have.
    SmallVector<std::pair<Value *, bool>, 2> conds;
};

struct FoundSite {
    CallInst *launch;
    std::string kernel;
    SmallVector<FoundPut, 2> puts;
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
bool afterRegion(CallInst *L, ArrayRef<BasicBlock *> hosts, Instruction *I) {
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
StoreInst *storeTo(Function &F, const Value *base, int64_t off, Instruction *at,
                   DominatorTree &DT, const DataLayout &DL) {
    StoreInst *found = nullptr;
    for (Instruction &I : instructions(F)) {
        auto *S = dyn_cast<StoreInst>(&I);
        if (!S) continue;
        APInt o(DL.getIndexTypeSizeInBits(S->getPointerOperand()->getType()), 0);
        const Value *b =
            S->getPointerOperand()->stripAndAccumulateConstantOffsets(DL, o, true);
        if (b != base || o.getSExtValue() != off) continue;
        if (found) return nullptr;
        found = S;
    }
    return found && DT.dominates(found, at) ? found : nullptr;
}

// The values a launch hands its kernel, by argument: the entries of the
// base-pointer array its __tgt_kernel_arguments points at (the third field,
// after two i32s). Null where an entry is stored more than once or not
// before the launch.
SmallVector<Value *, 16> launchArgs(CallInst *L, DominatorTree &DT, const DataLayout &DL) {
    SmallVector<Value *, 16> args;
    Function &F = *L->getFunction();
    const Value *kargs = L->getArgOperand(5)->stripPointerCasts();
    StoreInst *bp = storeTo(F, kargs, 8, L, DT, DL);
    if (!bp) return args;
    const Value *arr = bp->getValueOperand()->stripPointerCasts();
    DenseMap<int64_t, StoreInst *> slots;
    SmallPtrSet<StoreInst *, 4> ambiguous;
    for (Instruction &I : instructions(F)) {
        auto *S = dyn_cast<StoreInst>(&I);
        if (!S) continue;
        APInt o(DL.getIndexTypeSizeInBits(S->getPointerOperand()->getType()), 0);
        if (S->getPointerOperand()->stripAndAccumulateConstantOffsets(DL, o, true) != arr)
            continue;
        auto [it, fresh] = slots.try_emplace(o.getSExtValue(), S);
        if (!fresh) ambiguous.insert(it->second);
    }
    const int64_t slot = static_cast<int64_t>(DL.getPointerSize());
    int64_t last = -1;
    for (auto &[off, S] : slots) last = std::max(last, off);
    for (int64_t off = 0; off <= last; off += slot) {
        StoreInst *S = slots.lookup(off);
        args.push_back(S && !ambiguous.count(S) && DT.dominates(S, L) ? S->getValueOperand()
                                                                      : nullptr);
    }
    return args;
}

// Can V be computed before `at`: it is available there already, or a pure
// computation of values that are. No loads: the kernel may change memory.
bool canHoist(Value *V, Instruction *at, DominatorTree &DT, unsigned depth = 0) {
    auto *I = dyn_cast<Instruction>(V);
    if (!I || DT.dominates(I, at)) return true;
    if (depth > 32 || isa<PHINode>(I) || isa<LoadInst>(I) || isa<AllocaInst>(I) ||
        I->mayHaveSideEffects() || !isSafeToSpeculativelyExecute(I))
        return false;
    for (Value *op : I->operands())
        if (!canHoist(op, at, DT, depth + 1)) return false;
    return true;
}

Value *hoist(Value *V, Instruction *at, DominatorTree &DT, DenseMap<Value *, Value *> &done) {
    auto *I = dyn_cast<Instruction>(V);
    if (!I || DT.dominates(I, at)) return V;
    if (Value *c = done.lookup(I)) return c;
    Instruction *C = I->clone();
    for (unsigned k = 0; k < C->getNumOperands(); ++k)
        C->setOperand(k, hoist(I->getOperand(k), at, DT, done));
    C->insertInto(at->getParent(), at->getIterator());
    C->setName(I->getName() + ".post");
    done[I] = C;
    return C;
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

// Checks one put after launch L; fills P when it can be posted.
bool checkPut(CallInst *L, ArrayRef<BasicBlock *> hosts, CallInst *put, ArrayRef<Value *> args,
              DominatorTree &DT, PostDominatorTree &PDT, LoopInfo &LI, FoundPut &P,
              std::string &why) {
    BasicBlock *LB = L->getParent(), *PB = put->getParent();
    if (LI.getLoopFor(PB) != LI.getLoopFor(LB)) {
        why = "the put is in a loop the launch is not";
        return false;
    }
    // Every block on a path from the region to the put: from the launch, or
    // from the region run on the host.
    SmallPtrSet<BasicBlock *, 8> region{PB};
    SmallVector<BasicBlock *, 8> work{PB};
    while (!work.empty()) {
        BasicBlock *B = work.pop_back_val();
        if (B == LB || is_contained(hosts, B)) continue;
        for (BasicBlock *pred : predecessors(B)) {
            bool after = is_contained(hosts, pred) || afterRegion(L, hosts, pred->getTerminator());
            if (!after || LI.getLoopFor(pred) != LI.getLoopFor(LB)) {
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
    for (BasicBlock *B : region) {
        if (fallback && B != PB && DT.dominates(fallback, B)) continue;
        if (B != PB && any_of(hosts, [&](BasicBlock *H) { return DT.dominates(H, B); })) continue;
        for (Instruction &I : *B) {
            if (B == LB && (&I == L || !L->comesBefore(&I))) continue;
            if (B == PB && (&I == put || !I.comesBefore(put))) continue;
            if (!benign(I)) {
                why = "code between the launch and the put may synchronize or write the "
                      "source: " + (isa<CallBase>(I) ? calleeName(cast<CallBase>(I)).str()
                                                    : std::string(I.getOpcodeName()));
                return false;
            }
        }
    }
    // The branches that decide whether the put runs.
    P.conds.clear();
    BasicBlock *B = PB;
    for (unsigned n = 0; !PDT.dominates(B, LB); ++n) {
        BasicBlock *pred = B->getSinglePredecessor();
        auto *Br = pred ? dyn_cast<BranchInst>(pred->getTerminator()) : nullptr;
        if (!Br || pred == LB || n > 16) {
            why = "cannot tell before the launch whether the put runs";
            return false;
        }
        if (Br->isConditional()) P.conds.push_back({Br->getCondition(), Br->getSuccessor(0) == B});
        B = pred;
    }
    for (auto &[c, v] : P.conds)
        if (!canHoist(c, L, DT)) {
            why = "whether the put runs is decided by something the kernel may change";
            return false;
        }
    for (unsigned k = 0; k < 4; ++k)
        if (!canHoist(put->getArgOperand(k), L, DT)) {
            why = "argument " + std::to_string(k) + " of the put cannot be computed before "
                  "the launch";
            return false;
        }
    // The source: an offset from something the kernel is handed.
    const Value *obj = getUnderlyingObject(put->getArgOperand(2));
    for (unsigned j = 0; j < args.size(); ++j) {
        if (!args[j] || !args[j]->getType()->isPointerTy()) continue;
        if (getUnderlyingObject(args[j]) != obj) continue;
        P.put = put;
        P.srcArg = j;
        P.base = args[j];
        return true;
    }
    why = "the put's source is not an offset from anything the kernel is handed";
    return false;
}

bool findSite(CallInst *L, DominatorTree &DT, PostDominatorTree &PDT, LoopInfo &LI,
              const DataLayout &DL, FoundSite &S) {
    S.launch = L;
    S.kernel = launchedKernel(L);
    if (S.kernel.empty()) return false;
    SmallVector<Value *, 16> args = launchArgs(L, DT, DL);
    if (args.empty()) return false;
    Function &F = *L->getFunction();
    SmallVector<BasicBlock *, 2> hosts = hostRuns(L, S.kernel, DT);
    for (Instruction &I : instructions(F)) {
        auto *put = dyn_cast<CallInst>(&I);
        if (!put || calleeName(*put) != kHostPut || !afterRegion(L, hosts, put)) continue;
        // The launch nearest before the put only.
        bool nearer = false;
        for (Instruction &J : instructions(F))
            if (auto *L2 = dyn_cast<CallInst>(&J); L2 && L2 != L && calleeName(*L2) == kLaunch &&
                                                   afterRegion(L, hosts, L2) &&
                                                   afterRegion(L2, hostRuns(L2, launchedKernel(L2), DT), put))
                nearer = true;
        if (nearer) continue;
        if (S.puts.size() == 8) break;   // OMPX_PIPE_AFTER_MAX
        FoundPut P;
        std::string why;
        if (checkPut(L, hosts, put, args, DT, PDT, LI, P, why)) {
            S.puts.push_back(std::move(P));
        } else {
            out() << "[gicc-after] " << S.kernel << ": a put stays after the kernel: " << why
                  << "\n";
        }
    }
    return !S.puts.empty();
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
    DenseMap<Value *, Value *> hoisted;
    IRBuilder<> B(S.launch);
    for (unsigned i = 0; i < S.puts.size(); ++i) {
        FoundPut &P = S.puts[i];
        auto arg = [&](unsigned k) { return hoist(P.put->getArgOperand(k), S.launch, DT, hoisted); };
        Value *armed = B.getTrue();
        for (auto &[c, v] : P.conds) {
            Value *h = hoist(c, S.launch, DT, hoisted);
            armed = B.CreateAnd(armed, v ? h : B.CreateNot(h));
        }
        Value *src = arg(2);
        Value *rel = B.CreateSub(B.CreatePtrToInt(src, I64), B.CreatePtrToInt(P.base, I64));
        B.CreateCall(post, {hash, ConstantInt::get(I32, i), ConstantInt::get(I32, P.srcArg),
                            B.CreateZExt(armed, I32), arg(0), arg(1), src, rel,
                            B.CreateZExtOrTrunc(arg(3), I64)});
        IRBuilder<> A(P.put);
        A.CreateCall(done, {hash, ConstantInt::get(I32, i), P.put->getArgOperand(0),
                            P.put->getArgOperand(1), P.put->getArgOperand(2),
                            A.CreateZExtOrTrunc(P.put->getArgOperand(3), I64)});
        P.put->eraseFromParent();
    }
}

// ---- in-loop puts (LoopPutSite) ---------------------------------------------

constexpr uint64_t kMapLiteral = 0x100;   // OMP_MAP_LITERAL: handed over as is
constexpr int64_t kArgsTypes = 32, kArgsFlags = 64;   // in __tgt_kernel_arguments

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
    const ConstantDataArray *types = nullptr;
};

// Launch argument j as the kernel receives it, as a T; null unless the
// launch passes it as is (a scalar, an is_device_ptr pointer): a mapped
// pointer reaches the kernel translated.
Value *launchArg(IRBuilder<> &B, const LaunchView &LV, uint64_t j, Type *T) {
    if (j >= LV.args.size() || !LV.args[j] || !LV.types || j >= LV.types->getNumElements() ||
        !(LV.types->getElementAsInteger(j) & kMapLiteral))
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
    LV.args = launchArgs(L, DT, DL);
    if (Value *t = launchField(L, kArgsTypes, DT, DL))
        if (auto *G = dyn_cast<GlobalVariable>(t->stripPointerCasts());
            G && G->hasDefinitiveInitializer())
            LV.types = dyn_cast<ConstantDataArray>(G->getInitializer());
    if (LV.args.empty() || !LV.types) {
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
        for (const FoundPut &P : S.puts) srcArgs.push_back(static_cast<int64_t>(P.srcArg));
        out() << "[gicc-after] " << S.kernel << ": " << S.puts.size()
              << " put(s) after the launch can go from the kernel\n";
        if (discover) {
            recorded.push_back(json::Object{{"kernel", S.kernel}, {"src_args", std::move(srcArgs)}});
            return;
        }
        // Bracket only what the first compile recorded the same way: the
        // device side of this unit was built from that record.
        auto it = known.find(S.kernel);
        bool same = it != known.end() && it->second.srcArgs.size() == S.puts.size();
        for (unsigned i = 0; same && i < S.puts.size(); ++i)
            same = it->second.srcArgs[i] == S.puts[i].srcArg;
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
        SmallVector<CallInst *, 4> launches;
        for (Instruction &I : instructions(F))
            if (auto *CI = dyn_cast<CallInst>(&I); CI && calleeName(*CI) == kLaunch)
                launches.push_back(CI);
        if (launches.empty()) continue;
        auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);
        auto &PDT = FAM.getResult<PostDominatorTreeAnalysis>(F);
        auto &LI = FAM.getResult<LoopAnalysis>(F);
        for (CallInst *L : launches) {
            // The puts after the launch first: their check reads the code
            // between it and them, which the in-loop puts' end call joins.
            FoundSite S;
            if (hostPuts && findSite(L, DT, PDT, LI, M.getDataLayout(), S)) afterPuts(S, DT);
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
