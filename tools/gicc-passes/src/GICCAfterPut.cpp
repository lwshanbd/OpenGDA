// Puts after a kernel launch, host side (see include/GICCAfterPut.h).
#include "GICCAfterPut.h"
#include "GICCPassConfig.h"
#include "OmpKernel.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
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

namespace {

constexpr StringLiteral kLaunch  = "__tgt_target_kernel";
constexpr StringLiteral kHostPut = "ompx_put_host";

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
        // Another put: it only writes the peer's memory.
        if (n == kHostPut) return true;
        // The region's host fallback: the kernel itself, run on the host.
        if (n.starts_with("__omp_offloading_")) return true;
        if (isa<DbgInfoIntrinsic>(CB) || CB->isLifetimeStartOrEnd()) return true;
        return !CB->mayHaveSideEffects();
    }
    return !I.mayHaveSideEffects();
}

// Checks one put after launch L; fills P when it can be posted.
bool checkPut(CallInst *L, CallInst *put, ArrayRef<Value *> args, DominatorTree &DT,
              PostDominatorTree &PDT, LoopInfo &LI, FoundPut &P, std::string &why) {
    BasicBlock *LB = L->getParent(), *PB = put->getParent();
    if (LI.getLoopFor(PB) != LI.getLoopFor(LB)) {
        why = "the put is in a loop the launch is not";
        return false;
    }
    // Every block on a path from the launch to the put.
    SmallPtrSet<BasicBlock *, 8> region{PB};
    SmallVector<BasicBlock *, 8> work{PB};
    while (!work.empty()) {
        BasicBlock *B = work.pop_back_val();
        if (B == LB) continue;
        for (BasicBlock *pred : predecessors(B)) {
            if (!DT.dominates(LB, pred) || LI.getLoopFor(pred) != LI.getLoopFor(LB)) {
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
    for (BasicBlock *B : region) {
        if (fallback && B != PB && DT.dominates(fallback, B)) continue;
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
    for (Instruction &I : instructions(F)) {
        auto *put = dyn_cast<CallInst>(&I);
        if (!put || calleeName(*put) != kHostPut || !DT.dominates(L, put)) continue;
        // The launch nearest before the put only.
        bool nearer = false;
        for (Instruction &J : instructions(F))
            if (auto *L2 = dyn_cast<CallInst>(&J); L2 && L2 != L && calleeName(*L2) == kLaunch &&
                                                   DT.dominates(L, L2) && DT.dominates(L2, put))
                nearer = true;
        if (nearer) continue;
        if (S.puts.size() == 8) break;   // OMPX_PIPE_AFTER_MAX
        FoundPut P;
        std::string why;
        if (checkPut(L, put, args, DT, PDT, LI, P, why)) {
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
                                                I32, I32, I32, I32, Ptr, I64, I64);
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
                            B.CreateZExt(armed, I32), arg(0), arg(1), rel,
                            B.CreateZExtOrTrunc(arg(3), I64)});
        IRBuilder<> A(P.put);
        A.CreateCall(done, {hash, ConstantInt::get(I32, i), P.put->getArgOperand(0),
                            P.put->getArgOperand(1), P.put->getArgOperand(2),
                            A.CreateZExtOrTrunc(P.put->getArgOperand(3), I64)});
        P.put->eraseFromParent();
    }
}

}  // namespace

PreservedAnalyses GICCAfterPutPass::run(Module &M, ModuleAnalysisManager &MAM) {
    const Config &cfg = getConfig();
    const bool discover = cfg.mode == Mode::PutDiscover;
    const bool rewrite = cfg.mode == Mode::ChunkLower && cfg.metaDirSet;
    if (!discover && !rewrite) return PreservedAnalyses::all();
    Triple T(M.getTargetTriple());
    if (T.isNVPTX() || T.isAMDGPU()) return PreservedAnalyses::all();
    if (!M.getFunction(kHostPut) || !M.getFunction(kLaunch)) return PreservedAnalyses::all();
    auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
    StringMap<AfterPutSite> known;
    if (rewrite) known = readAfterPutSites();

    json::Array recorded;
    bool changed = false;
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
            FoundSite S;
            if (!findSite(L, DT, PDT, LI, M.getDataLayout(), S)) continue;
            json::Array srcArgs;
            for (const FoundPut &P : S.puts) srcArgs.push_back(static_cast<int64_t>(P.srcArg));
            out() << "[gicc-after] " << S.kernel << ": " << S.puts.size()
                  << " put(s) after the launch can go from the kernel\n";
            if (discover) {
                recorded.push_back(json::Object{{"kernel", S.kernel}, {"src_args", std::move(srcArgs)}});
                continue;
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
                continue;
            }
            bracket(M, S, DT);
            changed = true;
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
