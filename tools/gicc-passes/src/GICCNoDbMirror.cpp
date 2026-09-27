// GICC_MODE=chunk-lower, device side: mirror stores into posted
// ompx_put_no_db sources (see include/GICCNoDbMirror.h).
//
// What the quiet relies on, per word w of a posted source: if w is marked
// with the current epoch, the peer's copy of w equals ours. The pass keeps
// that true by construction:
//   - a store fully inside a posted source is repeated into the peer, and
//     marks the words it covers when it covers them whole;
//   - a store that is repeated but covers a word only in part leaves the
//     mark as it was (the peer got the same bytes we did);
//   - every other write that may touch a source -- a store straddling its
//     edge, an atomic, a memory intrinsic, a call the pass cannot see into
//     -- sets `poison`, and the quiet then sends the sources whole.
// Nothing here needs to know which kernel writes what: a word nobody marked
// is simply sent at the quiet, so a missed store costs time, not data.
//
// The state is the device global `ompx__nodb` (struct ompx_nodb_state in
// gicc/omp.h); its layout is repeated here as byte offsets.
#include "GICCNoDbMirror.h"
#include "GICCPassConfig.h"

#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

namespace gicc::pass {
namespace {

// struct ompx_nodb_state (gicc/omp.h).
constexpr uint64_t kLo = 0, kHi = 8, kN = 16, kEpoch = 20, kPoison = 24;
constexpr uint64_t kEntries = 32, kEntrySize = 32;
// struct ompx_nodb_entry.
constexpr uint64_t kSrc = 0, kBytes = 8, kPeer = 16, kMap = 24;
constexpr uint64_t kStateSize = kEntries + 8 * kEntrySize;

constexpr const char *kStateName = "ompx__nodb";
constexpr const char *kMirrorName = "__gicc_nodb_mirror";
// Marks the pass's own functions and the ones it already instrumented, so a
// second run (say, at the link as well) neither re-instruments nor wraps
// the mirror function -- which LTO may have renamed.
constexpr const char *kDoneAttr = "gicc-nodb";
constexpr const char *kRepeat = "gicc.repeat";

bool isDevice(const Module &M) {
    Triple T(M.getTargetTriple());
    return T.isNVPTX() || T.isAMDGPU();
}

// The object may be a posted source only if it can be heap memory: not a
// stack slot, not a global variable, not in the shared / private / constant
// address spaces (3, 4, 5 on both NVPTX and AMDGPU).
bool mayBeHeap(Value *Ptr) {
    unsigned AS = Ptr->getType()->getPointerAddressSpace();
    if (AS != 0 && AS != 1) return false;
    const Value *Obj = getUnderlyingObject(Ptr, /*MaxLookup=*/0);
    return !isa<AllocaInst>(Obj) && !isa<GlobalVariable>(Obj);
}

// Declarations that are known not to write memory a program allocated:
// the OpenMP device runtime, printf, the math libraries. GICC's own ompx_*
// device calls are not among them -- a get writes its destination through
// the proxy or the NIC -- only the device runtime's ompx_ intrinsics are.
bool isTrustedCallee(const Function *F) {
    if (F->onlyReadsMemory()) return true;
    StringRef N = F->getName();
    for (StringRef P : {"__kmpc_", "omp_", "__llvm_omp_", "__ockl_", "__ocml_",
                        "__nv_", "__assert", "ompx_thread_id", "ompx_block_id",
                        "ompx_block_dim", "ompx_grid_dim", "ompx_sync_",
                        "ompx_ballot", "ompx_shfl", "ompx_lane"})
        if (N.starts_with(P)) return true;
    return N == "vprintf" || N == "printf" || N == "malloc" || N == "free";
}

enum class Kind { Store, PoisonIfIn, PoisonIfActive };

struct Site {
    Instruction *I;
    Kind kind;
};

// An intrinsic that may write memory and takes a pointer may write through
// it (masked and buffer stores, bulk copies, target atomics). Markers that
// only name memory are the exception. Intrinsics without a pointer operand
// -- barriers, fences, special-register reads -- cannot reach a source.
bool writesThroughPointer(const IntrinsicInst *II) {
    if (!II->mayWriteToMemory() || II->isLifetimeStartOrEnd() ||
        isa<DbgInfoIntrinsic>(II) || II->isAssumeLikeIntrinsic())
        return false;
    switch (II->getIntrinsicID()) {
    case Intrinsic::invariant_start:
    case Intrinsic::invariant_end:
    case Intrinsic::prefetch:
        return false;
    default:
        break;
    }
    for (const Use &U : II->args())
        if (U->getType()->isPtrOrPtrVectorTy()) return true;
    return false;
}

std::optional<Site> classify(Instruction &I) {
    if (auto *S = dyn_cast<StoreInst>(&I)) {
        // A pipelined put's copy of a store, into the peer or onto itself:
        // the store it repeats is checked.
        if (S->hasMetadata(kRepeat)) return std::nullopt;
        if (!mayBeHeap(S->getPointerOperand())) return std::nullopt;
        if (S->getValueOperand()->getType()->isScalableTy())
            return Site{&I, Kind::PoisonIfActive};
        return Site{&I, Kind::Store};
    }
    if (auto *A = dyn_cast<AtomicRMWInst>(&I))
        return mayBeHeap(A->getPointerOperand()) ? std::optional(Site{&I, Kind::PoisonIfIn})
                                                 : std::nullopt;
    if (auto *X = dyn_cast<AtomicCmpXchgInst>(&I))
        return mayBeHeap(X->getPointerOperand()) ? std::optional(Site{&I, Kind::PoisonIfIn})
                                                 : std::nullopt;
    if (auto *MI = dyn_cast<AnyMemIntrinsic>(&I))
        return mayBeHeap(MI->getRawDest()) ? std::optional(Site{&I, Kind::PoisonIfIn})
                                           : std::nullopt;
    if (auto *II = dyn_cast<IntrinsicInst>(&I))
        return writesThroughPointer(II) ? std::optional(Site{&I, Kind::PoisonIfActive})
                                        : std::nullopt;
    if (auto *CB = dyn_cast<CallBase>(&I)) {
        if (CB->onlyReadsMemory()) return std::nullopt;
        const Function *F = CB->getCalledFunction();
        if (F && !F->isDeclaration()) return std::nullopt;   // instrumented itself
        if (F && isTrustedCallee(F)) return std::nullopt;
        return Site{&I, Kind::PoisonIfActive};
    }
    return std::nullopt;
}

class Instrumenter {
public:
    explicit Instrumenter(Module &M)
        : M(M), Ctx(M.getContext()), DL(M.getDataLayout()),
          I8(Type::getInt8Ty(Ctx)), I32(Type::getInt32Ty(Ctx)),
          I64(Type::getInt64Ty(Ctx)), Ptr(PointerType::get(Ctx, 0)) {}

    bool run();

private:
    GlobalVariable *state();
    Value *field(IRBuilder<> &B, Value *Base, uint64_t Off) {
        return B.CreateConstInBoundsGEP1_64(I8, Base, Off);
    }
    Value *stateBase(IRBuilder<> &B) {
        return B.CreateAddrSpaceCast(state(), Ptr);
    }
    Function *mirrorFn();
    void instrument(Function &F, ArrayRef<Site> Sites);
    Value *addr(IRBuilder<> &B, Value *P) {
        return B.CreatePtrToInt(B.CreateAddrSpaceCast(P, Ptr), I64);
    }
    void setPoison(IRBuilder<> &B) {
        B.CreateStore(ConstantInt::get(I32, 1), field(B, stateBase(B), kPoison));
    }

    Module &M;
    LLVMContext &Ctx;
    const DataLayout &DL;
    Type *I8, *I32, *I64;
    PointerType *Ptr;
    GlobalVariable *State = nullptr;
    Function *Mirror = nullptr;
};

// The TU that includes gicc/omp.h defines the state (weak, merged at the
// device link); any other TU refers to it, and a weak zero definition here
// keeps a program that never includes the header linking.
GlobalVariable *Instrumenter::state() {
    if (State) return State;
    State = M.getGlobalVariable(kStateName);
    if (!State) {
        auto *Ty = ArrayType::get(I8, kStateSize);
        State = new GlobalVariable(M, Ty, /*isConstant=*/false,
                                   GlobalValue::WeakAnyLinkage,
                                   ConstantAggregateZero::get(Ty), kStateName,
                                   nullptr, GlobalValue::NotThreadLocal,
                                   DL.getDefaultGlobalsAddressSpace());
        State->setAlignment(Align(8));
        State->setVisibility(GlobalValue::ProtectedVisibility);
    }
    return State;
}

// ptr __gicc_nodb_mirror(ptr p, i64 size): where to repeat a store of
// `size` bytes at `p`, or null. Marks the words the store covers whole;
// poisons the epoch for a store straddling a source's edge.
//
//   for (i = 0; i < n; ++i) {
//     off = p - e[i].src;
//     if (off < e[i].bytes) {
//       if (off + size > e[i].bytes) { poison = 1; return 0; }
//       if (((off | size) & 3) == 0)
//         for (k = 0; k < size / 4; ++k) e[i].map[off / 4 + k] = epoch;
//       return e[i].peer + off;
//     }
//     if (e[i].src > p && e[i].src - p < size) { poison = 1; return 0; }
//   }
//   return 0;
Function *Instrumenter::mirrorFn() {
    if (Mirror) return Mirror;
    auto *FTy = FunctionType::get(Ptr, {Ptr, I64}, false);
    Mirror = Function::Create(FTy, GlobalValue::InternalLinkage, kMirrorName, M);
    Mirror->addFnAttr(Attribute::NoInline);
    Mirror->addFnAttr(Attribute::Cold);
    Mirror->addFnAttr(Attribute::NoUnwind);
    Mirror->addFnAttr(kDoneAttr);
    Argument *P = Mirror->getArg(0), *Size = Mirror->getArg(1);

    auto *Entry = BasicBlock::Create(Ctx, "entry", Mirror);
    auto *Head = BasicBlock::Create(Ctx, "head", Mirror);
    auto *Body = BasicBlock::Create(Ctx, "body", Mirror);
    auto *Miss = BasicBlock::Create(Ctx, "miss", Mirror);
    auto *Next = BasicBlock::Create(Ctx, "next", Mirror);
    auto *Hit = BasicBlock::Create(Ctx, "hit", Mirror);
    auto *Inside = BasicBlock::Create(Ctx, "inside", Mirror);
    auto *MarkHead = BasicBlock::Create(Ctx, "mark.head", Mirror);
    auto *MarkBody = BasicBlock::Create(Ctx, "mark.body", Mirror);
    auto *Found = BasicBlock::Create(Ctx, "found", Mirror);
    auto *Poison = BasicBlock::Create(Ctx, "poison", Mirror);
    auto *None = BasicBlock::Create(Ctx, "none", Mirror);

    IRBuilder<> B(Entry);
    Value *S = stateBase(B);
    Value *N = B.CreateZExt(B.CreateLoad(I32, field(B, S, kN)), I64, "n");
    Value *Epoch = B.CreateTrunc(B.CreateLoad(I32, field(B, S, kEpoch)), I8, "epoch");
    Value *PI = B.CreatePtrToInt(P, I64);
    B.CreateBr(Head);

    B.SetInsertPoint(Head);
    PHINode *I = B.CreatePHI(I64, 2, "i");
    I->addIncoming(ConstantInt::get(I64, 0), Entry);
    B.CreateCondBr(B.CreateICmpULT(I, N), Body, None);

    B.SetInsertPoint(Body);
    Value *E = B.CreateGEP(I8, field(B, S, kEntries),
                           B.CreateMul(I, ConstantInt::get(I64, kEntrySize)), "e");
    Value *Src = B.CreatePtrToInt(B.CreateLoad(Ptr, field(B, E, kSrc)), I64, "src");
    Value *Bytes = B.CreateLoad(I64, field(B, E, kBytes), "bytes");
    Value *Off = B.CreateSub(PI, Src, "off");
    B.CreateCondBr(B.CreateICmpULT(Off, Bytes), Hit, Miss);

    B.SetInsertPoint(Miss);
    Value *Before = B.CreateICmpUGT(Src, PI);
    Value *Reaches = B.CreateICmpULT(B.CreateSub(Src, PI), Size);
    B.CreateCondBr(B.CreateAnd(Before, Reaches), Poison, Next);

    B.SetInsertPoint(Next);
    I->addIncoming(B.CreateAdd(I, ConstantInt::get(I64, 1)), Next);
    B.CreateBr(Head);

    B.SetInsertPoint(Hit);
    B.CreateCondBr(B.CreateICmpUGT(B.CreateAdd(Off, Size), Bytes), Poison, Inside);

    B.SetInsertPoint(Inside);
    Value *Q = B.CreateGEP(I8, B.CreateLoad(Ptr, field(B, E, kPeer)), Off, "q");
    Value *Whole = B.CreateICmpEQ(
        B.CreateAnd(B.CreateOr(Off, Size), ConstantInt::get(I64, 3)),
        ConstantInt::get(I64, 0));
    Value *Map = B.CreateLoad(Ptr, field(B, E, kMap), "map");
    Value *W0 = B.CreateLShr(Off, 2);
    Value *NW = B.CreateLShr(Size, 2);
    B.CreateCondBr(Whole, MarkHead, Found);

    B.SetInsertPoint(MarkHead);
    PHINode *K = B.CreatePHI(I64, 2, "k");
    K->addIncoming(ConstantInt::get(I64, 0), Inside);
    B.CreateCondBr(B.CreateICmpULT(K, NW), MarkBody, Found);

    B.SetInsertPoint(MarkBody);
    B.CreateStore(Epoch, B.CreateGEP(I8, Map, B.CreateAdd(W0, K)));
    K->addIncoming(B.CreateAdd(K, ConstantInt::get(I64, 1)), MarkBody);
    B.CreateBr(MarkHead);

    B.SetInsertPoint(Found);
    B.CreateRet(Q);

    B.SetInsertPoint(Poison);
    setPoison(B);
    B.CreateRet(ConstantPointerNull::get(Ptr));

    B.SetInsertPoint(None);
    B.CreateRet(ConstantPointerNull::get(Ptr));
    return Mirror;
}

// The range [lo, hi) is read once, at entry: a post that lands while the
// function runs is simply not seen, which only leaves words unmarked.
void Instrumenter::instrument(Function &F, ArrayRef<Site> Sites) {
    F.addFnAttr(kDoneAttr);
    // Splitting the entry block at a site must not strand an alloca after
    // it in a non-entry block, where it would become dynamic: gather the
    // entry block's constant-size allocas at its top first.
    BasicBlock &EB = F.getEntryBlock();
    Instruction *Top = &*EB.getFirstNonPHIOrDbgOrAlloca();
    for (Instruction &I : make_early_inc_range(EB))
        if (auto *AI = dyn_cast<AllocaInst>(&I))
            if (AI->isStaticAlloca() && !AI->comesBefore(Top)) AI->moveBefore(Top);
    IRBuilder<> B(&*EB.getFirstNonPHIOrDbgOrAlloca());
    Value *S = stateBase(B);
    Value *Lo = B.CreatePtrToInt(B.CreateLoad(Ptr, field(B, S, kLo)), I64, "nodb.lo");
    Value *Hi = B.CreatePtrToInt(B.CreateLoad(Ptr, field(B, S, kHi)), I64, "nodb.hi");
    Value *Span = B.CreateSub(Hi, Lo, "nodb.span");

    for (const Site &St : Sites) {
        Instruction *I = St.I;
        B.SetInsertPoint(I);
        // Every value the new blocks need is computed here, before the split.
        Value *Cond = nullptr;
        Value *P = nullptr, *Val = nullptr;
        uint64_t Size = 0;
        switch (St.kind) {
        case Kind::Store: {
            auto *SI = cast<StoreInst>(I);
            P = SI->getPointerOperand();
            Val = SI->getValueOperand();
            Size = DL.getTypeStoreSize(Val->getType()).getFixedValue();
            // A store that only reaches into [lo, hi) from below still
            // enters: the mirror function poisons it.
            Value *First = B.CreateSub(addr(B, P), Lo);
            Value *Reach = B.CreateAdd(Span, ConstantInt::get(I64, Size - 1));
            Cond = B.CreateICmpULT(B.CreateAdd(First, ConstantInt::get(I64, Size - 1)),
                                   Reach);
            break;
        }
        case Kind::PoisonIfIn: {
            Value *Ptr0;
            Value *Len;
            if (auto *MI = dyn_cast<AnyMemIntrinsic>(I)) {
                Ptr0 = MI->getRawDest();
                Len = B.CreateZExtOrTrunc(MI->getLength(), I64);
            } else {
                Ptr0 = getLoadStorePointerOperand(I);
                if (!Ptr0) Ptr0 = isa<AtomicRMWInst>(I)
                                      ? cast<AtomicRMWInst>(I)->getPointerOperand()
                                      : cast<AtomicCmpXchgInst>(I)->getPointerOperand();
                Type *T = isa<AtomicRMWInst>(I)
                              ? cast<AtomicRMWInst>(I)->getValOperand()->getType()
                              : cast<AtomicCmpXchgInst>(I)->getNewValOperand()->getType();
                Len = ConstantInt::get(I64, DL.getTypeStoreSize(T).getFixedValue());
            }
            // [a, a + len) meets [lo, hi): a < hi && a + len > lo.
            Value *A = addr(B, Ptr0);
            Cond = B.CreateAnd(B.CreateICmpULT(A, Hi),
                               B.CreateICmpUGT(B.CreateAdd(A, Len), Lo));
            break;
        }
        case Kind::PoisonIfActive:
            Cond = B.CreateICmpNE(Lo, Hi);
            break;
        }

        Instruction *Then = SplitBlockAndInsertIfThen(Cond, I, /*Unreachable=*/false);
        B.SetInsertPoint(Then);
        if (St.kind != Kind::Store) {
            setPoison(B);
            continue;
        }
        Value *Q = B.CreateCall(mirrorFn(), {B.CreateAddrSpaceCast(P, Ptr),
                                             ConstantInt::get(I64, Size)});
        Instruction *Do = SplitBlockAndInsertIfThen(B.CreateIsNotNull(Q), Then, false);
        B.SetInsertPoint(Do);
        // The peer's copy sits at the same offset modulo 16 (the runtime
        // posts only such pairs), so alignment up to 16 carries over.
        Align A = std::min(cast<StoreInst>(I)->getAlign(), Align(16));
        StoreInst *Copy = B.CreateAlignedStore(Val, Q, A);
        Copy->setMetadata(kRepeat, MDNode::get(Ctx, {}));
    }
}

bool Instrumenter::run() {
    SmallVector<std::pair<Function *, SmallVector<Site, 8>>, 16> Work;
    for (Function &F : M) {
        if (F.isDeclaration() || F.hasFnAttribute(kDoneAttr)) continue;
        SmallVector<Site, 8> Sites;
        for (Instruction &I : instructions(F))
            if (auto S = classify(I)) Sites.push_back(*S);
        if (!Sites.empty()) Work.emplace_back(&F, std::move(Sites));
    }
    for (auto &[F, Sites] : Work) instrument(*F, Sites);
    return !Work.empty();
}

}  // namespace

PreservedAnalyses GICCNoDbMirrorPass::run(Module &M, ModuleAnalysisManager &) {
    if (getConfig().mode != Mode::ChunkLower || !isDevice(M))
        return PreservedAnalyses::all();
    return Instrumenter(M).run() ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

}  // namespace gicc::pass
