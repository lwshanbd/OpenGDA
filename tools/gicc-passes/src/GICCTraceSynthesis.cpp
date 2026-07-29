#include "GICCTraceSynthesis.h"
#include "GICCHostDiscovery.h"
#include "MetadataIO.h"
#include "GICCPassConfig.h"
#include "KernelInventory.h"
#include "LaunchSiteInventory.h"
#include "MetadataIO.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <unordered_map>
#include <unordered_set>

using namespace llvm;

namespace gicc::pass {

namespace {

// Canonical argument order for the put_no_db / get_no_db placeholder
// call. Index 0 is the Runtime*, followed by these in this exact order.
const std::array<const char *, 6> &putGetArgOrder() {
    static const std::array<const char *, 6> a = {
        "target_rank", "dst_buf", "dst_off", "src_buf", "src_off", "size"};
    return a;
}

Type *typeFromStr(LLVMContext &Ctx, StringRef s) {
    if (s == "ptr")      return PointerType::getUnqual(Ctx);
    if (s == "i1")       return Type::getInt1Ty(Ctx);
    if (s == "i8")       return Type::getInt8Ty(Ctx);
    if (s == "i16")      return Type::getInt16Ty(Ctx);
    if (s == "i32")      return Type::getInt32Ty(Ctx);
    if (s == "i64")      return Type::getInt64Ty(Ctx);
    if (s == "f32")      return Type::getFloatTy(Ctx);
    if (s == "f64")      return Type::getDoubleTy(Ctx);
    // Default: i64 (sufficient for indices/offsets/sizes, the common
    // template-arg types). Pointers should always be tagged "ptr".
    return Type::getInt64Ty(Ctx);
}

// Convert an arbitrary integer Value `v` to type `target` via trunc /
// zext / sext as appropriate. Pointers pass through unchanged.
Value *coerceInt(IRBuilder<> &B, Value *v, Type *target) {
    if (v->getType() == target) return v;
    if (target->isPointerTy() || v->getType()->isPointerTy()) return v;
    if (!v->getType()->isIntegerTy() || !target->isIntegerTy()) return v;
    return B.CreateIntCast(v, target, /*isSigned=*/true);
}

// Evaluate an ArgRef in the trace function context. `traceFn->getArg(0)`
// is `rt`; `traceFn->getArg(k)` for k>=1 corresponds to kernel formal k.
//
// `currentIv` (optional): the live host-side loop induction variable for
// the enclosing loop. When `a.kind == LoopIv`, this value is substituted
// (and coerced to `expected` if needed). If null, a LoopIv leaf evaluates
// to undef — that's a malformed template and should never reach lowering.
Value *evalArgRef(IRBuilder<> &B, Function *traceFn, const ArgRef &a,
                  Type *expected, Value *currentIv = nullptr) {
    LLVMContext &Ctx = B.getContext();
    switch (a.kind) {
        case ArgRef::Kind::ConstI64: {
            auto *cv = ConstantInt::get(Type::getInt64Ty(Ctx), a.constVal,
                                         /*isSigned=*/true);
            return coerceInt(B, cv, expected ? expected
                                              : Type::getInt64Ty(Ctx));
        }
        case ArgRef::Kind::Param: {
            // Kernel formal index `paramIdx`; trace arg index is the
            // same (formal 0 is ctx and was dropped, but our index map
            // keeps formal i at trace arg i).
            unsigned ti = a.paramIdx;
            if (ti >= traceFn->arg_size()) {
                return UndefValue::get(expected ? expected
                                                 : Type::getInt64Ty(Ctx));
            }
            Value *v = traceFn->getArg(ti);
            return expected ? coerceInt(B, v, expected) : v;
        }
        case ArgRef::Kind::LoopIv: {
            Type *t = expected ? expected : Type::getInt64Ty(Ctx);
            if (!currentIv) return UndefValue::get(t);
            return coerceInt(B, currentIv, t);
        }
        case ArgRef::Kind::BinOp: {
            if (a.children.size() != 2) {
                return UndefValue::get(expected ? expected
                                                 : Type::getInt64Ty(Ctx));
            }
            Type *t = expected ? expected : Type::getInt64Ty(Ctx);
            Value *l = evalArgRef(B, traceFn, a.children[0], t, currentIv);
            Value *r = evalArgRef(B, traceFn, a.children[1], t, currentIv);
            if (a.opStr == "add") return B.CreateAdd(l, r);
            if (a.opStr == "sub") return B.CreateSub(l, r);
            if (a.opStr == "mul") return B.CreateMul(l, r);
            if (a.opStr == "shl") return B.CreateShl(l, r);
            if (a.opStr == "and") return B.CreateAnd(l, r);
            if (a.opStr == "or")  return B.CreateOr(l, r);
            if (a.opStr == "xor") return B.CreateXor(l, r);
            return UndefValue::get(t);
        }
        case ArgRef::Kind::Cast: {
            // Cast/Trunc/Sext/Zext: pass-through to the child; coerce
            // to the expected type if requested. (Reasonable for v1;
            // refine if signed/unsigned semantics matter.)
            if (a.children.empty())
                return UndefValue::get(expected ? expected
                                                 : Type::getInt64Ty(Ctx));
            return evalArgRef(B, traceFn, a.children[0], expected, currentIv);
        }
        case ArgRef::Kind::FieldLoad: {
            // host_mirror_of(formal[base_formal])[iv].field — resolve
            // the device pointer at trace time then read the field
            // from the host-side mirror.
            if (a.paramIdx >= traceFn->arg_size())
                return UndefValue::get(expected ? expected
                                                 : Type::getInt64Ty(Ctx));
            Value *devPtr = traceFn->getArg(a.paramIdx);
            Value *rt     = traceFn->getArg(0);

            Module *M = traceFn->getParent();
            Type   *ptrTy = PointerType::getUnqual(Ctx);
            FunctionCallee hostMirrorFn = M->getOrInsertFunction(
                "gicc_runtime_host_mirror_of",
                FunctionType::get(ptrTy, {ptrTy, ptrTy}, false));
            Value *hostBase =
                B.CreateCall(hostMirrorFn, {rt, devPtr}, "host_mirror");

            Type *i64Ty = Type::getInt64Ty(Ctx);
            Value *ivVal;
            if (a.children.empty()) {
                ivVal = ConstantInt::get(i64Ty, 0);
            } else {
                ivVal = evalArgRef(B, traceFn, a.children[0], i64Ty,
                                   currentIv);
            }
            Value *off = B.CreateMul(
                ivVal, ConstantInt::get(i64Ty, a.structElemSize), "elem_off");
            off = B.CreateAdd(
                off, ConstantInt::get(i64Ty, a.fieldByteOffset), "field_off");

            Value *fieldPtr = B.CreateGEP(
                B.getInt8Ty(), hostBase, off, "host_field_ptr");

            Type  *fieldTy  = typeFromStr(Ctx, a.fieldTypeStr);
            Value *fieldVal = B.CreateAlignedLoad(
                fieldTy, fieldPtr, Align(1), /*isVolatile=*/false,
                "host_field");

            if (!expected) return fieldVal;
            if (expected == fieldTy) return fieldVal;
            if (expected->isPointerTy() || fieldTy->isPointerTy())
                return fieldVal;
            if (expected->isIntegerTy() && fieldTy->isIntegerTy())
                return coerceInt(B, fieldVal, expected);
            return fieldVal;
        }
        case ArgRef::Kind::Derived:
        default:
            return UndefValue::get(expected ? expected
                                              : Type::getInt64Ty(Ctx));
    }
}

// Materialize a guard expression returning i1.
Value *evalGuard(IRBuilder<> &B, Function *traceFn, const GuardSpec &g) {
    LLVMContext &Ctx = B.getContext();
    if (g.kind == GuardSpec::Kind::Always) return B.getTrue();
    // FieldNotNull: the per-iteration check requires the iv, so it
    // can't be materialized at the outer (loop-invariant) guard layer.
    // Return True here and let the loop emitter insert the icmp
    // inside the loop body.
    if (g.kind == GuardSpec::Kind::FieldNotNull) return B.getTrue();
    if (g.paramIdx >= traceFn->arg_size()) return B.getFalse();
    Value *p = traceFn->getArg(g.paramIdx);

    if (g.kind == GuardSpec::Kind::ParamTruthy) {
        if (p->getType()->isIntegerTy(1)) return p;
        if (p->getType()->isIntegerTy())
            return B.CreateICmpNE(p, ConstantInt::get(p->getType(), 0));
        return B.getFalse();
    }
    if (g.kind == GuardSpec::Kind::ParamEqConst) {
        if (!p->getType()->isIntegerTy()) return B.getFalse();
        return B.CreateICmpEQ(p, ConstantInt::get(p->getType(), g.constVal));
    }
    if (g.kind == GuardSpec::Kind::ParamCmpConst) {
        if (!p->getType()->isIntegerTy()) return B.getFalse();
        return B.CreateICmp(static_cast<ICmpInst::Predicate>(g.pred), p,
                            ConstantInt::get(p->getType(), g.constVal));
    }
    // Unknown / BinOp: be safe — take the op every time so behavior is
    // never accidentally suppressed.
    return B.getTrue();
}

FunctionCallee getPlaceholder(Module &M, GICCOpKind kind) {
    LLVMContext &Ctx = M.getContext();
    Type *ptrTy = PointerType::getUnqual(Ctx);
    Type *i32Ty = Type::getInt32Ty(Ctx);
    Type *i64Ty = Type::getInt64Ty(Ctx);
    if (kind == GICCOpKind::PutNoDb || kind == GICCOpKind::GetNoDb) {
        // (ptr rt, i32 peer, i32 dst_buf, i64 dst_off,
        //  i32 src_buf, i64 src_off, i64 size)
        StringRef name = (kind == GICCOpKind::PutNoDb)
            ? "gicc.runtime.put_no_db.placeholder"
            : "gicc.runtime.get_no_db.placeholder";
        auto *FT = FunctionType::get(
            Type::getVoidTy(Ctx),
            {ptrTy, i32Ty, i32Ty, i64Ty, i32Ty, i64Ty, i64Ty},
            /*isVarArg=*/false);
        return M.getOrInsertFunction(name, FT);
    }
    return {};
}

// Batched form: (ptr rt, i32 n_ops, ptr peers, ptr dst_bufs,
//                ptr dst_offs, ptr src_bufs, ptr src_offs, ptr sizes).
// Matches gicc_runtime_dwq_enqueue_batched 1:1 so the dispatch
// lowering pass just renames the call.
FunctionCallee getBatchedPlaceholder(Module &M, GICCOpKind kind) {
    LLVMContext &Ctx = M.getContext();
    Type *ptrTy = PointerType::getUnqual(Ctx);
    Type *i32Ty = Type::getInt32Ty(Ctx);
    if (kind == GICCOpKind::PutNoDb || kind == GICCOpKind::GetNoDb) {
        StringRef name = (kind == GICCOpKind::PutNoDb)
            ? "gicc.runtime.put_no_db.batched.placeholder"
            : "gicc.runtime.get_no_db.batched.placeholder";
        auto *FT = FunctionType::get(
            Type::getVoidTy(Ctx),
            {ptrTy, i32Ty, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy},
            /*isVarArg=*/false);
        return M.getOrInsertFunction(name, FT);
    }
    return {};
}

GICCOpKind opKindFromStr(StringRef s) {
    if (s == "put_no_db") return GICCOpKind::PutNoDb;
    if (s == "get_no_db") return GICCOpKind::GetNoDb;
    if (s == "flush")     return GICCOpKind::Flush;
    return GICCOpKind::Quiet;
}

// Emit the placeholder call into the current insertion point, evaluating
// each ArgRef in `op.args`. If `currentIv` is non-null, ArgRef::LoopIv
// leaves are substituted with it. Caller is responsible for branching to
// the next block after this returns.
void emitPlaceholderCall(Module &M, IRBuilder<> &B, Function *traceFn,
                         const OpTemplate &op, GICCOpKind kind,
                         Value *currentIv) {
    LLVMContext &Ctx = M.getContext();
    auto callee = getPlaceholder(M, kind);
    SmallVector<Value *, 8> args;
    args.push_back(traceFn->getArg(0));  // rt

    Type *i32Ty = Type::getInt32Ty(Ctx);
    Type *i64Ty = Type::getInt64Ty(Ctx);
    for (size_t i = 0; i < putGetArgOrder().size(); ++i) {
        const char *argName = putGetArgOrder()[i];
        Type       *expected = (i == 0 || i == 1 || i == 3) ? i32Ty : i64Ty;
        auto        it = op.args.find(argName);
        Value      *v;
        if (it == op.args.end())
            v = ConstantInt::get(expected, 0);
        else
            v = evalArgRef(B, traceFn, it->second, expected, currentIv);
        args.push_back(v);
    }

    auto *CI = B.CreateCall(callee, args);
    auto *md = MDNode::get(Ctx, MDString::get(Ctx, op.siteId));
    CI->setMetadata("gicc.site_id", md);
}

// Emit a host-side loop around `op`: bound = evalArgRef(Param(ivParamIdx))
// coerced to i64; iv starts at op.loop.ivStart, increments by op.loop.ivStep.
//
//   bb_in  ──▶ %bound = …
//              br loop.head
//   loop.head:
//     %iv = phi i64 [ start, bb_in ], [ %iv.next, loop.body ]
//     %cmp = icmp slt i64 %iv, %bound
//     br i1 %cmp, label loop.body, label loop.exit
//   loop.body:
//     <emit placeholder with currentIv=%iv>
//     %iv.next = add i64 %iv, step
//     br label loop.head
//   loop.exit: (caller branches to contBB from here)
//
// On entry the IRBuilder is positioned at the bb where the bound expression
// should be materialized. On return the builder is positioned at the
// freshly-created loop.exit block; caller must terminate it.
// Emit a host-side loop that STAGES per-iteration args into
// stack-allocated arrays, then issues a single batched-placeholder
// call after the loop exits. The batched placeholder maps 1:1 to
// gicc_runtime_dwq_enqueue_batched, so dispatch lowering becomes a
// rename instead of N separate enqueue calls.
//
//   preBB:
//     %bound64    = …
//     %peer_arr   = alloca i32, i64 %bound64
//     %dst_buf_arr= alloca i32, i64 %bound64
//     %dst_off_arr= alloca i64, i64 %bound64
//     %src_buf_arr= alloca i32, i64 %bound64
//     %src_off_arr= alloca i64, i64 %bound64
//     %size_arr   = alloca i64, i64 %bound64
//     %count_ptr  = alloca i32
//     store i32 0, ptr %count_ptr
//     br loop.head
//   loop.head:
//     %iv  = phi i64 [0, preBB], [iv.next, loop.latch]
//     %cmp = icmp slt i64 %iv, %bound64
//     br i1 %cmp, label loop.body, label loop.exit
//   loop.body:           ; (optionally branches to guard.do via FieldNotNull)
//   guard.do:            ; (only present when op has FieldNotNull guard)
//     %c     = load i32, ptr %count_ptr
//     ; eval each arg, store to arrays[c]
//     %c_nxt = add i32 %c, 1
//     store i32 %c_nxt, ptr %count_ptr
//     br loop.latch
//   loop.latch:
//     %iv.next = add i64 %iv, step
//     br loop.head
//   loop.exit:
//     %n = load i32, ptr %count_ptr
//     call void @gicc.runtime.put_no_db.batched.placeholder(
//         ptr %rt, i32 %n, ptr %peer_arr, ptr %dst_buf_arr,
//         ptr %dst_off_arr, ptr %src_buf_arr, ptr %src_off_arr, ptr %size_arr)
void emitOpInLoop(Module &M, IRBuilder<> &B, Function *traceFn,
                  const OpTemplate &op, GICCOpKind kind) {
    LLVMContext &Ctx = M.getContext();
    Type *i32Ty = Type::getInt32Ty(Ctx);
    Type *i64Ty = Type::getInt64Ty(Ctx);

    // Materialize the bound (kernel formal `ivParamIdx`) as i64.
    ArgRef boundRef;
    boundRef.kind     = ArgRef::Kind::Param;
    boundRef.paramIdx = op.loop.ivParamIdx;
    Value *bound = evalArgRef(B, traceFn, boundRef, i64Ty, nullptr);

    // Stack-allocate the per-arg arrays in the SAME BB as `bound` was
    // computed so they dominate the loop. Sizes are i64 to match
    // alloca's preferred index type.
    Value *peerArr   = B.CreateAlloca(i32Ty, bound, "dwq.peers");
    Value *dstBufArr = B.CreateAlloca(i32Ty, bound, "dwq.dst_bufs");
    Value *dstOffArr = B.CreateAlloca(i64Ty, bound, "dwq.dst_offs");
    Value *srcBufArr = B.CreateAlloca(i32Ty, bound, "dwq.src_bufs");
    Value *srcOffArr = B.CreateAlloca(i64Ty, bound, "dwq.src_offs");
    Value *sizeArr   = B.CreateAlloca(i64Ty, bound, "dwq.sizes");
    Value *countPtr  = B.CreateAlloca(i32Ty, nullptr, "dwq.count");
    B.CreateStore(ConstantInt::get(i32Ty, 0), countPtr);

    BasicBlock *headBB  = BasicBlock::Create(Ctx, "loop.head." + op.siteId,
                                             traceFn);
    BasicBlock *bodyBB  = BasicBlock::Create(Ctx, "loop.body." + op.siteId,
                                             traceFn);
    BasicBlock *exitBB  = BasicBlock::Create(Ctx, "loop.exit." + op.siteId,
                                             traceFn);

    // Capture the predecessor (where we currently sit) so the head PHI
    // can reference it as the start incoming.
    BasicBlock *preBB = B.GetInsertBlock();
    B.CreateBr(headBB);

    // loop.head: PHI + icmp + cond-br
    B.SetInsertPoint(headBB);
    PHINode *iv = B.CreatePHI(i64Ty, 2, "iv");
    iv->addIncoming(ConstantInt::get(i64Ty, op.loop.ivStart, /*signed=*/true),
                    preBB);
    Value *cmp = B.CreateICmpSLT(iv, bound, "cmp");
    B.CreateCondBr(cmp, bodyBB, exitBB);

    // loop.body: optional per-iteration FieldNotNull guard, then
    // stage args[count++], then fall through to a latch BB that
    // increments the iv and branches back to head.
    B.SetInsertPoint(bodyBB);

    bool perIterGuard =
        (op.guard.kind == GuardSpec::Kind::FieldNotNull
         && !op.guard.fieldArg.empty());

    BasicBlock *stageBB;
    BasicBlock *latchBB;
    if (perIterGuard) {
        Type *ptrTy = PointerType::getUnqual(Ctx);
        Value *fv = evalArgRef(B, traceFn, op.guard.fieldArg[0], ptrTy, iv);
        Value *isNull = B.CreateICmpEQ(
            fv, ConstantPointerNull::get(cast<PointerType>(ptrTy)),
            "is_null");
        stageBB = BasicBlock::Create(Ctx, "guard.do." + op.siteId, traceFn);
        latchBB = BasicBlock::Create(Ctx, "loop.latch." + op.siteId, traceFn);
        // Enter staging only when field IS null (cross-node entry).
        B.CreateCondBr(isNull, stageBB, latchBB);
    } else {
        stageBB = bodyBB;
        latchBB = BasicBlock::Create(Ctx, "loop.latch." + op.siteId, traceFn);
    }

    // Staging: evaluate each arg, store into arrays[count], bump count.
    B.SetInsertPoint(stageBB);
    Value *count   = B.CreateLoad(i32Ty, countPtr, "count");
    Value *countI64 = B.CreateZExt(count, i64Ty);
    auto storeAt = [&](Value *arr, Type *eltTy, Value *val) {
        Value *addr = B.CreateGEP(eltTy, arr, countI64);
        B.CreateStore(val, addr);
    };
    // Same arg-name keys as emitPlaceholderCall but writing to arrays.
    auto evalNamed = [&](const char *name, Type *expected) -> Value * {
        auto it = op.args.find(name);
        if (it == op.args.end())
            return ConstantInt::get(expected, 0);
        return evalArgRef(B, traceFn, it->second, expected, iv);
    };
    storeAt(peerArr,   i32Ty, evalNamed("target_rank", i32Ty));
    storeAt(dstBufArr, i32Ty, evalNamed("dst_buf",     i32Ty));
    storeAt(dstOffArr, i64Ty, evalNamed("dst_off",     i64Ty));
    storeAt(srcBufArr, i32Ty, evalNamed("src_buf",     i32Ty));
    storeAt(srcOffArr, i64Ty, evalNamed("src_off",     i64Ty));
    storeAt(sizeArr,   i64Ty, evalNamed("size",        i64Ty));
    Value *countNext = B.CreateAdd(count, ConstantInt::get(i32Ty, 1));
    B.CreateStore(countNext, countPtr);
    B.CreateBr(latchBB);

    // Latch: increment iv, jump back to header.
    B.SetInsertPoint(latchBB);
    Value *next = B.CreateAdd(
        iv, ConstantInt::get(i64Ty, op.loop.ivStep, /*signed=*/true),
        "iv.next");
    B.CreateBr(headBB);
    iv->addIncoming(next, latchBB);

    // Exit: one batched-placeholder call carrying the final count + arrays.
    B.SetInsertPoint(exitBB);
    Value *finalCount = B.CreateLoad(i32Ty, countPtr, "final_count");
    auto callee = getBatchedPlaceholder(M, kind);
    auto *batchedCI = B.CreateCall(callee,
        {traceFn->getArg(0), finalCount,
         peerArr, dstBufArr, dstOffArr,
         srcBufArr, srcOffArr, sizeArr});
    auto *md = MDNode::get(Ctx, MDString::get(Ctx, op.siteId));
    batchedCI->setMetadata("gicc.site_id", md);

    // Caller continues at exitBB.
}

void emitOp(Module &M, IRBuilder<> &B, Function *traceFn,
            const OpTemplate &op, const KernelTemplate &t,
            BasicBlock *contBB) {
    LLVMContext &Ctx = M.getContext();
    GICCOpKind   kind = opKindFromStr(op.kind);

    // flush / quiet are device-side responsibilities (DeviceLowering
    // emits the lead-thread MMIO trigger in Phase 1); host trace skips
    // them.
    if (kind == GICCOpKind::Flush || kind == GICCOpKind::Quiet) {
        B.CreateBr(contBB);
        return;
    }

    // Degraded loop: builder recognized the call as loop-wrapped but
    // couldn't recover iv/bound. Emitting wrong code (e.g. one IPC for
    // the whole loop or N copies of the same offset) would silently
    // corrupt data; skip the op entirely instead.
    if (op.loop.inLoop && op.loop.degraded) {
        B.CreateBr(contBB);
        return;
    }

    // Always emit the guard branch first, even when wrapping in a loop:
    // the user pattern `if (cond) for(i=0;i<n;i++) put(...)` is common
    // (HK guard + HK loop) and we want both to be honored.
    Value *guard = evalGuard(B, traceFn, op.guard);
    BasicBlock *doBB = BasicBlock::Create(Ctx, "op." + op.siteId,
                                          traceFn, contBB);
    B.CreateCondBr(guard, doBB, contBB);
    B.SetInsertPoint(doBB);

    if (op.loop.inLoop && op.loop.ivBoundKnown) {
        emitOpInLoop(M, B, traceFn, op, kind);
        // Builder is now at loop.exit; thread to contBB.
        B.CreateBr(contBB);
        return;
    }

    // Non-loop path: single call inside doBB, branch to contBB.
    emitPlaceholderCall(M, B, traceFn, op, kind, /*currentIv=*/nullptr);
    B.CreateBr(contBB);
}

// Map launch CallInst arguments to trace function arguments. Pattern:
// the launch wrapper has signature
//   gicc::launch<K>(Runtime&, dim3, dim3, [shmem, stream,] kernel_user_args...)
// In IR the kernel_user_args land at the END of the call, so we take
// the last (params.size() - 1) arguments of the launch CallInst and
// prepend the Runtime* (always arg 0).
SmallVector<Value *, 16> buildTraceCallArgs(CallBase *launch,
                                            const KernelTemplate &t) {
    SmallVector<Value *, 16> out;
    out.push_back(launch->getArgOperand(0));            // Runtime*

    int userCount = static_cast<int>(t.params.size()) - 1;
    if (userCount < 0) userCount = 0;
    int totalArgs = static_cast<int>(launch->arg_size());
    int startIdx  = totalArgs - userCount;
    if (startIdx < 1) startIdx = 1;  // guard against malformed signatures

    for (int i = 0; i < userCount && (startIdx + i) < totalArgs; ++i) {
        out.push_back(launch->getArgOperand(startIdx + i));
    }
    return out;
}

void insertTraceCall(CallBase *launch, Function *traceFn,
                     const KernelTemplate &t) {
    IRBuilder<> B(launch);
    auto args = buildTraceCallArgs(launch, t);

    // The trace function expects a specific list of types. Coerce each
    // call-site argument to match by inserting trunc/zext where needed.
    SmallVector<Value *, 16> coerced;
    for (size_t i = 0; i < args.size() && i < traceFn->arg_size(); ++i) {
        Type *expected = traceFn->getArg(i)->getType();
        Value *v = args[i];
        if (v->getType() == expected) {
            coerced.push_back(v);
        } else if (expected->isPointerTy() || v->getType()->isPointerTy()) {
            coerced.push_back(v);  // raw cast not needed; opaque pointers
        } else if (v->getType()->isIntegerTy() && expected->isIntegerTy()) {
            coerced.push_back(B.CreateIntCast(v, expected, /*signed=*/true));
        } else {
            coerced.push_back(UndefValue::get(expected));
        }
    }
    while (coerced.size() < traceFn->arg_size())
        coerced.push_back(UndefValue::get(traceFn->getArg(coerced.size())
                                              ->getType()));
    B.CreateCall(traceFn, coerced);
}

}  // namespace

// Trace function signature: ptr %rt, then one parameter per kernel
// formal except the leading DeviceCtx* (formal 0). Returns the new
// Function* (declaration if it already existed). Public so the OpenMP
// host-discovery pass can reuse it.
Function *getOrCreateTraceFn(Module &M, const KernelTemplate &t) {
    LLVMContext &Ctx  = M.getContext();
    std::string  name = "gicc_trace_" + t.simpleName;
    if (auto *F = M.getFunction(name)) return F;

    SmallVector<Type *, 16> params;
    params.push_back(PointerType::getUnqual(Ctx));      // rt
    for (size_t i = 1; i < t.params.size(); ++i)
        params.push_back(typeFromStr(Ctx, t.params[i].typeStr));

    auto *FT = FunctionType::get(Type::getVoidTy(Ctx), params, false);
    auto *F  = Function::Create(FT, GlobalValue::InternalLinkage, name, &M);
    F->getArg(0)->setName("rt");
    for (size_t i = 1; i < t.params.size(); ++i)
        F->getArg(i)->setName(t.params[i].name.empty()
                                  ? ("arg" + std::to_string(i))
                                  : t.params[i].name);
    return F;
}

void emitTraceBody(Module &M, Function *traceFn, const KernelTemplate &t) {
    LLVMContext &Ctx = M.getContext();
    BasicBlock *entry = BasicBlock::Create(Ctx, "entry", traceFn);
    IRBuilder<>  B(entry);
    Type *i32Ty = Type::getInt32Ty(Ctx);
    Type *i64Ty = Type::getInt64Ty(Ctx);

    // Straight-line (non-loop) PUTs stage through ONE batched placeholder:
    // each op's guard block appends its evaluated args to stack arrays and
    // a single call is issued at the end, so per-descriptor enqueue cost
    // stays off the per-launch critical path. Loop ops keep their own
    // batched emission (emitOpInLoop); GETs and everything else keep the
    // per-op path.
    // OmpDwq-only: the HIP path's default IPC_OR_DWQ dispatch cannot lower
    // a batched placeholder, so keep its per-op emission unchanged.
    const bool batchStraight = getConfig().mode == Mode::OmpDwq;
    SmallVector<const OpTemplate *, 16> straightPuts;
    if (batchStraight)
        for (const auto &op : t.ops)
            if (opKindFromStr(op.kind) == GICCOpKind::PutNoDb &&
                !op.loop.inLoop)
                straightPuts.push_back(&op);

    Value *arrs[6] = {nullptr};
    Value *countPtr = nullptr;
    if (!straightPuts.empty()) {
        auto *n = B.getInt32(static_cast<int>(straightPuts.size()));
        Type *eltTy[6] = {i32Ty, i32Ty, i64Ty, i32Ty, i64Ty, i64Ty};
        static const char *arrName[6] = {"b.peers", "b.dst_bufs", "b.dst_offs",
                                         "b.src_bufs", "b.src_offs", "b.sizes"};
        for (int i = 0; i < 6; ++i)
            arrs[i] = B.CreateAlloca(eltTy[i], n, arrName[i]);
        countPtr = B.CreateAlloca(i32Ty, nullptr, "b.count");
        B.CreateStore(B.getInt32(0), countPtr);
    }

    BasicBlock *cur = entry;
    for (const auto &op : t.ops) {
        BasicBlock *next = BasicBlock::Create(
            Ctx, "after." + op.siteId, traceFn);
        B.SetInsertPoint(cur);
        bool isStraightPut = batchStraight &&
            opKindFromStr(op.kind) == GICCOpKind::PutNoDb && !op.loop.inLoop;
        if (isStraightPut) {
            Value *guard = evalGuard(B, traceFn, op.guard);
            BasicBlock *doBB = BasicBlock::Create(
                Ctx, "stage." + op.siteId, traceFn, next);
            B.CreateCondBr(guard, doBB, next);
            B.SetInsertPoint(doBB);
            Value *c = B.CreateLoad(i32Ty, countPtr);
            for (size_t i = 0; i < putGetArgOrder().size(); ++i) {
                Type *expected = (i == 0 || i == 1 || i == 3) ? i32Ty : i64Ty;
                auto  it = op.args.find(putGetArgOrder()[i]);
                Value *v = (it == op.args.end())
                               ? ConstantInt::get(expected, 0)
                               : evalArgRef(B, traceFn, it->second, expected,
                                            /*currentIv=*/nullptr);
                B.CreateStore(v, B.CreateGEP(expected, arrs[i], c));
            }
            B.CreateStore(B.CreateAdd(c, B.getInt32(1)), countPtr);
            B.CreateBr(next);
        } else {
            emitOp(M, B, traceFn, op, t, next);
        }
        cur = next;
    }
    B.SetInsertPoint(cur);
    if (!straightPuts.empty()) {
        Value *c = B.CreateLoad(i32Ty, countPtr);
        BasicBlock *enqBB = BasicBlock::Create(Ctx, "b.enq", traceFn);
        BasicBlock *retBB = BasicBlock::Create(Ctx, "b.ret", traceFn);
        B.CreateCondBr(B.CreateICmpNE(c, B.getInt32(0)), enqBB, retBB);
        B.SetInsertPoint(enqBB);
        auto callee = getBatchedPlaceholder(M, GICCOpKind::PutNoDb);
        auto *CI = B.CreateCall(callee, {traceFn->getArg(0), c, arrs[0],
                                         arrs[1], arrs[2], arrs[3], arrs[4],
                                         arrs[5]});
        auto *md = MDNode::get(
            Ctx, MDString::get(Ctx, straightPuts.front()->siteId));
        CI->setMetadata("gicc.site_id", md);
        B.CreateBr(retBB);
        B.SetInsertPoint(retBB);
    }
    B.CreateRetVoid();
}

PreservedAnalyses GICCTraceSynthesisPass::run(Module &M,
                                              ModuleAnalysisManager &) {
    const auto &cfg = getConfig();
    if (cfg.mode != Mode::Lower) return PreservedAnalyses::all();

    auto inv = collectLaunchInventory(M, cfg.metaDir);
    if (inv.sites.empty()) return PreservedAnalyses::all();

    // De-dup by kernel mangled name — multiple launch sites may share a
    // wrapper, but we want exactly one trace function per kernel.
    std::unordered_set<std::string>     emitted;
    std::unordered_map<std::string, Function *> traceByKernel;
    for (auto &site : inv.sites) {
        if (!site.haveTemplate) continue;
        if (emitted.insert(site.kernelMangled).second) {
            Function *traceFn =
                getOrCreateTraceFn(M, site.kernelTemplate);
            if (traceFn->isDeclaration())
                emitTraceBody(M, traceFn, site.kernelTemplate);
            traceByKernel[site.kernelMangled] = traceFn;
        }
    }

    // Insert calls.
    for (auto &site : inv.sites) {
        if (!site.haveTemplate) continue;
        Function *traceFn = traceByKernel[site.kernelMangled];
        if (!traceFn) continue;
        insertTraceCall(site.callsite, traceFn, site.kernelTemplate);
    }

    return PreservedAnalyses::none();
}

}  // namespace gicc::pass
