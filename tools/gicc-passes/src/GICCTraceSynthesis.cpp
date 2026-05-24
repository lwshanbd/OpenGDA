#include "GICCTraceSynthesis.h"
#include "GICCHostDiscovery.h"
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

// Trace function signature: ptr %rt, then one parameter per kernel
// formal except the leading DeviceCtx* (formal 0). Returns the new
// Function* (declaration if it already existed).
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
void emitOpInLoop(Module &M, IRBuilder<> &B, Function *traceFn,
                  const OpTemplate &op, GICCOpKind kind) {
    LLVMContext &Ctx = M.getContext();
    Type *i64Ty = Type::getInt64Ty(Ctx);

    // Materialize the bound (kernel formal `ivParamIdx`) as i64.
    ArgRef boundRef;
    boundRef.kind     = ArgRef::Kind::Param;
    boundRef.paramIdx = op.loop.ivParamIdx;
    Value *bound = evalArgRef(B, traceFn, boundRef, i64Ty, nullptr);

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
    // placeholder call, then fall through to a latch BB that increments
    // the iv and branches back to head.
    B.SetInsertPoint(bodyBB);

    bool perIterGuard =
        (op.guard.kind == GuardSpec::Kind::FieldNotNull
         && !op.guard.fieldArg.empty());

    BasicBlock *latchBB;
    if (perIterGuard) {
        // Evaluate host_mirror[iv].field with iv as the live PHI value,
        // compare against null, emit the placeholder only when the
        // field IS null (the ASF IPC-skip semantic: peer_recv_addr
        // == null means "cross-node — needs a DWQ descriptor").
        Type *ptrTy = PointerType::getUnqual(Ctx);
        Value *fv = evalArgRef(B, traceFn, op.guard.fieldArg[0], ptrTy, iv);
        Value *isNull = B.CreateICmpEQ(
            fv, ConstantPointerNull::get(cast<PointerType>(ptrTy)),
            "is_null");
        BasicBlock *doCallBB = BasicBlock::Create(
            Ctx, "guard.do." + op.siteId, traceFn);
        latchBB = BasicBlock::Create(Ctx, "loop.latch." + op.siteId, traceFn);
        B.CreateCondBr(isNull, doCallBB, latchBB);
        B.SetInsertPoint(doCallBB);
        emitPlaceholderCall(M, B, traceFn, op, kind, /*currentIv=*/iv);
        B.CreateBr(latchBB);
        B.SetInsertPoint(latchBB);
    } else {
        emitPlaceholderCall(M, B, traceFn, op, kind, /*currentIv=*/iv);
        latchBB = bodyBB;
    }

    Value *next = B.CreateAdd(
        iv, ConstantInt::get(i64Ty, op.loop.ivStep, /*signed=*/true),
        "iv.next");
    B.CreateBr(headBB);
    iv->addIncoming(next, latchBB);

    // Caller continues at exitBB.
    B.SetInsertPoint(exitBB);
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

void emitTraceBody(Module &M, Function *traceFn, const KernelTemplate &t) {
    LLVMContext &Ctx = M.getContext();
    BasicBlock *entry = BasicBlock::Create(Ctx, "entry", traceFn);
    IRBuilder<>  B(entry);

    BasicBlock *cur = entry;
    for (const auto &op : t.ops) {
        BasicBlock *next = BasicBlock::Create(
            Ctx, "after." + op.siteId, traceFn);
        B.SetInsertPoint(cur);
        emitOp(M, B, traceFn, op, t, next);
        cur = next;
    }
    B.SetInsertPoint(cur);
    B.CreateRetVoid();
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
