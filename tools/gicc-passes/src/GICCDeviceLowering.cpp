#include "GICCDeviceLowering.h"
#include "GICCPassConfig.h"
#include "KernelInventory.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

namespace gicc::pass {

namespace {

FunctionCallee getRuntimeHelper(Module &M, StringRef name, Type *ret,
                                ArrayRef<Type *> argTypes) {
    auto *FT = FunctionType::get(ret, argTypes, /*isVarArg=*/false);
    return M.getOrInsertFunction(name, FT);
}

// Lower gicc::flush(ctx) on AMDGCN to:
//   %tid = call i32 @llvm.amdgcn.workitem.id.x()
//   %bid = call i32 @llvm.amdgcn.workgroup.id.x()
//   %lead = and (icmp eq %tid, 0) (icmp eq %bid, 0)
//   br i1 %lead, label %do, label %skip
// do:
//   %addr = call ptr @gicc_runtime_trigger_addr(ptr %ctx)
//   %val  = call i64 @gicc_runtime_trigger_val(ptr %ctx)
//   store volatile i64 %val, ptr %addr, align 8, !nontemporal !1
//   fence syncscope("agent-system") release
//   br label %skip
// skip:
void lowerFlushAMDGCN(CallInst *CI) {
    LLVMContext &Ctx = CI->getContext();
    Module      *M   = CI->getModule();
    Value       *ctxArg = CI->getArgOperand(0);

    IRBuilder<> B(CI);
    auto tidFn = M->getOrInsertFunction(
        "llvm.amdgcn.workitem.id.x",
        FunctionType::get(B.getInt32Ty(), {}, false));
    auto bidFn = M->getOrInsertFunction(
        "llvm.amdgcn.workgroup.id.x",
        FunctionType::get(B.getInt32Ty(), {}, false));

    Value *tid  = B.CreateCall(tidFn);
    Value *bid  = B.CreateCall(bidFn);
    Value *t0   = B.CreateICmpEQ(tid, B.getInt32(0));
    Value *b0   = B.CreateICmpEQ(bid, B.getInt32(0));
    Value *lead = B.CreateAnd(t0, b0);

    BasicBlock *parent = CI->getParent();
    BasicBlock *skipBB = parent->splitBasicBlock(CI->getNextNode(), "flush.skip");
    BasicBlock *doBB   = BasicBlock::Create(Ctx, "flush.do",
                                             parent->getParent(), skipBB);

    parent->getTerminator()->eraseFromParent();
    B.SetInsertPoint(parent);
    B.CreateCondBr(lead, doBB, skipBB);

    B.SetInsertPoint(doBB);
    Type *ptrTy = PointerType::getUnqual(Ctx);
    auto trigAddrFn = getRuntimeHelper(*M, "gicc_runtime_trigger_addr",
                                       ptrTy, {ptrTy});
    auto trigValFn  = getRuntimeHelper(*M, "gicc_runtime_trigger_val",
                                       B.getInt64Ty(), {ptrTy});
    Value *addr = B.CreateCall(trigAddrFn, {ctxArg});
    Value *val  = B.CreateCall(trigValFn,  {ctxArg});
    auto *st = B.CreateAlignedStore(val, addr, Align(8), /*isVolatile=*/true);
    auto *nt = MDNode::get(Ctx,
        {ConstantAsMetadata::get(ConstantInt::get(B.getInt32Ty(), 1))});
    st->setMetadata(LLVMContext::MD_nontemporal, nt);
    B.CreateFence(AtomicOrdering::Release,
                  Ctx.getOrInsertSyncScopeID("agent-system"));
    B.CreateBr(skipBB);

    CI->eraseFromParent();
}

}  // namespace

PreservedAnalyses GICCDeviceLoweringPass::run(Module &M,
                                              ModuleAnalysisManager &) {
    const auto &cfg = getConfig();
    if (cfg.mode != Mode::Lower) return PreservedAnalyses::all();

    Triple T(M.getTargetTriple());
    bool isAMDGCN = T.isAMDGCN();
    bool isNVPTX  = T.isNVPTX();
    if (!isAMDGCN && !isNVPTX) return PreservedAnalyses::all();

    SmallVector<CallInst *, 16> putGetCalls;
    SmallVector<CallInst *, 16> flushCalls;

    for (Function &F : M) {
        if (F.isDeclaration() || !isGPUKernel(F)) continue;

        GICCKernelInfo info;
        collectGICCSites(F, info);
        for (const auto &s : info.sites) {
            switch (s.kind) {
                case GICCOpKind::PutNoDb:
                case GICCOpKind::GetNoDb:
                    putGetCalls.push_back(s.CI);
                    break;
                case GICCOpKind::Flush:
                    flushCalls.push_back(s.CI);
                    break;
                case GICCOpKind::Quiet:
                    // Quiet has no host-side IPC counterpart; v1 erases
                    // it like put_no_db. NVPTX path lowers it to a
                    // membar in Phase 4.
                    putGetCalls.push_back(s.CI);
                    break;
            }
        }
    }

    bool changed = false;
    for (CallInst *CI : putGetCalls) {
        CI->eraseFromParent();
        changed = true;
    }
    for (CallInst *CI : flushCalls) {
        if (isAMDGCN) {
            lowerFlushAMDGCN(CI);
            changed = true;
        }
        // NVPTX flush lowering lands in Phase 4.
    }

    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

}  // namespace gicc::pass
