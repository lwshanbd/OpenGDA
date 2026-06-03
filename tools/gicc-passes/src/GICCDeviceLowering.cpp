#include "GICCDeviceLowering.h"
#include "GICCPassConfig.h"
#include "KernelInventory.h"
#include "MetadataIO.h"
#include "DispatchDecision.h"

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

// Byte offsets of the relevant DeviceCtx fields (see
// src/gicc/platform/ofi/ofi_device.cuh, post-Phase-6 layout):
//
//   struct DeviceCtx {
//       volatile uint64_t *trigger_addr_;   // offset 0
//       uint64_t           trigger_val_;    // offset 8
//   };
//
// Loading these inline via byte-offset GEPs avoids cross-side helper
// calls (the device-side IR cannot link against host-only symbols).
// Kept stable by static_asserts on the C++ side.
constexpr unsigned kTriggerAddrOffset = 0;
constexpr unsigned kTriggerValOffset  = 8;

// Lower gicc::flush(ctx) on AMDGCN to:
//   %tid = call i32 @llvm.amdgcn.workitem.id.x()
//   %bid = call i32 @llvm.amdgcn.workgroup.id.x()
//   %lead = and (icmp eq %tid, 0) (icmp eq %bid, 0)
//   br i1 %lead, label %do, label %skip
// do:
//   %addr_ptr = getelementptr i8, ptr %ctx, i64 0
//   %addr     = load ptr, ptr %addr_ptr, align 8
//   %val_ptr  = getelementptr i8, ptr %ctx, i64 16
//   %val      = load i64, ptr %val_ptr,  align 8
//   store volatile i64 %val, ptr %addr, align 8, !nontemporal !1
//   fence release
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
    Type *i64Ty = Type::getInt64Ty(Ctx);
    Type *i8Ty  = Type::getInt8Ty(Ctx);

    Value *addrFieldPtr = B.CreateGEP(i8Ty, ctxArg,
                                      B.getInt64(kTriggerAddrOffset),
                                      "trigger.addr.ptr");
    Value *addr = B.CreateAlignedLoad(ptrTy, addrFieldPtr, Align(8),
                                      /*isVolatile=*/false, "trigger.addr");

    Value *valFieldPtr = B.CreateGEP(i8Ty, ctxArg,
                                     B.getInt64(kTriggerValOffset),
                                     "trigger.val.ptr");
    Value *val = B.CreateAlignedLoad(i64Ty, valFieldPtr, Align(8),
                                     /*isVolatile=*/false, "trigger.val");

    auto *st = B.CreateAlignedStore(val, addr, Align(8), /*isVolatile=*/true);
    auto *nt = MDNode::get(Ctx,
        {ConstantAsMetadata::get(ConstantInt::get(B.getInt32Ty(), 1))});
    st->setMetadata(LLVMContext::MD_nontemporal, nt);
    B.CreateFence(AtomicOrdering::Release, SyncScope::System);
    B.CreateBr(skipBB);

    CI->eraseFromParent();
}

// Wrap a PRESERVED device-side put/get/quiet call in a grid-wide lead-thread
// guard on AMDGCN:
//
//   %lead = (workitem.id.x == 0) && (workgroup.id.x == 0)
//   br i1 %lead, label %gicc.dev.do, label %gicc.dev.cont
//   gicc.dev.do:   call <the preserved op>; br %gicc.dev.cont
//   gicc.dev.cont: <rest of the original block>
//
// Rationale: in CPU-proxy mode the device op body is kept and pushes a
// TransferCmd into the proxy ring. Each gicc::put site is ONE logical
// transfer (the host-trace / DWQ path enqueues it exactly once), so the
// device push must also happen once — not once per thread. Without this
// guard a kernel launched with N threads pushes the same command N times,
// overflowing the bounded ring and hanging rt.reset()'s drain. This mirrors
// lowerFlushAMDGCN's own lead-thread gating so users don't have to hand-guard
// every comm kernel; the op's arguments are computed before the split point
// and therefore still dominate the moved call.
void wrapPreservedOpLeadThreadAMDGCN(CallInst *CI) {
    Module *M = CI->getModule();
    BasicBlock *parent = CI->getParent();

    // doBB := [CI, ...follow...]; cont := [...follow...]; doBB := [CI, br cont].
    BasicBlock *doBB   = parent->splitBasicBlock(CI, "gicc.dev.do");
    BasicBlock *contBB = doBB->splitBasicBlock(CI->getNextNode(), "gicc.dev.cont");

    IRBuilder<> B(parent->getTerminator());
    auto tidFn = M->getOrInsertFunction(
        "llvm.amdgcn.workitem.id.x",
        FunctionType::get(B.getInt32Ty(), {}, false));
    auto bidFn = M->getOrInsertFunction(
        "llvm.amdgcn.workgroup.id.x",
        FunctionType::get(B.getInt32Ty(), {}, false));
    Value *tid  = B.CreateCall(tidFn);
    Value *bid  = B.CreateCall(bidFn);
    Value *lead = B.CreateAnd(B.CreateICmpEQ(tid, B.getInt32(0)),
                              B.CreateICmpEQ(bid, B.getInt32(0)));

    Instruction *oldTerm = parent->getTerminator();   // the br doBB from split
    BranchInst::Create(doBB, contBB, lead, oldTerm);
    oldTerm->eraseFromParent();
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

    // Per-kernel buckets so we can decide put/get/quiet preservation
    // based on the kernel's `proxy_aware` bit. Flush is unconditional —
    // it always lowers to the lead-thread MMIO write regardless.
    SmallVector<CallInst *, 16> flushCalls;
    // List of (call, preserve-on-device) pairs for the put/get/quiet
    // bodies. preserve=true means a CPU_PROXY_ENQUEUE site exists in
    // this kernel — keep the device-side call so the put_no_db body in
    // ofi_device.cuh runs and pushes a TransferCmd into the proxy ring.
    SmallVector<std::pair<CallInst *, bool>, 16> putGetCalls;
    const auto &cfgRef = cfg;  // capture for the inner switch.

    for (Function &F : M) {
        if (F.isDeclaration() || !isGPUKernel(F)) continue;

        GICCKernelInfo info;
        collectGICCSites(F, info);
        if (info.sites.empty()) continue;

        // Direction 3: compute proxy_aware locally from hint.json. The
        // dispatch decision is a pure function of (site_id, hint), so
        // both passes (this device-side lowering and the host-side
        // dispatch lowering) can independently reach the same answer
        // without a JSON-mediated handshake.
        //
        // This removes the cross-pass ordering hazard that previously
        // required either (a) the host pass to run before this one in
        // the same invocation, or (b) a two-pass compile to populate
        // the JSON bit.
        //
        // Multi-TU caveat: hint.json must be identical across every TU
        // that emits the same kernel symbol, otherwise the .o files
        // disagree on whether the device-side put_no_db body is
        // preserved and the linker will silently pick one.
        //
        // Fallback: if no hint.json is supplied we keep the legacy JSON
        // read so existing build flows (where a prior compile or
        // external decider wrote proxy_aware) continue to work.
        bool proxyAware = false;
        if (!cfgRef.hintIn.empty()) {
            HintFile hint;
            if (readHintFile(cfgRef.hintIn, hint)) {
                proxyAware = kernelHasProxySite(info.sites, hint);
            } else {
                // Loud failure: silently falling back to JSON would
                // mask a typo'd GICC_HINT_IN path and produce a
                // mysteriously-wrong device body. Warn so the user
                // sees the misconfiguration.
                errs() << "[device-lowering] WARN: GICC_HINT_IN="
                       << cfgRef.hintIn << " could not be read; "
                       << "falling back to per-kernel JSON for "
                       << "proxy_aware. Fix the path or remove the "
                       << "env var to silence.\n";
            }
        }
        if (!proxyAware && !cfgRef.metaDir.empty()) {
            KernelTemplate kt;
            if (readKernelTemplate(cfgRef.metaDir, info.mangledName, kt))
                proxyAware = kt.proxy_aware;
        }

        for (const auto &s : info.sites) {
            switch (s.kind) {
                case GICCOpKind::PutNoDb:
                case GICCOpKind::GetNoDb:
                    // Preserve on device when this kernel routes some
                    // site to the CPU proxy; the device-side body
                    // pushes the TransferCmd into the proxy ring.
                    // Otherwise erase — host trace owns the work.
                    putGetCalls.emplace_back(s.CI, proxyAware);
                    break;
                case GICCOpKind::Flush:
                    flushCalls.push_back(s.CI);
                    break;
                case GICCOpKind::Quiet:
                    // Quiet has no host-side IPC counterpart; v1 erases
                    // it like put_no_db. Preserved on the device when
                    // proxy is in play so the device body (future:
                    // MMIO drain or membar) keeps running. NVPTX path
                    // lowers it to a membar in Phase 4.
                    putGetCalls.emplace_back(s.CI, proxyAware);
                    break;
            }
        }
    }

    bool changed = false;
    for (auto &pr : putGetCalls) {
        if (pr.second) {
            // Proxy-aware kernel: keep the device body, but gate it to a
            // single grid-wide lead thread so each logical op pushes the
            // proxy ring exactly once (see wrapPreservedOpLeadThreadAMDGCN).
            // NVPTX preservation stays unguarded until the Phase 4 backend
            // lands its own intrinsic lowering.
            if (isAMDGCN) {
                wrapPreservedOpLeadThreadAMDGCN(pr.first);
                changed = true;
            }
            continue;
        }
        pr.first->eraseFromParent();   // host trace owns the work.
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
