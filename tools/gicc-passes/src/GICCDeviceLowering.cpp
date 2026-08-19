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
// `slot` selects WHICH block runs this op, and how the ring lane is set.
//
//   slot < 0   every op on block 0, one ring. What was emitted before this
//              existed, and still the answer when there is nothing to
//              spread: one site, or a grid of one block.
//   slot >= 0  this op belongs to block `slot`, pushing on ring `slot`.
//              Independent sites released by the same completion point can
//              then push at the same time instead of queueing behind one
//              another on one thread.
//
// Taken modulo gridDim.x, which is not cosmetic: the pass cannot see the
// launch geometry, and a site assigned to a block that the launch does not
// have would never push at all -- silent data loss rather than a slow
// program. Modulo means every site is executed by exactly one block that
// exists, for any grid, degrading to the old behaviour at gridDim.x == 1.
//
// Spreading is only worth it when the completion waits are CONCURRENT.
// Measured: one thread draining N lanes in sequence costs about 5 us per
// lane and gains nothing, because the drain was never the bottleneck. N
// blocks draining their own lane pay that once, in parallel, which is why
// the win comes from the pushes overlapping and not from the extra rings.
// slot >= 0 : only block `slot` runs this op, pushing on ring `slot`.
// slot == kAllBlocks : every block below `nslots` runs it, each on its own
//   ring. That is what a completion point must do once the pushes have been
//   spread -- draining only ring 0 would let the kernel's quiet return with
//   the other rings still in flight, and an in-kernel read after it would
//   see nothing. rt.reset() would still drain them before the host went on,
//   so the program would look correct while the in-kernel ordering
//   guarantee was quietly gone.
static constexpr int kAllBlocks = -2;

// AMDGPU has intrinsics for the current workgroup/workitem id, but not for
// CUDA/HIP's gridDim.x or blockDim.x.  The previous lowering invented
// llvm.amdgcn.grid.size.x and llvm.amdgcn.workgroup.size.x declarations; opt
// accepted those as ordinary functions and the device linker then failed with
// undefined symbols.  Read the standard HSA kernel-dispatch packet instead:
//
//   byte  4: uint16_t workgroup_size_x
//   byte 12: uint32_t grid_size_x (workitems, not workgroups)
//
// llvm.amdgcn.dispatch.ptr is a real target intrinsic and returns a constant
// address-space pointer to that packet.  The quotient is HIP gridDim.x.
Value *amdgcnGridDimX(IRBuilder<> &B, Module *M) {
    Function *dispatchFn = Intrinsic::getDeclaration(
        M, Intrinsic::amdgcn_dispatch_ptr);
    Value *packet = B.CreateCall(dispatchFn, {}, "dispatch.ptr");
    Type *i8Ty  = B.getInt8Ty();
    Type *i16Ty = B.getInt16Ty();
    Type *i32Ty = B.getInt32Ty();
    Value *wgPtr = B.CreateGEP(i8Ty, packet, B.getInt64(4), "wg.x.ptr");
    Value *gsPtr = B.CreateGEP(i8Ty, packet, B.getInt64(12), "grid.x.ptr");
    Value *wg16 = B.CreateAlignedLoad(i16Ty, wgPtr, Align(2), false, "wg.x");
    Value *grid = B.CreateAlignedLoad(i32Ty, gsPtr, Align(4), false, "grid.x");
    Value *wg = B.CreateZExt(wg16, i32Ty, "wg.x.i32");
    Value *safeWg = B.CreateSelect(
        B.CreateICmpEQ(wg, B.getInt32(0)), B.getInt32(1), wg, "wg.x.safe");
    return B.CreateUDiv(grid, safeWg, "grid.dim.x");
}

void wrapPreservedOpBlockAMDGCN(CallInst *CI, int slot, int laneArgIdx,
                                int nslots = 0) {
    Module *M = CI->getModule();
    BasicBlock *parent = CI->getParent();

    BasicBlock *doBB   = parent->splitBasicBlock(CI, "gicc.dev.do");
    BasicBlock *contBB = doBB->splitBasicBlock(CI->getNextNode(), "gicc.dev.cont");

    IRBuilder<> B(parent->getTerminator());
    auto tidFn = M->getOrInsertFunction(
        "llvm.amdgcn.workitem.id.x",
        FunctionType::get(B.getInt32Ty(), {}, false));
    auto bidFn = M->getOrInsertFunction(
        "llvm.amdgcn.workgroup.id.x",
        FunctionType::get(B.getInt32Ty(), {}, false));
    Value *tid = B.CreateCall(tidFn);
    Value *bid = B.CreateCall(bidFn);
    Value *want = B.getInt32(0);
    Value *cond = nullptr;
    if (slot == kAllBlocks) {
        cond = B.CreateAnd(B.CreateICmpEQ(tid, B.getInt32(0)),
                           B.CreateICmpULT(bid, B.getInt32(nslots)));
    } else if (slot > 0) {
        Value *nblocks = amdgcnGridDimX(B, M);
        Value *safe = B.CreateSelect(
            B.CreateICmpEQ(nblocks, B.getInt32(0)), B.getInt32(1), nblocks);
        want = B.CreateURem(B.getInt32(slot), safe);
    }
    Value *lead = cond ? cond
                       : B.CreateAnd(B.CreateICmpEQ(tid, B.getInt32(0)),
                                     B.CreateICmpEQ(bid, want));

    Instruction *oldTerm = parent->getTerminator();
    BranchInst::Create(doBB, contBB, lead, oldTerm);
    oldTerm->eraseFromParent();

    // Push on (or drain) the ring belonging to the block that runs this op,
    // so sites that now run concurrently do not contend on one ring.
    const bool haveLane = laneArgIdx >= 0 &&
                          laneArgIdx < (int)CI->arg_size() &&
                          CI->getArgOperand(laneArgIdx)->getType()->isIntegerTy();
    if (haveLane && slot >= 0) {
        // The issuing block is `slot % gridDim.x`, so its ring lane must be
        // the same value.  Keeping the unmodded static slot here silently
        // sends to rings that no launched block will drain whenever the
        // number of sites exceeds the launch grid.
        Type *lt = CI->getArgOperand(laneArgIdx)->getType();
        IRBuilder<> LB(CI);
        CI->setArgOperand(laneArgIdx,
                          lt == want->getType()
                              ? want
                              : LB.CreateZExtOrTrunc(want, lt));
    } else if (haveLane && slot == kAllBlocks) {
        Type *lt = CI->getArgOperand(laneArgIdx)->getType();
        IRBuilder<> LB(CI);
        CI->setArgOperand(laneArgIdx,
                          lt == bid->getType() ? bid
                                               : LB.CreateZExtOrTrunc(bid, lt));
    }
}

void wrapPreservedOpLeadThreadAMDGCN(CallInst *CI) {
    wrapPreservedOpBlockAMDGCN(CI, /*slot=*/-1, /*laneArgIdx=*/-1);
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
    // Which block issues each preserved op, and for a completion point how
    // many rings it has to drain.
    DenseMap<CallInst *, int> slotOf, nslotOf;
    bool changed = false;
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
        bool     proxyAware = false;
        bool     haveHint   = false;
        HintFile hint;
        if (!cfgRef.hintIn.empty()) {
            if (readHintFile(cfgRef.hintIn, hint)) {
                haveHint   = true;
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

        // Fence scope, per completion point. The device header defaults the
        // argument to FENCE_SYSTEM, so a site the analysis says nothing
        // about keeps exactly the behaviour it had before this existed;
        // only a proven-weaker scope is written back. Weakening a fence
        // that was needed yields stale reads with nothing at runtime to
        // catch it, so this only ever moves in the direction the analysis
        // proved.
        KernelTemplate ktFence;
        const bool haveFence =
            !cfgRef.metaDir.empty() &&
            readKernelTemplate(cfgRef.metaDir, info.mangledName, ktFence);
        if (haveFence) {
            for (const auto &s : info.sites) {
                if (s.kind != GICCOpKind::Quiet) continue;
                auto it = std::find_if(
                    ktFence.ops.begin(), ktFence.ops.end(),
                    [&](const OpTemplate &o) { return o.siteId == s.siteId; });
                if (it == ktFence.ops.end() || it->fence_scope == 3) continue;
                // quiet(ctx, lane, fence): the fence is the last operand.
                const unsigned idx = s.CI->arg_size() - 1;
                if (idx < 2) continue;   // older signature, nothing to set
                Type *ty = s.CI->getArgOperand(idx)->getType();
                if (!ty->isIntegerTy()) continue;
                s.CI->setArgOperand(
                    idx, ConstantInt::get(ty, it->fence_scope));
                errs() << "[device-lowering] " << s.siteId
                       << ": fence scope " << it->fence_scope
                       << " (was 3=system)\n";
                changed = true;
            }
        }

        // A proxy quiet must drain every lane touched by a preserved proxy
        // transfer.  Batch ownership is defined by the first host/DWQ
        // completion point (often flush), so the trace template quite
        // correctly records the slot count on that point rather than on a
        // later quiet.  Device proxy pushes, however, are already live when
        // flush executes and are completed by quiet.  Derive the lane span
        // from the preserved transfer sites themselves instead of assuming
        // the quiet carries the count.  This also handles mixed kernels: a
        // trigger-only site does not force its device body to be preserved.
        int proxySlotCount = 0;
        if (haveFence && proxyAware) {
            for (const auto &s : info.sites) {
                if (s.kind != GICCOpKind::PutNoDb &&
                    s.kind != GICCOpKind::GetNoDb)
                    continue;
                bool preserve = proxyAware;  // legacy per-kernel metadata
                if (haveHint)
                    preserve = hintFor(hint, s.siteId).dispatch ==
                               DispatchKind::CpuProxyEnqueue;
                if (!preserve) continue;
                auto it = std::find_if(
                    ktFence.ops.begin(), ktFence.ops.end(),
                    [&](const OpTemplate &o) { return o.siteId == s.siteId; });
                if (it != ktFence.ops.end() && it->block_slot >= 0)
                    proxySlotCount =
                        std::max(proxySlotCount, it->block_slot + 1);
            }
        }

        for (const auto &s : info.sites) {
            int slot = -1;
            if (haveFence) {
                auto it = std::find_if(
                    ktFence.ops.begin(), ktFence.ops.end(),
                    [&](const OpTemplate &o) { return o.siteId == s.siteId; });
                if (it != ktFence.ops.end()) slot = it->block_slot;
            }
            switch (s.kind) {
                case GICCOpKind::PutNoDb:
                case GICCOpKind::GetNoDb: {
                    // Preservation is PER SITE, not per kernel. One
                    // kernel can legitimately mix a proxy-routed site
                    // (its descriptor is not host-knowable, so the
                    // device body must push the TransferCmd) with a
                    // trigger-routed site (the host trace owns it, so
                    // the device body must be erased). Deciding this
                    // per kernel means one proxy site drags every other
                    // site in the kernel onto the ring, and the routing
                    // the decider chose is silently not what runs.
                    bool preserve = proxyAware;   // legacy per-kernel path
                    if (haveHint) {
                        preserve = hintFor(hint, s.siteId).dispatch
                                   == DispatchKind::CpuProxyEnqueue;
                    }
                    putGetCalls.emplace_back(s.CI, preserve);
                    if (preserve) slotOf[s.CI] = slot;
                    break;
                }
                case GICCOpKind::Flush:
                    flushCalls.push_back(s.CI);
                    break;
                case GICCOpKind::Quiet:
                    // Quiet stays a per-KERNEL decision: it drains the
                    // ring, so it is preserved iff some site in this
                    // kernel actually uses the ring. proxySlotCount was
                    // derived above from the proxy-preserved sites, so the
                    // drain covers every lane they may have pushed to.
                    putGetCalls.emplace_back(s.CI, proxyAware);
                    if (proxyAware && proxySlotCount >= 2)
                        slotOf[s.CI] = kAllBlocks;
                    if (proxyAware && proxySlotCount >= 2)
                        nslotOf[s.CI] = proxySlotCount;
                    break;
            }
        }
    }

    for (auto &pr : putGetCalls) {
        if (pr.second) {
            // Proxy-aware kernel: keep the device body, but gate it to a
            // single grid-wide lead thread so each logical op pushes the
            // proxy ring exactly once (see wrapPreservedOpLeadThreadAMDGCN).
            // NVPTX preservation stays unguarded until the Phase 4 backend
            // lands its own intrinsic lowering.
            if (isAMDGCN) {
                const int slot = slotOf.count(pr.first) ? slotOf[pr.first] : -1;
                // put/get take the lane last; so does quiet, whose last
                // argument is the fence and whose lane is the one before.
                int laneIdx = -1;
                const unsigned n = pr.first->arg_size();
                if (n >= 8)      laneIdx = (int)n - 1;   // put/get(..., lane)
                else if (n == 3) laneIdx = 1;            // quiet(ctx, lane, fence)
                wrapPreservedOpBlockAMDGCN(
                    pr.first, slot, laneIdx,
                    nslotOf.count(pr.first) ? nslotOf[pr.first] : 0);
                if (slot >= 0 || slot == -2)
                    errs() << "[device-lowering] issue spread: slot " << slot
                           << "\n";
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
