#include "GICCOmpHostDiscovery.h"
#include "GICCPassConfig.h"
#include "GICCTraceSynthesis.h"
#include "MetadataIO.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

namespace gicc::pass {

namespace {

// Walk through pointer-preserving casts (bitcast / addrspacecast) so we
// can compare alloca identities.
static Value *stripPtrCastsLocal(Value *v) { return v->stripPointerCasts(); }

// Recover the underlying scalar host value behind a captured slot store.
// OpenMP firstprivate i32 scalars are widened with `zext i32 %x to i64`
// before being parked in the (uniformly i64) offload base-pointer slot;
// is_device_ptr values are stored as the raw pointer. We peel zext / sext
// / ptrtoint / inttoptr / trunc so the value handed to the trace call is
// the original kernel-formal SSA value (the trace-call insertion then
// coerces it to the trace-fn parameter type). We deliberately keep peeling
// shallow — only the trivial widening/casts OpenMP emits — so we never
// mis-attribute a computed value.
static Value *recoverScalar(Value *stored) {
    Value *v = stored;
    for (;;) {
        if (auto *ze = dyn_cast<ZExtInst>(v))      { v = ze->getOperand(0); continue; }
        if (auto *se = dyn_cast<SExtInst>(v))      { v = se->getOperand(0); continue; }
        if (auto *tr = dyn_cast<TruncInst>(v))     { v = tr->getOperand(0); continue; }
        if (auto *p2i = dyn_cast<PtrToIntInst>(v)) { v = p2i->getOperand(0); continue; }
        if (auto *i2p = dyn_cast<IntToPtrInst>(v)) { v = i2p->getOperand(0); continue; }
        break;
    }
    return v;
}

// Resolve the device source-mangled name for a __tgt_target_kernel call by
// matching its region_id operand against the global offloading entries in
// section "omp_offloading_entries", then stripping the
// "__omp_offloading_<hex>_<hex>_" prefix and "_l<line>" suffix from the
// entry's name string.
//
// Returns the empty string if no matching entry / name is found.
static std::string resolveSrcMangled(Module &M, Value *regionId) {
    Value *rid = stripPtrCastsLocal(regionId);

    // Find the offloading-entry struct whose operand 0 is this region_id.
    GlobalVariable *nameGV = nullptr;
    for (GlobalVariable &G : M.globals()) {
        if (!G.hasInitializer()) continue;
        if (G.getSection() != "omp_offloading_entries") continue;
        auto *cs = dyn_cast<ConstantStruct>(G.getInitializer());
        if (!cs || cs->getNumOperands() < 2) continue;
        Value *entryRid = cs->getOperand(0)->stripPointerCasts();
        if (entryRid != rid) continue;
        nameGV = dyn_cast<GlobalVariable>(cs->getOperand(1)->stripPointerCasts());
        break;
    }
    if (!nameGV || !nameGV->hasInitializer()) return {};
    auto *cda = dyn_cast<ConstantDataArray>(nameGV->getInitializer());
    if (!cda || !cda->isString()) return {};
    StringRef name = cda->getAsCString();  // "__omp_offloading_<hex>_<hex>_<Src>_l<line>"

    // This MUST mirror GICCOmpDeviceDiscovery::extractOmpKernelKey exactly so
    // the host looks the JSON up under the same key the device wrote. The
    // source-mangled portion begins at the first "_Z"; if there is no "_Z"
    // (e.g. the target region lives directly in `main`, an unmangled C name)
    // the device returns the FULL kernel name unchanged (no "_l" strip), so
    // we do the same here instead of failing.
    auto zpos = name.find("_Z");
    if (zpos == StringRef::npos)
        return name.str();
    StringRef src = name.drop_front(zpos);

    // Strip the trailing "_l<digits>" suffix.
    auto lpos = src.rfind("_l");
    if (lpos != StringRef::npos) {
        StringRef tail = src.drop_front(lpos + 2);
        bool allDigits = !tail.empty();
        for (char c : tail)
            if (c < '0' || c > '9') { allDigits = false; break; }
        if (allDigits) src = src.take_front(lpos);
    }
    return src.str();
}

// Given the __tgt_kernel_arguments struct pointer, recover the
// .offload_baseptrs array alloca: the struct's field-2 (byte offset 8)
// stores a pointer to that array. We scan the kernel_args alloca's users
// for the store into offset 8.
static Value *findBasePtrsArray(Value *kernelArgs, const DataLayout &DL) {
    Value *ka = kernelArgs->stripPointerCasts();
    for (User *U : ka->users()) {
        // Direct store into the struct base (offset 0) is the Version
        // field, not the array; we want offset 8.
        if (auto *gep = dyn_cast<GetElementPtrInst>(U)) {
            APInt off(64, 0);
            if (!gep->accumulateConstantOffset(DL, off)) continue;
            if (off.getZExtValue() != 8) continue;
            for (User *GU : gep->users()) {
                if (auto *st = dyn_cast<StoreInst>(GU)) {
                    if (st->getPointerOperand()->stripPointerCasts() == gep)
                        return st->getValueOperand()->stripPointerCasts();
                }
            }
        }
    }
    return nullptr;
}

// Walk the .offload_baseptrs array's stores and build a map from device
// param index -> recovered host SSA value. Slot K of the array holds the
// capture for device param (K + 1): slot 0 is the leading is_device_ptr /
// firstprivate capture which corresponds to device formal 1 (ctx), slot 1
// -> formal 2, ... (device formal 0 is the OpenMP dyn_env and is never
// captured into the array).
static void collectCaptures(Value *baseArr, const DataLayout &DL,
                            DenseMap<unsigned, Value *> &out) {
    Value *arr = baseArr->stripPointerCasts();
    auto record = [&](uint64_t byteOff, Value *stored) {
        if (byteOff % 8 != 0) return;
        unsigned slot = static_cast<unsigned>(byteOff / 8);
        unsigned devParam = slot + 1;  // slot 0 -> device formal 1
        out[devParam] = recoverScalar(stored);
    };
    // Slot 0 is a direct store into the array base (no GEP).
    for (User *U : arr->users()) {
        if (auto *st = dyn_cast<StoreInst>(U)) {
            if (st->getPointerOperand()->stripPointerCasts() == arr)
                record(0, st->getValueOperand());
        } else if (auto *gep = dyn_cast<GetElementPtrInst>(U)) {
            APInt off(64, 0);
            if (!gep->accumulateConstantOffset(DL, off)) continue;
            for (User *GU : gep->users()) {
                if (auto *st = dyn_cast<StoreInst>(GU)) {
                    if (st->getPointerOperand()->stripPointerCasts() == gep)
                        record(off.getZExtValue(), st->getValueOperand());
                }
            }
        }
    }
}

}  // namespace

PreservedAnalyses GICCOmpHostDiscoveryPass::run(Module &M,
                                                ModuleAnalysisManager &) {
    const auto &cfg = getConfig();
    if (cfg.mode != Mode::OmpDwq) return PreservedAnalyses::all();
    // Host module only — the device (amdgcn) module has no host launches.
    if (Triple(M.getTargetTriple()).isAMDGCN())
        return PreservedAnalyses::all();

    Function *tgtFn = M.getFunction("__tgt_target_kernel");
    if (!tgtFn) return PreservedAnalyses::all();

    const DataLayout &DL = M.getDataLayout();

    // Collect launch calls first (we mutate the IR while iterating).
    SmallVector<CallInst *, 4> launches;
    for (User *U : tgtFn->users())
        if (auto *CI = dyn_cast<CallInst>(U))
            if (CI->getCalledFunction() == tgtFn)
                launches.push_back(CI);

    bool changed = false;
    for (CallInst *CI : launches) {
        if (CI->arg_size() < 6) continue;
        Value *regionId   = CI->getArgOperand(4);
        Value *kernelArgs = CI->getArgOperand(5);

        std::string srcMangled = resolveSrcMangled(M, regionId);
        if (srcMangled.empty()) {
            errs() << "[omp-host-discovery] WARN: could not resolve src-mangled "
                      "name for __tgt_target_kernel\n";
            continue;
        }

        KernelTemplate t;
        if (!readKernelTemplate(cfg.metaDir, srcMangled, t)) {
            errs() << "[omp-host-discovery] WARN: no template for " << srcMangled
                   << " under " << cfg.metaDir << "\n";
            continue;
        }

        // Recover the captured host values keyed by device param index.
        Value *baseArr = findBasePtrsArray(kernelArgs, DL);
        if (!baseArr) {
            errs() << "[omp-host-discovery] WARN: could not find offload "
                      "base-pointer array for " << srcMangled << "\n";
            continue;
        }
        DenseMap<unsigned, Value *> captures;
        collectCaptures(baseArr, DL, captures);

        // GUARD FIX: the OpenMP target-init branch makes every op's guard
        // appear "unknown" (the `if (__tgt_target_kernel(...) != 0)`
        // fallback). In omp-dwq mode the host trace must ALWAYS run the
        // recovered op, so force the guard to Always before synthesis.
        for (auto &op : t.ops)
            if (op.guard.kind == GuardSpec::Kind::Unknown)
                op.guard.kind = GuardSpec::Kind::Always;

        // Synthesize / fetch the trace fn and its body (reused machinery).
        Function *traceFn = getOrCreateTraceFn(M, t);
        if (traceFn->isDeclaration())
            emitTraceBody(M, traceFn, t);

        // Build the trace-call argument list.
        //
        // Trace-fn signature (getOrCreateTraceFn): arg0 = rt (ptr), then
        // one arg per device formal 1..N-1, so trace arg index k == device
        // param index k for k>=1. evalArgRef(Param{paramIdx=k}) reads
        // traceFn->getArg(k) directly, so we feed each trace arg k the
        // recovered host value for device param k.
        //
        //   trace arg 0  = rt        -> live gicc::Runtime* obtained at the
        //                               launch site by calling the bridge C
        //                               ABI `gicc_runtime_current()`. The
        //                               trace body forwards this to
        //                               gicc_runtime_dwq_enqueue, which needs
        //                               the real Runtime (the old ctx-pointer
        //                               stand-in compiled but was WRONG).
        //   trace arg 1  = device param 1 (ctx, is_device_ptr) -> capture[1]
        //   trace arg k  = device param k (k=2..) captured scalar -> capture[k]
        IRBuilder<> B(CI);

        // Declare `ptr @gicc_runtime_current()` if absent, then call it just
        // before the trace call to source the live Runtime handle.
        FunctionCallee rtCur = M.getOrInsertFunction(
            "gicc_runtime_current",
            FunctionType::get(B.getPtrTy(), /*isVarArg=*/false));
        Value *rtVal = B.CreateCall(rtCur, {}, "gicc_rt");

        SmallVector<Value *, 16> callArgs(traceFn->arg_size(), nullptr);

        // arg0 (rt): the live Runtime from gicc_runtime_current().
        callArgs[0] = rtVal;
        for (unsigned k = 1; k < traceFn->arg_size(); ++k) {
            auto it = captures.find(k);
            callArgs[k] = (it != captures.end()) ? it->second : nullptr;
        }

        // Coerce each arg to the trace-fn parameter type (trunc/zext for
        // ints, pass-through for pointers); fall back to a typed zero so a
        // missing capture never produces a verifier error.
        for (unsigned k = 0; k < traceFn->arg_size(); ++k) {
            Type *expected = traceFn->getArg(k)->getType();
            Value *v = callArgs[k];
            if (!v) {
                callArgs[k] = expected->isPointerTy()
                    ? cast<Value>(ConstantPointerNull::get(
                          cast<PointerType>(expected)))
                    : cast<Value>(ConstantInt::get(expected, 0));
                continue;
            }
            if (v->getType() == expected) {
                callArgs[k] = v;
            } else if (expected->isPointerTy() && v->getType()->isPointerTy()) {
                callArgs[k] = v;  // opaque pointers
            } else if (expected->isPointerTy() && v->getType()->isIntegerTy()) {
                callArgs[k] = B.CreateIntToPtr(v, expected);
            } else if (expected->isIntegerTy() && v->getType()->isPointerTy()) {
                callArgs[k] = B.CreatePtrToInt(v, expected);
            } else if (expected->isIntegerTy() && v->getType()->isIntegerTy()) {
                callArgs[k] = B.CreateIntCast(v, expected, /*isSigned=*/true);
            } else {
                callArgs[k] = expected->isPointerTy()
                    ? cast<Value>(ConstantPointerNull::get(
                          cast<PointerType>(expected)))
                    : cast<Value>(ConstantInt::get(expected, 0));
            }
        }

        B.CreateCall(traceFn, callArgs);

        // Re-arm the DWQ trigger AFTER the trace enqueued this region's ops and
        // BEFORE the __tgt_target_kernel launch. The OpenMP app calls prepare()
        // before the region (to get d_ctx for is_device_ptr), so prepare()
        // snapshotted a stale trigger_val_=0; without this re-arm the kernel's
        // flush writes 0 and the queued DWQ descriptors never fire (reset()
        // would then hang on the completion counter). The HIP path doesn't need
        // it: gicc::launch calls prepare() AFTER the pass-inserted trace.
        FunctionCallee armTrig = M.getOrInsertFunction(
            "gicc_runtime_arm_dwq_trigger",
            FunctionType::get(B.getVoidTy(), {B.getPtrTy()}, /*isVarArg=*/false));
        B.CreateCall(armTrig, {rtVal});
        changed = true;

        errs() << "[omp-host-discovery] inserted trace call for " << srcMangled
               << " (" << captures.size() << " captures)\n";
    }

    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

}  // namespace gicc::pass
