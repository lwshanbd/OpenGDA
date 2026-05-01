#include "GICCDispatchLowering.h"
#include "GICCHostDiscovery.h"
#include "GICCPassConfig.h"
#include "MetadataIO.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <set>
#include <string>
#include <unordered_map>

using namespace llvm;

namespace gicc::pass {

namespace {

// IpcOrDwq: hybrid runtime branch. If the peer's IPC base ptr is non-null
// (peer is on the same node and the buffer is mapped), do an IPC
// hipMemcpyAsync on the IPC stream. Otherwise fall through to a DWQ
// enqueue. This is the new default when no hint.json is provided —
// matches the AST plugin's runtime behavior and avoids DWQ's per-op
// libfabric overhead on same-node halos.
enum class DispatchKind {
    IpcPush,         // force IPC path (assumes peer is mapped)
    DwqTrigger,      // force DWQ path
    DwqBatched,      // currently lowered same as DwqTrigger
    IpcOrDwq,        // runtime branch: IPC if mapped else DWQ
    CpuProxyEnqueue, // device-side enqueue to CPU proxy ring; host trace
                     // emits nothing for the site (the actual work is
                     // performed device-side and serviced by the CPU
                     // proxy thread). Required for HK-incapable sites.
    Unknown,
};

DispatchKind parseDispatch(StringRef s) {
    if (s == "IPC_PUSH")          return DispatchKind::IpcPush;
    if (s == "DWQ_TRIGGER")       return DispatchKind::DwqTrigger;
    if (s == "DWQ_BATCHED")       return DispatchKind::DwqBatched;
    if (s == "IPC_OR_DWQ")        return DispatchKind::IpcOrDwq;
    if (s == "CPU_PROXY_ENQUEUE") return DispatchKind::CpuProxyEnqueue;
    return DispatchKind::Unknown;
}

const char *dispatchName(DispatchKind d) {
    switch (d) {
        case DispatchKind::IpcPush:         return "IPC_PUSH";
        case DispatchKind::DwqTrigger:      return "DWQ_TRIGGER";
        case DispatchKind::DwqBatched:      return "DWQ_BATCHED";
        case DispatchKind::IpcOrDwq:        return "IPC_OR_DWQ";
        case DispatchKind::CpuProxyEnqueue: return "CPU_PROXY_ENQUEUE";
        case DispatchKind::Unknown:         return "UNKNOWN";
    }
    return "UNKNOWN";
}

struct SiteHint {
    DispatchKind dispatch     = DispatchKind::Unknown;
    int          streamIndex  = 0;  // 0 default; set by decider for IPC_PUSH dispatch
};

struct HintFile {
    // When no hint.json is supplied we default to the hybrid runtime
    // branch. Same-node peers go through IPC (low latency / high BW
    // SDMA) and off-node peers fall back to DWQ. Lit tests that
    // expect a hard DWQ default explicitly set GICC_HINT_IN to a
    // hint.json with default_dispatch="DWQ_TRIGGER".
    DispatchKind                          defaultDispatch = DispatchKind::IpcOrDwq;
    std::unordered_map<std::string,
                       SiteHint>          sites;
};

bool readHintFile(const std::string &path, HintFile &out) {
    auto bufOr = MemoryBuffer::getFile(path);
    if (!bufOr) return false;
    auto parsed = json::parse((*bufOr)->getBuffer());
    if (!parsed) {
        consumeError(parsed.takeError());
        return false;
    }
    const auto *root = parsed->getAsObject();
    if (!root) return false;
    if (auto def = root->getString("default_dispatch"))
        out.defaultDispatch = parseDispatch(*def);
    if (const auto *sites = root->getObject("sites")) {
        for (const auto &kv : *sites) {
            const auto *entry = kv.second.getAsObject();
            if (!entry) continue;
            auto disp = entry->getString("dispatch");
            if (!disp) continue;
            SiteHint sh;
            sh.dispatch = parseDispatch(*disp);
            if (auto v = entry->getInteger("stream_index"))
                sh.streamIndex = static_cast<int>(*v);
            out.sites[kv.first.str()] = sh;
        }
    }
    return true;
}

SiteHint hintFor(const HintFile &h, StringRef siteId) {
    auto it = h.sites.find(siteId.str());
    if (it != h.sites.end()) return it->second;
    SiteHint sh;
    sh.dispatch    = h.defaultDispatch;
    sh.streamIndex = 0;
    return sh;
}

// Recover the site_id metadata string attached to a placeholder call.
StringRef siteIdOf(CallInst *CI) {
    auto *md = CI->getMetadata("gicc.site_id");
    if (!md || md->getNumOperands() == 0) return {};
    if (auto *s = dyn_cast<MDString>(md->getOperand(0))) return s->getString();
    return {};
}

// Helper bundle: IR types + the runtime helper function callees that
// the three lowerings share. Looked up once per placeholder.
struct LoweringHelpers {
    Type *ptrTy;
    Type *i32Ty;
    Type *i64Ty;
    Type *i8Ty;
    Type *i1Ty;
    FunctionCallee peerBaseFn;
    FunctionCallee localBaseFn;
    FunctionCallee streamFn;         // gicc_runtime_ipc_stream(rt) — kept for legacy
    FunctionCallee indexedStreamFn;  // gicc_runtime_ipc_stream_indexed(rt, idx)
    FunctionCallee memcpyFn;
    FunctionCallee enqFn;
    FunctionCallee enqBatchedFn;
};

LoweringHelpers makeHelpers(Module *M) {
    LLVMContext &Ctx = M->getContext();
    LoweringHelpers H;
    H.ptrTy = PointerType::getUnqual(Ctx);
    H.i32Ty = Type::getInt32Ty(Ctx);
    H.i64Ty = Type::getInt64Ty(Ctx);
    H.i8Ty  = Type::getInt8Ty(Ctx);
    H.i1Ty  = Type::getInt1Ty(Ctx);
    Type *voidTy = Type::getVoidTy(Ctx);
    H.peerBaseFn = M->getOrInsertFunction(
        "gicc_runtime_peer_ipc_base",
        FunctionType::get(H.ptrTy, {H.ptrTy, H.i32Ty, H.i32Ty}, false));
    H.localBaseFn = M->getOrInsertFunction(
        "gicc_runtime_local_buf_base",
        FunctionType::get(H.ptrTy, {H.ptrTy, H.i32Ty}, false));
    H.streamFn = M->getOrInsertFunction(
        "gicc_runtime_ipc_stream",
        FunctionType::get(H.ptrTy, {H.ptrTy}, false));
    H.indexedStreamFn = M->getOrInsertFunction(
        "gicc_runtime_ipc_stream_indexed",
        FunctionType::get(H.ptrTy, {H.ptrTy, H.i32Ty}, false));
    H.memcpyFn = M->getOrInsertFunction(
        "hipMemcpyAsync",
        FunctionType::get(H.i32Ty,
            {H.ptrTy, H.ptrTy, H.i64Ty, H.i32Ty, H.ptrTy}, false));
    H.enqFn = M->getOrInsertFunction(
        "gicc_runtime_dwq_enqueue",
        FunctionType::get(voidTy,
            {H.ptrTy, H.i32Ty, H.i32Ty, H.i64Ty, H.i32Ty, H.i64Ty, H.i64Ty},
            false));
    // Batched enqueue: void(rt, n_ops, peers[], dst_bufs[], dst_offs[],
    // src_bufs[], src_offs[], sizes[]). All array params are ptr.
    H.enqBatchedFn = M->getOrInsertFunction(
        "gicc_runtime_dwq_enqueue_batched",
        FunctionType::get(voidTy,
            {H.ptrTy, H.i32Ty,
             H.ptrTy, H.ptrTy, H.ptrTy, H.ptrTy, H.ptrTy, H.ptrTy},
            false));
    return H;
}

// Emit the IPC body at IRBuilder B's insertion point:
//   %dst       = getelementptr i8, ptr %peerBase, i64 dst_off
//   %src_base  = call ptr @gicc_runtime_local_buf_base(rt, src_buf)
//   %src       = getelementptr i8, ptr %src_base, i64 src_off
//   %stream    = call ptr @gicc_runtime_ipc_stream_indexed(rt, stream_index)
//   call i32 @hipMemcpyAsync(ptr %dst, ptr %src, i64 size, i32 3, ptr %stream)
//
// peerBase must already be a non-null ptr to the peer's mapped buffer.
// streamIndex selects which IPC stream to use (0 = default single-stream).
void emitIpcBody(IRBuilder<> &B,
                 const LoweringHelpers &H,
                 Value *rt, Value *peerBase,
                 Value *dstOff, Value *srcBuf, Value *srcOff,
                 Value *size, int streamIndex = 0) {
    Value *dst     = B.CreateGEP(H.i8Ty, peerBase, dstOff);
    Value *srcBase = B.CreateCall(H.localBaseFn, {rt, srcBuf});
    Value *src     = B.CreateGEP(H.i8Ty, srcBase, srcOff);
    Value *idxVal  = ConstantInt::get(H.i32Ty, streamIndex);
    Value *stream  = B.CreateCall(H.indexedStreamFn, {rt, idxVal});
    // hipMemcpyDeviceToDevice = 3.
    B.CreateCall(H.memcpyFn,
                 {dst, src, size, B.getInt32(3), stream});
}

// Emit the DWQ enqueue at IRBuilder B's insertion point.
void emitDwqBody(IRBuilder<> &B,
                 const LoweringHelpers &H,
                 ArrayRef<Value *> args) {
    B.CreateCall(H.enqFn, args);
}

// Materialize the IPC_PUSH lowering: caller asserts the peer is
// IPC-mapped, so we issue the memcpy unconditionally. Replaces
//   call void @gicc.runtime.put_no_db.placeholder(rt, peer, dst_buf,
//                                                 dst_off, src_buf,
//                                                 src_off, size), !site_id
// with
//   %peer_base = call ptr @gicc_runtime_peer_ipc_base(rt, peer, dst_buf)
//   <emitIpcBody using stream_index from hint>
void lowerIpcPush(CallInst *placeholder, int streamIndex) {
    Module      *M = placeholder->getModule();
    IRBuilder<>  B(placeholder);
    LoweringHelpers H = makeHelpers(M);

    Value *rt     = placeholder->getArgOperand(0);
    Value *peer   = placeholder->getArgOperand(1);
    Value *dstBuf = placeholder->getArgOperand(2);
    Value *dstOff = placeholder->getArgOperand(3);
    Value *srcBuf = placeholder->getArgOperand(4);
    Value *srcOff = placeholder->getArgOperand(5);
    Value *size   = placeholder->getArgOperand(6);

    Value *peerBase = B.CreateCall(H.peerBaseFn, {rt, peer, dstBuf});
    emitIpcBody(B, H, rt, peerBase, dstOff, srcBuf, srcOff, size, streamIndex);
    placeholder->eraseFromParent();
}

// Materialize the DWQ_TRIGGER lowering. Replaces the placeholder with
// a single call to gicc_runtime_dwq_enqueue.
void lowerDwqTrigger(CallInst *placeholder) {
    Module      *M = placeholder->getModule();
    IRBuilder<>  B(placeholder);
    LoweringHelpers H = makeHelpers(M);

    SmallVector<Value *, 7> args(placeholder->args().begin(),
                                  placeholder->args().begin() + 7);
    emitDwqBody(B, H, args);
    placeholder->eraseFromParent();
}

// Materialize the IPC_OR_DWQ hybrid lowering. The placeholder is
// replaced by a runtime branch:
//
//   %peer_base = call ptr @gicc_runtime_peer_ipc_base(rt, peer, dst_buf)
//   %has_ipc   = icmp ne ptr %peer_base, null
//   br i1 %has_ipc, label %ipc, label %dwq
//   ipc:
//     <emitIpcBody using %peer_base>
//     br label %done
//   dwq:
//     <emitDwqBody>
//     br label %done
//   done:
//     <continuation of the original block>
//
// This lets the compiler emit one trace function that handles both
// same-node IPC peers and off-node DWQ peers without needing a hint.
void lowerIpcOrDwq(CallInst *placeholder, int streamIndex) {
    Module      *M  = placeholder->getModule();
    IRBuilder<>  B(placeholder);
    LoweringHelpers H = makeHelpers(M);

    Value *rt     = placeholder->getArgOperand(0);
    Value *peer   = placeholder->getArgOperand(1);
    Value *dstBuf = placeholder->getArgOperand(2);
    Value *dstOff = placeholder->getArgOperand(3);
    Value *srcBuf = placeholder->getArgOperand(4);
    Value *srcOff = placeholder->getArgOperand(5);
    Value *size   = placeholder->getArgOperand(6);

    // Compute %peer_base + branch condition before splitting the block.
    Value *peerBase = B.CreateCall(H.peerBaseFn, {rt, peer, dstBuf});
    Value *nullPtr  = ConstantPointerNull::get(
        cast<PointerType>(H.ptrTy));
    Value *hasIpc   = B.CreateICmpNE(peerBase, nullPtr, "has_ipc");

    // Split the placeholder's BB at the placeholder. SplitBlock returns
    // the successor block; the original block keeps everything before
    // the split (including %peer_base + %has_ipc) plus an unconditional
    // br to the successor that we'll replace.
    BasicBlock *origBB = placeholder->getParent();
    BasicBlock *doneBB = origBB->splitBasicBlock(placeholder, "ipc_or_dwq.done");

    // SplitBlock dropped the placeholder into doneBB. Erase it now —
    // the IPC/DWQ branches will emit the real work.
    placeholder->eraseFromParent();

    // Make the IPC + DWQ blocks. They both unconditionally jump to
    // doneBB so the rest of the original function falls through.
    Function   *F     = origBB->getParent();
    LLVMContext &Ctx  = F->getContext();
    BasicBlock *ipcBB = BasicBlock::Create(Ctx, "ipc", F, doneBB);
    BasicBlock *dwqBB = BasicBlock::Create(Ctx, "dwq", F, doneBB);

    // Replace origBB's tail (the unconditional br created by splitBB)
    // with a conditional br on %has_ipc.
    Instruction *origTerm = origBB->getTerminator();
    BranchInst::Create(ipcBB, dwqBB, hasIpc, origTerm);
    origTerm->eraseFromParent();

    // ipcBB: hipMemcpyAsync, then jump to done.
    {
        IRBuilder<> IB(ipcBB);
        emitIpcBody(IB, H, rt, peerBase, dstOff, srcBuf, srcOff, size, streamIndex);
        IB.CreateBr(doneBB);
    }
    // dwqBB: gicc_runtime_dwq_enqueue, then jump to done.
    {
        IRBuilder<> DB(dwqBB);
        Value *args[7] = {rt, peer, dstBuf, dstOff, srcBuf, srcOff, size};
        emitDwqBody(DB, H, args);
        DB.CreateBr(doneBB);
    }
}

// Materialize a DWQ_BATCHED group: collapse N consecutive placeholders
// in the same BB into a single gicc_runtime_dwq_enqueue_batched call.
//
// Stack-allocates 6 arrays of N entries each at the entry block of the
// containing function (so the allocas dominate every store). At the
// FIRST placeholder's position, stores each placeholder's args into
// the corresponding array slot, then issues one batched call. Erases
// all N placeholders.
//
// For N == 1, this is functionally equivalent to gicc_runtime_dwq_enqueue
// with a slight allocation/store penalty — used uniformly so the IR
// pattern is recognizable regardless of group size.
void lowerDwqBatched(ArrayRef<CallInst *> group) {
    if (group.empty()) return;
    Module      *M     = group.front()->getModule();
    Function    *F     = group.front()->getFunction();
    LLVMContext &Ctx   = M->getContext();
    LoweringHelpers H  = makeHelpers(M);
    const unsigned N   = group.size();

    // Stack arrays at function entry so they dominate every BB.
    IRBuilder<> EB(&F->getEntryBlock(), F->getEntryBlock().getFirstInsertionPt());
    auto allocArr = [&](Type *eltTy, const Twine &name) {
        return EB.CreateAlloca(eltTy, EB.getInt32(N), name);
    };
    Value *peerArr   = allocArr(H.i32Ty, "dwq.peers");
    Value *dstBufArr = allocArr(H.i32Ty, "dwq.dst_bufs");
    Value *dstOffArr = allocArr(H.i64Ty, "dwq.dst_offs");
    Value *srcBufArr = allocArr(H.i32Ty, "dwq.src_bufs");
    Value *srcOffArr = allocArr(H.i64Ty, "dwq.src_offs");
    Value *sizeArr   = allocArr(H.i64Ty, "dwq.sizes");

    // Insert stores + the batched call right BEFORE the first placeholder.
    IRBuilder<> B(group.front());
    Value *rt = group.front()->getArgOperand(0);
    for (unsigned i = 0; i < N; ++i) {
        CallInst *PH = group[i];
        Value *idx   = B.getInt32(i);
        Value *peer   = PH->getArgOperand(1);
        Value *dstBuf = PH->getArgOperand(2);
        Value *dstOff = PH->getArgOperand(3);
        Value *srcBuf = PH->getArgOperand(4);
        Value *srcOff = PH->getArgOperand(5);
        Value *size   = PH->getArgOperand(6);
        B.CreateStore(peer,   B.CreateGEP(H.i32Ty, peerArr,   idx));
        B.CreateStore(dstBuf, B.CreateGEP(H.i32Ty, dstBufArr, idx));
        B.CreateStore(dstOff, B.CreateGEP(H.i64Ty, dstOffArr, idx));
        B.CreateStore(srcBuf, B.CreateGEP(H.i32Ty, srcBufArr, idx));
        B.CreateStore(srcOff, B.CreateGEP(H.i64Ty, srcOffArr, idx));
        B.CreateStore(size,   B.CreateGEP(H.i64Ty, sizeArr,   idx));
    }
    B.CreateCall(H.enqBatchedFn,
                 {rt, B.getInt32(static_cast<int32_t>(N)),
                  peerArr, dstBufArr, dstOffArr,
                  srcBufArr, srcOffArr, sizeArr});

    for (CallInst *PH : group) PH->eraseFromParent();
}

}  // namespace

// Per-site HK info loaded from per-kernel JSON files via
// collectLaunchInventory. Used by the dispatch driver to enforce the
// HK / hint cross-check (Task 2).
struct SiteHKInfo {
    bool        hk_capable = true;   // default: assume HK if no JSON found
    std::string hk_fail_reason;
    std::string kernelMangled;       // for proxy_aware write-back
};

// Walk every launch site's kernel template once and index by site_id.
// Returns an empty map if no launches were found in M (e.g. unit-test
// IR with placeholders but no @llvm.global.annotations launch wrapper).
std::unordered_map<std::string, SiteHKInfo>
buildSiteHKMap(Module &M, const std::string &metaDir) {
    std::unordered_map<std::string, SiteHKInfo> out;
    auto inv = collectLaunchInventory(M, metaDir);
    for (const auto &site : inv.sites) {
        if (!site.haveTemplate) continue;
        for (const auto &op : site.kernelTemplate.ops) {
            SiteHKInfo info;
            info.hk_capable     = op.hk_capable;
            info.hk_fail_reason = op.hk_fail_reason;
            info.kernelMangled  = site.kernelMangled;
            out[op.siteId] = std::move(info);
        }
    }
    return out;
}

PreservedAnalyses GICCDispatchLoweringPass::run(Module &M,
                                                 ModuleAnalysisManager &) {
    const auto &cfg = getConfig();
    if (cfg.mode != Mode::Lower) return PreservedAnalyses::all();

    HintFile hints;
    if (!cfg.hintIn.empty()) {
        if (!readHintFile(cfg.hintIn, hints)) {
            errs() << "[dispatch-lowering] WARN: could not read hint "
                   << cfg.hintIn << "; falling back to default DWQ_TRIGGER\n";
        }
    }

    SmallVector<CallInst *, 32> placeholders;
    for (Function &F : M) {
        for (BasicBlock &BB : F) {
            for (Instruction &I : BB) {
                auto *CI = dyn_cast<CallInst>(&I);
                if (!CI) continue;
                auto *callee = CI->getCalledFunction();
                if (!callee) continue;
                StringRef n = callee->getName();
                if (n == "gicc.runtime.put_no_db.placeholder" ||
                    n == "gicc.runtime.get_no_db.placeholder") {
                    placeholders.push_back(CI);
                }
            }
        }
    }
    if (placeholders.empty()) return PreservedAnalyses::all();

    // Per-site HK capability map (from kernel JSON) for cross-check.
    auto hkMap = buildSiteHKMap(M, cfg.metaDir);

    // CPU_PROXY_ENQUEUE requires runtime support. Read the env var
    // exactly once so we get a consistent answer across all sites.
    static const bool proxyEnabled =
        std::getenv("GICC_PROXY_ENABLED") != nullptr;

    // Set of kernels that had at least one site lowered to
    // CPU_PROXY_ENQUEUE. Persisted back to per-kernel JSON at end of
    // pass so device-lowering (Task 8) knows to preserve the device-side
    // body for proxy sites.
    std::set<std::string> proxyAwareKernels;

    // Group consecutive same-BB placeholders that all want DwqBatched
    // into a single batched call. Walk in IR order so groups are
    // contiguous; flush a group whenever we hit a placeholder with a
    // different dispatch kind, a different BB, or the end of the list.
    SmallVector<CallInst *, 4> batchGroup;
    auto flushBatch = [&] {
        if (!batchGroup.empty()) {
            lowerDwqBatched(batchGroup);
            batchGroup.clear();
        }
    };

    for (auto *PH : placeholders) {
        StringRef siteId  = siteIdOf(PH);
        SiteHint  sh      = hintFor(hints, siteId);
        DispatchKind d    = sh.dispatch;

        // Cross-check 1: HK-incapable sites can only go to CPU_PROXY.
        // Look up by site_id; if no kernel JSON entry was found we
        // conservatively treat the site as HK-capable (default true)
        // — matches pre-Task 2 behavior so unit-test IR without launch
        // metadata still runs.
        auto hkIt = hkMap.find(siteId.str());
        const SiteHKInfo *hkInfo = (hkIt != hkMap.end()) ? &hkIt->second
                                                         : nullptr;
        if (hkInfo && !hkInfo->hk_capable &&
            d != DispatchKind::CpuProxyEnqueue) {
            report_fatal_error(
                Twine("gicc: site ") + siteId +
                " has hk_capable=false but hint requests " +
                dispatchName(d) +
                "; only CPU_PROXY_ENQUEUE accepts non-HK args. Reason: " +
                hkInfo->hk_fail_reason);
        }

        // Cross-check 2: CPU_PROXY hint requires runtime support.
        if (d == DispatchKind::CpuProxyEnqueue && !proxyEnabled) {
            report_fatal_error(
                Twine("gicc: hint requests CPU_PROXY_ENQUEUE for site ") +
                siteId +
                " but GICC_PROXY_ENABLED is not set. Either rebuild with "
                "-DGICC_ENABLE_CPU_PROXY=ON and set the env, or change "
                "the hint.");
        }

        if (d == DispatchKind::DwqBatched) {
            // Same BB as the running group? Add. Otherwise flush + start fresh.
            if (!batchGroup.empty() &&
                batchGroup.back()->getParent() != PH->getParent()) {
                flushBatch();
            }
            batchGroup.push_back(PH);
            continue;
        }
        flushBatch();

        switch (d) {
            case DispatchKind::IpcPush:
                lowerIpcPush(PH, sh.streamIndex);
                break;
            case DispatchKind::IpcOrDwq:
                lowerIpcOrDwq(PH, sh.streamIndex);
                break;
            case DispatchKind::DwqTrigger:
            case DispatchKind::Unknown:
                lowerDwqTrigger(PH);
                break;
            case DispatchKind::CpuProxyEnqueue: {
                // Lower the placeholder by ERASING it from the host
                // trace. The actual runtime work is done device-side
                // (see Task 8 for device-lowering changes that
                // preserve the device-side put_no_db body for proxy
                // sites). The host trace function does not stage
                // anything for this site.
                if (hkInfo && !hkInfo->kernelMangled.empty()) {
                    proxyAwareKernels.insert(hkInfo->kernelMangled);
                }
                PH->eraseFromParent();
                break;
            }
            case DispatchKind::DwqBatched:
                // Unreachable — handled above.
                break;
        }
    }
    flushBatch();

    // Drop the now-unused placeholder declaration so the linker doesn't
    // need a definition.
    for (StringRef n : {"gicc.runtime.put_no_db.placeholder",
                        "gicc.runtime.get_no_db.placeholder"}) {
        if (Function *F = M.getFunction(n))
            if (F->use_empty()) F->eraseFromParent();
    }

    // Write the proxy_aware bit back to per-kernel JSON for any kernel
    // that had a CPU_PROXY_ENQUEUE site. Read-modify-write: load,
    // toggle the bit if necessary, write back. Skips the I/O when the
    // bit is already set.
    for (const auto &kname : proxyAwareKernels) {
        KernelTemplate t;
        if (!readKernelTemplate(cfg.metaDir, kname, t)) continue;
        if (t.proxy_aware) continue;
        t.proxy_aware = true;
        if (!writeKernelTemplate(cfg.metaDir, t)) {
            errs() << "[dispatch-lowering] WARN: could not write "
                   << "proxy_aware=true to " << cfg.metaDir << "/"
                   << kname << ".json\n";
        }
    }

    return PreservedAnalyses::none();
}

}  // namespace gicc::pass
