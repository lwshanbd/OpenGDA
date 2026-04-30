#include "GICCDispatchLowering.h"
#include "GICCPassConfig.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

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
    IpcPush,    // force IPC path (assumes peer is mapped)
    DwqTrigger, // force DWQ path
    DwqBatched, // currently lowered same as DwqTrigger
    IpcOrDwq,   // runtime branch: IPC if mapped else DWQ
    Unknown,
};

DispatchKind parseDispatch(StringRef s) {
    if (s == "IPC_PUSH")    return DispatchKind::IpcPush;
    if (s == "DWQ_TRIGGER") return DispatchKind::DwqTrigger;
    if (s == "DWQ_BATCHED") return DispatchKind::DwqBatched;
    if (s == "IPC_OR_DWQ")  return DispatchKind::IpcOrDwq;
    return DispatchKind::Unknown;
}

struct HintFile {
    // When no hint.json is supplied we default to the hybrid runtime
    // branch. Same-node peers go through IPC (low latency / high BW
    // SDMA) and off-node peers fall back to DWQ. Lit tests that
    // expect a hard DWQ default explicitly set GICC_HINT_IN to a
    // hint.json with default_dispatch="DWQ_TRIGGER".
    DispatchKind                          defaultDispatch = DispatchKind::IpcOrDwq;
    std::unordered_map<std::string,
                       DispatchKind>      sites;
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
            out.sites[kv.first.str()] = parseDispatch(*disp);
        }
    }
    return true;
}

DispatchKind hintFor(const HintFile &h, StringRef siteId) {
    auto it = h.sites.find(siteId.str());
    if (it == h.sites.end()) return h.defaultDispatch;
    return it->second;
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
    FunctionCallee streamFn;
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
//   %stream    = call ptr @gicc_runtime_ipc_stream(rt)
//   call i32 @hipMemcpyAsync(ptr %dst, ptr %src, i64 size, i32 3, ptr %stream)
//
// peerBase must already be a non-null ptr to the peer's mapped buffer.
void emitIpcBody(IRBuilder<> &B,
                 const LoweringHelpers &H,
                 Value *rt, Value *peerBase,
                 Value *dstOff, Value *srcBuf, Value *srcOff,
                 Value *size) {
    Value *dst     = B.CreateGEP(H.i8Ty, peerBase, dstOff);
    Value *srcBase = B.CreateCall(H.localBaseFn, {rt, srcBuf});
    Value *src     = B.CreateGEP(H.i8Ty, srcBase, srcOff);
    Value *stream  = B.CreateCall(H.streamFn, {rt});
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
//   <emitIpcBody>
void lowerIpcPush(CallInst *placeholder) {
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
    emitIpcBody(B, H, rt, peerBase, dstOff, srcBuf, srcOff, size);
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
void lowerIpcOrDwq(CallInst *placeholder) {
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
        emitIpcBody(IB, H, rt, peerBase, dstOff, srcBuf, srcOff, size);
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
        StringRef siteId = siteIdOf(PH);
        DispatchKind d   = hintFor(hints, siteId);

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
                lowerIpcPush(PH);
                break;
            case DispatchKind::IpcOrDwq:
                lowerIpcOrDwq(PH);
                break;
            case DispatchKind::DwqTrigger:
            case DispatchKind::Unknown:
                lowerDwqTrigger(PH);
                break;
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

    return PreservedAnalyses::none();
}

}  // namespace gicc::pass
