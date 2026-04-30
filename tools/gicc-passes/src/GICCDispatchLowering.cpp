#include "GICCDispatchLowering.h"
#include "GICCPassConfig.h"

#include "llvm/ADT/SmallVector.h"
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

enum class DispatchKind { IpcPush, DwqTrigger, DwqBatched, Unknown };

DispatchKind parseDispatch(StringRef s) {
    if (s == "IPC_PUSH")    return DispatchKind::IpcPush;
    if (s == "DWQ_TRIGGER") return DispatchKind::DwqTrigger;
    if (s == "DWQ_BATCHED") return DispatchKind::DwqBatched;
    return DispatchKind::Unknown;
}

struct HintFile {
    DispatchKind                          defaultDispatch = DispatchKind::DwqTrigger;
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

// Materialize the IPC_PUSH lowering. Replaces
//   call void @gicc.runtime.put_no_db.placeholder(rt, peer, dst_buf,
//                                                 dst_off, src_buf,
//                                                 src_off, size), !site_id
// with
//   %peer_base = call ptr @gicc_runtime_peer_ipc_base(rt, peer, dst_buf)
//   %dst       = getelementptr i8, ptr %peer_base, i64 dst_off
//   %src_base  = call ptr @gicc_runtime_local_buf_base(rt, src_buf)
//   %src       = getelementptr i8, ptr %src_base, i64 src_off
//   %stream    = call ptr @gicc_runtime_ipc_stream(rt)
//   call i32 @hipMemcpyAsync(ptr %dst, ptr %src, i64 size, i32 3, ptr %stream)
void lowerIpcPush(CallInst *placeholder) {
    LLVMContext &Ctx = placeholder->getContext();
    Module      *M   = placeholder->getModule();
    IRBuilder<>  B(placeholder);

    Value *rt      = placeholder->getArgOperand(0);
    Value *peer    = placeholder->getArgOperand(1);
    Value *dstBuf  = placeholder->getArgOperand(2);
    Value *dstOff  = placeholder->getArgOperand(3);
    Value *srcBuf  = placeholder->getArgOperand(4);
    Value *srcOff  = placeholder->getArgOperand(5);
    Value *size    = placeholder->getArgOperand(6);

    Type *ptrTy = PointerType::getUnqual(Ctx);
    Type *i32Ty = Type::getInt32Ty(Ctx);
    Type *i64Ty = Type::getInt64Ty(Ctx);
    Type *i8Ty  = Type::getInt8Ty(Ctx);

    auto peerBaseFn = M->getOrInsertFunction(
        "gicc_runtime_peer_ipc_base",
        FunctionType::get(ptrTy, {ptrTy, i32Ty, i32Ty}, false));
    auto localBaseFn = M->getOrInsertFunction(
        "gicc_runtime_local_buf_base",
        FunctionType::get(ptrTy, {ptrTy, i32Ty}, false));
    auto streamFn = M->getOrInsertFunction(
        "gicc_runtime_ipc_stream",
        FunctionType::get(ptrTy, {ptrTy}, false));
    auto memcpyFn = M->getOrInsertFunction(
        "hipMemcpyAsync",
        FunctionType::get(i32Ty,
            {ptrTy, ptrTy, i64Ty, i32Ty, ptrTy}, false));

    Value *peerBase = B.CreateCall(peerBaseFn,  {rt, peer, dstBuf});
    Value *dst      = B.CreateGEP(i8Ty, peerBase, dstOff);
    Value *srcBase  = B.CreateCall(localBaseFn, {rt, srcBuf});
    Value *src      = B.CreateGEP(i8Ty, srcBase, srcOff);
    Value *stream   = B.CreateCall(streamFn, {rt});
    // hipMemcpyDeviceToDevice = 3.
    B.CreateCall(memcpyFn,
                 {dst, src, size, B.getInt32(3), stream});
    placeholder->eraseFromParent();
}

// Materialize the DWQ_TRIGGER lowering. Replaces the placeholder with
// a single call to gicc_runtime_dwq_enqueue.
void lowerDwqTrigger(CallInst *placeholder) {
    LLVMContext &Ctx = placeholder->getContext();
    Module      *M   = placeholder->getModule();
    IRBuilder<>  B(placeholder);

    Type *ptrTy = PointerType::getUnqual(Ctx);
    Type *i32Ty = Type::getInt32Ty(Ctx);
    Type *i64Ty = Type::getInt64Ty(Ctx);
    auto enqFn = M->getOrInsertFunction(
        "gicc_runtime_dwq_enqueue",
        FunctionType::get(B.getVoidTy(),
            {ptrTy, i32Ty, i32Ty, i64Ty, i32Ty, i64Ty, i64Ty}, false));

    SmallVector<Value *, 7> args(placeholder->args().begin(),
                                  placeholder->args().begin() + 7);
    B.CreateCall(enqFn, args);
    placeholder->eraseFromParent();
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

    for (auto *PH : placeholders) {
        StringRef siteId = siteIdOf(PH);
        DispatchKind d   = hintFor(hints, siteId);
        switch (d) {
            case DispatchKind::IpcPush:
                lowerIpcPush(PH);
                break;
            case DispatchKind::DwqTrigger:
            case DispatchKind::DwqBatched:   // v1 falls back to single-trigger
            case DispatchKind::Unknown:
                lowerDwqTrigger(PH);
                break;
        }
    }

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
