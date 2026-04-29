#include "GICCHKAnalysis.h"
#include "GICCPassConfig.h"
#include "HKAnalysis.h"
#include "KernelInventory.h"

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace gicc::pass {

namespace {

void printSourceLocation(raw_ostream &os, const Instruction *I,
                         const char *fallback) {
    if (auto loc = I->getDebugLoc()) {
        StringRef file;
        if (auto *scope = loc->getScope()) {
            if (auto *f = scope->getFile()) file = f->getFilename();
        }
        if (file.empty()) os << fallback;
        else              os << file;
        os << ':' << loc.getLine() << ':' << loc.getCol();
    } else {
        os << fallback << ":?:?";
    }
}

const char *argName(GICCOpKind kind, unsigned argIdx) {
    // Argument name table mirrors the user-visible gicc:: signatures.
    // put_no_db / get_no_db: (ctx, peer, dst_buf, dst_off, src_buf, src_off, size)
    // flush / quiet:        (ctx)
    static const char *putArgs[] = {
        "ctx", "peer", "dst_buf", "dst_off", "src_buf", "src_off", "size"};
    if (kind == GICCOpKind::PutNoDb || kind == GICCOpKind::GetNoDb) {
        if (argIdx < std::size(putArgs)) return putArgs[argIdx];
    }
    return "<arg>";
}

}  // namespace

PreservedAnalyses GICCHKAnalysisPass::run(Module &M, ModuleAnalysisManager &) {
    const auto &cfg = getConfig();
    if (cfg.mode == Mode::Passthrough) return PreservedAnalyses::all();

    sawError = false;
    StringRef tuName = M.getName();

    for (Function &F : M) {
        if (F.isDeclaration() || !isGPUKernel(F)) continue;

        GICCKernelInfo info;
        collectGICCSites(F, info);
        if (info.sites.empty()) continue;

        for (const auto &site : info.sites) {
            CallInst *CI = site.CI;
            // Skip the Runtime context pointer (arg 0). HK analysis only
            // matters for the data-shape arguments that the host-side
            // trace function will consume.
            for (unsigned ai = 1; ai < CI->arg_size(); ++ai) {
                Value *operand = CI->getArgOperand(ai);
                HKResult r = isHK(operand, &F);
                if (r.ok) continue;

                sawError = true;

                printSourceLocation(errs(), CI, tuName.empty() ? "?" : tuName.data());
                errs() << ": error: gicc::" << opKindName(site.kind)
                       << ": argument '" << argName(site.kind, ai)
                       << "' is not host-knowable\n";

                if (r.failOrigin) {
                    printSourceLocation(errs(), r.failOrigin,
                                        tuName.empty() ? "?" : tuName.data());
                    errs() << ": note: " << r.failReason << "\n";
                } else {
                    errs() << "note: " << r.failReason << "\n";
                }
            }
        }
    }
    return PreservedAnalyses::all();
}

}  // namespace gicc::pass
