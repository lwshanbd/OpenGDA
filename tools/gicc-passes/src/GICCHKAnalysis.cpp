#include "GICCHKAnalysis.h"
#include "GICCPassConfig.h"
#include "HKAnalysis.h"
#include "HostMirrorAnnotation.h"
#include "KernelInventory.h"
#include "MetadataIO.h"

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

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

    StringRef tuName = M.getName();

    for (Function &F : M) {
        if (F.isDeclaration() || !isGPUKernel(F)) continue;

        GICCKernelInfo info;
        collectGICCSites(F, info);
        if (info.sites.empty()) continue;

        // Per-kernel set of host-mirrored formals. Computed once per
        // kernel; threaded through isHK so loads of host-mirrored fields
        // are accepted instead of degrading to CPU_PROXY.
        std::vector<bool> hostMirrored = computeHostMirroredFormals(F);

        for (auto &site : info.sites) {
            CallInst *CI = site.CI;
            bool        siteHK   = true;
            std::string failReason;

            // Skip the Runtime context pointer (arg 0). HK analysis only
            // matters for the data-shape arguments that the host-side
            // trace function will consume.
            for (unsigned ai = 1; ai < CI->arg_size(); ++ai) {
                Value *operand = CI->getArgOperand(ai);
                HKResult r = isHK(operand, &F, &hostMirrored);
                if (r.ok) continue;

                // Record the first failing argument as the canonical
                // reason; surface a per-arg warning for each so users
                // can see the full diagnostic surface.
                if (siteHK) {
                    siteHK     = false;
                    failReason = "argument '" + std::string(argName(site.kind, ai))
                                 + "' is not host-knowable: " + r.failReason;
                }

                printSourceLocation(errs(), CI, tuName.empty() ? "?" : tuName.data());
                errs() << ": warning: gicc::" << opKindName(site.kind)
                       << ": argument '" << argName(site.kind, ai)
                       << "' is not host-knowable"
                       << " (will require CPU_PROXY_ENQUEUE dispatch)\n";

                if (r.failOrigin) {
                    printSourceLocation(errs(), r.failOrigin,
                                        tuName.empty() ? "?" : tuName.data());
                    errs() << ": note: " << r.failReason << "\n";
                } else {
                    errs() << "note: " << r.failReason << "\n";
                }
            }

            site.hk_capable     = siteHK;
            site.hk_fail_reason = failReason;
        }

        // Propagate the per-site capability bits into the on-disk
        // kernel template (written by GICCDeviceDiscovery) so the
        // host-side passes can read hk_capable directly. Match by
        // siteId -- the only stable join key between in-memory
        // GICCCallSite and on-disk OpTemplate.
        //
        // CROSS-TU / PARALLEL-BUILD HAZARD:
        // This is a read-modify-write on a shared JSON file. If the
        // same kernel is compiled in another translation unit (or
        // re-discovered under `make -j`), GICCDeviceDiscovery in that
        // other TU may overwrite this file *after* HK has set
        // hk_capable=false here, silently restoring the default
        // hk_capable=true. There is no file lock and no merge step.
        //
        // CURRENT SCOPE (safe):
        // Examples and minimod each compile any given kernel in
        // exactly one TU, so Discovery writes the JSON exactly once
        // per kernel and HK's later RMW is the last writer. The bug
        // only manifests when the same kernel symbol is emitted by
        // multiple TUs in one build.
        //
        // FUTURE FIX:
        // Fold HK's capability-bit write into Discovery's
        // writeKernelTemplate so there is a single writer per kernel
        // JSON, and drop this RMW block entirely.
        if ((cfg.mode == Mode::FeatureExtract || cfg.mode == Mode::Lower)
            && !cfg.metaDir.empty()) {
            KernelTemplate t;
            if (readKernelTemplate(cfg.metaDir, info.mangledName, t)) {
                bool changed = false;
                for (auto &op : t.ops) {
                    for (const auto &site : info.sites) {
                        if (op.siteId != site.siteId) continue;
                        if (op.hk_capable != site.hk_capable
                            || op.hk_fail_reason != site.hk_fail_reason) {
                            op.hk_capable     = site.hk_capable;
                            op.hk_fail_reason = site.hk_fail_reason;
                            changed = true;
                        }
                        break;
                    }
                }
                if (changed) {
                    if (!writeKernelTemplate(cfg.metaDir, t)) {
                        errs() << "[hk-analysis] WARN: failed to update template "
                               << "for " << info.mangledName << " under "
                               << cfg.metaDir << "\n";
                    }
                }
            }
        }
    }
    return PreservedAnalyses::all();
}

}  // namespace gicc::pass
