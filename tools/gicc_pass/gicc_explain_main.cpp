/**
 * gicc_explain_main.cpp — Standalone CLI that runs the GICC analysis pass
 * against an arbitrary .ll / .bc module.
 *
 * Usage:
 *   GICC_POLICY_FILE=policies/policy_tioga.json  \
 *   GICC_EXPLAIN=1 GICC_RANK_HINT=32             \
 *     gicc_explain  input.ll [-o output.ll]
 *
 * Intended for toy-kernel unit tests (Block M2-I) and for the rule-list
 * fitter's offline feature-dump step.
 */

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

// Forward declaration — we dlopen ourselves to pick up the plugin-registered
// pass. The plugin entry point is in GiccPass.cpp linked into libgicc_pass.so;
// here we just need to register it directly rather than going through the
// plugin loader, since gicc_explain is a monolithic binary.
namespace {
class GiccAnalysisPass;  // opaque; we rely on the shared .so when invoked via opt
}

// Since this CLI is for debugging ONLY, the simplest route is to link the
// pass object file into this binary and register the pass via the public
// NewPM entry. To keep the two binaries from fighting over llvmGetPassPluginInfo
// (which has weak linkage), we declare a single shim:
extern "C" ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo();

static cl::opt<std::string> InputFilename(cl::Positional,
    cl::desc("<input .ll/.bc>"), cl::Required);
static cl::opt<std::string> OutputFilename("o",
    cl::desc("Output .ll path (optional; default = no write)"),
    cl::value_desc("filename"), cl::init(""));

int main(int argc, char** argv) {
    InitLLVM X(argc, argv);
    cl::ParseCommandLineOptions(argc, argv, "gicc-explain — run the GICC analysis pass\n");

    LLVMContext ctx;
    SMDiagnostic err;
    std::unique_ptr<Module> M = parseIRFile(InputFilename, err, ctx);
    if (!M) {
        err.print(argv[0], errs());
        return 1;
    }

    PassBuilder PB;
    ModuleAnalysisManager MAM;
    CGSCCAnalysisManager CGAM;
    FunctionAnalysisManager FAM;
    LoopAnalysisManager LAM;
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    // Register our pass via the same entry point opt uses.
    auto pinfo = llvmGetPassPluginInfo();
    pinfo.RegisterPassBuilderCallbacks(PB);

    ModulePassManager MPM;
    if (auto ec = PB.parsePassPipeline(MPM, "gicc-analysis")) {
        errs() << "gicc_explain: " << toString(std::move(ec)) << "\n";
        return 2;
    }
    MPM.run(*M, MAM);

    if (!OutputFilename.empty()) {
        std::error_code ec;
        ToolOutputFile out(OutputFilename, ec, sys::fs::OF_TextWithCRLF);
        if (ec) { errs() << ec.message() << "\n"; return 3; }
        M->print(out.os(), /*AAW=*/nullptr);
        out.keep();
    }
    return 0;
}
