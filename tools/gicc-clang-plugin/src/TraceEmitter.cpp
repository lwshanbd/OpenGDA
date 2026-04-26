/**
 * TraceEmitter.cpp — see TraceEmitter.h for design intent.
 *
 * F1: build a kernel_trace<&K> specialization with an empty body and
 *     write it to /tmp/<basename>.gicc.cpp.
 *
 * F2 (this commit): for each gicc::put_no_db call in a validated kernel
 *     emit a host-side block that resolves the local source Buffer via
 *     rt.buffer_by_lkey(lkey), the destination base via
 *     rt.peer_buffer_base(peer, rkey), then calls rt.put_no_db with the
 *     resolved offsets. Argument source text is extracted with
 *     Lexer::getSourceText so the generated code mirrors what the user
 *     actually wrote (variable names, casts, etc.). Loops / threadIdx-
 *     indexed puts come in F3/F4.
 */
#include "TraceEmitter.h"

#include "clang/AST/PrettyPrinter.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"

#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>

namespace gicc_plugin {

namespace {

// Compute the sidecar path for a given main-file path. v1 uses /tmp so
// lit-test source trees (which are read-only) work without extra plumbing.
// Phase G will replace this with a CMake-build-dir-aware path.
std::string sidecar_path_for(const std::string& main_file) {
    std::string base = llvm::sys::path::stem(main_file).str();
    return std::string("/tmp/") + base + ".gicc.cpp";
}

// Render a kernel param list as "T0 n0, T1 n1, ..." with the leading
// DeviceCtx* (param 0) skipped. Uses a SuppressTagKeyword printing
// policy so we get "gicc::DeviceCtx*" rather than "class gicc::DeviceCtx*".
std::string param_decl_list(const clang::FunctionDecl* fd) {
    clang::PrintingPolicy pp(fd->getASTContext().getLangOpts());
    pp.SuppressTagKeyword = true;
    pp.Bool = true;
    std::string out;
    for (unsigned i = 1; i < fd->getNumParams(); i++) {
        if (i > 1) out += ", ";
        const auto* p = fd->getParamDecl(i);
        out += p->getType().getAsString(pp);
        out += " ";
        out += p->getNameAsString();
    }
    return out;
}

// Pull the verbatim source text of an Expr. We use getTokenRange so the
// final token is included (getCharRange would chop it off). For the
// straight-line-only F2 subset every arg should be a simple identifier
// or short expression; if Lexer returns empty (e.g. macro-expanded
// argument) we fall back to a literal "/*<unrenderable>*/" comment so
// the sidecar still parses.
std::string source_text(const clang::Expr* e, const clang::SourceManager& sm,
                        const clang::LangOptions& lo) {
    if (!e) return "/*null*/";
    auto range = clang::CharSourceRange::getTokenRange(e->getSourceRange());
    llvm::StringRef text = clang::Lexer::getSourceText(range, sm, lo);
    if (text.empty()) return "/*<unrenderable>*/";
    return text.str();
}

// Emit the host-side resolution + rt.put_no_db call for one
// gicc::put_no_db(ctx, local_addr, local_lkey, remote_addr, remote_rkey,
// size [, signaled]) call. We deliberately wrap each block in its own
// scope so the __src / __soff / __base / __doff temporaries don't
// collide across multiple lifted puts.
void emit_put_block(std::ostream& out, const clang::CallExpr* ce,
                    const clang::SourceManager& sm,
                    const clang::LangOptions& lo) {
    const std::string A  = source_text(ce->getArg(1), sm, lo);
    const std::string LK = source_text(ce->getArg(2), sm, lo);
    const std::string RA = source_text(ce->getArg(3), sm, lo);
    const std::string RK = source_text(ce->getArg(4), sm, lo);
    const std::string S  = source_text(ce->getArg(5), sm, lo);

    out << "        {\n";
    out << "            auto& __src = rt.buffer_by_lkey((uint32_t)("
        << LK << "));\n";
    out << "            size_t __soff = (uint64_t)(" << A
        << ") - (uint64_t)__src.addr;\n";
    out << "            uint64_t __base = rt.peer_buffer_base(peer, (uint32_t)("
        << RK << "));\n";
    out << "            size_t __doff = (uint64_t)(" << RA
        << ") - __base;\n";
    out << "            rt.put_no_db(__src, peer, (int)(" << RK
        << "), (size_t)(" << S << "), __soff, __doff);\n";
    out << "        }\n";
}

} // namespace

std::string TraceEmitter::emit(const KernelInfo& ki, const HKAnalysis& /*hk*/) {
    auto& sm = CI_.getSourceManager();
    const auto& lo = CI_.getLangOpts();
    auto file_id = sm.getMainFileID();
    auto file_ref = sm.getFileEntryRefForID(file_id);
    std::string main_file = file_ref
        ? std::string(file_ref->getName())
        : std::string("unknown");
    std::string sidecar = sidecar_path_for(main_file);

    const bool first_open = opened_sidecars_.insert(sidecar).second;
    std::ofstream out(sidecar, first_open ? std::ios::trunc : std::ios::app);
    if (!out) {
        llvm::errs() << "[gicc-plugin] FAIL to open sidecar " << sidecar << "\n";
        return sidecar;
    }

    if (first_open) {
        // Header: pull in everything the trace body needs. The user must
        // build the sidecar with the same GICC_PLATFORM_* / bootstrap
        // defines as the main TU; Phase G will arrange that automatically.
        out << "// AUTO-GENERATED by gicc-clang-plugin. DO NOT EDIT.\n";
        out << "// Trace specializations for kernels in " << main_file << "\n\n";
        out << "#include <hip/hip_runtime.h>\n";
        out << "#include \"gicc/gicc.hpp\"\n\n";
    }

    const std::string kname = ki.decl->getNameAsString();
    const std::string params = param_decl_list(ki.decl);

    // Forward-declare the kernel so &kname is a well-formed expression
    // inside the specialization. In v1 we deliberately avoid pulling the
    // user's TU header into the sidecar; the prototype here is the
    // minimum surface needed to take the address.
    {
        clang::PrintingPolicy pp(ki.decl->getASTContext().getLangOpts());
        pp.SuppressTagKeyword = true;
        pp.Bool = true;
        out << "__global__ void " << kname << "(";
        for (unsigned i = 0; i < ki.decl->getNumParams(); i++) {
            if (i > 0) out << ", ";
            const auto* p = ki.decl->getParamDecl(i);
            out << p->getType().getAsString(pp);
        }
        out << ");\n\n";
    }

    out << "namespace gicc {\n";
    out << "namespace detail {\n";
    out << "template<> struct kernel_trace<&" << kname << "> {\n";
    out << "    static void run(gicc::Runtime& rt, int peer, "
           "dim3 grid, dim3 block";
    if (!params.empty()) out << ", " << params;
    out << ") {\n";
    out << "        (void)rt; (void)peer; (void)grid; (void)block;\n";
    for (unsigned i = 1; i < ki.decl->getNumParams(); i++) {
        out << "        (void)" << ki.decl->getParamDecl(i)->getNameAsString()
            << ";\n";
    }

    // F2: lift every straight-line gicc::put_no_db call from the kernel
    // body into a host-side rt.put_no_db. Order is preserved (calls in
    // ki.calls are in source order from the AST visitor).
    for (const auto& call : ki.calls) {
        if (call.qualified_name != "gicc::put_no_db") continue;
        if (!call.expr || call.expr->getNumArgs() < 6) continue;
        emit_put_block(out, call.expr, sm, lo);
    }

    out << "    }\n";
    out << "};\n";
    out << "} // namespace detail\n";
    out << "} // namespace gicc\n\n";

    llvm::errs() << "[gicc-plugin]   trace emitted: " << sidecar
                 << " for kernel " << kname << "\n";
    return sidecar;
}

} // namespace gicc_plugin
