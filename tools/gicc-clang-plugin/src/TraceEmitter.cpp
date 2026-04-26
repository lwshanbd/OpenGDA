/**
 * TraceEmitter.cpp — see TraceEmitter.h for design intent.
 *
 * F1: build a kernel_trace<&K> specialization with an empty body and
 *     write it to /tmp/<basename>.gicc.cpp.
 *
 * F2: lift each straight-line gicc::put_no_db into a host-side block
 *     calling rt.put_no_db, with arg source text mirrored verbatim.
 *
 * F3 (this commit): walk the kernel body recursively. For ForStmt /
 *     WhileStmt / DoStmt / IfStmt with HK condition, emit the matching
 *     host-side control flow scaffold and recurse on the body. A nested
 *     gicc::put_no_db is then lifted with all surrounding HK control
 *     flow preserved. flush / quiet are dropped (no host effect).
 *     Statements we don't understand are skipped — they don't influence
 *     the schedule.
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

// Pull the verbatim source text of an Expr / Stmt. We use getTokenRange
// so the final token is included (getCharRange would chop it off). For
// macro-expanded ranges Lexer returns empty; we surface that as a
// placeholder so the sidecar still parses.
std::string source_text(const clang::Stmt* s, const clang::SourceManager& sm,
                        const clang::LangOptions& lo) {
    if (!s) return "/*null*/";
    auto range = clang::CharSourceRange::getTokenRange(s->getSourceRange());
    llvm::StringRef text = clang::Lexer::getSourceText(range, sm, lo);
    if (text.empty()) return "/*<unrenderable>*/";
    return text.str();
}

std::string source_text_expr(const clang::Expr* e,
                             const clang::SourceManager& sm,
                             const clang::LangOptions& lo) {
    return source_text(static_cast<const clang::Stmt*>(e), sm, lo);
}

// Trim trailing semicolon + whitespace from a text fragment. The init
// part of a ForStmt is a Stmt* (often a DeclStmt) whose source range
// usually does NOT include the trailing ';' — but be defensive.
std::string strip_trailing_semicolon(std::string s) {
    while (!s.empty() && (s.back() == ';' || s.back() == ' ' ||
                          s.back() == '\t' || s.back() == '\n')) {
        s.pop_back();
    }
    return s;
}

// Indent helper: emits N levels of 4-space indent.
void indent(std::ostream& out, int level) {
    for (int i = 0; i < level; i++) out << "    ";
}

// Emit the host-side resolution + rt.put_no_db call for one
// gicc::put_no_db(ctx, local_addr, local_lkey, remote_addr, remote_rkey,
// size [, signaled]) call. We deliberately wrap each block in its own
// scope so the _gicc_src / _gicc_soff / _gicc_base / _gicc_doff
// temporaries don't collide across multiple lifted puts. Names use a
// _gicc_ prefix (single underscore) rather than __ since identifiers
// starting with __ are reserved to the implementation per [lex.name]
// and could collide with libc++/libstdc++ internals.
void emit_put_block(std::ostream& out, int lvl, const clang::CallExpr* ce,
                    const clang::SourceManager& sm,
                    const clang::LangOptions& lo) {
    const std::string A  = source_text_expr(ce->getArg(1), sm, lo);
    const std::string LK = source_text_expr(ce->getArg(2), sm, lo);
    const std::string RA = source_text_expr(ce->getArg(3), sm, lo);
    const std::string RK = source_text_expr(ce->getArg(4), sm, lo);
    const std::string S  = source_text_expr(ce->getArg(5), sm, lo);

    indent(out, lvl);     out << "{\n";
    indent(out, lvl + 1); out << "auto& _gicc_src = rt.buffer_by_lkey((uint32_t)("
                              << LK << "));\n";
    indent(out, lvl + 1); out << "size_t _gicc_soff = (uint64_t)(" << A
                              << ") - (uint64_t)_gicc_src.addr;\n";
    indent(out, lvl + 1); out << "uint64_t _gicc_base = rt.peer_buffer_base(peer, (uint32_t)("
                              << RK << "));\n";
    indent(out, lvl + 1); out << "size_t _gicc_doff = (uint64_t)(" << RA
                              << ") - _gicc_base;\n";
    indent(out, lvl + 1); out << "rt.put_no_db(_gicc_src, peer, (int)(" << RK
                              << "), (size_t)(" << S << "), _gicc_soff, _gicc_doff);\n";
    indent(out, lvl);     out << "}\n";
}

// Forward decl — emit_stmt and emit_compound recurse mutually.
void emit_stmt(std::ostream& out, int lvl, const clang::Stmt* s,
               const HKAnalysis& hk, const clang::SourceManager& sm,
               const clang::LangOptions& lo);

void emit_compound(std::ostream& out, int lvl, const clang::CompoundStmt* cs,
                   const HKAnalysis& hk, const clang::SourceManager& sm,
                   const clang::LangOptions& lo) {
    if (!cs) return;
    for (const auto* child : cs->body()) {
        emit_stmt(out, lvl, child, hk, sm, lo);
    }
}

// Emit a host-side mirror of one kernel-body Stmt. Statements that
// don't affect the schedule (declarations of loop variables aside, plus
// gicc::flush / gicc::quiet, plus everything we don't recognize) are
// dropped silently.
void emit_stmt(std::ostream& out, int lvl, const clang::Stmt* s,
               const HKAnalysis& hk, const clang::SourceManager& sm,
               const clang::LangOptions& lo) {
    if (!s) return;

    // CompoundStmt → recurse with same indent level (caller already
    // emitted the opening '{').
    if (auto* cs = llvm::dyn_cast<clang::CompoundStmt>(s)) {
        emit_compound(out, lvl, cs, hk, sm, lo);
        return;
    }

    // ForStmt with HK cond → emit an equivalent host for-loop. The
    // init / cond / inc texts are extracted verbatim from source. If the
    // cond is NOT HK we conservatively drop the loop (any put inside
    // would have failed E5 in the validator anyway).
    if (auto* fs = llvm::dyn_cast<clang::ForStmt>(s)) {
        if (!hk.isHK(fs->getCond())) return;
        std::string init = strip_trailing_semicolon(
            source_text(fs->getInit(), sm, lo));
        std::string cond = source_text_expr(fs->getCond(), sm, lo);
        std::string inc  = source_text_expr(fs->getInc(),  sm, lo);
        indent(out, lvl); out << "for (" << init << "; " << cond << "; "
                              << inc << ") {\n";
        emit_stmt(out, lvl + 1, fs->getBody(), hk, sm, lo);
        indent(out, lvl); out << "}\n";
        return;
    }

    if (auto* ws = llvm::dyn_cast<clang::WhileStmt>(s)) {
        if (!hk.isHK(ws->getCond())) return;
        std::string cond = source_text_expr(ws->getCond(), sm, lo);
        indent(out, lvl); out << "while (" << cond << ") {\n";
        emit_stmt(out, lvl + 1, ws->getBody(), hk, sm, lo);
        indent(out, lvl); out << "}\n";
        return;
    }

    if (auto* ds = llvm::dyn_cast<clang::DoStmt>(s)) {
        if (!hk.isHK(ds->getCond())) return;
        std::string cond = source_text_expr(ds->getCond(), sm, lo);
        indent(out, lvl); out << "do {\n";
        emit_stmt(out, lvl + 1, ds->getBody(), hk, sm, lo);
        indent(out, lvl); out << "} while (" << cond << ");\n";
        return;
    }

    if (auto* is = llvm::dyn_cast<clang::IfStmt>(s)) {
        if (!hk.isHK(is->getCond())) return;
        std::string cond = source_text_expr(is->getCond(), sm, lo);
        indent(out, lvl); out << "if (" << cond << ") {\n";
        emit_stmt(out, lvl + 1, is->getThen(), hk, sm, lo);
        if (is->getElse()) {
            indent(out, lvl); out << "} else {\n";
            emit_stmt(out, lvl + 1, is->getElse(), hk, sm, lo);
        }
        indent(out, lvl); out << "}\n";
        return;
    }

    // gicc::* CallExpr — lift put_no_db, ignore flush / quiet.
    if (auto* ce = llvm::dyn_cast<clang::CallExpr>(s)) {
        auto* fd = ce->getDirectCallee();
        if (!fd) return;
        const std::string qn = fd->getQualifiedNameAsString();
        if (qn == "gicc::put_no_db") {
            if (ce->getNumArgs() < 6) return;
            emit_put_block(out, lvl, ce, sm, lo);
        }
        // gicc::flush, gicc::quiet, gicc::get_no_db (handled in F5),
        // and anything else: silently skip.
        return;
    }

    // Any other statement (DeclStmt for non-loop locals, raw expression
    // statements, returns, etc.) is irrelevant to the host-side trace.
    // Don't emit anything.
}

} // namespace

std::string TraceEmitter::emit(const KernelInfo& ki, const HKAnalysis& hk) {
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

    // F3: walk the kernel body recursively. The walker emits HK
    // for/while/do/if scaffolds verbatim and lifts any nested
    // put_no_db inside.
    if (auto* body = ki.decl->getBody()) {
        if (auto* cs = llvm::dyn_cast<clang::CompoundStmt>(body)) {
            emit_compound(out, 2, cs, hk, sm, lo);
        } else {
            emit_stmt(out, 2, body, hk, sm, lo);
        }
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
