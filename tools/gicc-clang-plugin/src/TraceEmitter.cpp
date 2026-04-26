/**
 * TraceEmitter.cpp — see TraceEmitter.h for design intent.
 *
 * F1: build a kernel_trace<&K> specialization with an empty body and
 *     write it to /tmp/<basename>.gicc.cpp.
 *
 * F2: lift each straight-line gicc::put_no_db into a host-side block
 *     calling rt.put_no_db, with arg source text mirrored verbatim.
 *
 * F3: walk the kernel body recursively. For ForStmt / WhileStmt /
 *     DoStmt / IfStmt with HK condition, emit the matching host-side
 *     control flow scaffold and recurse on the body.
 *
 * F4 (this commit): if a put_no_db's arg subtree references CUDA/HIP
 *     grid builtins (blockIdx.{x,y,z}, threadIdx.{x,y,z}), wrap the
 *     lifted call in synthetic host-side loops — one per referenced
 *     dim, nested z>y>x, blocks outside threads — and textually
 *     substitute the builtin references in the emitted source. The
 *     scan also follows DeclRefExprs to HK local variables so a
 *     "int i = blockIdx.x; ... put(..., a+i, ...)" pattern picks up
 *     the dim through `i`'s initializer; the local decl is then
 *     re-emitted inside the synthetic loops with the same textual
 *     substitution applied. blockDim.* / gridDim.* are rewritten to
 *     the host-side `block.*` / `grid.*` dim3 fields. This is v1 —
 *     the unroll-small-grid heuristic from spec §5.3 is deferred.
 */
#include "TraceEmitter.h"

#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <fstream>
#include <vector>

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

// ---------- Grid-builtin handling (F4) ----------

// One enum per builtin sub-axis we expand into a synthetic loop or
// substitute textually. Block/Thread dims drive synthetic loops;
// blockDim/gridDim are constants (= block.* / grid.*).
enum class Dim {
    BX, BY, BZ,   // blockIdx.{x,y,z} — synthetic for-loops
    TX, TY, TZ,   // threadIdx.{x,y,z}
};

const char* dim_var_name(Dim d) {
    switch (d) {
        case Dim::BX: return "_gicc_bx";
        case Dim::BY: return "_gicc_by";
        case Dim::BZ: return "_gicc_bz";
        case Dim::TX: return "_gicc_tx";
        case Dim::TY: return "_gicc_ty";
        case Dim::TZ: return "_gicc_tz";
    }
    return "?";
}

const char* dim_bound_expr(Dim d) {
    switch (d) {
        case Dim::BX: return "grid.x";
        case Dim::BY: return "grid.y";
        case Dim::BZ: return "grid.z";
        case Dim::TX: return "block.x";
        case Dim::TY: return "block.y";
        case Dim::TZ: return "block.z";
    }
    return "?";
}

// Map source-text → host-side replacement. blockIdx.* gets the loop
// var; threadIdx.* gets the loop var; blockDim.* / gridDim.* fold to
// the launch-time dim3 fields. v1 substitution is purely textual —
// see emit_with_substitution below.
struct SubRule { const char* needle; const char* repl; };
const std::array<SubRule, 12> subs = {{
    {"blockIdx.x",  "_gicc_bx"},
    {"blockIdx.y",  "_gicc_by"},
    {"blockIdx.z",  "_gicc_bz"},
    {"threadIdx.x", "_gicc_tx"},
    {"threadIdx.y", "_gicc_ty"},
    {"threadIdx.z", "_gicc_tz"},
    {"blockDim.x",  "block.x"},
    {"blockDim.y",  "block.y"},
    {"blockDim.z",  "block.z"},
    {"gridDim.x",   "grid.x"},
    {"gridDim.y",   "grid.y"},
    {"gridDim.z",   "grid.z"},
}};

// Apply textual builtin substitution to a source-text fragment. Naive
// string replace is good enough for v1 idioms (builtin written
// directly in put args or in a HK local's init); a user who has a
// variable literally called "blockIdx" will collide, but that is
// vanishingly rare and would already shadow the CUDA builtin.
std::string substitute_builtins(std::string text) {
    for (const auto& r : subs) {
        const std::string needle = r.needle;
        const std::string repl = r.repl;
        size_t pos = 0;
        while ((pos = text.find(needle, pos)) != std::string::npos) {
            text.replace(pos, needle.size(), repl);
            pos += repl.size();
        }
    }
    return text;
}

// AST-side detection of grid builtins inside an Expr subtree. Mirrors
// HKAnalysis's recognition of the HIP PseudoObjectExpr/MSPropertyRefExpr
// shape and the CUDA MemberExpr-on-DeclRef shape. Records which Dim
// values (BX..TZ) were seen so we know which synthetic loops to emit.
class BuiltinScanner : public clang::RecursiveASTVisitor<BuiltinScanner> {
public:
    explicit BuiltinScanner(llvm::DenseSet<int>& dims) : dims_(dims) {}

    bool VisitPseudoObjectExpr(clang::PseudoObjectExpr* poe) {
        const clang::Expr* syn = poe->getSyntacticForm();
        if (!syn) return true;
        auto* mpr = llvm::dyn_cast<clang::MSPropertyRefExpr>(
            syn->IgnoreParenImpCasts());
        if (!mpr) return true;
        const clang::Expr* base = mpr->getBaseExpr()->IgnoreParenImpCasts();
        std::string ty = base->getType().getUnqualifiedType().getAsString();
        std::string field = mpr->getPropertyDecl()->getNameAsString();
        recordIfBuiltin(ty, field);
        return true;
    }

    bool VisitMemberExpr(clang::MemberExpr* me) {
        const clang::Expr* base = me->getBase()->IgnoreParenImpCasts();
        std::string field = me->getMemberDecl()->getNameAsString();
        // CUDA: base is DeclRefExpr to "threadIdx" / "blockIdx" / etc.
        if (auto* dre = llvm::dyn_cast<clang::DeclRefExpr>(base)) {
            recordIfBuiltin(dre->getDecl()->getNameAsString(), field);
        }
        // HIP fallback: base type is __hip_builtin_*.
        std::string ty = base->getType().getUnqualifiedType().getAsString();
        recordIfBuiltin(ty, field);
        return true;
    }

private:
    void recordIfBuiltin(llvm::StringRef name_or_type, llvm::StringRef field) {
        const bool is_block = name_or_type.contains("blockIdx");
        const bool is_thread = name_or_type.contains("threadIdx");
        if (!is_block && !is_thread) return;
        if (field == "x") dims_.insert(static_cast<int>(is_block ? Dim::BX : Dim::TX));
        else if (field == "y") dims_.insert(static_cast<int>(is_block ? Dim::BY : Dim::TY));
        else if (field == "z") dims_.insert(static_cast<int>(is_block ? Dim::BZ : Dim::TZ));
    }

    llvm::DenseSet<int>& dims_;
};

// Walk an Expr subtree and collect every HK local VarDecl referenced
// via DeclRefExpr. We use this to follow "int i = blockIdx.x" through
// `i`'s use in put args back to the builtin, then recursively repeat
// on `i`'s init expression. Only HK locals are followed (per HKAnalysis);
// kernel params are HK but their init is not visible (they come from the
// launch site), so we stop the chain at them.
//
// Loop iterators declared in a ForStmt's init are excluded via the
// loop_iters set — they're already in scope from the host-side for-loop
// scaffold (F3) and re-emitting them would shadow the iterator and
// freeze it at its init value.
class LocalRefCollector : public clang::RecursiveASTVisitor<LocalRefCollector> {
public:
    LocalRefCollector(const HKAnalysis& hk,
                      const clang::FunctionDecl* kernel,
                      const llvm::DenseSet<const clang::VarDecl*>& loop_iters,
                      llvm::DenseSet<const clang::VarDecl*>& seen,
                      std::vector<const clang::VarDecl*>& ordered)
        : hk_(hk), kernel_(kernel), loop_iters_(loop_iters),
          seen_(seen), ordered_(ordered) {}

    bool VisitDeclRefExpr(clang::DeclRefExpr* dre) {
        auto* vd = llvm::dyn_cast<clang::VarDecl>(dre->getDecl());
        if (!vd) return true;
        // Skip kernel params: they aren't local decls and have no
        // body-side init we can re-emit.
        if (llvm::isa<clang::ParmVarDecl>(vd)) return true;
        // Skip ForStmt init iterators: already in scope from F3's
        // host-side for-loop scaffold.
        if (loop_iters_.count(vd)) return true;
        // Only follow HK locals (tracked via the analysis).
        if (!hk_.isHK(dre)) return true;
        // Must be a local of the kernel we're currently emitting.
        if (vd->getDeclContext() != kernel_) return true;
        if (seen_.insert(vd).second) ordered_.push_back(vd);
        return true;
    }

private:
    const HKAnalysis& hk_;
    const clang::FunctionDecl* kernel_;
    const llvm::DenseSet<const clang::VarDecl*>& loop_iters_;
    llvm::DenseSet<const clang::VarDecl*>& seen_;
    std::vector<const clang::VarDecl*>& ordered_;
};

// Pre-pass: collect every VarDecl that appears in a ForStmt's init
// (as the DeclStmt's decls). We do this once per kernel body and pass
// the resulting set into emit_put_block so its LocalRefCollector can
// skip those decls instead of re-emitting them and shadowing the
// iterator.
class LoopIterCollector : public clang::RecursiveASTVisitor<LoopIterCollector> {
public:
    explicit LoopIterCollector(llvm::DenseSet<const clang::VarDecl*>& set)
        : set_(set) {}
    bool VisitForStmt(clang::ForStmt* fs) {
        if (auto* ds = llvm::dyn_cast_or_null<clang::DeclStmt>(fs->getInit())) {
            for (auto* d : ds->decls()) {
                if (auto* vd = llvm::dyn_cast<clang::VarDecl>(d)) {
                    set_.insert(vd);
                }
            }
        }
        return true;
    }
private:
    llvm::DenseSet<const clang::VarDecl*>& set_;
};

// Stable iteration order for synthetic loops: outermost = z, then y,
// then x; blocks outside threads. Doesn't matter for correctness
// (all loops are independent) — pick this for predictability.
const Dim kDimOrder[] = {
    Dim::BZ, Dim::BY, Dim::BX, Dim::TZ, Dim::TY, Dim::TX
};

// Emit the put_no_db call body itself (no synthetic-loop wrapping).
// Each emitted sub-expression has builtin substitution applied so
// references to blockIdx.x etc. become _gicc_bx etc.
void emit_put_call(std::ostream& out, int lvl, const clang::CallExpr* ce,
                   const clang::SourceManager& sm,
                   const clang::LangOptions& lo) {
    const std::string A  = substitute_builtins(source_text_expr(ce->getArg(1), sm, lo));
    const std::string LK = substitute_builtins(source_text_expr(ce->getArg(2), sm, lo));
    const std::string RA = substitute_builtins(source_text_expr(ce->getArg(3), sm, lo));
    const std::string RK = substitute_builtins(source_text_expr(ce->getArg(4), sm, lo));
    const std::string S  = substitute_builtins(source_text_expr(ce->getArg(5), sm, lo));

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

// Render a HK local VarDecl as a host-side declaration with builtin
// substitution applied to its initializer. We deliberately don't reuse
// the user's full DeclStmt source text because that would include the
// trailing ';' and any sibling decls; instead synthesize the type via
// the printing policy and re-extract just the init expression text.
std::string render_hk_local(const clang::VarDecl* vd,
                            const clang::SourceManager& sm,
                            const clang::LangOptions& lo) {
    clang::PrintingPolicy pp(vd->getASTContext().getLangOpts());
    pp.SuppressTagKeyword = true;
    pp.Bool = true;
    std::string out = vd->getType().getAsString(pp) + " " +
                      vd->getNameAsString();
    if (vd->hasInit()) {
        out += " = " +
               substitute_builtins(source_text_expr(vd->getInit(), sm, lo));
    }
    out += ";";
    return out;
}

// Top-level lift for one put_no_db call:
//  1. Scan put args for grid builtin dims; transitively follow HK local
//     refs into their init exprs to pick up dims indirectly via
//     "int i = blockIdx.x;" patterns.
//  2. Emit one synthetic for-loop per referenced dim (z>y>x order,
//     blocks outside threads).
//  3. Inside the loop nest, re-emit each transitively-referenced HK
//     local with builtin substitution applied.
//  4. Emit the put_no_db call itself (also with substitution).
void emit_put_block(std::ostream& out, int lvl, const clang::CallExpr* ce,
                    const HKAnalysis& hk,
                    const clang::FunctionDecl* kernel,
                    const llvm::DenseSet<const clang::VarDecl*>& loop_iters,
                    const clang::SourceManager& sm,
                    const clang::LangOptions& lo) {
    // (1) Transitive scan of put args + HK local inits.
    llvm::DenseSet<int> dims;
    BuiltinScanner bs(dims);
    llvm::DenseSet<const clang::VarDecl*> seen_locals;
    std::vector<const clang::VarDecl*> ordered_locals;

    for (unsigned i = 1; i < ce->getNumArgs(); i++) {
        bs.TraverseStmt(const_cast<clang::Expr*>(ce->getArg(i)));
        LocalRefCollector lc(hk, kernel, loop_iters, seen_locals, ordered_locals);
        lc.TraverseStmt(const_cast<clang::Expr*>(ce->getArg(i)));
    }
    // Iterate (rather than range-for) because we may extend
    // ordered_locals while walking it.
    for (size_t i = 0; i < ordered_locals.size(); i++) {
        const clang::VarDecl* vd = ordered_locals[i];
        if (!vd->hasInit()) continue;
        bs.TraverseStmt(const_cast<clang::Expr*>(vd->getInit()));
        LocalRefCollector lc(hk, kernel, loop_iters, seen_locals, ordered_locals);
        lc.TraverseStmt(const_cast<clang::Expr*>(vd->getInit()));
    }

    // (2) Emit synthetic loops for every dim referenced.
    int loop_lvl = lvl;
    for (Dim d : kDimOrder) {
        if (!dims.count(static_cast<int>(d))) continue;
        indent(out, loop_lvl);
        out << "for (uint32_t " << dim_var_name(d) << " = 0; "
            << dim_var_name(d) << " < " << dim_bound_expr(d) << "; "
            << dim_var_name(d) << "++) {\n";
        loop_lvl++;
    }

    // (3) Re-emit transitively-referenced HK locals so the put can
    // reference them. We need a fresh scope around them + the put
    // (otherwise re-emitting the same put twice would redeclare them).
    indent(out, loop_lvl);     out << "{\n";
    for (const clang::VarDecl* vd : ordered_locals) {
        indent(out, loop_lvl + 1);
        out << render_hk_local(vd, sm, lo) << "\n";
    }

    // (4) The put itself (substitution applied inside).
    emit_put_call(out, loop_lvl + 1, ce, sm, lo);

    indent(out, loop_lvl);     out << "}\n";

    // Close synthetic loops in reverse order.
    for (Dim d : kDimOrder) {
        if (!dims.count(static_cast<int>(d))) continue;
        loop_lvl--;
        indent(out, loop_lvl); out << "}\n";
    }
}

// Forward decl — emit_stmt and emit_compound recurse mutually.
void emit_stmt(std::ostream& out, int lvl, const clang::Stmt* s,
               const HKAnalysis& hk, const clang::FunctionDecl* kernel,
               const llvm::DenseSet<const clang::VarDecl*>& loop_iters,
               const clang::SourceManager& sm,
               const clang::LangOptions& lo);

void emit_compound(std::ostream& out, int lvl, const clang::CompoundStmt* cs,
                   const HKAnalysis& hk, const clang::FunctionDecl* kernel,
                   const llvm::DenseSet<const clang::VarDecl*>& loop_iters,
                   const clang::SourceManager& sm,
                   const clang::LangOptions& lo) {
    if (!cs) return;
    for (const auto* child : cs->body()) {
        emit_stmt(out, lvl, child, hk, kernel, loop_iters, sm, lo);
    }
}

// Emit a host-side mirror of one kernel-body Stmt. Statements that
// don't affect the schedule (declarations of loop variables aside, plus
// gicc::flush / gicc::quiet, plus everything we don't recognize) are
// dropped silently.
void emit_stmt(std::ostream& out, int lvl, const clang::Stmt* s,
               const HKAnalysis& hk, const clang::FunctionDecl* kernel,
               const llvm::DenseSet<const clang::VarDecl*>& loop_iters,
               const clang::SourceManager& sm,
               const clang::LangOptions& lo) {
    if (!s) return;

    // CompoundStmt → recurse with same indent level (caller already
    // emitted the opening '{').
    if (auto* cs = llvm::dyn_cast<clang::CompoundStmt>(s)) {
        emit_compound(out, lvl, cs, hk, kernel, loop_iters, sm, lo);
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
        emit_stmt(out, lvl + 1, fs->getBody(), hk, kernel, loop_iters, sm, lo);
        indent(out, lvl); out << "}\n";
        return;
    }

    if (auto* ws = llvm::dyn_cast<clang::WhileStmt>(s)) {
        if (!hk.isHK(ws->getCond())) return;
        std::string cond = source_text_expr(ws->getCond(), sm, lo);
        indent(out, lvl); out << "while (" << cond << ") {\n";
        emit_stmt(out, lvl + 1, ws->getBody(), hk, kernel, loop_iters, sm, lo);
        indent(out, lvl); out << "}\n";
        return;
    }

    if (auto* ds = llvm::dyn_cast<clang::DoStmt>(s)) {
        if (!hk.isHK(ds->getCond())) return;
        std::string cond = source_text_expr(ds->getCond(), sm, lo);
        indent(out, lvl); out << "do {\n";
        emit_stmt(out, lvl + 1, ds->getBody(), hk, kernel, loop_iters, sm, lo);
        indent(out, lvl); out << "} while (" << cond << ");\n";
        return;
    }

    if (auto* is = llvm::dyn_cast<clang::IfStmt>(s)) {
        if (!hk.isHK(is->getCond())) return;
        std::string cond = source_text_expr(is->getCond(), sm, lo);
        indent(out, lvl); out << "if (" << cond << ") {\n";
        emit_stmt(out, lvl + 1, is->getThen(), hk, kernel, loop_iters, sm, lo);
        if (is->getElse()) {
            indent(out, lvl); out << "} else {\n";
            emit_stmt(out, lvl + 1, is->getElse(), hk, kernel, loop_iters, sm, lo);
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
            emit_put_block(out, lvl, ce, hk, kernel, loop_iters, sm, lo);
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

    // F3+F4: walk the kernel body recursively. The walker emits HK
    // for/while/do/if scaffolds verbatim and lifts any nested
    // put_no_db inside, with synthetic loops for grid builtins.
    //
    // Pre-pass: collect for-loop iterators so the put-block's local
    // re-emission doesn't shadow them (see LoopIterCollector).
    llvm::DenseSet<const clang::VarDecl*> loop_iters;
    if (auto* body = ki.decl->getBody()) {
        LoopIterCollector lic(loop_iters);
        lic.TraverseStmt(body);
        if (auto* cs = llvm::dyn_cast<clang::CompoundStmt>(body)) {
            emit_compound(out, 2, cs, hk, ki.decl, loop_iters, sm, lo);
        } else {
            emit_stmt(out, 2, body, hk, ki.decl, loop_iters, sm, lo);
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
