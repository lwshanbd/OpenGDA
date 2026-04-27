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
 * F4: if a put_no_db's arg subtree references CUDA/HIP grid builtins
 *     (blockIdx.{x,y,z}, threadIdx.{x,y,z}), wrap the lifted call in
 *     synthetic host-side loops — one per referenced dim, nested z>y>x,
 *     blocks outside threads — and textually substitute the builtin
 *     references in the emitted source. blockDim.* / gridDim.* are
 *     rewritten to the host-side `block.*` / `grid.*` dim3 fields.
 *
 * F5 (this commit): lift gicc::get_no_db the same way as put_no_db.
 *     Identical lookup machinery (buffer_by_lkey + peer_buffer_base);
 *     emits rt.get_no_db(local_dst, peer, dst_buf_idx, size, local_off,
 *     remote_off). local_addr/lkey is the read DESTINATION, not source —
 *     reflected in the emitted variable names (_gicc_dst / _gicc_loff /
 *     _gicc_roff) for readability.
 */
#include "TraceEmitter.h"

#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <utime.h>

#include <array>
#include <fstream>
#include <vector>

namespace gicc_plugin {

namespace {

// Compute the sidecar path for a given main-file path. The directory is
// configurable via -fplugin-arg-gicc-sidecar-dir=<path>; default `/tmp`
// keeps lit tests (which use read-only source trees) working without
// extra plumbing. CMake's gicc_target_plugin() macro sets this to a
// per-target subdirectory under the build tree.
//
// Note: the file extension is `.gicc.cpp` even though it's intended to
// be `-include`d as a header by the main TU. Naming as .cpp keeps
// existing lit-test FileChecks happy and makes it obvious the contents
// are full C++ (not just declarations).
std::string sidecar_path_for(const std::string& main_file,
                             const std::string& sidecar_dir) {
    std::string base = llvm::sys::path::stem(main_file).str();
    std::string dir = sidecar_dir.empty() ? std::string("/tmp") : sidecar_dir;
    // Trim trailing '/' so we don't emit double-slashes.
    while (!dir.empty() && dir.back() == '/') dir.pop_back();
    return dir + "/" + base + ".gicc.cpp";
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

// One of the two RMA verbs we lift. Picks the rt-method name and the
// emitted local variable names so the sidecar reads naturally — for a
// PUT the local buffer is the SOURCE, for a GET it is the DESTINATION.
enum class RmaOp { Put, Get };

struct OpNames {
    const char* rt_method;
    const char* local_var;   // _gicc_src for put, _gicc_dst for get
    const char* local_off;   // _gicc_soff for put, _gicc_loff for get
    const char* remote_off;  // _gicc_doff for put, _gicc_roff for get
};

OpNames names_for(RmaOp op) {
    if (op == RmaOp::Put) {
        return {"put_no_db", "_gicc_src", "_gicc_soff", "_gicc_doff"};
    }
    return {"get_no_db", "_gicc_dst", "_gicc_loff", "_gicc_roff"};
}

// Emit the rt.put_no_db / rt.get_no_db call body itself (no synthetic
// loop wrapping). Each emitted sub-expression has builtin substitution
// applied so references to blockIdx.x etc. become _gicc_bx etc.
//
// New (v1.5) put/get signature with PER-CALL peer + dst_buf:
//   ce->getArg(0)  ctx
//   ce->getArg(1)  peer       <-- per-call destination rank
//   ce->getArg(2)  dst_buf    <-- per-call destination buffer index
//   ce->getArg(3)  local_addr (src for put, dst for get)
//   ce->getArg(4)  local_lkey
//   ce->getArg(5)  remote_addr
//   ce->getArg(6)  remote_rkey  (ignored by host trace; consumed only by MLX5)
//   ce->getArg(7)  size
//   ce->getArg(8)  signaled (optional)
//
// peer + dst_buf come from the put_no_db call's own args (HK kernel
// params), NOT from the kernel_trace::run signature, so a single kernel
// can issue puts to multiple peers (e.g. jacobi halo: top + bottom).
void emit_rma_call(std::ostream& out, int lvl, RmaOp op,
                   const clang::CallExpr* ce,
                   const clang::SourceManager& sm,
                   const clang::LangOptions& lo) {
    const OpNames n = names_for(op);
    const std::string PEER = substitute_builtins(source_text_expr(ce->getArg(1), sm, lo));
    const std::string DBUF = substitute_builtins(source_text_expr(ce->getArg(2), sm, lo));
    const std::string A    = substitute_builtins(source_text_expr(ce->getArg(3), sm, lo));
    const std::string LK   = substitute_builtins(source_text_expr(ce->getArg(4), sm, lo));
    const std::string RA   = substitute_builtins(source_text_expr(ce->getArg(5), sm, lo));
    const std::string S    = substitute_builtins(source_text_expr(ce->getArg(7), sm, lo));

    indent(out, lvl);     out << "{\n";
    indent(out, lvl + 1); out << "int _gicc_peer = (int)(" << PEER << ");\n";
    indent(out, lvl + 1); out << "int _gicc_dbuf = (int)(" << DBUF << ");\n";
    indent(out, lvl + 1); out << "auto& " << n.local_var
                              << " = rt.buffer_by_lkey((uint32_t)("
                              << LK << "));\n";
    indent(out, lvl + 1); out << "size_t " << n.local_off << " = (uint64_t)(" << A
                              << ") - (uint64_t)" << n.local_var << ".addr;\n";
    indent(out, lvl + 1); out << "uint64_t _gicc_base = rt.peer_buffer_base(_gicc_peer, _gicc_dbuf);\n";
    indent(out, lvl + 1); out << "size_t " << n.remote_off << " = (uint64_t)(" << RA
                              << ") - _gicc_base;\n";
    indent(out, lvl + 1); out << "rt." << n.rt_method << "("
                              << n.local_var << ", _gicc_peer, _gicc_dbuf, (size_t)("
                              << S << "), "
                              << n.local_off << ", " << n.remote_off << ");\n";
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

// Top-level lift for one put_no_db / get_no_db call. The scaffolding
// — transitive HK-local + builtin scan, synthetic loop nest, local
// re-emission — is identical for put and get because their arg shape is
// the same. emit_rma_call picks the right rt method and variable names
// based on the op tag.
//
//  1. Scan args for grid builtin dims; transitively follow HK local
//     refs into their init exprs to pick up dims indirectly via
//     "int i = blockIdx.x;" patterns.
//  2. Emit one synthetic for-loop per referenced dim (z>y>x order,
//     blocks outside threads).
//  3. Inside the loop nest, re-emit each transitively-referenced HK
//     local with builtin substitution applied.
//  4. Emit the put / get call itself (also with substitution).
void emit_rma_block(std::ostream& out, int lvl, RmaOp op,
                    const clang::CallExpr* ce,
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

    // (4) The put / get itself (substitution applied inside).
    emit_rma_call(out, loop_lvl + 1, op, ce, sm, lo);

    indent(out, loop_lvl);     out << "}\n";

    // Close synthetic loops in reverse order.
    for (Dim d : kDimOrder) {
        if (!dims.count(static_cast<int>(d))) continue;
        loop_lvl--;
        indent(out, loop_lvl); out << "}\n";
    }
}

// Detect whether a Stmt subtree contains ANY gicc::put_no_db /
// gicc::get_no_db call. Used to short-circuit emission of control flow
// constructs (if / for / while / do) whose bodies have no liftable op —
// otherwise we'd emit `if (cond) { for (...) {} }` referencing device-only
// locals (e.g. `int k = blockIdx.x * blockDim.x + threadIdx.x;`) that
// have no host-side definition, which fails to compile.
class HasLiftableScanner
    : public clang::RecursiveASTVisitor<HasLiftableScanner> {
public:
    bool found = false;
    bool VisitCallExpr(clang::CallExpr* ce) {
        auto* fd = ce->getDirectCallee();
        if (!fd) return true;
        const std::string qn = fd->getQualifiedNameAsString();
        if (qn == "gicc::put_no_db" || qn == "gicc::get_no_db") {
            found = true;
            return false; // stop walking
        }
        return true;
    }
};

bool stmt_contains_liftable(const clang::Stmt* s) {
    if (!s) return false;
    HasLiftableScanner sc;
    sc.TraverseStmt(const_cast<clang::Stmt*>(s));
    return sc.found;
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
    // would have failed E5 in the validator anyway). We also drop the
    // entire construct if its body contains no liftable op — emitting
    // an empty host loop that references kernel-local iterators would
    // not compile (e.g. `for (int k = blockIdx.x * blockDim.x + ...)`
    // refers to device-only builtins).
    if (auto* fs = llvm::dyn_cast<clang::ForStmt>(s)) {
        if (!hk.isHK(fs->getCond())) return;
        if (!stmt_contains_liftable(fs->getBody())) return;
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
        if (!stmt_contains_liftable(ws->getBody())) return;
        std::string cond = source_text_expr(ws->getCond(), sm, lo);
        indent(out, lvl); out << "while (" << cond << ") {\n";
        emit_stmt(out, lvl + 1, ws->getBody(), hk, kernel, loop_iters, sm, lo);
        indent(out, lvl); out << "}\n";
        return;
    }

    if (auto* ds = llvm::dyn_cast<clang::DoStmt>(s)) {
        if (!hk.isHK(ds->getCond())) return;
        if (!stmt_contains_liftable(ds->getBody())) return;
        std::string cond = source_text_expr(ds->getCond(), sm, lo);
        indent(out, lvl); out << "do {\n";
        emit_stmt(out, lvl + 1, ds->getBody(), hk, kernel, loop_iters, sm, lo);
        indent(out, lvl); out << "} while (" << cond << ");\n";
        return;
    }

    if (auto* is = llvm::dyn_cast<clang::IfStmt>(s)) {
        if (!hk.isHK(is->getCond())) return;
        const bool then_has = stmt_contains_liftable(is->getThen());
        const bool else_has = stmt_contains_liftable(is->getElse());
        if (!then_has && !else_has) return;
        std::string cond = source_text_expr(is->getCond(), sm, lo);
        // If the condition references __device__-only builtins (a
        // common idiom: `if (threadIdx.x == 0) put_no_db(...)`), drop
        // the if and just recurse into its body — the synthetic loop
        // emitted by emit_rma_block already enumerates the per-thread
        // dim space for any RDMA ops it lifts. Keeping the cond would
        // require referencing _gicc_tx / _gicc_bx outside the loop
        // scope, which is harder than it's worth in v1.
        if (cond.find("threadIdx.") != std::string::npos ||
            cond.find("blockIdx.")  != std::string::npos) {
            if (then_has)
                emit_stmt(out, lvl, is->getThen(), hk, kernel, loop_iters, sm, lo);
            if (else_has)
                emit_stmt(out, lvl, is->getElse(), hk, kernel, loop_iters, sm, lo);
            return;
        }
        indent(out, lvl); out << "if (" << cond << ") {\n";
        if (then_has)
            emit_stmt(out, lvl + 1, is->getThen(), hk, kernel, loop_iters, sm, lo);
        if (else_has) {
            indent(out, lvl); out << "} else {\n";
            emit_stmt(out, lvl + 1, is->getElse(), hk, kernel, loop_iters, sm, lo);
        }
        indent(out, lvl); out << "}\n";
        return;
    }

    // gicc::* CallExpr — lift put_no_db / get_no_db, ignore flush / quiet.
    if (auto* ce = llvm::dyn_cast<clang::CallExpr>(s)) {
        auto* fd = ce->getDirectCallee();
        if (!fd) return;
        const std::string qn = fd->getQualifiedNameAsString();
        if (qn == "gicc::put_no_db") {
            if (ce->getNumArgs() < 6) return;
            emit_rma_block(out, lvl, RmaOp::Put, ce, hk, kernel,
                           loop_iters, sm, lo);
        } else if (qn == "gicc::get_no_db") {
            if (ce->getNumArgs() < 6) return;
            emit_rma_block(out, lvl, RmaOp::Get, ce, hk, kernel,
                           loop_iters, sm, lo);
        }
        // gicc::flush, gicc::quiet, and anything else: silently skip.
        return;
    }

    // Any other statement (DeclStmt for non-loop locals, raw expression
    // statements, returns, etc.) is irrelevant to the host-side trace.
    // Don't emit anything.
}

} // namespace

std::ostringstream& TraceEmitter::buffer_for(const std::string& sidecar,
                                             const std::string& main_file) {
    auto it = buffers_.find(sidecar);
    if (it == buffers_.end()) {
        // First touch this TU — emit the standard header.
        auto& buf = buffers_[sidecar];
        buf << "// AUTO-GENERATED by gicc-clang-plugin. DO NOT EDIT.\n";
        buf << "// Trace specializations for kernels in " << main_file << "\n\n";
        buf << "#include <hip/hip_runtime.h>\n";
        buf << "#include \"gicc/gicc.hpp\"\n\n";
        return buf;
    }
    return it->second;
}

std::string TraceEmitter::ensure_sidecar(const std::string& main_file) {
    std::string sidecar = sidecar_path_for(main_file, sidecar_dir_);
    (void)buffer_for(sidecar, main_file); // header-only buffer
    return sidecar;
}

void TraceEmitter::flush() {
    for (auto& kv : buffers_) {
        const std::string& sidecar = kv.first;
        const std::string  content = kv.second.str();
        // mtime-preservation: only rewrite if the new content differs
        // from what's on disk. The CMake helper uses the sidecar's
        // mtime to decide whether the main TU needs to be recompiled,
        // so spuriously bumping the mtime each build would create a
        // never-finishing rebuild loop.
        std::ifstream in(sidecar);
        if (in) {
            std::ostringstream cur;
            cur << in.rdbuf();
            if (cur.str() == content) {
                continue; // unchanged → leave mtime alone
            }
        }
        std::ofstream out(sidecar, std::ios::trunc);
        if (!out) {
            llvm::errs() << "[gicc-plugin] FAIL to open sidecar "
                         << sidecar << "\n";
            continue;
        }
        out << content;
        out.close();
        // Bump mtime forward so the about-to-be-emitted .o file isn't
        // accidentally newer than the sidecar — which would let make
        // skip the next rebuild and never pick up the fresh trace. The
        // main-TU compile finishes seconds AFTER we get here (this code
        // runs from inside HandleTranslationUnit, before object code
        // generation completes). +5 seconds is safely greater than any
        // realistic post-plugin codegen time on supported toolchains.
        struct stat st;
        if (stat(sidecar.c_str(), &st) == 0) {
            struct utimbuf ut;
            ut.actime  = st.st_atime;
            ut.modtime = st.st_mtime + 5;
            (void)utime(sidecar.c_str(), &ut);
        }
    }
}

std::string TraceEmitter::emit(const KernelInfo& ki, const HKAnalysis& hk) {
    auto& sm = CI_.getSourceManager();
    const auto& lo = CI_.getLangOpts();
    auto file_id = sm.getMainFileID();
    auto file_ref = sm.getFileEntryRefForID(file_id);
    std::string main_file = file_ref
        ? std::string(file_ref->getName())
        : std::string("unknown");
    std::string sidecar = sidecar_path_for(main_file, sidecar_dir_);

    std::ostringstream& out = buffer_for(sidecar, main_file);

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
    out << "    static void run(gicc::Runtime& rt, "
           "dim3 grid, dim3 block";
    if (!params.empty()) out << ", " << params;
    out << ") {\n";
    out << "        (void)rt; (void)grid; (void)block;\n";
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
