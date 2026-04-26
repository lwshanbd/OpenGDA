/**
 * Validator.cpp — see Validator.h for design intent.
 *
 * E3 (this commit): every put_no_db / get_no_db call has all non-ctx,
 * non-signaled arguments host-known. Argument names by index:
 *   0 ctx          (skip)
 *   1 local_addr
 *   2 local_lkey
 *   3 remote_addr
 *   4 remote_rkey
 *   5 size
 *   6 signaled     (optional, skip)
 *
 * Later commits add E4 (whitelist), E5 (non-HK control flow), and E6
 * (first-arg type) checks.
 */
#include "Validator.h"

#include "clang/AST/RecursiveASTVisitor.h"

namespace gicc_plugin {

namespace {

// Map argument index → human-readable name for put_no_db / get_no_db.
// Used in the diagnostic message to point the user at the wrong arg.
const char* arg_name_for_put(unsigned i) {
    static const char* names[] = {
        "ctx", "local_addr", "local_lkey", "remote_addr",
        "remote_rkey", "size", "signaled"
    };
    if (i < sizeof(names) / sizeof(names[0])) return names[i];
    return "<?>";
}

// Strip the leading "gicc::" namespace from a qualified name so the
// diagnostic prints "gicc::put_no_db" cleanly via the format string
// (which already includes the "gicc::" prefix).
std::string strip_gicc_prefix(const std::string& qn) {
    static const std::string prefix = "gicc::";
    if (qn.compare(0, prefix.size(), prefix) == 0) {
        return qn.substr(prefix.size());
    }
    return qn;
}

// v1 lift whitelist (spec §5.4): the only gicc:: device-side symbols a
// kernel may call. Anything else gicc-namespaced — auto-doorbell put/get,
// am::*, barrier_*, etc. — is a hard error since the plugin cannot
// statically pre-stage it for the OFI DWQ.
bool is_whitelisted(const std::string& qn) {
    return qn == "gicc::put_no_db" ||
           qn == "gicc::get_no_db" ||
           qn == "gicc::flush"     ||
           qn == "gicc::quiet";
}

// E5: walk the kernel CFG and flag any put_no_db / get_no_db call
// inside a control-flow construct gated by a non-HK condition. We
// override Traverse* so we can run the body with non_hk_depth_++ and
// restore it afterwards — VisitCallExpr alone can't see its enclosing
// statement.
//
// Important per spec §5.2: only the loop *condition* gates puts. The
// init and increment of a for() run outside the loop body for HK
// purposes (init runs once; we don't lift puts in the increment).
class CFGuardVisitor
    : public clang::RecursiveASTVisitor<CFGuardVisitor> {
public:
    CFGuardVisitor(const HKAnalysis& hk, clang::DiagnosticsEngine& diag,
                   Diags ids)
        : hk_(hk), diag_(diag), ids_(ids) {}

    bool ok() const { return ok_; }

    bool TraverseIfStmt(clang::IfStmt* is) {
        if (!is) return true;
        const bool cond_hk = hk_.isHK(is->getCond());
        if (cond_hk) {
            return clang::RecursiveASTVisitor<CFGuardVisitor>::TraverseIfStmt(is);
        }
        // Visit cond at the current depth (doesn't gate puts itself,
        // but might syntactically contain one), then visit then/else
        // at depth+1.
        if (auto* c = is->getCond()) TraverseStmt(c);
        non_hk_depth_++;
        if (auto* t = is->getThen()) TraverseStmt(t);
        if (auto* e = is->getElse()) TraverseStmt(e);
        non_hk_depth_--;
        return true;
    }

    bool TraverseForStmt(clang::ForStmt* fs) {
        if (!fs) return true;
        const bool cond_hk = hk_.isHK(fs->getCond());
        if (cond_hk) {
            return clang::RecursiveASTVisitor<CFGuardVisitor>::TraverseForStmt(fs);
        }
        if (auto* i = fs->getInit()) TraverseStmt(i);
        if (auto* c = fs->getCond()) TraverseStmt(c);
        if (auto* inc = fs->getInc()) TraverseStmt(inc);
        non_hk_depth_++;
        if (auto* b = fs->getBody()) TraverseStmt(b);
        non_hk_depth_--;
        return true;
    }

    bool TraverseWhileStmt(clang::WhileStmt* ws) {
        if (!ws) return true;
        const bool cond_hk = hk_.isHK(ws->getCond());
        if (cond_hk) {
            return clang::RecursiveASTVisitor<CFGuardVisitor>::TraverseWhileStmt(ws);
        }
        if (auto* c = ws->getCond()) TraverseStmt(c);
        non_hk_depth_++;
        if (auto* b = ws->getBody()) TraverseStmt(b);
        non_hk_depth_--;
        return true;
    }

    bool TraverseDoStmt(clang::DoStmt* ds) {
        if (!ds) return true;
        const bool cond_hk = hk_.isHK(ds->getCond());
        if (cond_hk) {
            return clang::RecursiveASTVisitor<CFGuardVisitor>::TraverseDoStmt(ds);
        }
        if (auto* c = ds->getCond()) TraverseStmt(c);
        non_hk_depth_++;
        if (auto* b = ds->getBody()) TraverseStmt(b);
        non_hk_depth_--;
        return true;
    }

    bool VisitCallExpr(clang::CallExpr* ce) {
        if (non_hk_depth_ == 0) return true;
        auto* fd = ce->getDirectCallee();
        if (!fd) return true;
        const std::string qn = fd->getQualifiedNameAsString();
        if (qn != "gicc::put_no_db" && qn != "gicc::get_no_db") return true;
        const std::string short_fn = qn.substr(std::string("gicc::").size());
        diag_.Report(ce->getBeginLoc(), ids_.put_in_non_hk_branch)
            << short_fn;
        ok_ = false;
        return true;
    }

private:
    const HKAnalysis& hk_;
    clang::DiagnosticsEngine& diag_;
    Diags ids_;
    int non_hk_depth_ = 0;
    bool ok_ = true;
};

} // namespace

// E6: kernel's first formal parameter must be gicc::DeviceCtx*. The
// gicc::launch wrapper synthesizes the ctx and prepends it before
// forwarding the user-visible args, so the kernel signature MUST start
// with that pointer or the launch ABI is broken.
//
// We resolve via the canonical CXXRecord path rather than QualType
// string-printing because the textual form of the type varies with
// PrintingPolicy ("class gicc::DeviceCtx *" vs "gicc::DeviceCtx *" vs
// elaborated forms with HIP).
namespace {
bool first_arg_is_device_ctx_ptr(const clang::FunctionDecl* fd) {
    if (!fd || fd->getNumParams() == 0) return false;
    auto qt = fd->getParamDecl(0)->getType();
    auto* pt = qt->getAs<clang::PointerType>();
    if (!pt) return false;
    auto pointee = pt->getPointeeType();
    auto* rd = pointee->getAsCXXRecordDecl();
    if (!rd) return false;
    return rd->getQualifiedNameAsString() == "gicc::DeviceCtx";
}
} // namespace

bool Validator::validate(const KernelInfo& ki, const HKAnalysis& hk) {
    bool ok = true;
    auto& diag = CI_.getDiagnostics();

    // E6: check first parameter type. Only enforced for kernels that
    // call into gicc — pure HIP/CUDA kernels are out of scope for the
    // unified-launch ABI and shouldn't trigger this diagnostic. A
    // kernel that uses any gicc:: API is, by definition, launched via
    // gicc::launch, which prepends the synthesized ctx.
    if (!ki.calls.empty() && !first_arg_is_device_ctx_ptr(ki.decl)) {
        clang::SourceLocation loc = ki.decl->getNumParams() > 0
            ? ki.decl->getParamDecl(0)->getBeginLoc()
            : ki.decl->getBeginLoc();
        diag.Report(loc, diags_.bad_first_arg)
            << ki.decl->getNameAsString();
        ok = false;
    }

    for (const auto& call : ki.calls) {
        const std::string short_fn = strip_gicc_prefix(call.qualified_name);

        // E4: reject any gicc:: call not in the v1 whitelist.
        if (!is_whitelisted(call.qualified_name)) {
            diag.Report(call.expr->getBeginLoc(), diags_.non_whitelist_call)
                << short_fn;
            ok = false;
            // Don't run the put-arg HK check on a non-whitelist call —
            // the diagnostic message already covers it.
            continue;
        }

        // E3: every put_no_db / get_no_db arg (except ctx + signaled)
        // must be HK.
        const bool is_put = (call.qualified_name == "gicc::put_no_db" ||
                             call.qualified_name == "gicc::get_no_db");
        if (!is_put) continue;

        auto* ce = call.expr;
        const unsigned n = ce->getNumArgs();

        // Skip ctx (arg 0). We check arg 6 (signaled) only if the
        // user passes a 7th arg, but per the spec it is allowed to be
        // any bool literal — still, exclude index 6 explicitly so a
        // common pattern like `, /*signaled*/ false` is never flagged.
        for (unsigned i = 1; i < n && i < 6; i++) {
            auto* a = ce->getArg(i);
            if (!hk.isHK(a)) {
                diag.Report(a->getBeginLoc(), diags_.non_hk_arg)
                    << short_fn
                    << arg_name_for_put(i);
                ok = false;
            }
        }
    }

    // E5: walk the kernel body's CFG and reject puts inside non-HK
    // guards.
    if (ki.decl && ki.decl->getBody()) {
        CFGuardVisitor cfg(hk, diag, diags_);
        cfg.TraverseStmt(ki.decl->getBody());
        if (!cfg.ok()) ok = false;
    }

    return ok;
}

} // namespace gicc_plugin
