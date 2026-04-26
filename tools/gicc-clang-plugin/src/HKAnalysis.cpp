/**
 * HKAnalysis.cpp — see HKAnalysis.h for design intent.
 *
 * Implementation notes:
 *   - We track HK at the VarDecl level (DeclStmt-introduced locals whose
 *     initializer is HK). A DeclRefExpr to a HK VarDecl, or to any
 *     ParmVarDecl of the kernel, is HK.
 *   - exprIsHK is a pure recursive test on the AST shape; it does NOT
 *     mutate state, so it is safe to call repeatedly from the validator.
 *   - run() does a single pre-pass to seed hk_decls_ from straight-line
 *     `T x = HK_init;` patterns. This is sufficient for v1 — assignments
 *     and more complex flows fall through as NOT_HK and any put depending
 *     on them will be caught by E3.
 *   - Debug printing uses Lexer::getSourceText so the printed snippet
 *     matches what the user wrote.
 */
#include "HKAnalysis.h"

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/raw_ostream.h"

namespace gicc_plugin {

HKAnalysis::HKAnalysis(clang::FunctionDecl* k) : kernel_(k) {}

bool HKAnalysis::declIsHK(const clang::ValueDecl* d) const {
    if (!d) return false;
    // Kernel parameters are always HK (passed via gicc::launch).
    if (auto* pvd = llvm::dyn_cast<clang::ParmVarDecl>(d)) {
        if (pvd->getDeclContext() == kernel_) return true;
    }
    return hk_decls_.count(d) != 0;
}

bool HKAnalysis::exprIsHK(const clang::Expr* e) const {
    if (!e) return false;
    e = e->IgnoreParenImpCasts();

    // Literals — always HK.
    if (llvm::isa<clang::IntegerLiteral>(e) ||
        llvm::isa<clang::FloatingLiteral>(e) ||
        llvm::isa<clang::CharacterLiteral>(e) ||
        llvm::isa<clang::CXXBoolLiteralExpr>(e) ||
        llvm::isa<clang::CXXNullPtrLiteralExpr>(e) ||
        llvm::isa<clang::ImplicitValueInitExpr>(e)) {
        return true;
    }

    // CUDA/HIP grid builtins: threadIdx.x, blockIdx.y, blockDim.z,
    // gridDim.x, etc.
    //
    // CUDA-style: MemberExpr whose base is DeclRef("threadIdx" | ...).
    // HIP-style: __hip_builtin_threadIdx_t implements .x as a property
    //   that lowers to a PseudoObjectExpr wrapping a call to __get_x()
    //   on a const __hip_builtin_threadIdx_t& base. The DeclRefExpr to
    //   the global `threadIdx` lives inside the OpaqueValueExpr chain.
    auto is_grid_builtin_name = [](llvm::StringRef n) {
        return n == "threadIdx" || n == "blockIdx" ||
               n == "blockDim"  || n == "gridDim";
    };
    auto is_grid_builtin_type = [](clang::QualType qt) {
        std::string s = qt.getUnqualifiedType().getAsString();
        return s.find("__hip_builtin_threadIdx_t") != std::string::npos ||
               s.find("__hip_builtin_blockIdx_t")  != std::string::npos ||
               s.find("__hip_builtin_blockDim_t")  != std::string::npos ||
               s.find("__hip_builtin_gridDim_t")   != std::string::npos;
    };
    if (auto* poe = llvm::dyn_cast<clang::PseudoObjectExpr>(e)) {
        const clang::Expr* syn = poe->getSyntacticForm();
        if (auto* mpr = llvm::dyn_cast<clang::MSPropertyRefExpr>(
                syn ? syn->IgnoreParenImpCasts() : nullptr)) {
            const clang::Expr* base = mpr->getBaseExpr()->IgnoreParenImpCasts();
            if (is_grid_builtin_type(base->getType())) return true;
        }
        // Fall through to NOT HK for other pseudo-object forms.
        return false;
    }
    if (auto* me = llvm::dyn_cast<clang::MemberExpr>(e)) {
        const clang::Expr* base = me->getBase()->IgnoreParenImpCasts();
        if (auto* dre = llvm::dyn_cast<clang::DeclRefExpr>(base)) {
            if (is_grid_builtin_name(dre->getDecl()->getNameAsString())) {
                return true;
            }
        }
        if (is_grid_builtin_type(base->getType())) return true;
        // Otherwise: MemberExpr on a non-builtin base loads memory → NOT HK.
        return false;
    }
    if (auto* dre = llvm::dyn_cast<clang::DeclRefExpr>(e)) {
        if (is_grid_builtin_name(dre->getDecl()->getNameAsString())) {
            return true;
        }
        // Constants declared `constexpr` or `const` integer with a
        // constant initializer are HK by C++ semantics; for v1 we treat
        // any kernel-param ValueDecl or HK-seeded VarDecl as HK and let
        // others fall through.
        if (auto* vd = llvm::dyn_cast<clang::VarDecl>(dre->getDecl())) {
            if (vd->isConstexpr()) return true;
            if (vd->getType().isConstQualified() && vd->hasInit() &&
                exprIsHK(vd->getInit())) {
                return true;
            }
        }
        return declIsHK(dre->getDecl());
    }

    // Pure arithmetic on HK operands.
    if (auto* bo = llvm::dyn_cast<clang::BinaryOperator>(e)) {
        // Comma, assignment, compound-assignment, member-pointer ops are
        // NOT pure arithmetic; reject them here.
        if (bo->isAssignmentOp() || bo->isCompoundAssignmentOp() ||
            bo->getOpcode() == clang::BO_Comma ||
            bo->getOpcode() == clang::BO_PtrMemD ||
            bo->getOpcode() == clang::BO_PtrMemI) {
            return false;
        }
        return exprIsHK(bo->getLHS()) && exprIsHK(bo->getRHS());
    }

    if (auto* uo = llvm::dyn_cast<clang::UnaryOperator>(e)) {
        switch (uo->getOpcode()) {
            case clang::UO_Plus:
            case clang::UO_Minus:
            case clang::UO_Not:
            case clang::UO_LNot:
                return exprIsHK(uo->getSubExpr());
            default:
                // UO_Deref, UO_AddrOf, ++ / --: NOT HK by default.
                return false;
        }
    }

    // Conditional (?:): all three subexprs HK → result HK.
    if (auto* co = llvm::dyn_cast<clang::ConditionalOperator>(e)) {
        return exprIsHK(co->getCond()) &&
               exprIsHK(co->getTrueExpr()) &&
               exprIsHK(co->getFalseExpr());
    }

    // Explicit cast on an HK operand stays HK (covers static_cast etc.).
    if (auto* cse = llvm::dyn_cast<clang::CastExpr>(e)) {
        return exprIsHK(cse->getSubExpr());
    }

    // Array subscripts, function calls, dereferences, member loads on
    // non-builtin bases: conservatively NOT HK.
    if (llvm::isa<clang::ArraySubscriptExpr>(e) ||
        llvm::isa<clang::CallExpr>(e) ||
        llvm::isa<clang::CXXThisExpr>(e)) {
        return false;
    }

    return false;
}

bool HKAnalysis::isHK(const clang::Expr* e) const {
    return exprIsHK(e);
}

// Pre-pass over the kernel body: any `auto x = HK_init;` or
// `T x = HK_init;` makes x HK going forward. We don't currently track
// re-assignment (would require dataflow); for v1 the lift-able subset
// uses const-initialized locals or direct param/builtin references.
namespace {
class SeedVisitor : public clang::RecursiveASTVisitor<SeedVisitor> {
public:
    SeedVisitor(HKAnalysis& a, llvm::DenseSet<const clang::ValueDecl*>& set)
        : a_(a), set_(set) {}
    bool VisitDeclStmt(clang::DeclStmt* ds) {
        for (auto* d : ds->decls()) {
            if (auto* vd = llvm::dyn_cast<clang::VarDecl>(d)) {
                if (vd->hasInit() && a_.isHK(vd->getInit())) {
                    set_.insert(vd);
                }
            }
        }
        return true;
    }
private:
    HKAnalysis& a_;
    llvm::DenseSet<const clang::ValueDecl*>& set_;
};

// Debug printer: walks DeclStmt initializers in the kernel body and
// prints "[hk]   expr '<src>': HK|NOT_HK" lines.
class DebugPrinter : public clang::RecursiveASTVisitor<DebugPrinter> {
public:
    DebugPrinter(const HKAnalysis& a, const clang::SourceManager& sm,
                 const clang::LangOptions& lo)
        : a_(a), sm_(sm), lo_(lo) {}
    bool VisitDeclStmt(clang::DeclStmt* ds) {
        for (auto* d : ds->decls()) {
            auto* vd = llvm::dyn_cast<clang::VarDecl>(d);
            if (!vd || !vd->hasInit()) continue;
            const clang::Expr* init = vd->getInit()->IgnoreParenImpCasts();
            llvm::StringRef text = clang::Lexer::getSourceText(
                clang::CharSourceRange::getTokenRange(init->getSourceRange()),
                sm_, lo_);
            llvm::errs() << "[hk]   expr '" << text << "': "
                         << (a_.isHK(init) ? "HK" : "NOT_HK") << "\n";
        }
        return true;
    }
private:
    const HKAnalysis& a_;
    const clang::SourceManager& sm_;
    const clang::LangOptions& lo_;
};
} // namespace

void HKAnalysis::run() {
    if (!kernel_ || !kernel_->getBody()) return;

    SeedVisitor seeder(*this, hk_decls_);
    seeder.TraverseStmt(kernel_->getBody());

    if (debug) {
        llvm::errs() << "[hk] kernel: " << kernel_->getNameAsString() << "\n";
        for (auto* p : kernel_->parameters()) {
            llvm::errs() << "[hk]   param " << p->getNameAsString()
                         << ": HK\n";
        }
        const auto& sm = kernel_->getASTContext().getSourceManager();
        const auto& lo = kernel_->getASTContext().getLangOpts();
        DebugPrinter dp(*this, sm, lo);
        dp.TraverseStmt(kernel_->getBody());
    }
}

} // namespace gicc_plugin
