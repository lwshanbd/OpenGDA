#include "AccessDecomposition.h"

#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/Operator.h"

using namespace llvm;

namespace gicc::pass {

AccessDecomposer::AccessDecomposer(ScalarEvolution &SE, const DataLayout &DL,
                                   Loop *L, PHINode *iv)
    : SE(SE), DL(DL), L(L), iv(iv), I64(Type::getInt64Ty(iv->getContext())) {}

// s with every outer truncate / extend removed.
const SCEV *AccessDecomposer::castBase(const SCEV *s) {
    while (auto *C = dyn_cast<SCEVCastExpr>(s)) s = C->getOperand();
    return s;
}

bool AccessDecomposer::eq(const SCEV *a, const SCEV *b) { return sameSCEV(SE, a, b); }

const SCEV *AccessDecomposer::coef(const Lin &a, int key) {
    for (auto &[k, c] : a.t)
        if (k == key) return c;
    return SE.getZero(I64);
}

void AccessDecomposer::addTerm(Lin &a, int key, const SCEV *c) {
    for (auto &[k, x] : a.t)
        if (k == key) { x = SE.getAddExpr(x, c); return; }
    a.t.push_back({key, c});
}

AccessDecomposer::Lin AccessDecomposer::combine(const Lin &a, const Lin &b, bool subtract) {
    Lin r;
    r.c0 = subtract ? SE.getMinusSCEV(a.c0, b.c0) : SE.getAddExpr(a.c0, b.c0);
    r.t = a.t;
    for (auto &[k, c] : b.t) addTerm(r, k, subtract ? SE.getNegativeSCEV(c) : c);
    // Drop terms that cancelled (e.g. X - (X - r)).
    r.t.erase(std::remove_if(r.t.begin(), r.t.end(),
                             [](auto &kc) { return kc.second->isZero(); }),
              r.t.end());
    return r;
}

AccessDecomposer::Lin AccessDecomposer::scale(const Lin &a, const SCEV *s) {
    Lin r;
    r.c0 = SE.getMulExpr(a.c0, s);
    for (auto &[k, c] : a.t) r.t.push_back({k, SE.getMulExpr(c, s)});
    return r;
}

bool sameSCEV(ScalarEvolution &SE, const SCEV *a, const SCEV *b) {
    if (a->getType() != b->getType()) {
        Type *T = SE.getWiderType(a->getType(), b->getType());
        a = SE.getNoopOrSignExtend(a, T);
        b = SE.getNoopOrSignExtend(b, T);
    }
    return SE.getMinusSCEV(a, b)->isZero();
}

const SCEV *exactDivide(ScalarEvolution &SE, const SCEV *a, const SCEV *b) {
    Type *T = SE.getWiderType(a->getType(), b->getType());
    a = SE.getNoopOrSignExtend(a, T);
    b = SE.getNoopOrSignExtend(b, T);
    if (sameSCEV(SE, a, b)) return SE.getOne(T);
    auto factors = [](const SCEV *s) {
        SmallVector<const SCEV *, 4> f;
        if (auto *M = dyn_cast<SCEVMulExpr>(s)) f.append(M->operands().begin(), M->operands().end());
        else f.push_back(s);
        return f;
    };
    SmallVector<const SCEV *, 4> fa = factors(a);
    for (const SCEV *x : factors(b)) {
        auto it = std::find_if(fa.begin(), fa.end(),
                               [&](const SCEV *y) { return sameSCEV(SE, x, y); });
        if (it == fa.end()) {
            // A constant factor may divide a larger constant.
            auto *cx = dyn_cast<SCEVConstant>(x);
            auto ic = std::find_if(fa.begin(), fa.end(),
                                   [](const SCEV *y) { return isa<SCEVConstant>(y); });
            if (!cx || ic == fa.end()) return nullptr;
            const APInt &A = cast<SCEVConstant>(*ic)->getAPInt();
            const APInt &B = cx->getAPInt();
            if (B.isZero() || !A.srem(B).isZero()) return nullptr;
            *ic = SE.getConstant(A.sdiv(B));
            continue;
        }
        fa.erase(it);
    }
    return fa.empty() ? SE.getOne(T) : SE.getMulExpr(fa);
}

// Division or remainder of X - q_0*R_0 - ... - q_{d-1}*R_{d-1} by a
// loop-invariant R: the digit q_d (or X - ... - q_d*R_d for a remainder).
std::optional<AccessDecomposer::Lin>
AccessDecomposer::digit(Instruction *I, bool remainder, std::string &why) {
    auto a = build(I->getOperand(0), why);
    if (!a) return std::nullopt;
    // The divisor read the way the instruction reads it: sign-extended for
    // sdiv / srem, zero-extended for udiv / urem.
    const SCEV *RS = SE.getSCEV(I->getOperand(1));
    if (!SE.isLoopInvariant(RS, L)) { why = "division by a loop-variant value"; return std::nullopt; }
    const bool isSigned =
        I->getOpcode() == Instruction::SDiv || I->getOpcode() == Instruction::SRem;
    const SCEV *R = isSigned ? SE.getNoopOrSignExtend(RS, I64) : SE.getNoopOrZeroExtend(RS, I64);

    // The dividend must be X minus a prefix of the digits, each times its
    // own radix, and nothing else.
    if (!a->c0->isZero() || !eq(coef(*a, -1), SE.getOne(I64))) {
        why = "a division of the loop index that is not a collapse decomposition";
        return std::nullopt;
    }
    unsigned d = 0;
    for (auto &[k, c] : a->t) {
        if (k < 0) continue;
        ++d;
        if (unsigned(k) >= chain.size() || !eq(c, SE.getNegativeSCEV(chain[k]))) {
            why = "the remainder of a collapse digit is not taken with that digit's radix";
            return std::nullopt;
        }
    }
    for (unsigned e = 0; e < d; ++e)
        if (!eq(coef(*a, e), SE.getNegativeSCEV(chain[e]))) {
            why = "collapse digits are not peeled outermost first";
            return std::nullopt;
        }
    if (d == chain.size()) chain.push_back(R);
    else if (!eq(chain[d], R)) {
        why = "collapse digit " + std::to_string(d) + " is divided by two different radices";
        return std::nullopt;
    }

    Lin r;
    if (remainder) {
        r = *a;
        addTerm(r, d, SE.getNegativeSCEV(R));
    } else {
        r.c0 = SE.getZero(I64);
        r.t.push_back({int(d), SE.getOne(I64)});
    }
    r.sx = r.ux = true;   // in [0, R) or [0, n_d): in range by construction
    return r;
}

// Why a value is not a single index: names the common case of a remainder
// X - q_d*R' whose R' is not the radix R the digit was divided by (clang
// masks the radix for 32-bit loops with a non-unit step).
std::string AccessDecomposer::whyNotSingle(const Lin &a) {
    const SCEV *cX = coef(a, -1);
    for (unsigned d = 0; d < chain.size(); ++d) {
        const SCEV *c = coef(a, d);
        if (!cX->isZero() && !c->isZero() &&
            !SE.getAddExpr(c, SE.getMulExpr(cX, chain[d]))->isZero())
            return "the remainder of collapse digit " + std::to_string(d) +
                   " is taken with a different radix than its division (" +
                   "cannot prove them equal)";
    }
    return "truncation of an index combining several loop dimensions";
}

// v as F << k, F built from the values shifted left by k: shl x, k -> x;
// sums and differences of such; constants whose low k bits are zero. k is
// taken from the first shl found and must be the same throughout.
std::optional<AccessDecomposer::Lin>
AccessDecomposer::buildShifted(Value *v, unsigned &k, std::string &why) {
    if (auto *C = dyn_cast<ConstantInt>(v)) {
        if (!k) { why = "arithmetic shift of the loop index"; return std::nullopt; }
        const APInt &A = C->getValue();
        if (A.countr_zero() < k) { why = "arithmetic shift of the loop index"; return std::nullopt; }
        Lin r;
        r.c0 = SE.getConstant(A.ashr(k).sextOrTrunc(64));
        return r;
    }
    auto *BO = dyn_cast<BinaryOperator>(v);
    if (!BO) { why = "arithmetic shift of the loop index"; return std::nullopt; }
    if (BO->getOpcode() == Instruction::Shl) {
        auto *K = dyn_cast<ConstantInt>(BO->getOperand(1));
        if (!K || (k && K->getZExtValue() != k)) {
            why = "arithmetic shift of the loop index";
            return std::nullopt;
        }
        k = K->getZExtValue();
        return build(BO->getOperand(0), why);
    }
    if (BO->getOpcode() == Instruction::Add || BO->getOpcode() == Instruction::Sub) {
        // Shifted operand first, so k is known before a constant is seen.
        Value *x = BO->getOperand(0), *y = BO->getOperand(1);
        const bool swap = isa<ConstantInt>(x);
        auto a = buildShifted(swap ? y : x, k, why);
        if (!a) return std::nullopt;
        auto b = buildShifted(swap ? x : y, k, why);
        if (!b) return std::nullopt;
        return BO->getOpcode() == Instruction::Sub && !swap ? combine(*a, *b, true)
             : BO->getOpcode() == Instruction::Sub       ? combine(*b, *a, true)
                                                         : combine(*a, *b, false);
    }
    why = "arithmetic shift of the loop index";
    return std::nullopt;
}

bool AccessDecomposer::sameLin(const Lin &a, const Lin &b) {
    Lin d = combine(a, b, true);
    if (!d.c0->isZero()) return false;
    return llvm::all_of(d.t, [](auto &kc) { return kc.second->isZero(); });
}

// At most one loop dimension, counted in digit coordinates: X itself is
// sum(q_d * R_d) + q_last, so X - q_0*R_0 is the single index q_last.
bool AccessDecomposer::singleIndex(const Lin &a) {
    const SCEV *cX = coef(a, -1);
    unsigned dims = cX->isZero() ? 0 : 1;
    for (unsigned d = 0; d < chain.size(); ++d)
        if (!SE.getAddExpr(coef(a, d), SE.getMulExpr(cX, chain[d]))->isZero())
            ++dims;
    return dims <= 1;
}

std::optional<AccessDecomposer::Lin> AccessDecomposer::build(Value *v, std::string &why) {
    if (auto it = memo.find(v); it != memo.end()) return it->second;
    auto done = [&](Lin r) { memo[v] = r; return std::optional<Lin>(r); };

    if (v == iv) {
        Lin r;
        r.c0 = SE.getZero(I64);
        r.t.push_back({-1, SE.getOne(I64)});
        r.sx = r.ux = true;   // X in [0, N), N < 2^63
        return done(r);
    }
    Type *T = v->getType();
    if (!T->isIntegerTy() || T->getIntegerBitWidth() > 64) {
        why = "index arithmetic on a non-integer or wider-than-64-bit value";
        return std::nullopt;
    }
    const SCEV *S = SE.getSCEV(v);
    if (SE.isLoopInvariant(S, L)) {
        Lin r;
        r.c0 = SE.getNoopOrSignExtend(S, I64);
        r.sx = true;
        r.ux = SE.isKnownNonNegative(S);
        return done(r);
    }

    auto *I = dyn_cast<Instruction>(v);
    if (!I) { why = "loop-variant value that is not an instruction"; return std::nullopt; }
    // A value reached again while it is being built is a recurrence (an
    // inner loop's IV, an accumulator), not an index of this loop.
    if (!inProgress.insert(I).second) {
        why = "loop-carried recurrence in the index";
        return std::nullopt;
    }
    struct Leave { DenseSet<Instruction *> &s; Instruction *i; ~Leave() { s.erase(i); } } leave{inProgress, I};
    auto *BO = dyn_cast<BinaryOperator>(I);
    auto nsw = [&] { return isa<OverflowingBinaryOperator>(I) && I->hasNoSignedWrap(); };
    auto nuw = [&] { return isa<OverflowingBinaryOperator>(I) && I->hasNoUnsignedWrap(); };

    switch (I->getOpcode()) {
    case Instruction::PHI: {
        auto *PN = cast<PHINode>(I);
        // A second IV stepping in lockstep with the IV: on every incoming
        // edge an extension of (or equal to) the IV's value there -- the
        // wide copy indvars keeps of a narrow IV. Its value is X.
        if (PN->getParent() == L->getHeader()) {
            bool lockstep = PN->getNumIncomingValues() == iv->getNumIncomingValues();
            for (unsigned k = 0; lockstep && k < PN->getNumIncomingValues(); ++k) {
                Value *a = PN->getIncomingValue(k);
                Value *b = iv->getIncomingValueForBlock(PN->getIncomingBlock(k));
                if (a == b) continue;
                auto *E = dyn_cast<CastInst>(a);
                lockstep = E && (isa<SExtInst>(E) || isa<ZExtInst>(E)) && E->getOperand(0) == b;
            }
            if (!lockstep) {
                why = "loop-carried recurrence in the index";
                return std::nullopt;
            }
            Lin r;
            r.c0 = SE.getZero(I64);
            r.t.push_back({-1, SE.getOne(I64)});
            r.sx = r.ux = true;   // X in [0, N), sext and zext agree
            return done(r);
        }
        // Merge of the same index computed on several paths (e.g. once per
        // branch and not CSE'd): all incoming values must agree.
        std::optional<Lin> r;
        for (Value *in : PN->incoming_values()) {
            auto x = build(in, why);
            if (!x) return std::nullopt;
            if (!r) { r = x; continue; }
            if (!sameLin(*r, *x)) {
                why = "a phi merging different indices";
                return std::nullopt;
            }
            r->sx &= x->sx;
            r->ux &= x->ux;
        }
        return done(*r);
    }
    case Instruction::Add:
    case Instruction::Sub: {
        auto a = build(BO->getOperand(0), why), b = build(BO->getOperand(1), why);
        if (!a || !b) return std::nullopt;
        Lin r = combine(*a, *b, I->getOpcode() == Instruction::Sub);
        r.sx = a->sx && b->sx && nsw();
        r.ux = a->ux && b->ux && nuw();
        // trunc(A + q) as instcombine narrows it, trunc(A) + trunc(q) with
        // no flags (a collapsed int loop's lb + digit): the same single
        // rebuilt loop index the trunc rule trusts to fit its type.
        auto *Tr0 = dyn_cast<TruncInst>(BO->getOperand(0)), *Tr1 = dyn_cast<TruncInst>(BO->getOperand(1));
        if (!r.sx && I->getType()->getIntegerBitWidth() < 64 && (Tr0 || Tr1) &&
            (a->t.empty() || b->t.empty()) && !r.t.empty() && singleIndex(r)) {
            usedTrunc = true;
            r.sx = true;
        }
        return done(r);
    }
    case Instruction::Mul: {
        auto a = build(BO->getOperand(0), why), b = build(BO->getOperand(1), why);
        if (!a || !b) return std::nullopt;
        if (!a->t.empty() && !b->t.empty()) {
            why = "product of two loop-variant values";
            return std::nullopt;
        }
        const Lin &var = a->t.empty() ? *b : *a, &inv = a->t.empty() ? *a : *b;
        const SCEV *factor = inv.c0;
        // q_d * R' with R' the digit's radix up to casts (clang divides by
        // the sign-extended trip count and multiplies back by the
        // zero-extended one): use the radix and leave R' == R_d for the
        // kernel to prove.
        if (var.c0->isZero() && var.t.size() == 1 && var.t[0].first >= 0 &&
            var.t[0].second->isOne()) {
            const SCEV *R = chain[var.t[0].first];
            if (!eq(factor, R) && castBase(factor) == castBase(R)) {
                if (llvm::none_of(assumed, [&](auto &p) { return p.first == R && p.second == factor; }))
                    assumed.push_back({R, factor});
                factor = R;
            }
        }
        Lin r = scale(var, factor);
        r.sx = var.sx && inv.sx && nsw();
        r.ux = var.ux && inv.ux && nuw();
        return done(r);
    }
    case Instruction::Shl: {
        auto *K = dyn_cast<ConstantInt>(BO->getOperand(1));
        auto a = build(BO->getOperand(0), why);
        if (!a) return std::nullopt;
        if (!K) { why = "shift of the loop index by a non-constant"; return std::nullopt; }
        Lin r = scale(*a, SE.getConstant(I64, uint64_t(1) << K->getZExtValue()));
        r.sx = a->sx && nsw();
        r.ux = a->ux && nuw();
        return done(r);
    }
    case Instruction::AShr: {
        // E >>s m where E = F << k (k >= m): sext of F truncated to
        // (width - k) bits, times 2^(k-m). instcombine writes int index
        // arithmetic this way: sext(trunc(q)) * 4 is (q << 32) >>s 30, and
        // sext(trunc(q) + 1) is ((q << 32) + (1 << 32)) >>s 32.
        auto *M = dyn_cast<ConstantInt>(BO->getOperand(1));
        unsigned k = 0;
        auto F = M ? buildShifted(BO->getOperand(0), k, why) : std::nullopt;
        if (!M || !F || k < M->getZExtValue()) {
            if (why.empty() || !M) why = "arithmetic shift of the loop index";
            return std::nullopt;
        }
        if (!singleIndex(*F)) {
            why = whyNotSingle(*F);
            return std::nullopt;
        }
        usedTrunc = true;
        Lin r = scale(*F, SE.getConstant(I64, uint64_t(1) << (k - M->getZExtValue())));
        r.sx = true;
        return done(r);
    }
    case Instruction::UDiv:
    case Instruction::SDiv:
        if (auto r = digit(I, false, why)) return done(*r);
        return std::nullopt;
    case Instruction::URem:
    case Instruction::SRem:
        if (auto r = digit(I, true, why)) return done(*r);
        return std::nullopt;
    case Instruction::SExt: {
        auto a = build(I->getOperand(0), why);
        if (!a) return std::nullopt;
        if (!a->sx) {
            why = "sign extension of index arithmetic that may have wrapped (no nsw)";
            return std::nullopt;
        }
        Lin r = *a;
        r.sx = true;
        return done(r);
    }
    case Instruction::ZExt: {
        auto a = build(I->getOperand(0), why);
        if (!a) return std::nullopt;
        if (!a->ux) {
            why = "zero extension of index arithmetic that may have wrapped (no nuw)";
            return std::nullopt;
        }
        Lin r = *a;
        r.sx = r.ux = true;
        return done(r);
    }
    case Instruction::Trunc: {
        auto a = build(I->getOperand(0), why);
        if (!a) return std::nullopt;
        if (!singleIndex(*a)) {
            why = whyNotSingle(*a);
            return std::nullopt;
        }
        // One rebuilt loop index: the source loop variable, which fits the
        // type it was declared with.
        usedTrunc = true;
        Lin r = *a;
        r.sx = true;
        r.ux = false;
        return done(r);
    }
    default:
        why = std::string("unsupported operation '") + I->getOpcodeName() +
              "' on the loop index";
        return std::nullopt;
    }
}

std::optional<AccessDecomposer::Lin>
AccessDecomposer::buildPointer(Value *p, Value *&base, std::string &why) {
    const SCEV *S = SE.getSCEV(p);
    if (SE.isLoopInvariant(S, L)) {
        auto *B = dyn_cast<SCEVUnknown>(SE.getPointerBase(S));
        if (!B) { why = "address has no single base pointer"; return std::nullopt; }
        base = B->getValue();
        Lin r;
        r.c0 = SE.getNoopOrSignExtend(SE.getMinusSCEV(S, B), I64);
        return r;
    }
    auto *G = dyn_cast<GEPOperator>(p);
    if (!G) { why = "address is not a base pointer plus index arithmetic"; return std::nullopt; }
    auto acc = buildPointer(G->getPointerOperand(), base, why);
    if (!acc) return std::nullopt;
    for (auto GTI = gep_type_begin(G), E = gep_type_end(G); GTI != E; ++GTI) {
        Value *idx = GTI.getOperand();
        if (StructType *ST = GTI.getStructTypeOrNull()) {
            uint64_t off = DL.getStructLayout(ST)->getElementOffset(
                cast<ConstantInt>(idx)->getZExtValue());
            acc->c0 = SE.getAddExpr(acc->c0, SE.getConstant(I64, off));
            continue;
        }
        if (idx->getType()->isVectorTy()) { why = "vector GEP index"; return std::nullopt; }
        auto li = build(idx, why);
        if (!li) return std::nullopt;
        // GEP sign-extends narrower indices; i64 arithmetic modulo 2^64 is
        // exact for an address.
        if (idx->getType()->getIntegerBitWidth() < 64 && !li->sx) {
            why = "narrow GEP index that may have wrapped";
            return std::nullopt;
        }
        uint64_t size = DL.getTypeAllocSize(GTI.getIndexedType());
        *acc = combine(*acc, scale(*li, SE.getConstant(I64, size)), false);
    }
    return acc;
}

std::optional<AccessDecomp> AccessDecomposer::decompose(Value *ptr, std::string &why) {
    AccessDecomp D;
    auto A = buildPointer(ptr, D.base, why);
    if (!A) return std::nullopt;
    D.offset = A->c0;
    D.trustedTrunc = usedTrunc;
    D.assumedEqual = assumed;
    D.radix = chain;

    // X = sum_{d<K} q_d R_d + q_K: fold the IV's coefficient into the digits.
    const SCEV *cX = coef(*A, -1);
    const unsigned K = chain.size();
    for (unsigned d = 0; d < K; ++d)
        D.stride.push_back(SE.getAddExpr(coef(*A, d), SE.getMulExpr(cX, chain[d])));
    D.stride.push_back(cX);

    D.extent.push_back(nullptr);   // n_1 = N / R_1, known to the kernel only
    for (unsigned d = 1; d < K; ++d) {
        const SCEV *n = exactDivide(SE, chain[d - 1], chain[d]);
        if (!n) {
            why = "collapse radix " + std::to_string(d - 1) +
                  " is not a multiple of radix " + std::to_string(d);
            return std::nullopt;
        }
        D.extent.push_back(n);
    }
    if (K > 0) D.extent.push_back(chain[K - 1]);
    return D;
}

}  // namespace gicc::pass
