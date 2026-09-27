#pragma once
// Decompose a store address in a (possibly collapsed) OpenMP worksharing
// loop into a multi-dimensional box.
//
// collapse(D) gives one loop whose IV X runs over [0, N) and rebuilds the
// source indices with divisions: q_1 = X / R_1, q_2 = (X - q_1*R_1) / R_2,
// ..., q_D = X - q_1*R_1 - ... - q_{D-1}*R_{D-1}. If every radix divides
// the one before it (R_{d-1} = n_d * R_d) and N = n_1 * R_1, the digits
// q_d in [0, n_d) visit every point of the D-dimensional box exactly once
// as X runs over [0, N) -- plain mixed-radix arithmetic, whatever the
// source looked like. When the address is base + offset + sum(stride_d *
// q_d), the loop writes exactly that box.
//
// The address is rebuilt from the IR rather than from SCEV, so that the
// casts of 32-bit index arithmetic can be judged with the flags on the
// instructions:
//   - sext / zext need the operand to equal its math value read signed /
//     unsigned: nsw / nuw arithmetic on such operands does (a signed
//     overflow would be undefined behaviour in the source);
//   - trunc (also as shl + ashr, instcombine's sext(trunc(x))) is accepted
//     only on a single rebuilt loop index (one digit plus a loop-invariant
//     offset). That value is a source loop variable, or a source index
//     computed in the narrow type that the optimizer folded (i*m + j into
//     the collapsed IV): either way it fits that type, a signed overflow
//     in the source being undefined behaviour. This is a trusted
//     assumption -- a source that deliberately wraps a 64-bit index into
//     an int would break it -- reported through AccessDecomp::trustedTrunc.
//   - i64 arithmetic without flags is taken modulo 2^64, which is exact
//     for an address; a division's dividend must instead be recognised as
//     X minus earlier digits, whose value is in range by construction.
// Radices are trip-count products; they are taken to be positive and
// below 2^63, as trip counts are.

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"

#include <optional>
#include <string>

namespace gicc::pass {

struct AccessDecomp {
    llvm::Value *base = nullptr;         // loop-invariant base pointer
    const llvm::SCEV *offset = nullptr;  // bytes from base (i64, loop-invariant)
    // Outermost dimension first; dims = stride.size().
    llvm::SmallVector<const llvm::SCEV *, 4> stride;  // bytes per step of q_d
    // extent[d] for d >= 1 (extent[0] is null: N / R_1 comes from the
    // kernel's trip count). 1-D loops have one dimension and no radix.
    llvm::SmallVector<const llvm::SCEV *, 4> extent;
    llvm::SmallVector<const llvm::SCEV *, 4> radix;   // R_1 .. R_{D-1}
    bool trustedTrunc = false;           // relied on the loop-variable trunc rule
    // Radix pairs (R_d, R') taken as equal because they differ only by
    // casts; the box is valid only once the caller proves each pair.
    llvm::SmallVector<std::pair<const llvm::SCEV *, const llvm::SCEV *>, 2> assumedEqual;
};

// a / b when b divides a symbolically (equal, or b's factors are a subset of
// a's, constants dividing exactly); null otherwise.
const llvm::SCEV *exactDivide(llvm::ScalarEvolution &SE, const llvm::SCEV *a,
                              const llvm::SCEV *b);

// One per worksharing loop: every store in it must agree on the digits.
class AccessDecomposer {
public:
    AccessDecomposer(llvm::ScalarEvolution &SE, const llvm::DataLayout &DL,
                     llvm::Loop *L, llvm::PHINode *iv);

    // The box `ptr` walks as the loop runs, or nullopt with `why` set.
    std::optional<AccessDecomp> decompose(llvm::Value *ptr, std::string &why);

    // Radices found so far, outermost first.
    const llvm::SmallVector<const llvm::SCEV *, 4> &radices() const { return chain; }

private:
    // A value as a math integer: c0 + sum(coef * key), key -1 the IV and
    // key d >= 0 digit d; coefficients and c0 are loop-invariant i64 SCEVs.
    // sx / ux: the IR value, read signed / unsigned in its own width,
    // equals that math value (otherwise it only agrees modulo 2^width).
    struct Lin {
        const llvm::SCEV *c0 = nullptr;
        llvm::SmallVector<std::pair<int, const llvm::SCEV *>, 4> t;
        bool sx = false, ux = false;
    };
    const llvm::SCEV *coef(const Lin &a, int key);
    void addTerm(Lin &a, int key, const llvm::SCEV *c);
    Lin combine(const Lin &a, const Lin &b, bool subtract);
    Lin scale(const Lin &a, const llvm::SCEV *s);
    bool singleIndex(const Lin &a);
    bool sameLin(const Lin &a, const Lin &b);
    static const llvm::SCEV *castBase(const llvm::SCEV *s);
    std::string whyNotSingle(const Lin &a);

    llvm::DenseMap<llvm::Value *, Lin> memo;
    llvm::DenseSet<llvm::Instruction *> inProgress;
    std::optional<Lin> build(llvm::Value *v, std::string &why);
    std::optional<Lin> buildPointer(llvm::Value *p, llvm::Value *&base, std::string &why);
    std::optional<Lin> buildShifted(llvm::Value *v, unsigned &k, std::string &why);
    std::optional<Lin> digit(llvm::Instruction *I, bool remainder, std::string &why);
    bool eq(const llvm::SCEV *a, const llvm::SCEV *b);

    llvm::ScalarEvolution &SE;
    const llvm::DataLayout &DL;
    llvm::Loop *L;
    llvm::PHINode *iv;
    llvm::Type *I64;
    llvm::SmallVector<const llvm::SCEV *, 4> chain;   // radix of digit d
    bool usedTrunc = false;
    llvm::SmallVector<std::pair<const llvm::SCEV *, const llvm::SCEV *>, 2> assumed;
};

}  // namespace gicc::pass
