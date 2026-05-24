#include "HostMirrorAnnotation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"

#include <vector>

using namespace llvm;

namespace gicc::pass {

namespace {

// Pull the underlying C-string out of an annotation string operand.
// Clang lowers an `annotate("foo")` to a private constant whose
// initializer is a [N x i8] holding "foo\00". The pointer that gets
// stored into @llvm.global.annotations is typically a direct reference
// to that global; sometimes it's the same global wrapped in a GEP /
// bitcast / addrspacecast. Strip those constant ops.
const ConstantDataArray *getAnnotationCDA(const Constant *C) {
    if (!C) return nullptr;
    if (auto *CE = dyn_cast<ConstantExpr>(C)) {
        if (CE->getOpcode() == Instruction::GetElementPtr ||
            CE->getOpcode() == Instruction::BitCast ||
            CE->getOpcode() == Instruction::AddrSpaceCast) {
            return getAnnotationCDA(CE->getOperand(0));
        }
        return nullptr;
    }
    if (auto *GV = dyn_cast<GlobalVariable>(C)) {
        if (!GV->hasInitializer()) return nullptr;
        return dyn_cast<ConstantDataArray>(GV->getInitializer());
    }
    return nullptr;
}

// Strip ConstantExpr casts to recover the underlying function pointer.
const Function *getAnnotationFunc(const Constant *C) {
    if (!C) return nullptr;
    if (auto *F = dyn_cast<Function>(C)) return F;
    if (auto *CE = dyn_cast<ConstantExpr>(C)) {
        if (CE->getOpcode() == Instruction::BitCast ||
            CE->getOpcode() == Instruction::AddrSpaceCast) {
            return getAnnotationFunc(CE->getOperand(0));
        }
    }
    return nullptr;
}

// Iterate every (funcPtr, annStr) tuple in @llvm.global.annotations
// whose funcPtr matches `target`. `consume(StringRef annStr)` is called
// with the annotation string (trailing NUL stripped).
template <typename Consume>
void forEachAnnotationOn(const Function &target, Consume consume) {
    const Module *M = target.getParent();
    if (!M) return;
    const GlobalVariable *GA = M->getGlobalVariable("llvm.global.annotations");
    if (!GA || !GA->hasInitializer()) return;
    const auto *arr = dyn_cast<ConstantArray>(GA->getInitializer());
    if (!arr) return;

    for (const Use &U : arr->operands()) {
        const auto *entry = dyn_cast<ConstantStruct>(U.get());
        if (!entry || entry->getNumOperands() < 2) continue;

        const Function *F = getAnnotationFunc(entry->getOperand(0));
        if (F != &target) continue;

        const ConstantDataArray *cda =
            getAnnotationCDA(entry->getOperand(1));
        if (!cda || !cda->isString()) continue;
        StringRef raw = cda->getAsCString();   // strips trailing NUL
        consume(raw);
    }
}

}  // namespace

std::set<std::string>
getHostMirroredFormalNames(const Function &F) {
    std::set<std::string> out;
    const StringRef prefix = "gicc_kernel_host_mirror=";
    forEachAnnotationOn(F, [&](StringRef ann) {
        if (!ann.starts_with(prefix)) return;
        out.insert(ann.drop_front(prefix.size()).str());
    });
    return out;
}

std::set<unsigned>
getHostMirroredFormalParamIndices(const Function &F) {
    std::set<unsigned> out;
    const StringRef prefix = "gicc_kernel_host_mirror_param=";
    forEachAnnotationOn(F, [&](StringRef ann) {
        if (!ann.starts_with(prefix)) return;
        StringRef rest = ann.drop_front(prefix.size());
        unsigned idx = 0;
        if (!rest.getAsInteger(10, idx)) out.insert(idx);
    });
    return out;
}

std::vector<bool>
computeHostMirroredFormals(const Function &F) {
    std::vector<bool> out(F.arg_size(), false);
    auto names = getHostMirroredFormalNames(F);
    auto idxs  = getHostMirroredFormalParamIndices(F);

    if (!names.empty()) {
        for (const Argument &A : F.args()) {
            StringRef nm = A.getName();
            if (!nm.empty() && names.count(nm.str()))
                out[A.getArgNo()] = true;
        }
    }
    for (unsigned i : idxs) {
        if (i < out.size()) out[i] = true;
    }
    return out;
}

}  // namespace gicc::pass
