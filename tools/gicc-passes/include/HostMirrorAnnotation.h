#pragma once

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include <set>
#include <string>

namespace gicc::pass {

// Walk @llvm.global.annotations looking for entries of the form
//     gicc_kernel_host_mirror=<name>
// attached to `F`. Returns the set of <name> strings.
std::set<std::string>
getHostMirroredFormalNames(const llvm::Function &F);

// Walk @llvm.global.annotations looking for entries of the form
//     gicc_kernel_host_mirror_param=<idx>
// attached to `F`. Returns the set of integer indices. Used as a
// fallback when value names are stripped by the build pipeline.
std::set<unsigned>
getHostMirroredFormalParamIndices(const llvm::Function &F);

// Convenience: combine the two — for each formal of F, decide whether
// it should be marked host_mirrored. Match by name first; fall back to
// positional. Returns a vector of length F.arg_size().
std::vector<bool>
computeHostMirroredFormals(const llvm::Function &F);

}  // namespace gicc::pass
