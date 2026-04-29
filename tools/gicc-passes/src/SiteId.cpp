#include "SiteId.h"
#include "KernelInventory.h"

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/Path.h"

#include <sstream>

using namespace llvm;

namespace gicc::pass {

const char *opKindName(GICCOpKind k) {
    switch (k) {
        case GICCOpKind::PutNoDb: return "put_no_db";
        case GICCOpKind::GetNoDb: return "get_no_db";
        case GICCOpKind::Flush:   return "flush";
        case GICCOpKind::Quiet:   return "quiet";
    }
    return "?";
}

std::string buildSiteId(const Instruction *I, const Function *kernel,
                        unsigned callIndex) {
    std::string tuBase = "?";
    std::string line   = "?";
    if (const auto loc = I->getDebugLoc()) {
        if (auto *scope = loc->getScope()) {
            if (auto *file = scope->getFile()) {
                tuBase = sys::path::filename(file->getFilename()).str();
            }
        }
        line = std::to_string(loc.getLine());
    }
    std::ostringstream os;
    os << tuBase << ':' << line << ':' << kernel->getName().str()
       << "::" << callIndex;
    return os.str();
}

}  // namespace gicc::pass
