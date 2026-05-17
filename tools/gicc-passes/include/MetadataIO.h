#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace gicc::pass {

struct ArgRef {
    // LoopIv: refers to the host-side loop induction variable that the
    // trace function materializes when emitting a loop op. Carries no
    // payload — the trace builder substitutes the live %iv value in IR.
    enum class Kind { Param, ConstI64, BinOp, Cast, Derived, LoopIv };
    Kind                 kind        = Kind::Derived;
    unsigned             paramIdx    = 0;       // for Kind::Param
    int64_t              constVal    = 0;       // for Kind::ConstI64
    std::string          opStr;                 // for BinOp ("add", "shl", ...) / Cast
    std::vector<ArgRef>  children;
};

struct GuardSpec {
    enum class Kind { Always, ParamTruthy, ParamEqConst, BinOp, Unknown };
    Kind     kind    = Kind::Always;
    unsigned paramIdx = 0;
    int64_t  constVal = 0;
};

// Description of a loop containing a GICC call site. Populated by the
// TraceTemplateBuilder when it detects that the call's parent BB has a
// canonical "for (i = start; i < bound; i += step)" loop with the bound
// being a kernel formal parameter (host-knowable). Iv-dependent
// arguments at the call site are stored as ArgRef expressions whose
// leaves are ArgRef::Kind::LoopIv (instead of a kernel formal); the
// host-side trace function substitutes the materialized loop counter.
//
// (Named OpLoopInfo to avoid collision with llvm::LoopInfo in callers.)
struct OpLoopInfo {
    bool      inLoop      = false;  // false → op runs once unconditionally
    unsigned  ivParamIdx  = 0;      // kernel formal that bounds the loop
    bool      ivBoundKnown = false; // true if ivParamIdx is meaningful
    int64_t   ivStart     = 0;
    int64_t   ivStep      = 1;
    bool      degraded    = false;  // recognized as loop but iv/bound not
                                     // resolvable; trace synthesizer treats
                                     // as "skip" so we don't silently emit
                                     // wrong code.
};

struct OpTemplate {
    std::string                  siteId;
    std::string                  kind;          // "put_no_db" / "get_no_db" / "flush" / "quiet"
    GuardSpec                    guard;
    OpLoopInfo                   loop;          // loop containing this op (if any)
    // Map from canonical arg-name (e.g. "target_rank", "size") to its
    // ArgRef expression. Order is not significant; readers should look
    // up by name.
    std::map<std::string, ArgRef> args;
    // HK Analysis capability bit, mirrored from GICCCallSite. True when
    // every non-ctx argument of this op is host-knowable. Defaults to
    // true so older JSON files (without the field) round-trip with the
    // pre-soft behavior.
    bool                         hk_capable = true;
    // Diagnostic explaining why hk_capable is false (e.g. "arg 6 not
    // host-knowable: depends on threadIdx.x"). Empty when hk_capable.
    std::string                  hk_fail_reason;
    // Static count of arithmetic / FP instructions in basic blocks
    // that dominate this call site (including the call's own BB,
    // counting only instructions ordered before the call within it).
    // Used as a coarse "how much compute precedes this comm op" feature
    // for the ML decider. Computed at device-discovery time via the
    // DominatorTree; -1 means "not computed" (older JSON / DT absent).
    int                          compute_before = -1;
};

struct ParamInfo {
    std::string name;        // formal name as it appears in IR (may be empty)
    std::string typeStr;     // "i32" / "i64" / "ptr" / ...
};

struct KernelTemplate {
    std::string             mangledName;
    std::string             simpleName;
    std::vector<ParamInfo>  params;
    std::vector<OpTemplate> ops;
    // Set to true by GICCDispatchLowering when at least one of this
    // kernel's call sites was lowered to CPU_PROXY_ENQUEUE. The
    // device-side lowering pass (Task 8) reads this bit to decide
    // whether to preserve the device-side put_no_db body so the proxy
    // ring enqueue stays in the kernel. Defaults to false so kernel
    // JSON files written before Task 2 round-trip with the original
    // "host trace owns everything" semantics.
    bool                    proxy_aware = false;
};

// Serialize `t` as JSON to ${metaDir}/<mangledName>.json. Creates the
// directory (recursively) if it doesn't exist. Returns false on I/O
// error.
bool writeKernelTemplate(const std::string &metaDir,
                         const KernelTemplate &t);

// Inverse of writeKernelTemplate. Populates `out` from
// ${metaDir}/<mangledName>.json. Returns false if the file is absent
// or malformed.
bool readKernelTemplate(const std::string &metaDir,
                        const std::string &mangledName,
                        KernelTemplate    &out);

}  // namespace gicc::pass
