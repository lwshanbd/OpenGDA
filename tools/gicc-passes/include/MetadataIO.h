#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace gicc::pass {

struct ArgRef {
    // Kinds:
    //   Param      - kernel formal at index paramIdx
    //   ConstI64   - literal int64
    //   BinOp/Cast - arithmetic / type conversion on `children`
    //   Derived    - placeholder; trace synth emits undef
    //   LoopIv     - host-side loop induction variable (no payload; the
    //                trace builder substitutes the live %iv value in IR)
    //   FieldLoad  - `host_mirror_of(formal[paramIdx])[iv].field` where
    //                paramIdx names a kernel formal annotated as
    //                gicc_host_mirror.  baseFormalIdx is the formal,
    //                structElemSize is sizeof(struct), fieldByteOffset
    //                is offsetof(struct, field), fieldTypeStr is the
    //                loaded type ("i32" / "i64" / "ptr" / ...).  iv lives
    //                in children[0] as a nested ArgRef (typically LoopIv,
    //                but any HK expression works).  Trace synth emits
    //                a call to gicc_runtime_host_mirror_of(rt, formal)
    //                followed by a byte-GEP + load.
    enum class Kind { Param, ConstI64, BinOp, Cast, Derived, LoopIv, FieldLoad };
    Kind                 kind        = Kind::Derived;
    unsigned             paramIdx    = 0;       // for Kind::Param, Kind::FieldLoad
    int64_t              constVal    = 0;       // for Kind::ConstI64
    std::string          opStr;                 // for BinOp ("add", "shl", ...) / Cast
    std::vector<ArgRef>  children;
    // FieldLoad-only:
    int64_t              structElemSize  = 0;
    int64_t              fieldByteOffset = 0;
    std::string          fieldTypeStr;          // "i32" / "i64" / "ptr"
};

struct GuardSpec {
    // Kinds:
    //   Always         - unconditional
    //   ParamTruthy    - `if (kernel_formal[paramIdx]) op(...)`
    //   ParamEqConst   - `if (kernel_formal == constVal)`
    //   FieldNotNull   - `if (host_mirror[iv].field != nullptr)`; field
    //                    locator carried in `fieldArg` (a FieldLoad ArgRef).
    //                    Used by ASF's IPC-skip pattern
    //                    `if (transfers[i].peer_recv_addr != nullptr) continue;`
    //   Unknown        - guarded but pass can't model; degraded.
    enum class Kind { Always, ParamTruthy, ParamEqConst, BinOp, FieldNotNull, Unknown };
    Kind     kind     = Kind::Always;
    unsigned paramIdx = 0;
    int64_t  constVal = 0;
    // FieldNotNull-only: the field-load expression we compare against null.
    // Stored as a single-element vector to keep the type forward-declared
    // (ArgRef is defined just above).
    std::vector<ArgRef> fieldArg;
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
    bool      ivBoundKnown = false; // true if the bound is recoverable
    bool      ivBoundIsConst = false; // bound is a literal, not a formal
    int64_t   ivBoundConst = 0;     // the literal, when ivBoundIsConst
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
    // Arithmetic executed BETWEEN this op and the kernel's completion
    // point (its quiet/flush, or the kernel exit). This is the static
    // proxy for issue-to-first-use distance: compute_before says how
    // much work preceded the communication, compute_after says how much
    // is available to hide it behind. -1 when not measured.
    int                          compute_after  = -1;
    // Compile-time trip count of the enclosing loop when ScalarEvolution
    // can prove one, else -1. This is the op count per communication
    // phase, which the dispatch decision is highly sensitive to.
    long long                    trip_count     = -1;
    // False when compute_after had to skip a loop whose trip count is a
    // runtime value, i.e. the distance is a lower bound rather than an
    // estimate. The decider refuses to lean on an inexact distance.
    bool                         distance_exact = true;
    // How many transfers reach the wire on the same completion point as
    // this one, counting a loop body trip_count times. The compiler sees
    // this whole group before the first transfer is issued; a runtime
    // sees its members one at a time and cannot know how many follow.
    // Needs the CFG to compute, so unlike the reuse predicates below it
    // is carried rather than derived. -1 when not computed.
    long long                    batch_size          = -1;
    // For a flush/quiet: the weakest memory fence scope that is still
    // sound here (0 none, 1 block, 2 device, 3 system). Needs forward
    // reachability, so it is carried rather than derived. Defaults to the
    // strongest, which is what every site emitted before the pass could
    // choose, so an unset value can only be conservative.
    int                          fence_scope         = 3;
};

// Is this descriptor expression the same on every iteration of the
// enclosing loop?
//
// Conservative by construction: only a kernel formal or a literal counts
// as invariant. A Derived leaf means the builder could not model the
// value at all, so claiming invariance would be a guess — and these
// predicates gate hoisting the descriptor out of the loop, where a wrong
// answer silently sends stale bytes.
inline bool isLoopInvariant(const ArgRef &a) {
    switch (a.kind) {
        case ArgRef::Kind::LoopIv:   return false;
        case ArgRef::Kind::Derived:  return false;
        case ArgRef::Kind::Param:
        case ArgRef::Kind::ConstI64: return true;
        // A field load is invariant exactly when its index is: the host
        // mirror itself does not change during a launch.
        case ArgRef::Kind::BinOp:
        case ArgRef::Kind::Cast:
        case ArgRef::Kind::FieldLoad: break;
    }
    for (const auto &c : a.children)
        if (!isLoopInvariant(c)) return false;
    return true;
}

inline bool argInvariant(const OpTemplate &op, const char *name) {
    auto it = op.args.find(name);
    return it != op.args.end() && isLoopInvariant(it->second);
}

// The registered buffers this op names do not vary, so registration and
// address resolution can be hoisted even when the OFFSETS vary per
// iteration — the ordinary stencil shape, and the reason this is a
// separate predicate rather than implied by descriptorReusable.
inline bool bufferReusable(const OpTemplate &op) {
    if (op.kind != "put_no_db" && op.kind != "get_no_db") return false;
    return argInvariant(op, "dst_buf") && argInvariant(op, "src_buf");
}

// Every field of the descriptor repeats, so the host can stage it ONCE
// and trigger it trip_count times instead of restaging per iteration.
// Requires a loop worth amortising over, and one the pass understood.
inline bool descriptorReusable(const OpTemplate &op) {
    return bufferReusable(op) && op.loop.inLoop && !op.loop.degraded &&
           argInvariant(op, "target_rank") && argInvariant(op, "dst_off") &&
           argInvariant(op, "src_off") && argInvariant(op, "size");
}

// Structural equality of two descriptor expressions. Deliberately syntactic:
// two different spellings of the same value compare unequal, which costs an
// optimisation but never authorises a wrong one.
inline bool sameExpr(const ArgRef &a, const ArgRef &b) {
    if (a.kind != b.kind) return false;
    switch (a.kind) {
        case ArgRef::Kind::ConstI64:  return a.constVal == b.constVal;
        case ArgRef::Kind::Param:     return a.paramIdx == b.paramIdx;
        case ArgRef::Kind::LoopIv:    return true;
        case ArgRef::Kind::Derived:   return false;   // unknown != unknown
        case ArgRef::Kind::FieldLoad:
            if (a.paramIdx != b.paramIdx ||
                a.fieldByteOffset != b.fieldByteOffset ||
                a.structElemSize != b.structElemSize) return false;
            break;
        case ArgRef::Kind::BinOp:
        case ArgRef::Kind::Cast:
            if (a.opStr != b.opStr) return false;
            break;
    }
    if (a.children.size() != b.children.size()) return false;
    for (size_t i = 0; i < a.children.size(); ++i)
        if (!sameExpr(a.children[i], b.children[i])) return false;
    return true;
}

// Coefficient of the induction variable in an affine expression, returned
// as an expression rather than a number because the stride is usually a
// kernel formal. `hasIv` distinguishes "loop-invariant" (coefficient zero)
// from "not affine at all", which must not be conflated: the first is
// mergeable-with-itself, the second is unanalysable.
inline bool ivCoefficient(const ArgRef &a, ArgRef &coef, bool &hasIv) {
    auto oneOf = [](int64_t v) {
        ArgRef r; r.kind = ArgRef::Kind::ConstI64; r.constVal = v; return r;
    };
    switch (a.kind) {
        case ArgRef::Kind::ConstI64:
        case ArgRef::Kind::Param:
        case ArgRef::Kind::FieldLoad:
            hasIv = false; coef = oneOf(0); return true;
        case ArgRef::Kind::LoopIv:
            hasIv = true;  coef = oneOf(1); return true;
        case ArgRef::Kind::Derived:
            return false;
        case ArgRef::Kind::Cast:
            return a.children.size() == 1 &&
                   ivCoefficient(a.children[0], coef, hasIv);
        case ArgRef::Kind::BinOp:
            break;
    }
    if (a.children.size() != 2) return false;
    ArgRef ca, cb; bool ia = false, ib = false;
    if (!ivCoefficient(a.children[0], ca, ia)) return false;
    if (!ivCoefficient(a.children[1], cb, ib)) return false;

    if (a.opStr == "add" || a.opStr == "sub") {
        hasIv = ia || ib;
        if (ia && ib) return false;          // iv on both sides: give up
        coef = ia ? ca : cb;
        // (invariant - iv*c) would need a negated coefficient; refuse
        // rather than emit one with the wrong sign.
        if (a.opStr == "sub" && ib) return false;
        return true;
    }
    if (a.opStr == "mul" || a.opStr == "shl") {
        if (ia && ib) return false;          // iv*iv is not affine
        if (!ia && !ib) { hasIv = false; coef = oneOf(0); return true; }
        // The iv-free side is the stride. For shl the iv must be on the
        // left, and the stride is 1 << c.
        const ArgRef &other = ia ? a.children[1] : a.children[0];
        if (a.opStr == "shl") {
            if (!ia || other.kind != ArgRef::Kind::ConstI64 ||
                other.constVal < 0 || other.constVal > 62) return false;
            hasIv = true; coef = oneOf(int64_t(1) << other.constVal);
            return true;
        }
        // Only a unit iv coefficient is handled; i*a*b would need the
        // product of two expressions, which there is no ArgRef for.
        const ArgRef &ivSideCoef = ia ? ca : cb;
        if (!(ivSideCoef.kind == ArgRef::Kind::ConstI64 &&
              ivSideCoef.constVal == 1)) return false;
        hasIv = true; coef = other; return true;
    }
    return false;
}

// Do consecutive iterations of this transfer land exactly `size` apart at
// BOTH ends? If so the loop's transfers are one contiguous region and any
// run of them may be issued as a single larger transfer.
//
// This is the property a runtime provably cannot establish: when it sees
// transfer i it does not know where i+1 will go, and by the time it does,
// i has already been issued. Merging is also the reason a decision like
// this cannot be expressed as picking from a fixed menu — which runs to
// merge depends on the program's address pattern, so the action's arity
// follows the program rather than the model.
inline bool transfersAreAdjacent(const OpTemplate &op) {
    if (op.kind != "put_no_db" && op.kind != "get_no_db") return false;
    if (!op.loop.inLoop || op.loop.degraded) return false;
    auto sz = op.args.find("size");
    if (sz == op.args.end() || !isLoopInvariant(sz->second)) return false;
    for (const char *f : {"dst_off", "src_off"}) {
        auto it = op.args.find(f);
        if (it == op.args.end()) return false;
        ArgRef coef; bool hasIv = false;
        if (!ivCoefficient(it->second, coef, hasIv)) return false;
        if (!hasIv || !sameExpr(coef, sz->second)) return false;
    }
    return true;
}

struct ParamInfo {
    std::string name;        // formal name as it appears in IR (may be empty)
    std::string typeStr;     // "i32" / "i64" / "ptr" / ...
    // True when this formal is annotated as carrying a host-side mirror
    // (via __attribute__((annotate("gicc_kernel_host_mirror=<name>"))))
    // on the kernel function).  Field loads of the form `formal[iv].field`
    // become HK-capable when this bit is set: the trace synthesizer
    // resolves the device pointer at trace time via
    // gicc_runtime_host_mirror_of() then reads the field from the host
    // mirror.  See ArgRef::Kind::FieldLoad.
    bool        host_mirrored = false;
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
