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

// Source-free expression recovered from device IR.  This is separate from
// ArgRef: ArgRef is intentionally host-evaluable, while a producer address or
// predicate may depend on GPU launch coordinates.  Unknown leaves are kept in
// the tree so feature extraction can fail closed without hiding where
// recovery stopped.
struct DeviceExpr {
    enum class Kind {
        Param,
        ConstI64,
        Builtin,
        BinOp,
        Cast,
        Compare,
        Select,
        Unknown,
    };
    Kind                    kind = Kind::Unknown;
    unsigned                paramIdx = 0;
    int64_t                 constVal = 0;
    std::string             opStr;
    std::string             typeStr;
    std::vector<DeviceExpr> children;
};

struct ProducerPredicateFact {
    DeviceExpr condition;
    bool       requiredValue = true;
};

// One ordinary producer store, expressed as a byte interval rooted in a
// kernel pointer formal and the exact controlling predicates required to
// reach it.  `domain_exact` is only a local IR-recovery fact: it does not bind
// a transfer buffer handle or prove that this domain matches a transfer.
struct ProducerStoreDomainFact {
    unsigned                           pointerParam = 0;
    DeviceExpr                         byteOffset;
    uint64_t                           byteSize = 0;
    bool                               addressExact = false;
    std::vector<ProducerPredicateFact> predicates;
    bool                               predicatesExact = false;
    bool                               domainExact = false;
    // One compiler-recovered single-entry CFG edge that gates the whole
    // producer computation containing this store. The predicate index names
    // the corresponding entry in `predicates`; device LTO recomputes it.
    bool                               partitionRegionExact = false;
    unsigned                           partitionPredicateIndex = 0;
    std::string                        partitionRegionReason;
    std::string                        reason;
};

// One recognized atomic write between a transfer group and its completion.
// The first fission candidate does not reassociate atomics.  Instead it may
// use an exact, shared kernel-formal control predicate to prove that every
// atomic is unreachable on the optimized path, retaining the original fused
// launch when that predicate is enabled.
struct ProducerAtomicDomainFact {
    unsigned                           pointerParam = 0;
    std::string                        operation;
    bool                               resultUnused = false;
    std::vector<ProducerPredicateFact> predicates;
    bool                               predicatesExact = false;
    bool                               domainExact = false;
    std::string                        reason;
};

// One convergent or noduplicate call between a transfer group and its
// completion that is not a compiler-recognized, freely replicable GPU
// identity query. Phase fission may duplicate the surrounding kernel control
// flow, so every such call must either be preserved in one phase or be
// unreachable on the
// optimized path.  These predicates intentionally contain only exact direct
// i1 kernel-formal guards; unlike a complete store domain they remain useful
// when the call itself is nested in a loop.
struct ProducerPhaseSensitiveDomainFact {
    std::string                        operation;
    std::vector<ProducerPredicateFact> guardPredicates;
    bool                               guardPredicatesExact = false;
    bool                               domainExact = false;
    std::string                        reason;
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

// Source-free facts about the writes between one compiler-discovered PUT
// group and its mandatory flush.  These facts deliberately stop short of a
// legality claim: formal-rooted writes are only the first prerequisite for
// producer-frontier fission.  Host buffer identity, exact byte intervals,
// an exhaustive/disjoint iteration-domain partition, and side-effect
// partitioning must still be proved before a transform can be advertised.
struct ProducerFrontierFacts {
    bool                  analyzed = false;
    bool                  write_footprint_known = false;
    std::string           completion_site_id;
    std::vector<unsigned> ordinary_store_params;
    std::vector<unsigned> atomic_write_params;
    // Narrow runtime-guard shape for the first fission candidate. True only
    // when every transfer in the group uses one shared i32 source-buffer
    // formal and all ordinary producer stores use one pointer formal. Host
    // LTO may compare these two kernel argument slots at runtime and retain
    // the untouched fused launch on the false edge.
    bool                  buffer_identity_guardable = false;
    unsigned              producer_pointer_param = 0;
    unsigned              source_buffer_index_param = 0;
    std::string           buffer_identity_guard_reason;
    // Candidate pointer identities for a possible guarded early trigger.
    // `noalias` alone does not prove whole-allocation disjointness, so this
    // relation is not a legality result: host LTO must additionally prove or
    // check that every write-root allocation misses the transfer interval.
    bool                  source_identity_guardable = false;
    std::vector<unsigned> source_pointer_candidates;
    unsigned              source_identity_buffer_index_param = 0;
    std::string           source_identity_guard_reason;
    bool                  producer_domains_known = false;
    std::vector<ProducerStoreDomainFact> producer_store_domains;
    bool                  atomic_domains_known = false;
    std::vector<ProducerAtomicDomainFact> producer_atomic_domains;
    bool                  phase_sensitive_domains_known = false;
    std::vector<ProducerPhaseSensitiveDomainFact>
                          producer_phase_sensitive_domains;
    unsigned              ordinary_store_sites = 0;
    unsigned              atomic_write_sites = 0;
    unsigned              phase_sensitive_sites = 0;
    unsigned              unknown_write_sites = 0;
    std::string           reason;
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
    // Site ID of the first compiler-discovered flush/quiet that releases this
    // transfer group.  Empty means the host-side reset closes the group or the
    // CFG analysis could not name a completion point.
    std::string                  completion_site_id;
    // A conservative compiler proof for moving a shared DWQ trigger from the
    // completion point to immediately after the last static transfer site in
    // this group.  The proof is deliberately stronger than alias analysis:
    // without a mapping from a registered-buffer handle to an LLVM pointer,
    // *any* intervening instruction that may write memory rejects the move.
    // The lowering pass repeats the proof on the final device IR.
    bool                         group_early_trigger_analyzed = false;
    bool                         group_early_trigger_legal    = false;
    std::string                  group_early_trigger_reason;
    // Compiler-recovered write footprint between this transfer's group and
    // its flush.  This is richer model input, not permission to transform.
    ProducerFrontierFacts        producer_frontier;
    // For a flush/quiet: the weakest memory fence scope that is still
    // sound here (0 none, 1 block, 2 device, 3 system). Needs forward
    // reachability, so it is carried rather than derived. Defaults to the
    // strongest, which is what every site emitted before the pass could
    // choose, so an unset value can only be conservative.
    int                          fence_scope         = 3;
    // Which block issues this transfer, when the sites released by one
    // completion point are spread across blocks so their pushes overlap.
    // -1 means "not spread": everything on block 0, which is what was
    // emitted before this existed. On a completion point it instead holds
    // how many slots its group used, so the drain covers every ring that
    // was pushed to.
    int                          block_slot          = -1;
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

// Fold an expression to a constant, when it is one. Only the operators the
// builder actually emits for offsets; anything else is "not a constant",
// which is the safe answer for every caller here.
inline bool constOf(const ArgRef &a, int64_t &out) {
    switch (a.kind) {
        case ArgRef::Kind::ConstI64: out = a.constVal; return true;
        case ArgRef::Kind::Cast:
            return a.children.size() == 1 && constOf(a.children[0], out);
        case ArgRef::Kind::BinOp: {
            if (a.children.size() != 2) return false;
            int64_t l, r;
            if (!constOf(a.children[0], l) || !constOf(a.children[1], r))
                return false;
            if (a.opStr == "add") { out = l + r; return true; }
            if (a.opStr == "sub") { out = l - r; return true; }
            if (a.opStr == "mul") { out = l * r; return true; }
            if (a.opStr == "shl") {
                if (r < 0 || r > 62) return false;
                out = l << r; return true;
            }
            return false;
        }
        default: return false;
    }
}

// Fold an expression with the induction variable taken as zero -- the
// constant part of an affine offset, i.e. where the run starts. Returns
// false if anything else fails to fold, because an offset that is a kernel
// formal has unknown alignment and nothing may be claimed from it.
inline bool constAtIvZero(const ArgRef &a, int64_t &out) {
    if (a.kind == ArgRef::Kind::LoopIv) { out = 0; return true; }
    switch (a.kind) {
        case ArgRef::Kind::ConstI64: out = a.constVal; return true;
        case ArgRef::Kind::Cast:
            return a.children.size() == 1 && constAtIvZero(a.children[0], out);
        case ArgRef::Kind::BinOp: {
            if (a.children.size() != 2) return false;
            int64_t l, r;
            if (!constAtIvZero(a.children[0], l) ||
                !constAtIvZero(a.children[1], r))
                return false;
            if (a.opStr == "add") { out = l + r; return true; }
            if (a.opStr == "sub") { out = l - r; return true; }
            if (a.opStr == "mul") { out = l * r; return true; }
            if (a.opStr == "shl") {
                if (r < 0 || r > 62) return false;
                out = l << r; return true;
            }
            return false;
        }
        default: return false;
    }
}

// The widest power-of-two element a copy of this transfer could legally
// use, in bytes.
//
// A wide vector load has to stay inside one contiguous run and start
// aligned. Both bounds come from the descriptor: the transfer size, the
// base offsets, and -- when the transfer sits in a loop -- the stride
// between iterations, since a run ends where the next gap begins. The
// answer is the largest power of two dividing all of them.
//
// This is the knob whose legality is worth proving: measured, a vector
// width past the contiguous run does not run slowly on a strided face, it
// FAULTS. So anything that cannot be folded to a constant yields 1, which
// is always safe, and the cap is 16 because that is the widest load the
// copy kernels have.
inline int maxVectorBytes(const OpTemplate &op) {
    if (op.kind != "put_no_db" && op.kind != "get_no_db") return 1;
    auto sz = op.args.find("size");
    int64_t bound = 0;
    if (sz == op.args.end() || !constOf(sz->second, bound) || bound <= 0)
        return 1;

    auto accumulate = [&](int64_t v) {
        if (v < 0) v = -v;
        if (v == 0) return;                       // 0 constrains nothing
        int64_t a = bound, b = v;                 // gcd
        while (b) { int64_t t = a % b; a = b; b = t; }
        bound = a;
    };

    for (const char *f : {"dst_off", "src_off"}) {
        auto it = op.args.find(f);
        if (it == op.args.end()) return 1;
        // BOTH parts of an affine offset constrain the answer: the stride
        // bounds the contiguous run, and the constant addend sets where it
        // starts. Taking only the stride claims 16-byte alignment for
        // `iv*4096 + 2`, which is wrong in the direction that faults.
        int64_t addend = 0;
        if (!constAtIvZero(it->second, addend)) return 1;   // base unknown
        accumulate(addend);
        ArgRef coef; bool hasIv = false;
        if (!ivCoefficient(it->second, coef, hasIv)) return 1;
        if (hasIv) {
            int64_t stride = 0;
            if (!constOf(coef, stride)) return 1;
            accumulate(stride);
        }
    }

    int v = 1;
    while (v < 16 && (bound % (v * 2)) == 0) v *= 2;
    return v;
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
    // Device-IR argument attributes used only as compiler proofs. They are
    // serialized so final LTO and source-free schedule analysis can re-check
    // the same pointer relation without application source text.
    bool        noalias = false;
    bool        readonly = false;
};

struct KernelTemplate {
    std::string             mangledName;
    std::string             simpleName;
    std::vector<ParamInfo>  params;
    std::vector<OpTemplate> ops;
    // Written only after the final device LTO pass successfully materializes
    // the exact producer/remainder partition. The explicit hint-driven host
    // pass requires this one-way attestation before cloning a launch, so a
    // device-side rejection cannot leave a host-only two-phase schedule.
    // Device discovery rewrites the template with the default false value on
    // every compile, preventing an earlier failed attempt from inheriting it.
    bool                    producer_fission_device_materialized = false;
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
