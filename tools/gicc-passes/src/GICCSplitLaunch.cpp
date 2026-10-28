// A kernel split around a deferred fence (see include/GICCSplitLaunch.h).
#include "GICCSplitLaunch.h"
#include "GICCAfterPut.h"
#include "GICCPassConfig.h"
#include "OmpKernel.h"
#include "TuneConfig.h"
#include "WriteSet.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <cstdio>
#include <cstdlib>

using namespace llvm;

namespace gicc::pass {

namespace {

constexpr StringLiteral kWindow  = "ompx__split_window";
constexpr StringLiteral kLaunch  = "__tgt_target_kernel";
constexpr StringLiteral kDefer   = "ompx__fence_defer";
constexpr StringLiteral kFlush   = "ompx__fence_flush";
constexpr StringLiteral kBegin   = "ompx__split_begin";
constexpr StringLiteral kNext    = "ompx__split_next";
// Marks the launches the rewrite made, which no flush goes before.
constexpr StringLiteral kSplitMD = "gicc.split";

// __tgt_kernel_arguments: the loop trip count the runtime sizes the launch by.
constexpr uint64_t kArgsTripcount = 56;
constexpr uint64_t kArgsBytes = 104;   // sizeof(KernelArgsTy), LLVM 19 to 22
// __kmpc_{distribute,for}_static_init: the last-iteration flag.
constexpr unsigned kInitLast = 3;

// ompx_split_access in gicc/omp.h.
constexpr unsigned kDimsMax = 8;
constexpr uint64_t kAccBox = 0, kAccStride = 8, kAccExtent = 8 + 8 * kDimsMax,
                   kAccOffset = 8 + 16 * kDimsMax, kAccDims = kAccOffset + 8,
                   kAccLoopDims = kAccDims + 4, kAccStore = kAccDims + 8,
                   kAccElem = kAccDims + 12, kAccMapped = kAccDims + 16,
                   kAccBytes = kAccDims + 24;

raw_ostream &out() { return errs(); }

SmallString<256> recordFile(StringRef kernel) {
    char name[64];
    snprintf(name, sizeof(name), "split-%016llx.json",
             static_cast<unsigned long long>(afterPutHash(kernel)));
    SmallString<256> path(getConfig().metaDir);
    sys::path::append(path, name);
    return path;
}

void writeRecord(StringRef kernel, json::Object rec) {
    SmallString<256> path = recordFile(kernel);
    sys::fs::create_directories(sys::path::parent_path(path));
    std::error_code ec;
    raw_fd_ostream os(path, ec);
    if (ec) {
        out() << "[gicc-split] cannot write " << path << ": " << ec.message() << "\n";
        return;
    }
    rec["kernel"] = kernel;
    os << json::Value(std::move(rec));
}

// ---- device side --------------------------------------------------------------

// Whether the kernel, or anything it runs, may send or fetch by itself:
// reads the GiOMP device context or the pipelined-put list, makes an atomic
// access, or calls what the pass cannot see into. `what` says which.
bool communicates(Function &K, std::string &what) {
    SmallPtrSet<Function *, 16> seen;
    SmallVector<Function *, 16> work{&K};
    while (!work.empty()) {
        Function *F = work.pop_back_val();
        if (!seen.insert(F).second || F->getName() == kWindow) continue;
        for (Instruction &I : instructions(*F)) {
            if (isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I)) {
                what = "an atomic in " + F->getName().str();
                return true;
            }
            if (auto *Ld = dyn_cast<LoadInst>(&I)) {
                auto *G = dyn_cast<GlobalVariable>(getUnderlyingObject(Ld->getPointerOperand()));
                if (G && (G->getName() == "ompx__ctx" || G->getName() == "ompx__pipe_deferred")) {
                    what = (F->getName() + " reads " + G->getName()).str();
                    return true;
                }
                continue;
            }
            auto *CB = dyn_cast<CallBase>(&I);
            if (!CB) continue;
            for (Value *A : CB->args())
                if (auto *H = dyn_cast<Function>(A->stripPointerCasts()); H && !H->isDeclaration())
                    work.push_back(H);
            Function *G = CB->getCalledFunction();
            if (!G) {
                what = CB->isInlineAsm() ? "inline assembly" : "an indirect call";
                return true;
            }
            if (!G->isDeclaration()) {
                work.push_back(G);
                continue;
            }
            StringRef n = G->getName();
            if (n.starts_with("llvm.") || n.starts_with("__kmpc_") || n.starts_with("omp_") ||
                n.starts_with("__nv_") || n.starts_with("__ocml_") || n.starts_with("__ockl_"))
                continue;
            what = ("a call to " + n).str();
            return true;
        }
    }
    return false;
}

// Whether the loop has a lastprivate: something reads the last-iteration
// flag a worksharing init sets, which a launch over part of the loop would
// set on the wrong iteration.
bool readsLastFlag(CallInst *init) {
    auto *A = dyn_cast<AllocaInst>(getUnderlyingObject(init->getArgOperand(kInitLast)));
    if (!A) return true;
    for (User *U : A->users())
        if (isa<LoadInst>(U)) return true;
    return false;
}

// The kernel's boxes and iteration count as expressions of its formals, or
// nullopt with `why`.
std::optional<json::Object> plan(const OmpKernel &KI, FunctionAnalysisManager &FAM,
                                 std::string &why) {
    if (!KI.spmd()) {
        why = "not an SPMD kernel";
        return std::nullopt;
    }
    // Its second launch runs on teams of its own count.
    if (asksTeam(KI.K)) {
        why = "it asks which team it runs in";
        return std::nullopt;
    }
    auto *sched = dyn_cast<ConstantInt>(KI.distInit->getArgOperand(kmpc::InitSched));
    if (!sched || sched->getSExtValue() != kmpc::SchedDistChunked) {
        why = "its distribute loop is not chunked";
        return std::nullopt;
    }
    DominatorTree DT(KI.K);
    if (!KI.distBounds(DT)) {
        why = "cannot find its distribute loop's bounds";
        return std::nullopt;
    }
    if (readsLastFlag(KI.distInit)) {
        why = "it has a lastprivate";
        return std::nullopt;
    }
    for (Instruction &I : instructions(*KI.outlined))
        if (isCallTo(I, "__kmpc_for_static_init") && readsLastFlag(cast<CallInst>(&I))) {
            why = "it has a lastprivate";
            return std::nullopt;
        }
    auto W = collectAccesses(KI, FAM, why);
    if (!W) return std::nullopt;
    if (!W->iterations) {
        why = "its loop's iteration count is unknown (does it start at 0?)";
        return std::nullopt;
    }
    auto iters = loopPutExpr(W->iterations, KI.K);
    if (!iters) {
        why = "its iteration count is not a function of what it is handed";
        return std::nullopt;
    }
    json::Array accesses;
    auto add = [&](const BoxWrite &b, bool store) {
        auto *A = dyn_cast<Argument>(b.obj);
        if (!A || A->getParent() != &KI.K || A->getArgNo() == 0) {
            why = "an access through something it is not handed";
            return false;
        }
        const size_t dims = b.stride.size();
        if (dims == 0 || dims > kDimsMax || b.extent.size() != dims || b.innerDims >= dims) {
            why = "an access of " + std::to_string(dims) + " dims";
            return false;
        }
        auto off = loopPutExpr(b.offset, KI.K);
        json::Array stride, extent;
        bool ok = off.has_value();
        for (size_t d = 0; ok && d < dims; ++d) {
            auto s = loopPutExpr(b.stride[d], KI.K), e = loopPutExpr(b.extent[d], KI.K);
            ok = s && e;
            if (ok) {
                stride.push_back(std::move(*s));
                extent.push_back(std::move(*e));
            }
        }
        if (!ok) {
            why = std::string("the box of a ") + (store ? "store" : "load") +
                  " is not a function of what the kernel is handed";
            return false;
        }
        accesses.push_back(json::Object{
            {"store", store},
            {"arg", static_cast<int64_t>(A->getArgNo())},
            {"offset", std::move(*off)},
            {"stride", std::move(stride)},
            {"extent", std::move(extent)},
            {"loop_dims", static_cast<int64_t>(dims - b.innerDims)},
            {"elem", static_cast<int64_t>(b.storeSize)}});
        return true;
    };
    for (const BoxWrite &b : W->boxes)
        if (!add(b, true)) return std::nullopt;
    for (const BoxWrite &b : W->reads)
        if (!add(b, false)) return std::nullopt;
    return json::Object{{"iterations", std::move(*iters)}, {"accesses", std::move(accesses)}};
}

// The kernel's distribute loop runs [lo, hi) when the launch says so
// (ompx__split_window, read at its start): its bounds are replaced before
// the dist-chunk pass deals its chunks, which then fit the range. Clang's
// loop also clamps each chunk to the loop's last iteration and runs while
// a chunk starts before the end, both from values of its own rather than
// the bounds it handed the init: those uses become the range's too.
void lowerRange(const OmpKernel &KI, ScalarEvolution &SE) {
    Module &M = *KI.K.getParent();
    LLVMContext &C = M.getContext();
    Type *I64 = Type::getInt64Ty(C), *I32 = Type::getInt32Ty(C);
    auto *Ptr = PointerType::get(C, 0);
    FunctionCallee window = M.getOrInsertFunction(kWindow, I32, Ptr, Ptr);
    BasicBlock &entry = KI.K.getEntryBlock();
    IRBuilder<> B(&*entry.getFirstInsertionPt());
    const unsigned AS = KI.DL.getAllocaAddrSpace();
    AllocaInst *loSlot = B.CreateAlloca(I64, AS, nullptr, "gicc.split.lo");
    AllocaInst *hiSlot = B.CreateAlloca(I64, AS, nullptr, "gicc.split.hi");
    Value *range = B.CreateICmpNE(
        B.CreateCall(window, {B.CreateAddrSpaceCast(loSlot, Ptr), B.CreateAddrSpaceCast(hiSlot, Ptr)}),
        ConstantInt::get(I32, 0), "gicc.split.range");
    Value *lo = B.CreateLoad(I64, loSlot, "gicc.split.lo");
    Value *hi = B.CreateLoad(I64, hiSlot, "gicc.split.hi");
    DominatorTree DT(KI.K);
    auto [lbSt, ubSt] = *KI.distBounds(DT);
    Value *ub0 = ubSt->getValueOperand();
    Type *T = ub0->getType();
    // The loop's last iteration and its end, as the loop's own code spells
    // them after the init.
    const SCEV *last = SE.getSCEV(ub0), *end = SE.getAddExpr(last, SE.getOne(T));
    SmallVector<std::pair<Use *, bool>, 8> uses;
    for (BasicBlock &BB : KI.K) {
        if (!DT.dominates(KI.distInit->getParent(), &BB)) continue;
        for (Instruction &I : BB) {
            if (&BB == KI.distInit->getParent() && !KI.distInit->comesBefore(&I)) continue;
            if (!isa<ICmpInst>(I) && !isa<MinMaxIntrinsic>(I) && !isa<SelectInst>(I)) continue;
            for (Use &U : I.operands()) {
                if (U->getType() != T || isa<Constant>(U.get())) continue;
                const SCEV *x = SE.getSCEV(U.get());
                if (x == last) uses.push_back({&U, false});
                else if (x == end) uses.push_back({&U, true});
            }
        }
    }
    IRBuilder<> At(lbSt);
    Value *newLb = At.CreateSelect(range, At.CreateSExtOrTrunc(lo, T), lbSt->getValueOperand());
    lbSt->setOperand(0, newLb);
    At.SetInsertPoint(ubSt);
    Value *hiT = At.CreateSExtOrTrunc(hi, T);
    Value *newLast = At.CreateSelect(range, At.CreateSub(hiT, ConstantInt::get(T, 1)), ub0);
    ubSt->setOperand(0, newLast);
    if (uses.empty()) return;
    // The end, where the bounds are stored: it dominates the loop.
    Value *newEnd = At.CreateAdd(newLast, ConstantInt::get(T, 1), "gicc.split.end");
    for (auto [U, isEnd] : uses) U->set(isEnd ? newEnd : newLast);
}

// Whether a kernel the tuning file says nothing about is split: GICC_SPLIT=1.
// Off by default: the second launch and the record's exchange cost tens of
// microseconds a step, which a short epoch's transfers do not repay.
bool splitDefault() {
    const char *e = std::getenv("GICC_SPLIT");
    return e && *e == '1';
}

bool runDevice(Module &M, ModuleAnalysisManager &MAM) {
    auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
    Function *window = M.getFunction(kWindow);
    const bool canSplit = window && !window->isDeclaration();
    bool changed = false;
    for (Function &K : M) {
        if (K.isDeclaration() || !isOffloadKernel(K)) continue;
        json::Object rec;
        json::Object site{{"kind", "kernel"}, {"id", K.getName()}, {"option", "split"}};
        std::string what, why;
        const bool comm = communicates(K, what);
        rec["comm"] = comm;
        std::optional<json::Object> P;
        std::optional<OmpKernel> KI;
        if (comm) why = "it may communicate itself: " + what;
        else if (!canSplit) why = "the unit does not include gicc/omp_pipeline.h";
        else if (!tune::split(K.getName()).value_or(splitDefault()))
            why = "not asked for (GICC_SPLIT=1, or split: true in the tuning file)";
        else if (auto located = OmpKernel::locate(K, why)) {
            KI.emplace(std::move(*located));
            P = plan(*KI, FAM, why);
        }
        site["splittable"] = P.has_value();
        if (P) {
            lowerRange(*KI, FAM.getResult<ScalarEvolutionAnalysis>(K));
            FAM.invalidate(K, PreservedAnalyses::none());
            if (std::getenv("GICC_SPLIT_DUMP")) KI->K.print(out());
            rec["split"] = std::move(*P);
            changed = true;
            out() << "[gicc-split] " << K.getName() << ": split around a deferred fence\n";
        } else {
            site["why"] = why;
            out() << "[gicc-split] " << K.getName() << ": not split: " << why << "\n";
        }
        tune::record(std::move(site));
        writeRecord(K.getName(), std::move(rec));
    }
    return changed;
}

// ---- host side ----------------------------------------------------------------

struct Record {
    bool comm = true;
    std::optional<json::Object> split;
};

StringMap<Record> readRecords() {
    StringMap<Record> recs;
    std::error_code ec;
    for (sys::fs::directory_iterator it(getConfig().metaDir, ec), end; it != end && !ec;
         it.increment(ec)) {
        StringRef name = sys::path::filename(it->path());
        if (!name.starts_with("split-") || !name.ends_with(".json")) continue;
        auto buf = MemoryBuffer::getFile(it->path());
        if (!buf) continue;
        Expected<json::Value> v = json::parse((*buf)->getBuffer());
        if (!v) {
            consumeError(v.takeError());
            continue;
        }
        const json::Object *o = v->getAsObject();
        auto kernel = o ? o->getString("kernel") : std::nullopt;
        auto comm = o ? o->getBoolean("comm") : std::nullopt;
        if (!kernel || !comm) continue;
        Record r;
        r.comm = *comm;
        if (const json::Object *s = o->getObject("split")) r.split = *s;
        recs[*kernel] = std::move(r);
    }
    return recs;
}

// Calls that neither see device memory nor wait on another rank.
bool benignCall(const Function &F) {
    if (F.doesNotAccessMemory()) return true;
    static const StringSet<> names = {
        "MPI_Wtime", "MPI_Wtick", "MPI_Comm_rank", "MPI_Comm_size", "MPI_Initialized",
        "clock_gettime", "gettimeofday", "time", "clock",
        "omp_get_wtime", "omp_get_wtick", "omp_get_thread_num", "omp_get_num_threads",
        "omp_get_max_threads", "omp_get_default_device", "omp_get_initial_device",
        "omp_get_num_devices", "omp_in_parallel", "__kmpc_global_thread_num",
        "printf", "fprintf", "puts", "fputs", "putchar", "fputc", "fflush", "sprintf",
        "snprintf", "vprintf", "vfprintf", "perror", "strlen", "strcmp", "strncmp",
        "strcpy", "strncpy", "strchr", "strstr", "memcmp", "atoi", "atol", "atof",
        "strtol", "strtoul", "strtoll", "strtoull", "strtod", "getenv", "malloc", "calloc",
        "realloc", "free", "aligned_alloc", "posix_memalign", "_Znwm", "_Znam", "_ZdlPv",
        "_ZdaPv", "_ZdlPvm", "_ZdaPvm", "sqrt", "sqrtf", "exp", "expf", "log", "logf",
        "pow", "powf", "sin", "cos", "tan", "fabs", "floor", "ceil", "fmod"};
    return names.contains(F.getName());
}

// A callee that is one of several functions of the unit, chosen by a select
// or a phi (`(c ? f : g)(...)`): each flushes itself where it must.
bool callsDefinedOnly(const Value *callee, unsigned depth = 0) {
    callee = callee->stripPointerCasts();
    if (auto *F = dyn_cast<Function>(callee)) return !F->isDeclaration();
    if (depth > 4) return false;
    if (auto *S = dyn_cast<SelectInst>(callee))
        return callsDefinedOnly(S->getTrueValue(), depth + 1) &&
               callsDefinedOnly(S->getFalseValue(), depth + 1);
    if (auto *P = dyn_cast<PHINode>(callee))
        return all_of(P->incoming_values(),
                      [&](const Value *v) { return callsDefinedOnly(v, depth + 1); });
    return false;
}

bool hasAddressTaken(const Function &F) {
    return F.hasAddressTaken(nullptr, /*IgnoreCallbackUses=*/false,
                             /*IgnoreAssumeLikeCalls=*/true, /*IgnoreLLVMUsed=*/true);
}

// Rewrites launch L of a split kernel; false (and nothing changed) when the
// launch does not wait for it or does not hand over what its boxes need.
bool rewriteLaunch(Module &M, CallInst *L, StringRef kernel, const json::Object &P,
                   DominatorTree &DT, std::string &why) {
    if (!launchWaits(L, DT)) {
        why = "the launch may not wait for the kernel (nowait)";
        return false;
    }
    if (L->getFunction()->getName().contains(".omp_outlined")) {
        why = "the launch is in a host parallel region";
        return false;
    }
    HostLaunch H = readHostLaunch(L, DT);
    const json::Array *acc = P.getArray("accesses");
    const json::Value *itersV = P.get("iterations");
    if (H.args.empty() || !acc || !itersV) {
        why = "cannot read what the launch hands the kernel";
        return false;
    }
    LLVMContext &C = M.getContext();
    Type *I64 = Type::getInt64Ty(C), *I32 = Type::getInt32Ty(C), *I8 = Type::getInt8Ty(C);
    auto *Ptr = PointerType::get(C, 0);
    Function &F = *L->getFunction();
    IRBuilder<> EB(&*F.getEntryBlock().getFirstInsertionPt());
    const unsigned n = acc->size();
    AllocaInst *arr = EB.CreateAlloca(ArrayType::get(I8, std::max(1u, n) * kAccBytes), nullptr,
                                      "gicc.split.acc");
    arr->setAlignment(Align(8));

    IRBuilder<> B(L);
    Instruction *before = L->getPrevNode();
    auto undo = [&](const char *reason) {
        while (L->getPrevNode() != before) L->getPrevNode()->eraseFromParent();
        if (arr->use_empty()) arr->eraseFromParent();
        why = reason;
        return false;
    };
    auto i64 = [&](const json::Value *e) -> Value * {
        Value *v = e ? buildLaunchExpr(*e, B, H) : nullptr;
        if (!v) return nullptr;
        if (v->getType()->isPointerTy()) return B.CreatePtrToInt(v, I64);
        return v->getType()->isIntegerTy() ? B.CreateSExtOrTrunc(v, I64) : nullptr;
    };
    auto field = [&](uint64_t off) { return B.CreateConstInBoundsGEP1_64(I8, arr, off); };
    for (unsigned i = 0; i < n; ++i) {
        const json::Object *a = (*acc)[i].getAsObject();
        auto argNo = a ? a->getInteger("arg") : std::nullopt;
        auto store = a ? a->getBoolean("store") : std::nullopt;
        auto loopDims = a ? a->getInteger("loop_dims") : std::nullopt;
        auto elem = a ? a->getInteger("elem") : std::nullopt;
        const json::Array *stride = a ? a->getArray("stride") : nullptr;
        const json::Array *extent = a ? a->getArray("extent") : nullptr;
        if (!argNo || !store || !loopDims || !elem || !stride || !extent || *argNo < 1 ||
            stride->size() != extent->size() || stride->empty() || stride->size() > kDimsMax)
            return undo("its record does not read");
        // Formal n is launch argument n - 1.
        json::Value baseE = json::Object{{"op", "arg"}, {"ty", "ptr"}, {"n", *argNo}};
        Value *base = buildLaunchExpr(baseE, B, H);
        // A mapped variable reaches the kernel translated: the runtime finds
        // it in the heap only if ompx_bind put it there.
        bool mapped = false;
        if (!base && static_cast<size_t>(*argNo - 1) < H.args.size() && H.args[*argNo - 1] &&
            H.args[*argNo - 1]->getType()->isPointerTy()) {
            base = H.args[*argNo - 1];
            mapped = true;
        }
        Value *off = i64(a->get("offset"));
        if (!base || !off)
            return undo("a box's base or offset is not computed from what the launch hands over");
        const uint64_t at = i * kAccBytes;
        // A mapped variable by its own host address, which the runtime looks
        // up; a device address with the offset applied.
        B.CreateStore(mapped ? base : B.CreateGEP(I8, base, off), field(at + kAccBox));
        B.CreateStore(mapped ? off : ConstantInt::get(I64, 0), field(at + kAccOffset));
        for (unsigned d = 0; d < stride->size(); ++d) {
            Value *s = i64(&(*stride)[d]), *e = i64(&(*extent)[d]);
            if (!s || !e) return undo("a box's dims are not computed from what the launch hands over as is");
            B.CreateStore(s, field(at + kAccStride + 8 * d));
            B.CreateStore(e, field(at + kAccExtent + 8 * d));
        }
        B.CreateStore(ConstantInt::get(I32, stride->size()), field(at + kAccDims));
        B.CreateStore(ConstantInt::get(I32, *loopDims), field(at + kAccLoopDims));
        B.CreateStore(ConstantInt::get(I32, *store ? 1 : 0), field(at + kAccStore));
        B.CreateStore(ConstantInt::get(I32, *elem), field(at + kAccElem));
        B.CreateStore(ConstantInt::get(I32, mapped ? 1 : 0), field(at + kAccMapped));
    }
    Value *iters = i64(itersV);
    if (!iters) return undo("its iteration count is not computed from what the launch hands over");

    FunctionCallee begin = M.getOrInsertFunction(kBegin, I32, I64, I32, Ptr, I64);
    FunctionCallee next = M.getOrInsertFunction(kNext, I32, Ptr);
    Value *split = B.CreateICmpNE(
        B.CreateCall(begin, {ConstantInt::get(I64, afterPutHash(kernel)), ConstantInt::get(I32, n),
                             arr, iters}),
        ConstantInt::get(I32, 0));
    AllocaInst *trip = EB.CreateAlloca(I64, nullptr, "gicc.split.trip");
    // libomptarget rewrites the launch's arguments in place (NumArgs, the
    // teams' unused dims): each launch of a range starts from a copy of them
    // as the region left them.
    Value *kargs = L->getArgOperand(5);
    uint64_t kargsBytes = kArgsBytes;
    if (auto *KA = dyn_cast<AllocaInst>(kargs->stripPointerCasts()))
        if (auto sz = KA->getAllocationSize(M.getDataLayout())) kargsBytes = sz->getFixedValue();
    AllocaInst *saved = EB.CreateAlloca(ArrayType::get(I8, kargsBytes), nullptr, "gicc.split.kargs");
    saved->setAlignment(Align(8));
    B.CreateMemCpy(saved, Align(8), kargs, Align(8), kargsBytes);
    MDNode *mark = MDNode::get(C, {});
    L->setMetadata(kSplitMD, mark);
    // split ? (while (next(&trip)) { kargs.Tripcount = trip; launch; }) : launch
    Instruction *thenT, *elseT;
    SplitBlockAndInsertIfThenElse(split, L->getIterator(), &thenT, &elseT);
    BasicBlock *tail = L->getParent(), *thenBB = thenT->getParent();
    L->moveBefore(elseT->getIterator());
    BasicBlock *head = BasicBlock::Create(C, "gicc.split.next", &F, tail);
    BasicBlock *body = BasicBlock::Create(C, "gicc.split.launch", &F, tail);
    thenT->eraseFromParent();
    BranchInst::Create(head, thenBB);
    IRBuilder<> Hd(head);
    PHINode *sum = Hd.CreatePHI(L->getType(), 2, "gicc.split.ret");
    sum->addIncoming(ConstantInt::get(L->getType(), 0), thenBB);
    Value *more = Hd.CreateICmpNE(Hd.CreateCall(next, {trip}), ConstantInt::get(I32, 0));
    Hd.CreateCondBr(more, body, tail);
    IRBuilder<> Bd(body);
    Bd.CreateMemCpy(kargs, Align(8), saved, Align(8), kargsBytes);
    Bd.CreateStore(Bd.CreateLoad(I64, trip), Bd.CreateConstInBoundsGEP1_64(I8, kargs, kArgsTripcount));
    auto *part = cast<CallInst>(L->clone());
    Bd.Insert(part);
    sum->addIncoming(Bd.CreateOr(sum, part), body);
    Bd.CreateBr(head);
    PHINode *ret = PHINode::Create(L->getType(), 2, "gicc.split.result");
    ret->insertBefore(tail->begin());
    L->replaceAllUsesWith(ret);
    ret->addIncoming(sum, head);
    ret->addIncoming(L, elseT->getParent());
    return true;
}

// ompx_fence, and ompx_quiet with an ompx_barrier after it in the same
// block and nothing between but benign calls, become ompx__fence_defer.
unsigned deferFences(Module &M) {
    LLVMContext &C = M.getContext();
    FunctionCallee defer = M.getOrInsertFunction(kDefer, Type::getVoidTy(C));
    unsigned n = 0;
    SmallVector<CallInst *, 8> fences, quiets;
    for (Function &F : M)
        for (Instruction &I : instructions(F))
            if (auto *CI = dyn_cast<CallInst>(&I)) {
                StringRef callee = calleeName(*CI);
                if (callee == "ompx_fence") fences.push_back(CI);
                else if (callee == "ompx_quiet_host") quiets.push_back(CI);
            }
    for (CallInst *CI : fences) {
        CallInst::Create(defer, "", CI->getIterator());
        CI->eraseFromParent();
        ++n;
    }
    for (CallInst *Q : quiets) {
        CallInst *barrier = nullptr;
        for (Instruction *I = Q->getNextNode(); I; I = I->getNextNode()) {
            auto *CB = dyn_cast<CallBase>(I);
            if (!CB || isa<IntrinsicInst>(CB)) continue;
            if (calleeName(*CB) == "ompx_barrier") barrier = cast<CallInst>(CB);
            else if (CB->getCalledFunction() && benignCall(*CB->getCalledFunction())) continue;
            break;
        }
        if (!barrier) continue;
        CallInst::Create(defer, "", Q->getIterator());
        Q->eraseFromParent();
        barrier->eraseFromParent();
        ++n;
    }
    return n;
}

// ompx__fence_flush before every call that may see what a deferred fence
// orders or wait on another rank, and before each return to code outside
// the unit.
unsigned insertFlushes(Module &M) {
    LLVMContext &C = M.getContext();
    FunctionCallee flush = M.getOrInsertFunction(kFlush, Type::getVoidTy(C));
    unsigned n = 0;
    for (Function &F : M) {
        if (F.isDeclaration()) continue;
        SmallVector<Instruction *, 16> at;
        for (Instruction &I : instructions(F)) {
            if (auto *CB = dyn_cast<CallBase>(&I)) {
                if (isa<IntrinsicInst>(CB) || CB->hasMetadata(kSplitMD)) continue;
                const Function *G = CB->getCalledFunction();
                if (G && !G->isDeclaration()) continue;   // flushes itself where it must
                if (!G && callsDefinedOnly(CB->getCalledOperand())) continue;
                if (G && (G->getName().starts_with("ompx_") || benignCall(*G))) continue;
                at.push_back(CB);
            } else if (isa<ReturnInst>(I) &&
                       (!F.hasLocalLinkage() || hasAddressTaken(F) || F.getName() == "main")) {
                at.push_back(&I);
            }
        }
        for (Instruction *I : at) {
            CallInst::Create(flush, "", I->getIterator());
            ++n;
            if (std::getenv("GICC_SPLIT_DUMP")) {
                auto *CB = dyn_cast<CallBase>(I);
                out() << "[gicc-split] flush in " << F.getName() << " before "
                      << (CB ? calleeName(*CB) : StringRef("return")) << "\n";
            }
        }
    }
    return n;
}

bool runHost(Module &M, ModuleAnalysisManager &MAM) {
    if (!M.getFunction(kLaunch)) return false;
    StringMap<Record> recs = readRecords();
    if (recs.empty()) return false;
    auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
    SmallVector<std::pair<CallInst *, std::string>, 8> launches;
    for (Function &F : M)
        for (Instruction &I : instructions(F))
            if (auto *CI = dyn_cast<CallInst>(&I); CI && calleeName(*CI) == kLaunch)
                launches.push_back({CI, launchedKernelOf(CI)});
    // A fence is deferred only past kernels that cannot communicate.
    std::string block;
    if (sharesHostMemory(M)) block = "the unit requires unified shared memory";
    for (auto &[L, kernel] : launches) {
        if (!block.empty()) break;
        auto it = recs.find(kernel);
        if (it == recs.end()) block = "kernel " + kernel + " has no record (rebuild?)";
        else if (it->second.comm) block = "kernel " + kernel + " may communicate itself";
    }
    if (!block.empty()) {
        out() << "[gicc-split] no fence deferred: " << block << "\n";
        return false;
    }
    unsigned rewritten = 0;
    for (auto &[L, kernel] : launches) {
        const Record &r = recs.find(kernel)->second;
        if (!r.split) continue;
        auto &DT = FAM.getResult<DominatorTreeAnalysis>(*L->getFunction());
        std::string why;
        if (rewriteLaunch(M, L, kernel, *r.split, DT, why)) {
            FAM.invalidate(*L->getFunction(), PreservedAnalyses::none());
            out() << "[gicc-split] " << kernel << ": launch split around a deferred fence\n";
            if (std::getenv("GICC_SPLIT_DUMP")) L->getFunction()->print(out());
            ++rewritten;
        } else {
            out() << "[gicc-split] " << kernel << ": launch not split: " << why << "\n";
        }
    }
    if (!rewritten) return false;
    const unsigned fences = deferFences(M);
    const unsigned flushes = insertFlushes(M);
    out() << "[gicc-split] " << fences << " fence(s) deferred, " << flushes
          << " flush point(s)\n";
    return true;
}

}  // namespace

PreservedAnalyses GICCSplitLaunchPass::run(Module &M, ModuleAnalysisManager &MAM) {
    const Config &cfg = getConfig();
    if (cfg.mode != Mode::ChunkLower || !cfg.metaDirSet) return PreservedAnalyses::all();
    const Triple T(M.getTargetTriple());
    const bool changed = T.isNVPTX() || T.isAMDGPU() ? runDevice(M, MAM) : runHost(M, MAM);
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

}  // namespace gicc::pass
