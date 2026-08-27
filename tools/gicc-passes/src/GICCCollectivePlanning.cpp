#include "GICCCollectivePlanning.h"

#include "GICCPassConfig.h"
#include "HostMirrorAnnotation.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

namespace gicc::pass {
namespace {

constexpr StringLiteral AnchorPrefix = "gicc.collective.anchor.v1";
constexpr StringLiteral CandidatePrefix = "gicc.collective.candidate.v1";
constexpr StringLiteral InventorySchema = "gicc-collective-inventory-v1";
constexpr StringLiteral HintSchema = "gicc-collective-hint-v1";

using Fields = std::map<std::string, std::string>;

struct CatalogEntry {
    Function *function = nullptr;
    Fields fields;
    std::string catalogId;
};

struct AnchorEntry {
    Function *function = nullptr;
    Fields fields;
    std::string anchorId;
    unsigned countArg = 0;
    uint64_t elementBytes = 0;
    std::optional<unsigned> ppnArg;
    std::vector<uint64_t> thresholds;
};

struct Opportunity {
    CallInst *call = nullptr;
    AnchorEntry *anchor = nullptr;
    std::string opportunityId;
    unsigned callOrdinal = 0;
    unsigned loopDepth = 0;
    uint64_t arithmeticBefore = 0;
    uint64_t arithmeticAfter = 0;
    std::vector<CatalogEntry *> candidates;
};

struct Rule {
    std::optional<uint64_t> maxBytes;
    Function *function = nullptr;
    std::string targetId;
};

struct Selection {
    enum class Kind { Uniform, SizePolicy } kind = Kind::Uniform;
    std::string candidateId;
    std::vector<Rule> rules;
};

// The ROCm opt/clang binaries used on Tioga are statically linked and do not
// export llvm::SHA256.  Keep this tiny, self-contained SHA-256 here so the
// plugin can independently recompute content IDs without linking a second copy
// of LLVMSupport into the process.
class LocalSHA256 {
public:
    void update(StringRef input) {
        for (unsigned char byte : input.bytes()) {
            buffer_[bufferSize_++] = byte;
            if (bufferSize_ == 64) {
                transform(buffer_);
                bitLength_ += 512;
                bufferSize_ = 0;
            }
        }
    }

    std::array<uint8_t, 32> final() {
        uint64_t totalBits = bitLength_ + static_cast<uint64_t>(bufferSize_) * 8;
        buffer_[bufferSize_++] = 0x80;
        if (bufferSize_ > 56) {
            while (bufferSize_ < 64) buffer_[bufferSize_++] = 0;
            transform(buffer_);
            bufferSize_ = 0;
        }
        while (bufferSize_ < 56) buffer_[bufferSize_++] = 0;
        for (int shift = 56; shift >= 0; shift -= 8)
            buffer_[bufferSize_++] = static_cast<uint8_t>(totalBits >> shift);
        transform(buffer_);
        std::array<uint8_t, 32> output{};
        for (unsigned i = 0; i < 8; ++i) {
            output[4 * i] = static_cast<uint8_t>(state_[i] >> 24);
            output[4 * i + 1] = static_cast<uint8_t>(state_[i] >> 16);
            output[4 * i + 2] = static_cast<uint8_t>(state_[i] >> 8);
            output[4 * i + 3] = static_cast<uint8_t>(state_[i]);
        }
        return output;
    }

private:
    static uint32_t rotate(uint32_t value, unsigned amount) {
        return (value >> amount) | (value << (32 - amount));
    }

    void transform(const uint8_t block[64]) {
        static constexpr uint32_t K[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
            0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
            0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
            0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
            0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        };
        uint32_t words[64];
        for (unsigned i = 0; i < 16; ++i) {
            words[i] = (static_cast<uint32_t>(block[4 * i]) << 24) |
                       (static_cast<uint32_t>(block[4 * i + 1]) << 16) |
                       (static_cast<uint32_t>(block[4 * i + 2]) << 8) |
                       static_cast<uint32_t>(block[4 * i + 3]);
        }
        for (unsigned i = 16; i < 64; ++i) {
            uint32_t s0 = rotate(words[i - 15], 7) ^
                          rotate(words[i - 15], 18) ^ (words[i - 15] >> 3);
            uint32_t s1 = rotate(words[i - 2], 17) ^
                          rotate(words[i - 2], 19) ^ (words[i - 2] >> 10);
            words[i] = words[i - 16] + s0 + words[i - 7] + s1;
        }
        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (unsigned i = 0; i < 64; ++i) {
            uint32_t sum1 = rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25);
            uint32_t choose = (e & f) ^ (~e & g);
            uint32_t temp1 = h + sum1 + choose + K[i] + words[i];
            uint32_t sum0 = rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22);
            uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = sum0 + majority;
            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    uint8_t buffer_[64]{};
    unsigned bufferSize_ = 0;
    uint64_t bitLength_ = 0;
    uint32_t state_[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    };
};

std::string digestId(StringRef kind, const std::vector<std::string> &parts) {
    LocalSHA256 hash;
    hash.update(kind);
    for (const std::string &part : parts) {
        hash.update(StringRef("\0", 1));
        hash.update(part);
    }
    const auto bytes = hash.final();
    static constexpr char Hex[] = "0123456789abcdef";
    std::string encoded;
    encoded.reserve(24);
    for (unsigned i = 0; i < 12; ++i) {
        encoded.push_back(Hex[bytes[i] >> 4]);
        encoded.push_back(Hex[bytes[i] & 0x0f]);
    }
    return (kind + ":" + encoded).str();
}

std::vector<std::string> fieldParts(const Fields &fields) {
    std::vector<std::string> parts;
    parts.reserve(fields.size());
    for (const auto &[key, value] : fields)
        parts.push_back(key + "=" + value);
    return parts;
}

bool parseUnsigned(StringRef text, uint64_t &out) {
    if (text.empty() || text.getAsInteger(10, out)) return false;
    return true;
}

std::optional<Fields> parseAnnotation(StringRef annotation,
                                      StringRef prefix) {
    if (annotation == prefix) return Fields{};
    if (!annotation.starts_with(prefix) ||
        annotation.size() <= prefix.size() ||
        annotation[prefix.size()] != ';')
        return std::nullopt;
    Fields result;
    StringRef rest = annotation.drop_front(prefix.size() + 1);
    while (!rest.empty()) {
        auto split = rest.split(';');
        StringRef item = split.first;
        rest = split.second;
        auto kv = item.split('=');
        if (kv.first.empty() || kv.second.empty() ||
            kv.second.contains('=') || result.count(kv.first.str()))
            return std::nullopt;
        result[kv.first.str()] = kv.second.str();
    }
    return result;
}

bool hasCoreContract(const Fields &fields) {
    auto family = fields.find("family");
    auto contract = fields.find("contract");
    return family != fields.end() && !family->second.empty() &&
           contract != fields.end() && !contract->second.empty();
}

std::vector<uint64_t> parseThresholds(const Fields &fields, bool &ok) {
    ok = true;
    std::vector<uint64_t> result;
    auto it = fields.find("thresholds_bytes");
    if (it == fields.end() || it->second.empty()) return result;
    StringRef rest(it->second);
    while (!rest.empty()) {
        auto split = rest.split(',');
        uint64_t value = 0;
        if (!parseUnsigned(split.first, value) || value == 0 ||
            (!result.empty() && value <= result.back())) {
            ok = false;
            return {};
        }
        result.push_back(value);
        rest = split.second;
    }
    return result;
}

bool parseAnchor(Function &function, StringRef annotation, AnchorEntry &out) {
    auto fields = parseAnnotation(annotation, AnchorPrefix);
    if (!fields || !hasCoreContract(*fields)) return false;
    auto count = fields->find("count_arg");
    auto width = fields->find("element_bytes");
    uint64_t countArg = 0, elementBytes = 0;
    if (count == fields->end() || width == fields->end() ||
        !parseUnsigned(count->second, countArg) ||
        !parseUnsigned(width->second, elementBytes) || elementBytes == 0 ||
        countArg >= function.arg_size())
        return false;
    if (!function.getFunctionType()->getParamType(countArg)->isIntegerTy())
        return false;
    std::optional<unsigned> ppnArg;
    if (auto ppn = fields->find("ppn_arg"); ppn != fields->end()) {
        uint64_t value = 0;
        if (!parseUnsigned(ppn->second, value) || value >= function.arg_size() ||
            !function.getFunctionType()->getParamType(value)->isIntegerTy())
            return false;
        ppnArg = static_cast<unsigned>(value);
    }
    bool thresholdsOk = false;
    std::vector<uint64_t> thresholds = parseThresholds(*fields, thresholdsOk);
    if (!thresholdsOk || thresholds.size() > 8) return false;
    out.function = &function;
    out.fields = std::move(*fields);
    out.countArg = static_cast<unsigned>(countArg);
    out.elementBytes = elementBytes;
    out.ppnArg = ppnArg;
    out.thresholds = std::move(thresholds);
    out.anchorId = digestId("anchor", fieldParts(out.fields));
    return true;
}

bool parseCandidate(Function &function, StringRef annotation,
                    CatalogEntry &out) {
    auto fields = parseAnnotation(annotation, CandidatePrefix);
    if (!fields || !hasCoreContract(*fields) ||
        !fields->count("algorithm") || fields->at("algorithm").empty())
        return false;
    out.function = &function;
    out.fields = std::move(*fields);
    out.catalogId = digestId("catalog", fieldParts(out.fields));
    return true;
}

std::string contractKey(const Fields &fields) {
    return fields.at("family") + std::string("\0", 1) + fields.at("contract");
}

std::optional<uint64_t> constantUnsigned(const Value *value) {
    const auto *constant = dyn_cast<ConstantInt>(value);
    if (!constant || constant->isNegative() || constant->getBitWidth() > 64)
        return std::nullopt;
    return constant->getZExtValue();
}

bool isArithmetic(const Instruction &instruction) {
    return instruction.isBinaryOp() || isa<CmpInst>(instruction) ||
           isa<SelectInst>(instruction) || isa<GetElementPtrInst>(instruction);
}

void computeContext(Opportunity &opportunity) {
    Function *parent = opportunity.call->getFunction();
    DominatorTree DT(*parent);
    LoopInfo LI(DT);
    opportunity.loopDepth = LI.getLoopDepth(opportunity.call->getParent());
    bool before = true;
    for (Instruction &instruction : instructions(*parent)) {
        if (&instruction == opportunity.call) {
            before = false;
            continue;
        }
        if (!isArithmetic(instruction)) continue;
        if (before) ++opportunity.arithmeticBefore;
        else        ++opportunity.arithmeticAfter;
    }
}

std::vector<Opportunity> discover(
    Module &module, std::vector<AnchorEntry> &anchors,
    std::vector<CatalogEntry> &catalog) {
    std::map<Function *, AnchorEntry *> anchorByFunction;
    std::map<std::string, std::vector<CatalogEntry *>> candidatesByContract;
    std::set<std::string> catalogIds;
    for (Function &function : module) {
        for (const std::string &annotation : getFunctionAnnotations(function)) {
            AnchorEntry anchor;
            if (parseAnchor(function, annotation, anchor)) {
                anchors.push_back(std::move(anchor));
                anchorByFunction[&function] = &anchors.back();
                continue;
            }
            CatalogEntry candidate;
            if (parseCandidate(function, annotation, candidate)) {
                if (!catalogIds.insert(candidate.catalogId).second) {
                    errs() << "gicc: duplicate collective catalog ID "
                           << candidate.catalogId << "\n";
                    continue;
                }
                catalog.push_back(std::move(candidate));
            }
        }
    }
    // Vectors can reallocate while discovering annotations. Rebuild pointer
    // maps only after their storage is final.
    anchorByFunction.clear();
    for (AnchorEntry &anchor : anchors)
        anchorByFunction[anchor.function] = &anchor;
    for (CatalogEntry &candidate : catalog)
        candidatesByContract[contractKey(candidate.fields)].push_back(&candidate);

    std::vector<Opportunity> opportunities;
    for (Function &parent : module) {
        unsigned ordinal = 0;
        for (BasicBlock &block : parent) {
            for (Instruction &instruction : block) {
                auto *call = dyn_cast<CallInst>(&instruction);
                if (!call || call->isInlineAsm()) continue;
                Function *callee = call->getCalledFunction();
                auto anchorIt = anchorByFunction.find(callee);
                if (anchorIt == anchorByFunction.end()) continue;
                AnchorEntry *anchor = anchorIt->second;
                Opportunity opportunity;
                opportunity.call = call;
                opportunity.anchor = anchor;
                opportunity.callOrdinal = ordinal++;
                opportunity.opportunityId = digestId(
                    "opportunity",
                    {anchor->fields.at("family"),
                     anchor->fields.at("contract"), parent.getName().str(),
                     std::to_string(opportunity.callOrdinal)});
                auto candidatesIt = candidatesByContract.find(
                    contractKey(anchor->fields));
                if (candidatesIt != candidatesByContract.end()) {
                    for (CatalogEntry *candidate : candidatesIt->second) {
                        if (candidate->function != anchor->function &&
                            candidate->function->getFunctionType() ==
                                anchor->function->getFunctionType() &&
                            candidate->function->getReturnType()->isVoidTy())
                            opportunity.candidates.push_back(candidate);
                    }
                }
                std::sort(opportunity.candidates.begin(),
                          opportunity.candidates.end(),
                          [](const CatalogEntry *left,
                             const CatalogEntry *right) {
                              return left->catalogId < right->catalogId;
                          });
                computeContext(opportunity);
                opportunities.push_back(std::move(opportunity));
            }
        }
    }
    return opportunities;
}

json::Value fieldsJSON(const Fields &fields) {
    json::Object result;
    for (const auto &[key, value] : fields) result[key] = value;
    return json::Value(std::move(result));
}

json::Value valueFact(const Value *value) {
    json::Object result;
    if (auto constant = constantUnsigned(value)) {
        result["kind"] = "constant";
        result["value"] = *constant;
    } else if (const auto *argument = dyn_cast<Argument>(value)) {
        result["kind"] = "function_argument";
        result["argument_index"] = static_cast<int64_t>(argument->getArgNo());
    } else {
        result["kind"] = "dynamic_ir_expression";
    }
    return json::Value(std::move(result));
}

json::Value inventoryJSON(const Module &module,
                          const std::vector<Opportunity> &opportunities) {
    json::Array records;
    for (const Opportunity &opportunity : opportunities) {
        const AnchorEntry &anchor = *opportunity.anchor;
        json::Object record;
        record["opportunity_id"] = opportunity.opportunityId;
        record["family"] = anchor.fields.at("family");
        record["contract"] = anchor.fields.at("contract");
        record["anchor_id"] = anchor.anchorId;
        record["anchor_descriptor"] = fieldsJSON(anchor.fields);

        json::Object callFacts;
        callFacts["count"] = valueFact(
            opportunity.call->getArgOperand(anchor.countArg));
        callFacts["element_bytes"] = anchor.elementBytes;
        if (auto count = constantUnsigned(
                opportunity.call->getArgOperand(anchor.countArg));
            count && *count <= std::numeric_limits<uint64_t>::max() /
                                  anchor.elementBytes)
            callFacts["message_bytes"] = *count * anchor.elementBytes;
        else
            callFacts["message_bytes"] = nullptr;
        if (anchor.ppnArg)
            callFacts["ranks_per_node"] = valueFact(
                opportunity.call->getArgOperand(*anchor.ppnArg));
        else
            callFacts["ranks_per_node"] = nullptr;
        callFacts["loop_depth"] =
            static_cast<int64_t>(opportunity.loopDepth);
        callFacts["arithmetic_before"] = opportunity.arithmeticBefore;
        callFacts["arithmetic_after"] = opportunity.arithmeticAfter;
        callFacts["direct_call"] = true;
        callFacts["void_return"] = true;
        record["call_facts"] = std::move(callFacts);

        json::Array thresholds;
        for (uint64_t threshold : anchor.thresholds)
            thresholds.push_back(threshold);
        record["compiler_policy_thresholds_bytes"] = std::move(thresholds);

        json::Array candidates;
        for (const CatalogEntry *candidate : opportunity.candidates) {
            json::Object item;
            item["catalog_id"] = candidate->catalogId;
            item["descriptor"] = fieldsJSON(candidate->fields);
            item["compiler_legality"] = json::Object{
                {"same_family", true},
                {"same_semantic_contract", true},
                {"exact_function_type", true},
                {"void_call_materializer", true},
            };
            candidates.push_back(std::move(item));
        }
        record["candidates"] = std::move(candidates);
        records.push_back(std::move(record));
    }
    json::Object root;
    root["schema_version"] = InventorySchema;
    root["compiler_only"] = true;
    root["target_triple"] = module.getTargetTriple();
    root["source_visible"] = false;
    root["opportunities"] = std::move(records);
    return json::Value(std::move(root));
}

bool writeInventory(StringRef path, const json::Value &inventory) {
    std::error_code error;
    raw_fd_ostream output(path, error, sys::fs::OF_Text);
    if (error) return false;
    output << formatv("{0:2}", inventory) << "\n";
    return true;
}

Function *targetById(const Opportunity &opportunity, StringRef id) {
    if (opportunity.anchor->anchorId == id)
        return opportunity.anchor->function;
    for (CatalogEntry *candidate : opportunity.candidates)
        if (candidate->catalogId == id) return candidate->function;
    return nullptr;
}

std::string uniformPlanId(const Opportunity &opportunity,
                          StringRef catalogId) {
    return digestId("candidate",
                    {opportunity.opportunityId, "uniform", catalogId.str()});
}

std::string policyPlanId(const Opportunity &opportunity,
                         const std::vector<Rule> &rules) {
    std::vector<std::string> parts{
        opportunity.opportunityId, "size_policy",
        std::to_string(opportunity.anchor->elementBytes)};
    for (const Rule &rule : rules) {
        std::string bound = rule.maxBytes ? std::to_string(*rule.maxBytes) : "*";
        parts.push_back(bound + ":" + rule.targetId);
    }
    return digestId("candidate", parts);
}

bool hasOnlyKeys(const json::Object &object,
                 std::initializer_list<StringRef> allowed) {
    std::set<std::string> allowedSet;
    for (StringRef key : allowed) allowedSet.insert(key.str());
    for (const auto &item : object)
        if (!allowedSet.count(item.first.str())) return false;
    return true;
}

bool parseSelection(const json::Object &object,
                    const Opportunity &opportunity, Selection &selection) {
    auto kind = object.getString("kind");
    auto candidateId = object.getString("candidate_id");
    if (!kind || !candidateId) return false;
    selection.candidateId = candidateId->str();
    if (*kind == "uniform") {
        if (!hasOnlyKeys(object, {"kind", "candidate_id", "target_id"}))
            return false;
        auto targetId = object.getString("target_id");
        if (!targetId) return false;
        Function *target = targetById(opportunity, *targetId);
        if (!target || selection.candidateId !=
                           uniformPlanId(opportunity, *targetId))
            return false;
        selection.kind = Selection::Kind::Uniform;
        selection.rules = {{std::nullopt, target, targetId->str()}};
        return true;
    }
    if (*kind != "size_policy" ||
        !hasOnlyKeys(object, {"kind", "candidate_id", "rules"}))
        return false;
    const json::Array *ruleValues = object.getArray("rules");
    if (!ruleValues || ruleValues->size() !=
                           opportunity.anchor->thresholds.size() + 1)
        return false;
    selection.kind = Selection::Kind::SizePolicy;
    for (size_t index = 0; index < ruleValues->size(); ++index) {
        const json::Object *ruleObject = (*ruleValues)[index].getAsObject();
        if (!ruleObject ||
            !hasOnlyKeys(*ruleObject, {"max_bytes", "target_id"}))
            return false;
        auto targetId = ruleObject->getString("target_id");
        const json::Value *maxValue = ruleObject->get("max_bytes");
        if (!targetId || !maxValue) return false;
        Function *target = targetById(opportunity, *targetId);
        if (!target) return false;
        Rule rule;
        rule.function = target;
        rule.targetId = targetId->str();
        if (index < opportunity.anchor->thresholds.size()) {
            auto maximum = maxValue->getAsUINT64();
            if (!maximum || *maximum != opportunity.anchor->thresholds[index])
                return false;
            rule.maxBytes = *maximum;
        } else if (!maxValue->getAsNull()) {
            return false;
        }
        selection.rules.push_back(rule);
    }
    return selection.candidateId == policyPlanId(opportunity, selection.rules);
}

bool readSelections(StringRef path,
                    const std::vector<Opportunity> &opportunities,
                    std::map<std::string, Selection> &selections) {
    auto buffer = MemoryBuffer::getFile(path);
    if (!buffer) return false;
    auto parsed = json::parse((*buffer)->getBuffer());
    if (!parsed) {
        consumeError(parsed.takeError());
        return false;
    }
    const json::Object *root = parsed->getAsObject();
    if (!root || !hasOnlyKeys(*root, {"version", "schema_version",
                                     "selections", "llm_metadata"}))
        return false;
    auto version = root->getInteger("version");
    auto schema = root->getString("schema_version");
    const json::Object *values = root->getObject("selections");
    if (!version || *version != 1 || !schema || *schema != HintSchema ||
        !values || values->size() != opportunities.size())
        return false;
    std::map<std::string, const Opportunity *> byId;
    for (const Opportunity &opportunity : opportunities)
        byId[opportunity.opportunityId] = &opportunity;
    for (const auto &item : *values) {
        auto found = byId.find(item.first.str());
        const json::Object *object = item.second.getAsObject();
        if (found == byId.end() || !object) return false;
        Selection selection;
        if (!parseSelection(*object, *found->second, selection)) return false;
        selections[item.first.str()] = std::move(selection);
    }
    return selections.size() == opportunities.size();
}

CallInst *createCandidateCall(IRBuilder<> &builder, const CallInst &oldCall,
                              Function &candidate,
                              ArrayRef<Value *> arguments,
                              StringRef candidateId, StringRef targetId) {
    CallInst *call = builder.CreateCall(candidate.getFunctionType(), &candidate,
                                        arguments);
    call->setCallingConv(candidate.getCallingConv());
    call->setDebugLoc(oldCall.getDebugLoc());
    LLVMContext &context = call->getContext();
    call->setMetadata("gicc.collective.candidate_id", MDNode::get(
        context, MDString::get(context, candidateId)));
    call->setMetadata("gicc.collective.target_id", MDNode::get(
        context, MDString::get(context, targetId)));
    return call;
}

bool materializeUniform(Opportunity &opportunity,
                        const Selection &selection) {
    Function *target = selection.rules.front().function;
    opportunity.call->setCalledFunction(target);
    opportunity.call->setCallingConv(target->getCallingConv());
    LLVMContext &context = opportunity.call->getContext();
    opportunity.call->setMetadata("gicc.collective.candidate_id", MDNode::get(
        context, MDString::get(context, selection.candidateId)));
    opportunity.call->setMetadata("gicc.collective.target_id", MDNode::get(
        context, MDString::get(
            context, selection.rules.front().targetId)));
    return true;
}

bool materializePolicy(Opportunity &opportunity,
                       const Selection &selection) {
    CallInst *oldCall = opportunity.call;
    if (!oldCall->getType()->isVoidTy() || oldCall->isTailCall() ||
        oldCall->hasOperandBundles())
        return false;
    const unsigned countArg = opportunity.anchor->countArg;
    if (countArg >= oldCall->arg_size() ||
        !oldCall->getArgOperand(countArg)->getType()->isIntegerTy())
        return false;

    SmallVector<Value *, 16> arguments(oldCall->args());
    BasicBlock *original = oldCall->getParent();
    Function *parent = original->getParent();
    BasicBlock *continuation = original->splitBasicBlock(
        oldCall->getIterator(), "gicc.collective.cont");
    original->getTerminator()->eraseFromParent();

    IRBuilder<> initial(original);
    Value *count = arguments[countArg];
    Value *count64 = initial.CreateZExtOrTrunc(count, initial.getInt64Ty(),
                                               "gicc.collective.count");
    Value *bytes = initial.CreateMul(
        count64, initial.getInt64(opportunity.anchor->elementBytes),
        "gicc.collective.bytes");

    BasicBlock *testBlock = original;
    for (size_t index = 0; index + 1 < selection.rules.size(); ++index) {
        const Rule &rule = selection.rules[index];
        BasicBlock *callBlock = BasicBlock::Create(
            parent->getContext(), "gicc.collective.bin", parent,
            continuation);
        BasicBlock *nextTest = BasicBlock::Create(
            parent->getContext(), "gicc.collective.test", parent,
            continuation);
        IRBuilder<> testBuilder(testBlock);
        Value *condition = testBuilder.CreateICmpULE(
            bytes, testBuilder.getInt64(*rule.maxBytes));
        testBuilder.CreateCondBr(condition, callBlock, nextTest);
        IRBuilder<> callBuilder(callBlock);
        createCandidateCall(callBuilder, *oldCall, *rule.function,
                            arguments, selection.candidateId, rule.targetId);
        callBuilder.CreateBr(continuation);
        testBlock = nextTest;
    }
    IRBuilder<> fallbackBuilder(testBlock);
    createCandidateCall(fallbackBuilder, *oldCall,
                        *selection.rules.back().function, arguments,
                        selection.candidateId,
                        selection.rules.back().targetId);
    fallbackBuilder.CreateBr(continuation);
    oldCall->eraseFromParent();
    return true;
}

}  // namespace

PreservedAnalyses GICCCollectivePlanningPass::run(
    Module &module, ModuleAnalysisManager &) {
    const Config &config = getConfig();
    if (config.mode == Mode::Passthrough) return PreservedAnalyses::all();
    Triple triple(module.getTargetTriple());
    if (triple.isAMDGPU() || triple.isNVPTX())
        return PreservedAnalyses::all();

    std::vector<AnchorEntry> anchors;
    std::vector<CatalogEntry> catalog;
    // Reserve before annotation discovery so pointers installed by discover()
    // remain stable even for moderately large compiler catalogs.
    anchors.reserve(32);
    catalog.reserve(128);
    std::vector<Opportunity> opportunities =
        discover(module, anchors, catalog);
    if (opportunities.empty()) return PreservedAnalyses::all();

    if (!config.collectiveOut.empty()) {
        if (!writeInventory(config.collectiveOut,
                            inventoryJSON(module, opportunities)))
            errs() << "gicc: cannot write collective inventory "
                   << config.collectiveOut << "\n";
    }
    if (config.mode != Mode::Lower || config.collectiveHintIn.empty())
        return PreservedAnalyses::all();

    std::map<std::string, Selection> selections;
    if (!readSelections(config.collectiveHintIn, opportunities, selections)) {
        errs() << "gicc: rejected collective hint " << config.collectiveHintIn
               << "; preserving compiler anchor calls\n";
        return PreservedAnalyses::all();
    }

    bool changed = false;
    // Size policies split blocks, invalidating iterators but not the CallInst
    // pointers of other opportunities. Each original call is transformed once.
    for (Opportunity &opportunity : opportunities) {
        const Selection &selection = selections.at(opportunity.opportunityId);
        bool applied = selection.kind == Selection::Kind::Uniform
                           ? materializeUniform(opportunity, selection)
                           : materializePolicy(opportunity, selection);
        if (!applied) {
            errs() << "gicc: collective plan failed final IR checks for "
                   << opportunity.opportunityId << "; preserving anchor\n";
            continue;
        }
        changed = true;
    }
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

}  // namespace gicc::pass
