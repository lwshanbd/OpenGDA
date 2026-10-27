#include "TuneConfig.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

using namespace llvm;

namespace gicc::pass::tune {
namespace {

struct Choices {
    StringMap<SiteChoice> sites;
    StringMap<uint64_t>   chunks;
};

[[noreturn]] void bad(const std::string &path, const std::string &why) {
    report_fatal_error(Twine("GICC_TUNE_IN ") + path + ": " + why);
}

Choices load() {
    Choices c;
    const char *path = std::getenv("GICC_TUNE_IN");
    if (!path || !*path) return c;
    auto buf = MemoryBuffer::getFile(path);
    if (!buf) bad(path, "cannot be read");
    Expected<json::Value> v = json::parse((*buf)->getBuffer());
    if (!v) bad(path, toString(v.takeError()));
    const json::Object *top = v->getAsObject();
    if (!top) bad(path, "is not a JSON object");
    if (const json::Object *sites = top->getObject("sites"))
        for (const auto &[id, val] : *sites) {
            const json::Object *o = val.getAsObject();
            if (!o) bad(path, "site " + id.str() + " is not an object");
            SiteChoice s;
            if (auto p = o->get("pipeline")) {
                if (!p->getAsBoolean()) bad(path, "site " + id.str() + ": pipeline must be a bool");
                s.pipeline = *p->getAsBoolean();
            }
            if (auto g = o->getString("grain")) {
                if (*g == "element") s.grain = ChunkGrain::Element;
                else if (*g == "block") s.grain = ChunkGrain::Block;
                else bad(path, "site " + id.str() + ": grain must be 'element' or 'block'");
            }
            c.sites[id] = s;
        }
    if (const json::Object *kernels = top->getObject("kernels"))
        for (const auto &[id, val] : *kernels) {
            const json::Object *o = val.getAsObject();
            auto n = o ? o->getInteger("dist_chunk") : std::nullopt;
            if (!n || *n < 0) bad(path, "kernel " + id.str() + ": dist_chunk must be a count");
            c.chunks[id] = static_cast<uint64_t>(*n);
        }
    return c;
}

const Choices &choices() {
    static const Choices c = load();
    return c;
}

}  // namespace

SiteChoice site(StringRef id) {
    // A site's own entry wins field by field over the "*" entry.
    const auto &all = choices().sites;
    SiteChoice s;
    if (auto it = all.find("*"); it != all.end()) s = it->second;
    if (auto it = all.find(id); it != all.end()) {
        if (it->second.pipeline) s.pipeline = it->second.pipeline;
        if (it->second.grain) s.grain = it->second.grain;
    }
    return s;
}

std::optional<uint64_t> distChunk(StringRef kernel) {
    auto it = choices().chunks.find(kernel);
    if (it == choices().chunks.end()) return std::nullopt;
    return it->second;
}

bool anyDistChunk() { return !choices().chunks.empty(); }

void record(json::Object rec) {
    const char *path = std::getenv("GICC_SITES_OUT");
    if (!path || !*path) return;
    std::error_code ec;
    raw_fd_ostream os(path, ec, sys::fs::OF_Append | sys::fs::OF_Text);
    if (ec) report_fatal_error(Twine("GICC_SITES_OUT ") + path + ": " + ec.message());
    os << json::Value(std::move(rec)) << "\n";
}

}  // namespace gicc::pass::tune
