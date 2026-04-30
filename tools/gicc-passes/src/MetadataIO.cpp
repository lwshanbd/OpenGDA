#include "MetadataIO.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace gicc::pass {

namespace {

const char *argRefKindStr(ArgRef::Kind k) {
    switch (k) {
        case ArgRef::Kind::Param:    return "param";
        case ArgRef::Kind::ConstI64: return "const_i64";
        case ArgRef::Kind::BinOp:    return "binop";
        case ArgRef::Kind::Cast:     return "cast";
        case ArgRef::Kind::Derived:  return "derived";
        case ArgRef::Kind::LoopIv:   return "loop_iv";
    }
    return "derived";
}

bool parseArgRefKind(StringRef s, ArgRef::Kind &out) {
    if (s == "param")     { out = ArgRef::Kind::Param;    return true; }
    if (s == "const_i64") { out = ArgRef::Kind::ConstI64; return true; }
    if (s == "binop")     { out = ArgRef::Kind::BinOp;    return true; }
    if (s == "cast")      { out = ArgRef::Kind::Cast;     return true; }
    if (s == "derived")   { out = ArgRef::Kind::Derived;  return true; }
    if (s == "loop_iv")   { out = ArgRef::Kind::LoopIv;   return true; }
    return false;
}

const char *guardKindStr(GuardSpec::Kind k) {
    switch (k) {
        case GuardSpec::Kind::Always:        return "always";
        case GuardSpec::Kind::ParamTruthy:   return "param_truthy";
        case GuardSpec::Kind::ParamEqConst:  return "param_eq_const";
        case GuardSpec::Kind::BinOp:         return "binop";
        case GuardSpec::Kind::Unknown:       return "unknown";
    }
    return "unknown";
}

bool parseGuardKind(StringRef s, GuardSpec::Kind &out) {
    if (s == "always")          { out = GuardSpec::Kind::Always;       return true; }
    if (s == "param_truthy")    { out = GuardSpec::Kind::ParamTruthy;  return true; }
    if (s == "param_eq_const")  { out = GuardSpec::Kind::ParamEqConst; return true; }
    if (s == "binop")           { out = GuardSpec::Kind::BinOp;        return true; }
    if (s == "unknown")         { out = GuardSpec::Kind::Unknown;      return true; }
    return false;
}

json::Value argRefToJSON(const ArgRef &a) {
    json::Object o;
    o["kind"] = argRefKindStr(a.kind);
    switch (a.kind) {
        case ArgRef::Kind::Param:
            o["param"] = static_cast<int64_t>(a.paramIdx);
            break;
        case ArgRef::Kind::ConstI64:
            o["value"] = a.constVal;
            break;
        case ArgRef::Kind::BinOp:
        case ArgRef::Kind::Cast: {
            o["op"] = a.opStr;
            json::Array children;
            for (const auto &c : a.children) children.push_back(argRefToJSON(c));
            o["children"] = std::move(children);
            break;
        }
        case ArgRef::Kind::Derived:
        case ArgRef::Kind::LoopIv:
            break;
    }
    return json::Value(std::move(o));
}

bool argRefFromJSON(const json::Value &v, ArgRef &out) {
    const auto *o = v.getAsObject();
    if (!o) return false;
    auto kindStr = o->getString("kind");
    if (!kindStr) return false;
    if (!parseArgRefKind(*kindStr, out.kind)) return false;
    switch (out.kind) {
        case ArgRef::Kind::Param: {
            auto p = o->getInteger("param");
            if (!p) return false;
            out.paramIdx = static_cast<unsigned>(*p);
            break;
        }
        case ArgRef::Kind::ConstI64: {
            auto v2 = o->getInteger("value");
            if (!v2) return false;
            out.constVal = *v2;
            break;
        }
        case ArgRef::Kind::BinOp:
        case ArgRef::Kind::Cast: {
            auto op = o->getString("op");
            if (!op) return false;
            out.opStr = op->str();
            const auto *children = o->getArray("children");
            if (children) {
                for (const auto &c : *children) {
                    ArgRef child;
                    if (!argRefFromJSON(c, child)) return false;
                    out.children.push_back(std::move(child));
                }
            }
            break;
        }
        case ArgRef::Kind::Derived:
        case ArgRef::Kind::LoopIv:
            break;
    }
    return true;
}

json::Value guardToJSON(const GuardSpec &g) {
    json::Object o;
    o["kind"] = guardKindStr(g.kind);
    if (g.kind == GuardSpec::Kind::ParamTruthy ||
        g.kind == GuardSpec::Kind::ParamEqConst) {
        o["param"] = static_cast<int64_t>(g.paramIdx);
    }
    if (g.kind == GuardSpec::Kind::ParamEqConst) {
        o["value"] = g.constVal;
    }
    return json::Value(std::move(o));
}

bool guardFromJSON(const json::Value &v, GuardSpec &out) {
    const auto *o = v.getAsObject();
    if (!o) return false;
    auto kindStr = o->getString("kind");
    if (!kindStr) return false;
    if (!parseGuardKind(*kindStr, out.kind)) return false;
    if (out.kind == GuardSpec::Kind::ParamTruthy ||
        out.kind == GuardSpec::Kind::ParamEqConst) {
        auto p = o->getInteger("param");
        if (!p) return false;
        out.paramIdx = static_cast<unsigned>(*p);
    }
    if (out.kind == GuardSpec::Kind::ParamEqConst) {
        auto v2 = o->getInteger("value");
        if (!v2) return false;
        out.constVal = *v2;
    }
    return true;
}

json::Value loopToJSON(const OpLoopInfo &L) {
    json::Object o;
    o["in_loop"]  = L.inLoop;
    if (L.inLoop) {
        if (L.ivBoundKnown) {
            o["iv_param"] = static_cast<int64_t>(L.ivParamIdx);
        }
        o["iv_start"] = L.ivStart;
        o["iv_step"]  = L.ivStep;
        if (L.degraded) o["degraded"] = true;
    }
    return json::Value(std::move(o));
}

bool loopFromJSON(const json::Value &v, OpLoopInfo &out) {
    const auto *o = v.getAsObject();
    if (!o) return false;
    if (auto b = o->getBoolean("in_loop")) out.inLoop = *b;
    else                                    out.inLoop = false;
    if (!out.inLoop) return true;
    if (auto p = o->getInteger("iv_param")) {
        out.ivParamIdx   = static_cast<unsigned>(*p);
        out.ivBoundKnown = true;
    }
    if (auto s = o->getInteger("iv_start")) out.ivStart = *s;
    if (auto s = o->getInteger("iv_step"))  out.ivStep  = *s;
    if (auto d = o->getBoolean("degraded")) out.degraded = *d;
    return true;
}

json::Value templateToJSON(const KernelTemplate &t) {
    json::Object root;
    root["version"]         = 1;
    root["kernel_mangled"]  = t.mangledName;
    root["kernel_simple"]   = t.simpleName;

    json::Array params;
    for (size_t i = 0; i < t.params.size(); ++i) {
        json::Object p;
        p["idx"]  = static_cast<int64_t>(i);
        p["name"] = t.params[i].name;
        p["type"] = t.params[i].typeStr;
        params.push_back(std::move(p));
    }
    root["params"] = std::move(params);

    json::Array ops;
    for (const auto &op : t.ops) {
        json::Object o;
        o["site_id"] = op.siteId;
        o["kind"]    = op.kind;
        o["guard"]   = guardToJSON(op.guard);
        if (op.loop.inLoop) o["loop"] = loopToJSON(op.loop);

        json::Object args;
        for (const auto &kv : op.args) {
            args[kv.first] = argRefToJSON(kv.second);
        }
        o["args"] = std::move(args);
        ops.push_back(std::move(o));
    }
    root["ops"] = std::move(ops);
    return json::Value(std::move(root));
}

bool templateFromJSON(const json::Value &v, KernelTemplate &out) {
    const auto *o = v.getAsObject();
    if (!o) return false;
    auto mangled = o->getString("kernel_mangled");
    if (!mangled) return false;
    out.mangledName = mangled->str();
    if (auto simple = o->getString("kernel_simple"))
        out.simpleName = simple->str();

    out.params.clear();
    if (const auto *params = o->getArray("params")) {
        for (const auto &pv : *params) {
            const auto *po = pv.getAsObject();
            if (!po) continue;
            ParamInfo info;
            if (auto n = po->getString("name")) info.name    = n->str();
            if (auto t = po->getString("type")) info.typeStr = t->str();
            out.params.push_back(std::move(info));
        }
    }

    out.ops.clear();
    if (const auto *ops = o->getArray("ops")) {
        for (const auto &ov : *ops) {
            const auto *oo = ov.getAsObject();
            if (!oo) continue;
            OpTemplate op;
            if (auto s = oo->getString("site_id")) op.siteId = s->str();
            if (auto k = oo->getString("kind"))    op.kind   = k->str();
            if (const auto *g = oo->get("guard")) {
                if (!guardFromJSON(*g, op.guard)) return false;
            }
            if (const auto *l = oo->get("loop")) {
                if (!loopFromJSON(*l, op.loop)) return false;
            }
            if (const auto *a = oo->getObject("args")) {
                for (const auto &kv : *a) {
                    ArgRef ref;
                    if (!argRefFromJSON(kv.second, ref)) return false;
                    op.args[kv.first.str()] = std::move(ref);
                }
            }
            out.ops.push_back(std::move(op));
        }
    }
    return true;
}

}  // namespace

bool writeKernelTemplate(const std::string &metaDir, const KernelTemplate &t) {
    if (auto ec = sys::fs::create_directories(metaDir)) return false;

    SmallString<256> path(metaDir);
    sys::path::append(path, t.mangledName + ".json");

    std::error_code ec;
    raw_fd_ostream out(path, ec, sys::fs::OF_Text);
    if (ec) return false;

    json::Value root = templateToJSON(t);
    out << formatv("{0:2}", root) << "\n";
    return !out.has_error();
}

bool readKernelTemplate(const std::string &metaDir,
                        const std::string &mangledName,
                        KernelTemplate    &out) {
    SmallString<256> path(metaDir);
    sys::path::append(path, mangledName + ".json");

    auto bufOrErr = MemoryBuffer::getFile(path);
    if (!bufOrErr) return false;
    auto parsed = json::parse((*bufOrErr)->getBuffer());
    if (!parsed) {
        consumeError(parsed.takeError());
        return false;
    }
    return templateFromJSON(*parsed, out);
}

}  // namespace gicc::pass
