/**
 * mini_json.hpp — Single-header recursive-descent JSON parser.
 *
 * Scope: exactly what the M2 selector's policy files need.
 *   - null, bool, number (stored as double; callers can cast),
 *   - string (no \u escapes beyond ASCII),
 *   - array, object.
 *   - UTF-8 pass-through inside strings; no re-encoding.
 *
 * Not RFC-8259 strict. Does not support trailing commas, comments, or Unicode
 * escapes outside \" \\ \/ \n \r \t \b \f. Sufficient for hand-authored
 * policy_<platform>.json and emitter-written calibration JSONs.
 */
#pragma once

#include <cctype>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace gicc {
namespace mini_json {

class Value;
using Array  = std::vector<Value>;
using Object = std::map<std::string, Value>;

class Value {
public:
    using Storage = std::variant<std::nullptr_t, bool, double, std::string, Array, Object>;

    Value() : storage_(nullptr) {}
    Value(std::nullptr_t) : storage_(nullptr) {}
    Value(bool b) : storage_(b) {}
    Value(double d) : storage_(d) {}
    Value(int i) : storage_((double)i) {}
    Value(std::string s) : storage_(std::move(s)) {}
    Value(Array a) : storage_(std::move(a)) {}
    Value(Object o) : storage_(std::move(o)) {}

    bool is_null()   const { return std::holds_alternative<std::nullptr_t>(storage_); }
    bool is_bool()   const { return std::holds_alternative<bool>(storage_); }
    bool is_number() const { return std::holds_alternative<double>(storage_); }
    bool is_string() const { return std::holds_alternative<std::string>(storage_); }
    bool is_array()  const { return std::holds_alternative<Array>(storage_); }
    bool is_object() const { return std::holds_alternative<Object>(storage_); }

    bool        as_bool()   const { return std::get<bool>(storage_); }
    double      as_double() const { return std::get<double>(storage_); }
    int         as_int()    const { return (int)std::get<double>(storage_); }
    const std::string& as_string() const { return std::get<std::string>(storage_); }
    const Array&  as_array()  const { return std::get<Array>(storage_); }
    const Object& as_object() const { return std::get<Object>(storage_); }

    // Object-key accessor that returns a pointer (null if missing) rather than throwing.
    const Value* find(const std::string& key) const {
        if (!is_object()) return nullptr;
        const auto& o = std::get<Object>(storage_);
        auto it = o.find(key);
        return it == o.end() ? nullptr : &it->second;
    }

private:
    Storage storage_;
};

struct ParseError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class Parser {
public:
    explicit Parser(const std::string& src) : s_(src), p_(0) {}

    Value parse() {
        skip_ws();
        Value v = parse_value();
        skip_ws();
        if (p_ != s_.size()) throw ParseError("trailing garbage at offset " + std::to_string(p_));
        return v;
    }

private:
    void skip_ws() {
        while (p_ < s_.size() && (s_[p_]==' '||s_[p_]=='\t'||s_[p_]=='\n'||s_[p_]=='\r')) ++p_;
    }

    bool match(char c) {
        skip_ws();
        if (p_ < s_.size() && s_[p_] == c) { ++p_; return true; }
        return false;
    }

    void expect(char c) {
        if (!match(c)) throw ParseError(std::string("expected '") + c + "' at offset " + std::to_string(p_));
    }

    Value parse_value() {
        skip_ws();
        if (p_ >= s_.size()) throw ParseError("unexpected eof");
        char c = s_[p_];
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return Value(parse_string());
        if (c == 't' || c == 'f') return Value(parse_bool());
        if (c == 'n') { consume_literal("null"); return Value(nullptr); }
        if (c == '-' || (c >= '0' && c <= '9')) return Value(parse_number());
        throw ParseError(std::string("unexpected char '") + c + "' at offset " + std::to_string(p_));
    }

    Value parse_object() {
        expect('{');
        Object out;
        skip_ws();
        if (match('}')) return Value(std::move(out));
        while (true) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            expect(':');
            Value v = parse_value();
            out.emplace(std::move(key), std::move(v));
            skip_ws();
            if (match('}')) return Value(std::move(out));
            expect(',');
        }
    }

    Value parse_array() {
        expect('[');
        Array out;
        skip_ws();
        if (match(']')) return Value(std::move(out));
        while (true) {
            out.emplace_back(parse_value());
            skip_ws();
            if (match(']')) return Value(std::move(out));
            expect(',');
        }
    }

    std::string parse_string() {
        skip_ws();
        expect('"');
        std::string out;
        while (p_ < s_.size()) {
            char c = s_[p_++];
            if (c == '"') return out;
            if (c == '\\') {
                if (p_ >= s_.size()) throw ParseError("unterminated escape");
                char esc = s_[p_++];
                switch (esc) {
                    case '"':  out.push_back('"');  break;
                    case '\\': out.push_back('\\'); break;
                    case '/':  out.push_back('/');  break;
                    case 'n':  out.push_back('\n'); break;
                    case 'r':  out.push_back('\r'); break;
                    case 't':  out.push_back('\t'); break;
                    case 'b':  out.push_back('\b'); break;
                    case 'f':  out.push_back('\f'); break;
                    default:   throw ParseError(std::string("bad escape \\") + esc);
                }
            } else {
                out.push_back(c);
            }
        }
        throw ParseError("unterminated string");
    }

    bool parse_bool() {
        if (s_.compare(p_, 4, "true")  == 0) { p_ += 4; return true; }
        if (s_.compare(p_, 5, "false") == 0) { p_ += 5; return false; }
        throw ParseError("expected true/false at offset " + std::to_string(p_));
    }

    void consume_literal(const char* lit) {
        size_t n = std::char_traits<char>::length(lit);
        if (s_.compare(p_, n, lit) != 0)
            throw ParseError(std::string("expected '") + lit + "' at offset " + std::to_string(p_));
        p_ += n;
    }

    double parse_number() {
        size_t start = p_;
        if (s_[p_] == '-') ++p_;
        while (p_ < s_.size() && (std::isdigit((unsigned char)s_[p_])
               || s_[p_]=='.' || s_[p_]=='e' || s_[p_]=='E'
               || s_[p_]=='+' || s_[p_]=='-')) ++p_;
        std::string tok = s_.substr(start, p_ - start);
        if (tok.empty()) throw ParseError("empty number");
        try {
            return std::stod(tok);
        } catch (const std::exception&) {
            throw ParseError("bad number literal: " + tok);
        }
    }

    const std::string& s_;
    size_t p_;
};

inline Value parse(const std::string& src) {
    Parser p(src);
    return p.parse();
}

inline Value parse_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw ParseError("cannot open '" + path + "'");
    std::stringstream ss;
    ss << f.rdbuf();
    return parse(ss.str());
}

}  // namespace mini_json
}  // namespace gicc
