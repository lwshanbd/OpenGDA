#ifndef OPENGDA_LOG_HPP
#define OPENGDA_LOG_HPP

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace opengda {

// ============================================================================
// Log Levels
// ============================================================================
enum LogLevel {
    LOG_ERROR  = 0,  // Critical errors
    LOG_WARN   = 1,  // Warnings
    LOG_INFO   = 2,  // General information
    LOG_DEBUG  = 3,  // Debug information
    LOG_TRACE  = 4   // Detailed trace
};

// ============================================================================
// Log Context
// ============================================================================
class LogContext {
public:
    static LogContext& instance() {
        static LogContext ctx;
        return ctx;
    }

    void init(int rank);
    bool should_log(LogLevel level, const char* tag);
    FILE* get_output();

    LogLevel get_level() const { return level_; }

private:
    LogContext();
    ~LogContext();

    LogLevel level_;
    FILE* output_;
    int rank_;
    bool initialized_;
};

// ============================================================================
// Core Logging Function
// ============================================================================
void log_impl(LogLevel level, const char* tag, const char* file, int line,
              const char* fmt, ...) __attribute__((format(printf, 5, 6)));

} // namespace opengda

// ============================================================================
// Public Macros - Always compiled, runtime filtered
// ============================================================================

#define OPENGDA_Log(level, tag, fmt, ...)                                      \
    do {                                                                       \
        if (opengda::LogContext::instance().should_log(level, tag)) {         \
            opengda::log_impl(level, tag, __FILE__, __LINE__,                 \
                            fmt, ##__VA_ARGS__);                               \
        }                                                                      \
    } while (0)

// Convenience macros for specific levels
#define OPENGDA_Error(tag, fmt, ...)   OPENGDA_Log(opengda::LOG_ERROR, tag, fmt, ##__VA_ARGS__)
#define OPENGDA_Warn(tag, fmt, ...)    OPENGDA_Log(opengda::LOG_WARN, tag, fmt, ##__VA_ARGS__)
#define OPENGDA_Info(tag, fmt, ...)    OPENGDA_Log(opengda::LOG_INFO, tag, fmt, ##__VA_ARGS__)

// ============================================================================
// Debug Macros - Only compiled when DEBUG_OFI is ON
// ============================================================================

#ifdef DEBUG_OFI
    #define OPENGDA_DBG_Log(level, tag, fmt, ...) OPENGDA_Log(level, tag, fmt, ##__VA_ARGS__)
    #define OPENGDA_Debug(tag, fmt, ...)          OPENGDA_Log(opengda::LOG_DEBUG, tag, fmt, ##__VA_ARGS__)
    #define OPENGDA_Trace(tag, fmt, ...)          OPENGDA_Log(opengda::LOG_TRACE, tag, fmt, ##__VA_ARGS__)

    // Debug assertions
    #define OPENGDA_DBG_Assert(expr, fmt, ...)                                 \
        do {                                                                   \
            if (!(expr)) {                                                     \
                fprintf(stderr, "[ASSERT FAILED] %s:%d: %s\n",                 \
                        __FILE__, __LINE__, #expr);                            \
                fprintf(stderr, fmt "\n", ##__VA_ARGS__);                      \
                abort();                                                       \
            }                                                                  \
        } while (0)
#else
    // No-op in release builds (zero overhead)
    #define OPENGDA_DBG_Log(level, tag, fmt, ...) do { (void)sizeof(fmt); } while (0)
    #define OPENGDA_Debug(tag, fmt, ...)          do { (void)sizeof(fmt); } while (0)
    #define OPENGDA_Trace(tag, fmt, ...)          do { (void)sizeof(fmt); } while (0)
    #define OPENGDA_DBG_Assert(expr, fmt, ...)    do { (void)sizeof(expr); } while (0)
#endif

// ============================================================================
// Production Assertions - Always compiled
// ============================================================================

#define OPENGDA_Assert(expr, fmt, ...)                                         \
    do {                                                                       \
        if (!(expr)) {                                                         \
            fprintf(stderr, "[ASSERT FAILED] %s:%d: %s\n",                     \
                    __FILE__, __LINE__, #expr);                                \
            fprintf(stderr, fmt "\n", ##__VA_ARGS__);                          \
            abort();                                                           \
        }                                                                      \
    } while (0)

#endif // OPENGDA_LOG_HPP
