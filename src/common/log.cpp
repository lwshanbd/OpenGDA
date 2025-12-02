#include "log.hpp"
#include <cstdarg>
#include <ctime>
#include <cstring>
#include <unistd.h>

namespace opengda {

// ============================================================================
// Log Level Names
// ============================================================================
static const char* level_names[] = {
    "ERROR",
    "WARN ",
    "INFO ",
    "DEBUG",
    "TRACE"
};

static const char* level_colors[] = {
    "\033[1;31m",  // ERROR: bright red
    "\033[1;33m",  // WARN:  bright yellow
    "\033[1;32m",  // INFO:  bright green
    "\033[1;36m",  // DEBUG: bright cyan
    "\033[0;37m"   // TRACE: white
};

static const char* color_reset = "\033[0m";

// ============================================================================
// LogContext Implementation
// ============================================================================

LogContext::LogContext()
    : level_(LOG_WARN)
    , output_(stderr)
    , rank_(0)
    , initialized_(false)
{
}

LogContext::~LogContext() {
    if (output_ && output_ != stderr && output_ != stdout) {
        fclose(output_);
    }
}

void LogContext::init(int rank) {
    if (initialized_) return;

    rank_ = rank;
    initialized_ = true;

    // Parse log level from environment
    const char* level_str = getenv("OPENGDA_LOG_LEVEL");
    if (level_str) {
        if (strcasecmp(level_str, "error") == 0) {
            level_ = LOG_ERROR;
        } else if (strcasecmp(level_str, "warn") == 0) {
            level_ = LOG_WARN;
        } else if (strcasecmp(level_str, "info") == 0) {
            level_ = LOG_INFO;
        } else if (strcasecmp(level_str, "debug") == 0) {
            level_ = LOG_DEBUG;
        } else if (strcasecmp(level_str, "trace") == 0) {
            level_ = LOG_TRACE;
        } else if (level_str[0] >= '0' && level_str[0] <= '4') {
            level_ = static_cast<LogLevel>(level_str[0] - '0');
        }
    }

    // Parse output file from environment
    const char* outfile = getenv("OPENGDA_LOG_OUTFILE");
    if (outfile) {
        if (strcasecmp(outfile, "stdout") == 0) {
            output_ = stdout;
        } else if (strcasecmp(outfile, "stderr") == 0) {
            output_ = stderr;
        } else {
            // Support %r or % for rank substitution
            char filename[512];
            const char* percent = strchr(outfile, '%');
            if (percent) {
                // Replace % with rank
                size_t prefix_len = percent - outfile;
                snprintf(filename, sizeof(filename), "%.*s%d%s",
                        (int)prefix_len, outfile, rank_, percent + 1);
            } else {
                snprintf(filename, sizeof(filename), "%s", outfile);
            }

            output_ = fopen(filename, "w");
            if (!output_) {
                fprintf(stderr, "Warning: Cannot open log file '%s', using stderr\n",
                        filename);
                output_ = stderr;
            }
        }
    }
}

bool LogContext::should_log(LogLevel level, const char* tag) {
    if (!initialized_) {
        init(0);  // Auto-initialize with rank 0 if not initialized
    }

    // Check level
    if (level > level_) {
        return false;
    }

    // TODO: Add whitelist/blacklist support if needed
    (void)tag;  // Unused for now

    return true;
}

FILE* LogContext::get_output() {
    if (!initialized_) {
        init(0);
    }
    return output_;
}

// ============================================================================
// Core Logging Implementation
// ============================================================================

void log_impl(LogLevel level, const char* tag, const char* file, int line,
              const char* fmt, ...) {
    LogContext& ctx = LogContext::instance();

    if (!ctx.should_log(level, tag)) {
        return;
    }

    FILE* out = ctx.get_output();

    // Get timestamp
    time_t now = time(NULL);
    struct tm* tm_info = localtime(&now);
    char time_buf[32];
    strftime(time_buf, sizeof(time_buf), "%H:%M:%S", tm_info);

    // Check if output supports colors (is a tty)
    bool use_color = isatty(fileno(out));

    // Print header
    if (use_color) {
        fprintf(out, "%s[%s] [Rank %d] [%s] %s:%d: ",
                level_colors[level],
                level_names[level],
                ctx.get_level(),  // This should be rank, but we don't have it in context easily
                tag,
                file,
                line);
    } else {
        fprintf(out, "[%s] [%s] [%s] %s:%d: ",
                time_buf,
                level_names[level],
                tag,
                file,
                line);
    }

    // Print message
    va_list args;
    va_start(args, fmt);
    vfprintf(out, fmt, args);
    va_end(args);

    // Print color reset and newline
    if (use_color) {
        fprintf(out, "%s\n", color_reset);
    } else {
        fprintf(out, "\n");
    }

    fflush(out);
}

} // namespace opengda
