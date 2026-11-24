// ompx_select.h - which of a kernel's device versions a launch runs.
//
// Internal to the GiOMP runtimes (ompx_ofi.cpp, ompx_ib.cpp). The
// gicc-passes plugin keeps a kernel the host puts after twice on the device
// (GICCAfterPut.h): the one rewritten to send those puts itself, and a plain
// copy as the compiler made it. Whether sending from the kernel pays cannot
// be told ahead: it hides the transfer behind the kernel's own work, but the
// rewritten kernel plans, counts and mirrors, and that has cost a kernel
// more than the transfer it hid. So the runtime measures. Each kernel is
// tried both ways over its first launches, a fixed number of pairs in an
// order that follows no period of the program's, and timed from one launch
// to the next: that takes in the kernel, the puts after it and the fence
// that waits for them. From then on it runs the way that was faster, and
// the plain copy unless another won by a margin -- so a program whose
// kernels gain nothing from sending runs as it would have without the
// rewrite, less the trial launches.
//
// Where the plugin could, there is a third way (version 2, "early"): the
// plain copy run in two ranges, first the iterations that write what the
// puts send, then, the puts made, the rest. It is tried with the others.
//
// The schedule is a function of the kernel and of how often it has been
// launched, nothing else, so every rank of an SPMD program tries the same
// version at the same launch: the ranks wait for one another at each fence,
// and a rank timing its own choice against the others' mixed ones would
// measure theirs. One kernel is tried at a time, in the order they first
// launch; the others run plain meanwhile.
//
// OMPX_PIPE_SELECT=1 always runs the sender (what a build without the plain
// copy does), 0 always the plain copy, 2 the early way where there is one
// (else plain); unset or "auto" measures. OMPX_PIPE_SELECT_PAIRS sets the
// number of trial rounds (default 5), each every version once.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace gicc_select {

struct Kernel {
    int state = 0;              // 0 waiting its turn, 1 on trial, 2 decided
    int choice = 0;             // once decided: 0 plain, 1 the sender, 2 early
    int versions = 2;           // 3 when the launch has the early way
    long long launches = 0;     // since its trial began
    int last = -1;              // what the previous launch ran, -1 if not timed
    double t_last = 0;
    long long waited = 0, tick_seen = 0;
    std::vector<double> t[3];   // seconds from a launch to the next, by version
    double median[3] = {0, 0, 0};
};

struct State {
    int mode = -1;              // OMPX_PIPE_SELECT: -1 measure, else the version
    int pairs = 5;
    bool read_env = false;
    unsigned long long on_trial = 0;   // the kernel being tried, 0 if none
    long long tick = 0;                // launches of the kernel on trial
    std::unordered_map<unsigned long long, Kernel> kernels;
    std::vector<unsigned long long> order;   // as first launched
};

inline State& state() {
    static State s;
    return s;
}

// The first launch of each version is not timed: it loads it.
// A waiting kernel takes the trial over from one not launched in this many
// of its own launches (a kernel of the set-up, launched once).
constexpr long long kStale = 32;
// The sender must be this much faster to be chosen.
constexpr double kMargin = 0.01;

inline double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

inline void decide(State& s, Kernel& k) {
    k.choice = 0;
    for (int v = 0; v < k.versions; ++v) k.median[v] = median(k.t[v]);
    for (int v = 1; v < k.versions && !k.t[0].empty(); ++v)
        if (!k.t[v].empty() && k.median[v] < k.median[k.choice] * (k.choice ? 1 : 1 - kMargin))
            k.choice = v;
    k.state = 2;
    k.last = -1;
    s.on_trial = 0;
}

// Which version this launch of `kernel` runs: 0 the plain copy, 1 the
// sender, 2 the early way (only when `early` says the launch has it).
// `armed` says a put is posted for the kernel to send; `now` is the time in
// seconds.
inline int select(unsigned long long kernel, bool armed, bool early, double now) {
    State& s = state();
    if (!s.read_env) {
        s.read_env = true;
        if (const char* e = std::getenv("OMPX_PIPE_SELECT"); e && *e && std::strcmp(e, "auto"))
            s.mode = *e == '0' ? 0 : *e == '2' ? 2 : 1;
        if (const char* e = std::getenv("OMPX_PIPE_SELECT_PAIRS"); e && std::atoi(e) > 0)
            s.pairs = std::atoi(e);
    }
    if (s.mode >= 0) return armed ? (s.mode == 2 && !early ? 0 : s.mode) : 0;
    auto found = s.kernels.find(kernel);
    if (found == s.kernels.end()) {
        found = s.kernels.emplace(kernel, Kernel{}).first;
        s.order.push_back(kernel);
    }
    Kernel& k = found->second;
    if (early) k.versions = 3;
    if (k.state == 2) return armed ? k.choice : 0;
    if (k.state == 0) {
        if (s.on_trial != 0) {
            if (k.tick_seen != s.tick) {
                k.tick_seen = s.tick;
                k.waited = 0;
            }
            if (++k.waited <= kStale) return 0;
            decide(s, s.kernels[s.on_trial]);   // on what it has, if anything
        }
        s.on_trial = kernel;
        k.state = 1;
    }
    ++s.tick;
    if (k.last >= 0) k.t[k.last].push_back(now - k.t_last);
    const long long n = k.launches++;
    k.t_last = now;
    const int V = k.versions;
    if (n >= (long long)V * (1 + s.pairs)) {
        decide(s, k);
        return armed ? k.choice : 0;
    }
    int v;
    if (n < V) {
        v = (int)n;
        k.last = -1;
    } else {
        // Round by round, every version once, in an order from a hash of
        // the kernel and the round: balanced, and in step with no period of
        // the program's.
        const long long j = n - V;
        unsigned long long h = kernel + 0x9e3779b97f4a7c15ull * (unsigned long long)(j / V + 1);
        h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ull;
        h = (h ^ (h >> 27)) * 0x94d049bb133111ebull;
        const int first = (int)((h >> 33) % (unsigned)V);
        const int step = V == 3 && ((h >> 20) & 1) ? 2 : 1;
        v = (first + step * (int)(j % V)) % V;
        k.last = v;
    }
    // A launch with nothing to send runs plain whatever the schedule says,
    // and counts for it all the same: the other ranks are on the schedule.
    return armed ? v : 0;
}

inline void report(int rank) {
    static const char* const names[3] = {"plain", "the sender", "early"};
    const State& s = state();
    for (unsigned long long id : s.order) {
        const Kernel& k = s.kernels.at(id);
        if (k.state != 2) {
            std::fprintf(stderr, "[giomp] rank %d: kernel %016llx: version not decided (%lld "
                         "trial launches)\n", rank, id, k.launches);
            continue;
        }
        std::fprintf(stderr, "[giomp] rank %d: kernel %016llx: runs %s; launch to launch, "
                     "median of %zu, us:", rank, id, names[k.choice], k.t[0].size());
        for (int v = 0; v < k.versions; ++v)
            std::fprintf(stderr, " %s %.1f", names[v], k.median[v] * 1e6);
        std::fprintf(stderr, "\n");
    }
}

}  // namespace gicc_select
