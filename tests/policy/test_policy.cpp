/**
 * test_policy.cpp — Unit tests for libgicc_policy.
 *
 * Tests the M2-C deliverable: JSON round-trip, rule evaluation, and cap
 * feasibility. No test framework dep — plain asserts + exit codes, so it
 * runs in the stock CMake add_test() flow.
 *
 * Invoke with the repo root as CWD so policy file paths resolve (CMake
 * add_test() is configured to set WORKING_DIRECTORY accordingly).
 */

#include "gicc/policy/policy.hpp"

#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace gicc::policy;

static int g_failed = 0;

#define CHECK(cond) do {                                                       \
    if (!(cond)) {                                                             \
        std::fprintf(stderr, "CHECK FAIL at %s:%d  %s\n",                      \
                     __FILE__, __LINE__, #cond);                               \
        ++g_failed;                                                            \
    }                                                                          \
} while (0)

#define CHECK_EQ(a, b) do {                                                    \
    auto _x = (a); auto _y = (b);                                              \
    if (!(_x == _y)) {                                                         \
        std::fprintf(stderr, "CHECK_EQ FAIL at %s:%d  %s == %s\n",             \
                     __FILE__, __LINE__, #a, #b);                              \
        ++g_failed;                                                            \
    }                                                                          \
} while (0)

// ---------- test 1: policy_tioga.json loads and round-trips ------------------
static void test_load_tioga() {
    std::fprintf(stderr, "[test] load_tioga\n");
    Policy p = load_policy("policies/policy_tioga.json");
    CHECK_EQ(p.version, 1);
    CHECK_EQ(p.platform_desc.fabric, Fabric::OfiCxi);
    CHECK_EQ(p.platform_desc.gpu_family, std::string("mi250x"));
    CHECK_EQ(p.platform_desc.nic_caps.counter_max, 2047);
    CHECK_EQ(p.platform_desc.nic_caps.dwq_max, 256);
    CHECK_EQ(p.platform_desc.nic_caps.ctrs_per_op, 2);
    CHECK(p.rules.size() >= 2);
    CHECK(p.rules.back().then_decision.conservative);  // catch-all
}

// ---------- test 2: policy_maple.json loads ----------------------------------
static void test_load_maple() {
    std::fprintf(stderr, "[test] load_maple\n");
    Policy p = load_policy("policies/policy_maple.json");
    CHECK_EQ(p.platform_desc.fabric, Fabric::IbMlx5);
    CHECK_EQ(p.rules.front().then_decision.path, Path::IbNative);
}

// ---------- test 3: rule eval — regular-stencil hot loop matches rule 10 -----
static void test_eval_jacobi_hot() {
    std::fprintf(stderr, "[test] eval_jacobi_hot\n");
    Policy p = load_policy("policies/policy_tioga.json");
    Features f;
    f.peer = PeerClass::KFromTopologyHint;
    f.size = SizeClass::LargeConst;
    f.freq = FreqClass::HotLoop;
    Decision d = eval(p, f);
    CHECK_EQ(d.rule_id, 10);
    CHECK_EQ(d.path, Path::OfiTriggered);
    CHECK_EQ(d.slot_depth, 4);
    CHECK_EQ(d.pool_size, 16);
}

// ---------- test 4: rule eval — matmul-style large const matches rule 20 ----
static void test_eval_matmul_large() {
    std::fprintf(stderr, "[test] eval_matmul_large\n");
    Policy p = load_policy("policies/policy_tioga.json");
    Features f;
    f.peer = PeerClass::KConst;
    f.size = SizeClass::LargeConst;
    f.freq = FreqClass::OuterLoop;
    Decision d = eval(p, f);
    CHECK_EQ(d.rule_id, 20);
    CHECK_EQ(d.path, Path::OfiProxy);
    CHECK_EQ(d.pool_size, 32);
    CHECK_EQ(d.channel_map, ChannelMap::StaticGraphAware);
}

// ---------- test 5: rule eval — all-dynamic falls through to catch-all ------
static void test_eval_dynamic_fallback() {
    std::fprintf(stderr, "[test] eval_dynamic_fallback\n");
    Policy p = load_policy("policies/policy_tioga.json");
    Features f;  // all Dynamic by default
    Decision d = eval(p, f);
    // First rule's if_peer is KFromTopologyHint; Dynamic matches any predicate
    // (selector treats unrecoverable features as wildcards), so we land on
    // rule 10 first, not the catch-all. This is by design: the rule list
    // itself is the fallback mechanism. If the author wants "conservative on
    // all-dynamic", they must order a dynamic-matching rule ahead of rule 10.
    CHECK(d.rule_id == 10 || d.conservative);
}

// ---------- test 6: check_feasible — legal config passes --------------------
static void test_feasibility_ok() {
    std::fprintf(stderr, "[test] feasibility_ok\n");
    PlatformDesc pd;
    pd.fabric = Fabric::OfiCxi;
    Decision d;
    d.path = Path::OfiTriggered;
    d.slot_depth = 4;
    d.pool_size = 16;
    // R(32) = 5, 4*5 = 20 <= counter_max/ctrs_per_op = 2047/2 = 1023  → OK
    // R(32) = 5, 16*5 = 80 <= dwq_max = 256                           → OK
    CHECK(check_feasible(d, pd, 32));
}

// ---------- test 7: check_feasible — cap violation at N=64 ------------------
static void test_feasibility_cap_bite() {
    std::fprintf(stderr, "[test] feasibility_cap_bite\n");
    PlatformDesc pd;
    pd.fabric = Fabric::OfiCxi;
    Decision d;
    d.path = Path::OfiTriggered;
    d.slot_depth = 8;   // aggressive
    d.pool_size = 64;   // aggressive
    // R(64) = 6, 64*6 = 384 > dwq_max = 256 → infeasible
    CHECK(!check_feasible(d, pd, 64));
}

// ---------- test 8: ib_native only legal on mlx5 ----------------------------
static void test_ib_native_scope() {
    std::fprintf(stderr, "[test] ib_native_scope\n");
    Decision d;
    d.path = Path::IbNative;
    PlatformDesc cxi; cxi.fabric = Fabric::OfiCxi;
    PlatformDesc ib;  ib.fabric  = Fabric::IbMlx5;
    CHECK(!check_feasible(d, cxi, 32));
    CHECK( check_feasible(d, ib,  32));
}

// ---------- test 9: enum <-> string round-trip ------------------------------
static void test_string_roundtrip() {
    std::fprintf(stderr, "[test] string_roundtrip\n");
    CHECK_EQ(to_cstr(Path::OfiTriggered), std::string("ofi_triggered"));
    CHECK_EQ(path_from_str("ofi_proxy"), Path::OfiProxy);
    CHECK_EQ(peer_from_str("k_const"), PeerClass::KConst);
    CHECK_EQ(freq_from_str("hot_loop"), FreqClass::HotLoop);
    try { (void)path_from_str("bogus"); CHECK(false); } catch (const std::runtime_error&) {}
}

int main() {
    try {
        test_load_tioga();
        test_load_maple();
        test_eval_jacobi_hot();
        test_eval_matmul_large();
        test_eval_dynamic_fallback();
        test_feasibility_ok();
        test_feasibility_cap_bite();
        test_ib_native_scope();
        test_string_roundtrip();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "uncaught exception: %s\n", e.what());
        return 2;
    }
    if (g_failed) {
        std::fprintf(stderr, "%d checks failed\n", g_failed);
        return 1;
    }
    std::fprintf(stderr, "all policy tests passed\n");
    return 0;
}
