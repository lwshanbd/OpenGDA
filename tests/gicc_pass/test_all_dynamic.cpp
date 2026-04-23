// Toy kernel: every selector feature is runtime-dynamic.
//   peer_class: function-arg peer → dynamic
//   size_class: no ConstantInt operand → dynamic
//   freq_class: irregular loop without a statically-resolvable trip count
//               (upper bound is itself runtime) → dynamic
// On policy_tioga.json Dynamic wildcards match the first rule (rule 10).
// The test asserts the pass handles all-dynamic input safely (doesn't crash)
// and emits a decision — not strictly the catch-all unless the rule list
// is authored to put a conservative rule ahead; that is a policy choice,
// not a pass correctness issue.
namespace gicc {
struct DeviceCtx {};
void put(DeviceCtx* ctx, void* src, void* dst, unsigned long size, int peer);
}

int sink;

void test_all_dynamic(int peer, unsigned long size, int niter) {
    gicc::DeviceCtx ctx;
    for (int i = 0; i < niter; i++) {
        gicc::put(&ctx, (void*)0, (void*)0, size, peer);
        sink += i;
    }
}
