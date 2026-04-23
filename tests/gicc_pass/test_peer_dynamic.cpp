// Toy kernel: peer is a runtime value (function arg), size is a literal.
//   peer_class: no ConstantInt among {peer=arg} → dynamic
//   size_class: 4 MiB → large_const
//   freq_class: hot loop
// On policy_tioga.json the policy's rules are ordered with rule 10 first
// (k_from_topology_hint); Dynamic matches any predicate, so rule 10 fires.
// Expected: rule_id=10, path=ofi_triggered.
namespace gicc {
struct DeviceCtx {};
void put(DeviceCtx* ctx, void* src, void* dst, unsigned long size, int peer);
}

int sink;

void test_peer_dynamic(int runtime_peer) {
    gicc::DeviceCtx ctx;
    for (int i = 0; i < 1000; i++) {
        gicc::put(&ctx, (void*)0, (void*)0, (1UL << 22), runtime_peer);
        sink += i;
    }
}
