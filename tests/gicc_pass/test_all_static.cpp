// Toy kernel: every selector feature is statically resolvable.
//   peer_class: literal integer peer argument → k_const
//   size_class: literal byte size → bucketed (2 MiB → large_const)
//   freq_class: enclosing loop with static trip count 1000 → hot_loop
// Expected decision on policy_tioga.json: rule 20 (k_const + large_const)
// → path=ofi_proxy, pool_size=32.
namespace gicc {
struct DeviceCtx {};
void put(DeviceCtx* ctx, void* src, void* dst, unsigned long size, int peer);
void barrier();
}

int sink;

void test_all_static() {
    gicc::DeviceCtx ctx;
    for (int i = 0; i < 1000; i++) {
        gicc::put(&ctx, (void*)0, (void*)0, (1UL << 21), 42);
        gicc::barrier();
        sink += i;
    }
}
