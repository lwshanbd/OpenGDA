#!/usr/bin/env python3
"""Prepare instrumented Minimod sources in an isolated build copy.

The authoritative Minimod worktree is never edited.  ``build.sh`` copies it
under build_ofi and this script performs two exact, fail-loud rewrites there:

* standard: add route counters to the existing serial/overlap source;
* o5: additionally replace the scalar halo arguments with a two-element
  descriptor table and compile byte-identical mirrored/unmirrored kernels.

The O5 pair differs only by the host-mirror function annotation.  Both modes
live in one executable and are selected at runtime, so source input, messages,
topology, and runtime state are held constant.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED = {
    "target_3d.cpp": "228b7efe7039fb2a4abb9a4d38009dc24dce7d60722735300d9a7021e6096c1d",
    "data_setup.cpp": "caab6c3f8dde09d2e7bcf883d49ff58b44a8efbaf33845ac30f35c6649e6dfa1",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def add_route_counter(text: str) -> str:
    old = '''    printf("CHECKSUM rank %d = %016llx\\n", rank, hash);

    *min_u = _min_u;
'''
    new = '''    printf("CHECKSUM rank %d = %016llx\\n", rank, hash);
    printf("GICC_ROUTE rank %d staged=%llu pushed=%llu\\n", rank,
           (unsigned long long)gicc_rt->staged_ops(),
           (unsigned long long)gicc_rt->proxy_pushes());

    *min_u = _min_u;
'''
    return replace_once(text, old, new, "route counter")


O5_KERNELS = r'''// One descriptor element.  The host creates two immutable tables, one for
// each ping-pong wavefield buffer, and registers a host mirror for each.
struct GiccMinimodHaloDesc {
    int target;
    int buf;
    size_t dst_off;
    size_t src_off;
    size_t bytes;
    // Null means the transfer is active.  A non-null sentinel means skip;
    // this is the compiler-supported host-mirrored guard shape used by ASF.
    const void* peer_recv_addr;
};

static GiccMinimodHaloDesc h_desc_u[2], h_desc_v[2];
static GiccMinimodHaloDesc *d_desc_u = nullptr, *d_desc_v = nullptr;

// The bodies of these two kernels are intentionally byte-for-byte identical.
// The annotation is the only semantic input that differs: it lets the host
// trace read desc[i] before launch and therefore makes IPC/DWQ legal.
__global__ void GICC_KERNEL_HOST_MIRROR(desc)
GICC_KERNEL_HOST_MIRROR_PARAM(1)
halo_kernel_mirrored(gicc::DeviceCtx* ctx,
                     const GiccMinimodHaloDesc* desc)
{
    for (int i = 0; i < 2; ++i) {
        if (desc[i].peer_recv_addr != nullptr) continue;
        gicc::put(ctx, desc[i].target, desc[i].buf, desc[i].dst_off,
                  desc[i].buf, desc[i].src_off, desc[i].bytes);
    }
    gicc::flush(ctx);
}

__global__ void halo_kernel_unmirrored(gicc::DeviceCtx* ctx,
                                       const GiccMinimodHaloDesc* desc)
{
    for (int i = 0; i < 2; ++i) {
        if (desc[i].peer_recv_addr != nullptr) continue;
        gicc::put(ctx, desc[i].target, desc[i].buf, desc[i].dst_off,
                  desc[i].buf, desc[i].src_off, desc[i].bytes);
    }
    gicc::flush(ctx);
}

extern "C" void gicc_minimod_desc_init() {
    const int left_target = (ngpu + rank - 1) % ngpu;
    const int right_target = (rank + 1) % ngpu;

    auto fill = [&](GiccMinimodHaloDesc* table, int buf) {
        table[0] = {left_target, buf, left_neighbor_dst_offset,
                    my_left_src_offset, halo_size,
                    rank == 0 ? reinterpret_cast<const void*>(1) : nullptr};
        table[1] = {right_target, buf, right_neighbor_dst_offset,
                    my_right_src_offset, halo_size,
                    rank == ngpu - 1 ? reinterpret_cast<const void*>(1) : nullptr};
    };
    fill(h_desc_u, buf_u_idx);
    fill(h_desc_v, buf_v_idx);

    HIP_CHECK(hipMalloc(&d_desc_u, sizeof(h_desc_u)));
    HIP_CHECK(hipMalloc(&d_desc_v, sizeof(h_desc_v)));
    HIP_CHECK(hipMemcpy(d_desc_u, h_desc_u, sizeof(h_desc_u),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_desc_v, h_desc_v, sizeof(h_desc_v),
                        hipMemcpyHostToDevice));
    gicc_rt->register_host_mirror(d_desc_u, h_desc_u,
                                  sizeof(GiccMinimodHaloDesc), 2);
    gicc_rt->register_host_mirror(d_desc_v, h_desc_v,
                                  sizeof(GiccMinimodHaloDesc), 2);
}

extern "C" void gicc_minimod_desc_finalize() {
    if (d_desc_u) HIP_CHECK(hipFree(d_desc_u));
    if (d_desc_v) HIP_CHECK(hipFree(d_desc_v));
    d_desc_u = d_desc_v = nullptr;
}
'''


def make_o5_target(text: str) -> str:
    include_anchor = '#include "gicc/gicc.hpp"\n'
    text = replace_once(
        text,
        include_anchor,
        include_anchor + '#include "gicc/platform/ofi/annotations.hpp"\n',
        "O5 annotation include",
    )
    start = text.index("__global__ void halo_kernel(")
    marker = "// =============================================================================\n// Main target_3d function"
    end = text.index(marker, start)
    text = text[:start] + O5_KERNELS + "\n" + text[end:]

    old = '''static void issue_halo() {
    int left_target  = (ngpu + rank - 1) % ngpu;
    int right_target = (rank + 1) % ngpu;
    int has_left  = (rank != 0)        ? 1 : 0;
    int has_right = (rank != ngpu - 1) ? 1 : 0;
    // Which buffer holds v this iteration (the two ping-pong).
    int my_buf = (d_v == buf_v) ? buf_v_idx : buf_u_idx;

    gicc::launch<halo_kernel>(*gicc_rt, dim3(64), dim3(256),
        left_target, right_target,
        my_buf,
        my_left_src_offset,  left_neighbor_dst_offset,
        my_right_src_offset, right_neighbor_dst_offset,
        halo_size,
        has_left, has_right);
}
'''
    new = '''static bool gicc_use_unmirrored() {
    static int cached = -1;
    if (cached < 0) {
        const char* e = getenv("GICC_MINIMOD_O5_MODE");
        cached = (e && strcmp(e, "unmirrored") == 0) ? 1 : 0;
    }
    return cached != 0;
}

static void issue_halo() {
    GiccMinimodHaloDesc* desc = (d_v == buf_v) ? d_desc_v : d_desc_u;
    if (gicc_use_unmirrored()) {
        gicc::launch<halo_kernel_unmirrored>(*gicc_rt, dim3(64), dim3(256),
                                              desc);
    } else {
        gicc::launch<halo_kernel_mirrored>(*gicc_rt, dim3(64), dim3(256),
                                            desc);
    }
}
'''
    return replace_once(text, old, new, "O5 issue_halo")


def make_o5_setup(text: str) -> str:
    include_anchor = '#include "gicc/platform/ofi/internal/gpu_device_context.hpp"\n'
    includes = include_anchor + '''
extern "C" void gicc_minimod_desc_init();
extern "C" void gicc_minimod_desc_finalize();
'''
    text = replace_once(text, include_anchor, includes, "O5 declarations")

    old_mode = '''    const char* dwq_env = getenv("GICC_DWQ_MODE");
    bool dwq_mode = (dwq_env == nullptr) || (atoi(dwq_env) != 0);
    if (dwq_mode) {
        gicc_rt->enable_host_wait_mode();
        if (gicc_rt->rank() == 0)
            fprintf(stderr, "[minimod-unified] transport = DWQ (GPU trigger)\\n");
    } else {
        if (gicc_rt->rank() == 0)
            fprintf(stderr, "[minimod-unified] transport = CPU proxy\\n");
    }
'''
    new_mode = '''    // O5 contains one trigger/IPC kernel and one proxy-only kernel in the
    // same executable. Keep both engines live so the runtime-mode comparison
    // changes only which byte-identical kernel body is launched.
    gicc_rt->enable_host_wait_mode();
    gicc_rt->enable_mixed_dispatch();
    if (gicc_rt->rank() == 0)
        fprintf(stderr, "[minimod-o5] mixed dispatch enabled\\n");
'''
    text = replace_once(text, old_mode, new_mode, "O5 runtime mode")

    init_anchor = '''    right_neighbor_dst_offset = (size_t)(lx - 4) * stride * sizeof(float);

    MPI_Barrier(MPI_COMM_WORLD);
'''
    init_new = '''    right_neighbor_dst_offset = (size_t)(lx - 4) * stride * sizeof(float);

    gicc_minimod_desc_init();
    MPI_Barrier(MPI_COMM_WORLD);
'''
    text = replace_once(text, init_anchor, init_new, "O5 descriptor init")

    fini_anchor = '''    delete gicc_rt;
    gicc_rt = nullptr;
'''
    fini_new = '''    gicc_minimod_desc_finalize();
    delete gicc_rt;
    gicc_rt = nullptr;
'''
    return replace_once(text, fini_anchor, fini_new, "O5 descriptor finalize")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--mode", choices=("standard", "o5"), required=True)
    ap.add_argument("--allow-source-drift", action="store_true")
    args = ap.parse_args()

    target_dir = args.work / "targets" / "hip_gicc_unified"
    target = target_dir / "target_3d.cpp"
    setup = target_dir / "data_setup.cpp"
    for path in (target, setup):
        expected = EXPECTED[path.name]
        actual = sha256(path)
        if actual != expected and not args.allow_source_drift:
            raise SystemExit(
                f"refusing source drift in {path}: {actual} != {expected}")

    target_text = target.read_text()
    setup_text = add_route_counter(setup.read_text())
    if args.mode == "o5":
        target_text = make_o5_target(target_text)
        setup_text = make_o5_setup(setup_text)

    target.write_text(target_text)
    setup.write_text(setup_text)
    print(f"prepared {args.mode} sources under {args.work}")


if __name__ == "__main__":
    main()
