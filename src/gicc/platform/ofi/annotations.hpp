/**
 * annotations.hpp - LTO-pass-visible kernel annotations
 *
 * Macros here lower to `__attribute__((annotate("...")))`, which Clang
 * preserves into LLVM IR as `@llvm.global.annotations` entries. The
 * GICC LTO pass walks those entries to recognize kernels (and kernel
 * formals) that opt in to specific code-generation behaviors.
 *
 * `GICC_KERNEL_HOST_MIRROR(name)`
 *   Marks a kernel formal as carrying a host-side mirror. Required when
 *   the kernel does `gicc::put(ctx, formal[iv].field, ...)`: without
 *   this annotation, HK Analysis treats the load as non-HK (it's a
 *   device-memory read) and the call site falls back to CPU_PROXY.
 *   With it, the pass synthesizes a host trace function that reads the
 *   field via `gicc_runtime_host_mirror_of(rt, formal)` at trace time
 *   and pre-stages one DWQ descriptor / IPC memcpy per element.
 *
 *   `name` must match the formal's source-level identifier exactly. The
 *   pass falls back to a positional form
 *   `__attribute__((annotate("gicc_kernel_host_mirror_param=<idx>")))`
 *   when the build strips value names.
 *
 *   The host side must call `Runtime::register_host_mirror(dev_ptr,
 *   host_ptr, sizeof(elem), count)` before each launch so the lookup
 *   resolves.
 *
 *   See docs/superpowers/specs/2026-05-24-asf-pass-driven-dwq.md.
 */
#pragma once

#define GICC_KERNEL_HOST_MIRROR(name) \
    __attribute__((annotate("gicc_kernel_host_mirror=" #name)))

#define GICC_KERNEL_HOST_MIRROR_PARAM(idx) \
    __attribute__((annotate("gicc_kernel_host_mirror_param=" #idx)))
