/*
 * proxy_ring_defs.hpp - Shared ProxyRing alias + capacity constant.
 *
 * Tiny header pulled in by both:
 *   - proxy/proxy_thread.hpp (the worker that owns the host pointer), and
 *   - ofi_device.cuh (the kernel-visible put_no_db that pushes commands).
 *
 * Splitting this out lets ofi_device.cuh stay free of proxy_thread.hpp's
 * <thread> / <unordered_set> includes.
 */
#pragma once

#include "d2h_ring.cuh"
#include <cstdint>

namespace gicc {
namespace proxy {

inline constexpr uint32_t kProxyRingCapacity = 4096;
using ProxyRing = D2HRing<kProxyRingCapacity>;

} // namespace proxy
} // namespace gicc
