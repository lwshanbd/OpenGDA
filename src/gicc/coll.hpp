// src/gicc/coll.hpp
//
// Collective reduction helpers built on top of Bootstrap::allgather_fixed.
// These live outside Bootstrap so both BootstrapMPI and BootstrapPMI2 share
// one implementation and new reduction ops can be added without touching
// the Bootstrap surface.
#pragma once

#include "gicc/bootstrap/bootstrap.hpp"

#include <cstddef>

namespace gicc::coll {

template<class T>
inline T allreduce_sum(Bootstrap& bs, const T& local) {
    auto all = bs.template allgather_fixed<T>(local);
    T acc = all.empty() ? T{} : all[0];
    for (std::size_t i = 1; i < all.size(); ++i) acc = acc + all[i];
    return acc;
}

template<class T>
inline T allreduce_min(Bootstrap& bs, const T& local) {
    auto all = bs.template allgather_fixed<T>(local);
    T acc = all.empty() ? local : all[0];
    for (std::size_t i = 1; i < all.size(); ++i)
        if (all[i] < acc) acc = all[i];
    return acc;
}

template<class T>
inline T allreduce_max(Bootstrap& bs, const T& local) {
    auto all = bs.template allgather_fixed<T>(local);
    T acc = all.empty() ? local : all[0];
    for (std::size_t i = 1; i < all.size(); ++i)
        if (acc < all[i]) acc = all[i];
    return acc;
}

} // namespace gicc::coll
