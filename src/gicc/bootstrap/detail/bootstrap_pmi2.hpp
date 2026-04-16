// src/gicc/bootstrap/detail/bootstrap_pmi2.hpp
//
// PMI2-backed implementation of the gicc::Bootstrap contract.
// Selected when GICC_BOOTSTRAP_PMI2 is defined. Collective primitives are
// emulated on top of PMI2 KVS put/get/fence. Intended for Cray PMI2 on
// Slingshot systems (e.g., Tioga).
#pragma once

#include <pmi2.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace gicc::detail {

class BootstrapPMI2 {
public:
    BootstrapPMI2() : BootstrapPMI2(nullptr, nullptr) {}

    BootstrapPMI2(int* /*argc*/, char*** /*argv*/) {
        int spawned = 0, appnum = 0;
        int rc = PMI2_Init(&spawned, &size_, &rank_, &appnum);
        if (rc != PMI2_SUCCESS) {
            throw std::runtime_error(
                "BootstrapPMI2: PMI2_Init failed (rc=" + std::to_string(rc) + ")");
        }
        owned_ = true;

        char host[256] = {0};
        gethostname(host, sizeof(host) - 1);
        hostname_.assign(host);

        read_env_local_identity();
        publish_hostname_and_fence();
    }

    BootstrapPMI2(int argc, char** argv)
        : BootstrapPMI2(&argc, &argv) {}

    ~BootstrapPMI2() {
        if (owned_) PMI2_Finalize();
    }

    BootstrapPMI2(const BootstrapPMI2&) = delete;
    BootstrapPMI2& operator=(const BootstrapPMI2&) = delete;

    int rank()       const noexcept { return rank_; }
    int size()       const noexcept { return size_; }
    int local_rank() const noexcept { return local_rank_; }
    int local_size() const noexcept { return local_size_; }

    std::vector<bool> locality_map() const {
        std::vector<bool> same(size_, false);
        char val[256];
        for (int i = 0; i < size_; ++i) {
            std::string key = "gicc.host." + std::to_string(i);
            int vallen = 0;
            int rc = PMI2_KVS_Get(NULL, PMI2_ID_NULL, key.c_str(),
                                  val, sizeof(val), &vallen);
            if (rc != PMI2_SUCCESS) {
                throw std::runtime_error(
                    "BootstrapPMI2::locality_map: KVS_Get failed for " + key);
            }
            same[i] = (hostname_ == val);
        }
        return same;
    }

    void barrier() {
        int rc = PMI2_KVS_Fence();
        if (rc != PMI2_SUCCESS) {
            throw std::runtime_error("BootstrapPMI2::barrier: KVS_Fence failed");
        }
    }

    std::vector<std::vector<uint8_t>>
    allgather(const void* data, int len) {
        const uint64_t epoch = next_epoch();
        publish_blob(kvs_key("ag", epoch, rank_),
                     reinterpret_cast<const uint8_t*>(data), len);
        barrier();
        std::vector<std::vector<uint8_t>> out(size_);
        for (int i = 0; i < size_; ++i) {
            out[i] = fetch_blob(kvs_key("ag", epoch, i));
        }
        return out;
    }

    template<class T>
    std::vector<T> allgather_fixed(const T& value) {
        auto raw = allgather(&value, sizeof(T));
        std::vector<T> out(size_);
        for (int i = 0; i < size_; ++i) {
            if ((int)raw[i].size() != (int)sizeof(T)) {
                throw std::runtime_error(
                    "BootstrapPMI2::allgather_fixed: size mismatch");
            }
            std::memcpy(&out[i], raw[i].data(), sizeof(T));
        }
        return out;
    }

    template<class T>
    void broadcast(T& value, int root = 0) {
        const uint64_t epoch = next_epoch();
        if (rank_ == root) {
            publish_blob(kvs_key("bc", epoch, root),
                         reinterpret_cast<const uint8_t*>(&value), sizeof(T));
        }
        barrier();
        if (rank_ != root) {
            auto blob = fetch_blob(kvs_key("bc", epoch, root));
            if ((int)blob.size() != (int)sizeof(T)) {
                throw std::runtime_error(
                    "BootstrapPMI2::broadcast: size mismatch");
            }
            std::memcpy(&value, blob.data(), sizeof(T));
        }
    }

    void send(const void* buf, int len, int dest, int tag = 0) {
        const uint64_t epoch = next_epoch();
        publish_blob(kvs_key("p2p", epoch, rank_, dest, tag),
                     reinterpret_cast<const uint8_t*>(buf), len);
        barrier();
    }
    void recv(void* buf, int len, int src, int tag = 0) {
        barrier();
        auto blob = fetch_blob(kvs_key("p2p", next_epoch_peek(), src, rank_, tag));
        if ((int)blob.size() != len) {
            throw std::runtime_error("BootstrapPMI2::recv: size mismatch");
        }
        std::memcpy(buf, blob.data(), len);
    }
    void sendrecv(const void* sbuf, void* rbuf, int len, int peer, int tag = 0) {
        const uint64_t epoch = next_epoch();
        publish_blob(kvs_key("p2p", epoch, rank_, peer, tag),
                     reinterpret_cast<const uint8_t*>(sbuf), len);
        barrier();
        auto blob = fetch_blob(kvs_key("p2p", epoch, peer, rank_, tag));
        if ((int)blob.size() != len) {
            throw std::runtime_error("BootstrapPMI2::sendrecv: size mismatch");
        }
        std::memcpy(rbuf, blob.data(), len);
    }

    double wtime() const noexcept {
        using clock = std::chrono::steady_clock;
        auto now = clock::now().time_since_epoch();
        return std::chrono::duration<double>(now).count();
    }

private:
    int rank_ = -1;
    int size_ = 0;
    int local_rank_ = -1;
    int local_size_ = 0;
    bool owned_ = false;
    std::string hostname_;
    uint64_t epoch_counter_ = 0;

    uint64_t next_epoch() { return epoch_counter_++; }
    uint64_t next_epoch_peek() { return epoch_counter_ - 1; }

    void read_env_local_identity() {
        const char* lr_vars[] = {
            "SLURM_LOCALID", "FLUX_TASK_LOCAL_ID",
            "OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", nullptr};
        for (int i = 0; lr_vars[i] && local_rank_ < 0; ++i) {
            if (const char* v = std::getenv(lr_vars[i])) local_rank_ = std::atoi(v);
        }
        if (local_rank_ < 0) local_rank_ = 0;

        const char* ls_vars[] = {
            "SLURM_NTASKS_PER_NODE", "FLUX_LOCAL_RANKS", nullptr};
        for (int i = 0; ls_vars[i] && local_size_ <= 0; ++i) {
            if (const char* v = std::getenv(ls_vars[i])) local_size_ = std::atoi(v);
        }
        if (local_size_ <= 0) local_size_ = 1;
    }

    void publish_hostname_and_fence() {
        std::string key = "gicc.host." + std::to_string(rank_);
        int rc = PMI2_KVS_Put(key.c_str(), hostname_.c_str());
        if (rc != PMI2_SUCCESS) {
            throw std::runtime_error("BootstrapPMI2: KVS_Put(hostname) failed");
        }
        rc = PMI2_KVS_Fence();
        if (rc != PMI2_SUCCESS) {
            throw std::runtime_error("BootstrapPMI2: KVS_Fence(hostname) failed");
        }
    }

    static std::string hex_encode(const uint8_t* data, int len) {
        static const char* digits = "0123456789abcdef";
        std::string out(2 * len, '0');
        for (int i = 0; i < len; ++i) {
            out[2 * i]     = digits[(data[i] >> 4) & 0xF];
            out[2 * i + 1] = digits[data[i] & 0xF];
        }
        return out;
    }
    static std::vector<uint8_t> hex_decode(const char* s) {
        int n = (int)std::strlen(s);
        if (n % 2 != 0) {
            throw std::runtime_error("BootstrapPMI2: odd-length hex blob");
        }
        std::vector<uint8_t> out(n / 2);
        for (int i = 0; i < n / 2; ++i) {
            unsigned hi = 0, lo = 0;
            std::sscanf(s + 2 * i,     "%1x", &hi);
            std::sscanf(s + 2 * i + 1, "%1x", &lo);
            out[i] = (uint8_t)((hi << 4) | lo);
        }
        return out;
    }

    template<class... Xs>
    static std::string kvs_key(const char* tag, Xs... xs) {
        std::string out = std::string("gicc.") + tag;
        append_parts(out, xs...);
        return out;
    }
    static void append_parts(std::string&) {}
    template<class Head, class... Tail>
    static void append_parts(std::string& s, Head h, Tail... rest) {
        s += ".";
        s += std::to_string(h);
        append_parts(s, rest...);
    }

    void publish_blob(const std::string& key, const uint8_t* data, int len) {
        auto hex = hex_encode(data, len);
        int rc = PMI2_KVS_Put(key.c_str(), hex.c_str());
        if (rc != PMI2_SUCCESS) {
            throw std::runtime_error(
                "BootstrapPMI2::publish_blob: KVS_Put(" + key + ") failed");
        }
    }
    std::vector<uint8_t> fetch_blob(const std::string& key) {
        std::vector<char> buf(16 * 1024);
        int vallen = 0;
        int rc = PMI2_KVS_Get(NULL, PMI2_ID_NULL, key.c_str(),
                              buf.data(), (int)buf.size(), &vallen);
        if (rc != PMI2_SUCCESS) {
            throw std::runtime_error(
                "BootstrapPMI2::fetch_blob: KVS_Get(" + key + ") failed");
        }
        return hex_decode(buf.data());
    }
};

} // namespace gicc::detail
