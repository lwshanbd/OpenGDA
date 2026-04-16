// src/gicc/bootstrap/detail/bootstrap_pmi2.hpp
//
// PMI2-backed implementation of the gicc::Bootstrap contract.
// Selected when GICC_BOOTSTRAP_PMI2 is defined. Collective primitives are
// emulated on top of PMI2 KVS put/get/fence. Intended for Cray PMI2 on
// Slingshot systems (e.g., Tioga).
#pragma once

#include <pmi2.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
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
        if (gethostname(host, sizeof(host) - 1) != 0) {
            throw std::runtime_error("BootstrapPMI2: gethostname failed");
        }
        host[sizeof(host) - 1] = '\0';
        if (host[0] == '\0') {
            throw std::runtime_error("BootstrapPMI2: gethostname returned empty string");
        }
        hostname_.assign(host);

        publish_hostname_and_fence();
        // locality_map() requires the hostname KVS entries; resolve local
        // identity from the map (authoritative) with env vars as a hint.
        resolve_local_identity();
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
        // Per-(src,dst,tag) sequence so asymmetric send/recv patterns (e.g. a
        // gather on rank 0 pulling from each peer) stay in sync — a global
        // counter would diverge between sender and receiver.
        const uint64_t seq = next_p2p_seq(rank_, dest, tag);
        publish_blob(kvs_key("p2p", seq, rank_, dest, tag),
                     reinterpret_cast<const uint8_t*>(buf), len);
        barrier();
    }
    void recv(void* buf, int len, int src, int tag = 0) {
        barrier();
        const uint64_t seq = next_p2p_seq(src, rank_, tag);
        auto blob = fetch_blob(kvs_key("p2p", seq, src, rank_, tag));
        if ((int)blob.size() != len) {
            throw std::runtime_error("BootstrapPMI2::recv: size mismatch");
        }
        std::memcpy(buf, blob.data(), len);
    }
    void sendrecv(const void* sbuf, void* rbuf, int len, int peer, int tag = 0) {
        const uint64_t send_seq = next_p2p_seq(rank_, peer, tag);
        const uint64_t recv_seq = next_p2p_seq(peer, rank_, tag);
        publish_blob(kvs_key("p2p", send_seq, rank_, peer, tag),
                     reinterpret_cast<const uint8_t*>(sbuf), len);
        barrier();
        auto blob = fetch_blob(kvs_key("p2p", recv_seq, peer, rank_, tag));
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

    // Abort all ranks. PMI2_Abort notifies the resource manager.
    [[noreturn]] static void abort(int code = 1, const char* msg = "BootstrapPMI2::abort") {
        PMI2_Abort(1 /*flag: abort all*/, msg);
        std::_Exit(code);  // fallback if PMI2_Abort returns
    }

private:
    int rank_ = -1;
    int size_ = 0;
    int local_rank_ = -1;
    int local_size_ = 0;
    bool owned_ = false;
    std::string hostname_;
    std::map<std::tuple<int,int,int>, uint64_t> p2p_seq_;  // key: (src,dst,tag)

    uint64_t next_p2p_seq(int src, int dst, int tag) {
        return p2p_seq_[{src, dst, tag}]++;
    }

    // Resolve local_rank_ / local_size_. Prefer launcher env vars; if the
    // launcher-provided values are inconsistent with the KVS-derived locality
    // map we trust the map and emit a warning — silently taking (0,1) masks
    // misconfiguration and collapses every rank onto GPU 0.
    void resolve_local_identity() {
        const char* lr_vars[] = {
            "SLURM_LOCALID", "FLUX_TASK_LOCAL_ID",
            "OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", nullptr};
        int env_lr = -1;
        for (int i = 0; lr_vars[i] && env_lr < 0; ++i) {
            if (const char* v = std::getenv(lr_vars[i])) env_lr = std::atoi(v);
        }
        const char* ls_vars[] = {
            "SLURM_NTASKS_PER_NODE", "FLUX_LOCAL_RANKS", nullptr};
        int env_ls = -1;
        for (int i = 0; ls_vars[i] && env_ls < 0; ++i) {
            if (const char* v = std::getenv(ls_vars[i])) env_ls = std::atoi(v);
        }

        auto same = locality_map();
        int derived_lr = 0, derived_ls = 0;
        for (int i = 0; i < size_; ++i) {
            if (!same[i]) continue;
            if (i < rank_) ++derived_lr;
            ++derived_ls;
        }

        if (env_lr >= 0 && env_lr != derived_lr) {
            std::fprintf(stderr,
                "BootstrapPMI2 rank %d: env local_rank=%d disagrees with "
                "hostname-derived %d; using derived.\n",
                rank_, env_lr, derived_lr);
        }
        if (env_ls > 0 && env_ls != derived_ls) {
            std::fprintf(stderr,
                "BootstrapPMI2 rank %d: env local_size=%d disagrees with "
                "hostname-derived %d; using derived.\n",
                rank_, env_ls, derived_ls);
        }
        local_rank_ = derived_lr;
        local_size_ = derived_ls;
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
    static int hex_nibble(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
        if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
        return -1;
    }
    static std::vector<uint8_t> hex_decode(const char* s, int n) {
        if (n % 2 != 0) {
            throw std::runtime_error("BootstrapPMI2: odd-length hex blob");
        }
        std::vector<uint8_t> out(n / 2);
        for (int i = 0; i < n / 2; ++i) {
            int hi = hex_nibble(s[2 * i]);
            int lo = hex_nibble(s[2 * i + 1]);
            if (hi < 0 || lo < 0) {
                throw std::runtime_error("BootstrapPMI2: non-hex byte in blob");
            }
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
        // Grow the buffer until PMI2_KVS_Get reports a vallen that fits.
        // A fixed 16 KiB buffer silently truncates larger allgathers.
        std::vector<char> buf(16 * 1024);
        for (int attempt = 0; attempt < 4; ++attempt) {
            int vallen = 0;
            int rc = PMI2_KVS_Get(NULL, PMI2_ID_NULL, key.c_str(),
                                  buf.data(), (int)buf.size(), &vallen);
            if (rc != PMI2_SUCCESS) {
                throw std::runtime_error(
                    "BootstrapPMI2::fetch_blob: KVS_Get(" + key + ") failed");
            }
            if (vallen < 0) {
                throw std::runtime_error(
                    "BootstrapPMI2::fetch_blob: negative vallen for " + key);
            }
            if (vallen < (int)buf.size()) {
                // vallen includes the trailing NUL on some implementations;
                // hex_decode only consumes the hex chars.
                int hex_len = vallen;
                while (hex_len > 0 && buf[hex_len - 1] == '\0') --hex_len;
                return hex_decode(buf.data(), hex_len);
            }
            buf.resize(buf.size() * 4);
        }
        throw std::runtime_error(
            "BootstrapPMI2::fetch_blob: value for " + key +
            " exceeds max buffer (4 MiB)");
    }
};

} // namespace gicc::detail
