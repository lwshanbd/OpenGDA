// src/gicc/bootstrap/detail/bootstrap_mpi.hpp
//
// MPI-backed implementation of the gicc::Bootstrap contract.
// Selected when GICC_BOOTSTRAP_MPI is defined. Not intended to be included
// directly by user code — go through gicc/bootstrap/bootstrap.hpp.
#pragma once

#include <mpi.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

namespace gicc::detail {

#define GICC_MPI_CHECK(call) do {                                         \
    int gicc_rc_ = (call);                                                \
    if (gicc_rc_ != MPI_SUCCESS) {                                        \
        char gicc_err_[MPI_MAX_ERROR_STRING] = {0};                       \
        int gicc_len_ = 0;                                                \
        MPI_Error_string(gicc_rc_, gicc_err_, &gicc_len_);                \
        throw std::runtime_error(                                         \
            std::string("BootstrapMPI: " #call " failed: ") + gicc_err_); \
    }                                                                     \
} while (0)

class BootstrapMPI {
public:
    BootstrapMPI() : BootstrapMPI(nullptr, nullptr) {}

    BootstrapMPI(int* argc, char*** argv) {
        int already = 0;
        GICC_MPI_CHECK(MPI_Initialized(&already));
        if (!already) {
            GICC_MPI_CHECK(MPI_Init(argc, argv));
            owned_ = true;
        }

        comm_ = MPI_COMM_WORLD;
        GICC_MPI_CHECK(MPI_Comm_rank(comm_, &rank_));
        GICC_MPI_CHECK(MPI_Comm_size(comm_, &size_));

        MPI_Comm local_comm;
        GICC_MPI_CHECK(MPI_Comm_split_type(comm_, MPI_COMM_TYPE_SHARED, rank_,
                                           MPI_INFO_NULL, &local_comm));
        GICC_MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank_));
        GICC_MPI_CHECK(MPI_Comm_size(local_comm, &local_size_));
        GICC_MPI_CHECK(MPI_Comm_free(&local_comm));
    }

    BootstrapMPI(int argc, char** argv)
        : BootstrapMPI(&argc, &argv) {}

    ~BootstrapMPI() {
        if (owned_) {
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (!finalized) MPI_Finalize();
        }
    }

    BootstrapMPI(const BootstrapMPI&) = delete;
    BootstrapMPI& operator=(const BootstrapMPI&) = delete;

    int rank()       const noexcept { return rank_; }
    int size()       const noexcept { return size_; }
    int local_rank() const noexcept { return local_rank_; }
    int local_size() const noexcept { return local_size_; }

    std::vector<bool> locality_map() const {
        // Compute a per-node color by taking the min global rank inside the
        // shared-memory subcommunicator, then Allgather it across COMM_WORLD.
        MPI_Comm local_comm;
        GICC_MPI_CHECK(MPI_Comm_split_type(comm_, MPI_COMM_TYPE_SHARED, rank_,
                                           MPI_INFO_NULL, &local_comm));
        int lsz = 0;
        GICC_MPI_CHECK(MPI_Comm_size(local_comm, &lsz));
        std::vector<int> local_globals(lsz);
        int my_global = rank_;
        GICC_MPI_CHECK(MPI_Allgather(&my_global, 1, MPI_INT,
                                     local_globals.data(), 1, MPI_INT, local_comm));
        int my_color = *std::min_element(local_globals.begin(), local_globals.end());
        GICC_MPI_CHECK(MPI_Comm_free(&local_comm));

        std::vector<int> all_colors(size_);
        GICC_MPI_CHECK(MPI_Allgather(&my_color, 1, MPI_INT,
                                     all_colors.data(), 1, MPI_INT, comm_));
        std::vector<bool> out(size_, false);
        for (int i = 0; i < size_; ++i) out[i] = (all_colors[i] == my_color);
        return out;
    }

    void barrier() { GICC_MPI_CHECK(MPI_Barrier(comm_)); }

    std::vector<std::vector<uint8_t>>
    allgather(const void* data, int len) {
        std::vector<int> lens(size_);
        GICC_MPI_CHECK(MPI_Allgather(&len, 1, MPI_INT,
                                     lens.data(), 1, MPI_INT, comm_));

        std::vector<int> displs(size_);
        int total = 0;
        for (int i = 0; i < size_; ++i) { displs[i] = total; total += lens[i]; }

        std::vector<uint8_t> flat(total);
        GICC_MPI_CHECK(MPI_Allgatherv(data, len, MPI_BYTE,
                                      flat.data(), lens.data(), displs.data(),
                                      MPI_BYTE, comm_));

        std::vector<std::vector<uint8_t>> out(size_);
        for (int i = 0; i < size_; ++i) {
            out[i].assign(flat.begin() + displs[i],
                          flat.begin() + displs[i] + lens[i]);
        }
        return out;
    }

    template<class T>
    std::vector<T> allgather_fixed(const T& value) {
        std::vector<T> out(size_);
        GICC_MPI_CHECK(MPI_Allgather(&value, sizeof(T), MPI_BYTE,
                                     out.data(), sizeof(T), MPI_BYTE, comm_));
        return out;
    }

    template<class T>
    void broadcast(T& value, int root = 0) {
        GICC_MPI_CHECK(MPI_Bcast(&value, sizeof(T), MPI_BYTE, root, comm_));
    }

    void send(const void* buf, int len, int dest, int tag = 0) {
        GICC_MPI_CHECK(MPI_Send(buf, len, MPI_BYTE, dest, tag, comm_));
    }
    void recv(void* buf, int len, int src, int tag = 0) {
        GICC_MPI_CHECK(MPI_Recv(buf, len, MPI_BYTE, src, tag, comm_, MPI_STATUS_IGNORE));
    }
    void sendrecv(const void* sbuf, void* rbuf, int len, int peer, int tag = 0) {
        GICC_MPI_CHECK(MPI_Sendrecv(sbuf, len, MPI_BYTE, peer, tag,
                                    rbuf, len, MPI_BYTE, peer, tag,
                                    comm_, MPI_STATUS_IGNORE));
    }

    double wtime() const noexcept { return MPI_Wtime(); }

    // Abort all ranks. Safe to call from any rank; others need not cooperate.
    [[noreturn]] static void abort(int code = 1, const char* /*msg*/ = nullptr) {
        int already = 0;
        MPI_Initialized(&already);
        if (already) MPI_Abort(MPI_COMM_WORLD, code);
        std::_Exit(code);
    }

private:
    MPI_Comm comm_ = MPI_COMM_NULL;
    int rank_ = 0;
    int size_ = 0;
    int local_rank_ = 0;
    int local_size_ = 0;
    bool owned_ = false;
};

} // namespace gicc::detail
