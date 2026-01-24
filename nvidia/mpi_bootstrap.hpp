/**
 * mpi_bootstrap.hpp - MPI-based bootstrap for address exchange (no PMI)
 *
 * Uses MPI_Allgather for collective address exchange.
 */
#pragma once

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

class MpiBootstrap {
public:
    int rank;
    int size;
    int local_rank;
    MPI_Comm comm;
    bool initialized_mpi;

    explicit MpiBootstrap(MPI_Comm communicator = MPI_COMM_WORLD)
        : rank(0), size(0), local_rank(0), comm(communicator), initialized_mpi(false)
    {
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);

        if (!mpi_initialized) {
            MPI_Init(NULL, NULL);
            initialized_mpi = true;
        }

        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        // Determine local rank using shared memory communicator
        MPI_Comm local_comm;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
        MPI_Comm_rank(local_comm, &local_rank);
        MPI_Comm_free(&local_comm);
    }

    ~MpiBootstrap() {
        if (initialized_mpi) {
            MPI_Finalize();
        }
    }

    // No copy
    MpiBootstrap(const MpiBootstrap&) = delete;
    MpiBootstrap& operator=(const MpiBootstrap&) = delete;

    // Global barrier
    void barrier() {
        MPI_Barrier(comm);
    }

    // Allgather variable-length data
    // Returns a vector of vectors, one per rank
    std::vector<std::vector<uint8_t>> allgather(const void* send_data, int send_len) {
        // First gather all sizes
        std::vector<int> all_sizes(size);
        MPI_Allgather(&send_len, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, comm);

        // Calculate displacements
        std::vector<int> displs(size);
        int total_size = 0;
        for (int i = 0; i < size; i++) {
            displs[i] = total_size;
            total_size += all_sizes[i];
        }

        // Gather all data
        std::vector<uint8_t> recv_buf(total_size);
        MPI_Allgatherv(send_data, send_len, MPI_BYTE,
                       recv_buf.data(), all_sizes.data(), displs.data(), MPI_BYTE, comm);

        // Split into per-rank vectors
        std::vector<std::vector<uint8_t>> result(size);
        for (int i = 0; i < size; i++) {
            result[i].assign(recv_buf.begin() + displs[i],
                             recv_buf.begin() + displs[i] + all_sizes[i]);
        }

        return result;
    }

    // Allgather fixed-size data
    template<typename T>
    std::vector<T> allgather_fixed(const T& value) {
        std::vector<T> result(size);
        MPI_Allgather(&value, sizeof(T), MPI_BYTE,
                      result.data(), sizeof(T), MPI_BYTE, comm);
        return result;
    }

    // Broadcast from root
    template<typename T>
    void broadcast(T& value, int root = 0) {
        MPI_Bcast(&value, sizeof(T), MPI_BYTE, root, comm);
    }

    // Send/recv for point-to-point
    void send(const void* buf, int len, int dest, int tag = 0) {
        MPI_Send(buf, len, MPI_BYTE, dest, tag, comm);
    }

    void recv(void* buf, int len, int src, int tag = 0) {
        MPI_Recv(buf, len, MPI_BYTE, src, tag, comm, MPI_STATUS_IGNORE);
    }

    // Exchange data with a peer (send to peer, receive from peer)
    void exchange(const void* send_buf, void* recv_buf, int len, int peer, int tag = 0) {
        MPI_Sendrecv(send_buf, len, MPI_BYTE, peer, tag,
                     recv_buf, len, MPI_BYTE, peer, tag,
                     comm, MPI_STATUS_IGNORE);
    }
};
