/**
 * mpi_bootstrap.hpp - MPI-based bootstrap (mirrors src/gicc/util/mpi_bootstrap.hpp)
 *
 * The CXI backend additionally uses PMI2 internally for libfabric endpoint
 * address exchange (inside GdaComm), but buffer-metadata exchange in the
 * unified gicc::Runtime path goes through MPI_Allgather here so that user
 * code matches the mlx5 backend exactly.
 */
#pragma once

#include <mpi.h>
#include <cstdint>
#include <vector>

namespace gicc {

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

        MPI_Comm local_comm;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
        MPI_Comm_rank(local_comm, &local_rank);
        MPI_Comm_free(&local_comm);
    }

    ~MpiBootstrap() {
        if (initialized_mpi) MPI_Finalize();
    }

    MpiBootstrap(const MpiBootstrap&) = delete;
    MpiBootstrap& operator=(const MpiBootstrap&) = delete;

    void barrier() { MPI_Barrier(comm); }

    template<typename T>
    std::vector<T> allgather_fixed(const T& value) {
        std::vector<T> result(size);
        MPI_Allgather(&value, sizeof(T), MPI_BYTE,
                      result.data(), sizeof(T), MPI_BYTE, comm);
        return result;
    }
};

} // namespace gicc
