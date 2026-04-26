/*
 * dwq_work_builder.hpp - Deferred Work Queue operation builder
 *
 * Holds persistent structures for DWQ operations that must remain valid
 * until the operation completes.
 */
#pragma once

#include <rdma/fabric.h>
#include <rdma/fi_cxi_ext.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_atomic.h>
#include <rdma/fi_trigger.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

class DwqWorkBuilder {
public:
    // Persistent structures for RMA write (must outlive the operation)
    struct fi_deferred_work work;
    struct fi_op_rma op_rma;
    struct fi_msg_rma msg_rma;
    struct iovec iov;
    struct fi_rma_iov rma_iov;
    void* stored_rma_desc;  // Must persist - desc is void**, points here

    // Persistent structures for atomic operation
    struct fi_deferred_work atomic_work;
    struct fi_op_atomic op_atomic;
    struct fi_msg_atomic atomic_msg;
    struct fi_ioc atomic_iov;
    struct fi_rma_ioc atomic_rma_iov;
    void* stored_atomic_desc;  // Must persist - desc is void**, points here

    int rank;  // For error messages

    explicit DwqWorkBuilder(int rank_) : rank(rank_),
        stored_rma_desc(nullptr), stored_atomic_desc(nullptr) {
        memset(&work, 0, sizeof(work));
        memset(&op_rma, 0, sizeof(op_rma));
        memset(&msg_rma, 0, sizeof(msg_rma));
        memset(&iov, 0, sizeof(iov));
        memset(&rma_iov, 0, sizeof(rma_iov));
        memset(&atomic_work, 0, sizeof(atomic_work));
        memset(&op_atomic, 0, sizeof(op_atomic));
        memset(&atomic_msg, 0, sizeof(atomic_msg));
        memset(&atomic_iov, 0, sizeof(atomic_iov));
        memset(&atomic_rma_iov, 0, sizeof(atomic_rma_iov));
    }

    // No copy/move
    DwqWorkBuilder(const DwqWorkBuilder&) = delete;
    DwqWorkBuilder& operator=(const DwqWorkBuilder&) = delete;

    // Queue an RMA write operation
    // Triggered by trigger_cntr reaching threshold
    // On completion, increments completion_cntr
    void queue_rma_write(
        struct fid_domain* domain,
        struct fid_ep* ep,
        void* src_buf,
        void* desc,
        size_t size,
        fi_addr_t dest_addr,
        uint64_t remote_addr,
        uint64_t remote_key,
        struct fid_cntr* trigger_cntr,
        struct fid_cntr* completion_cntr,
        uint64_t threshold)
    {
        // Setup iovec for source
        iov.iov_base = src_buf;
        iov.iov_len = size;

        // Setup remote address
        rma_iov.addr = remote_addr;
        rma_iov.len = size;
        rma_iov.key = remote_key;

        // Setup message
        // CRITICAL: desc is void**, must point to persistent storage
        stored_rma_desc = desc;
        msg_rma.msg_iov = &iov;
        msg_rma.desc = &stored_rma_desc;
        msg_rma.iov_count = 1;
        msg_rma.addr = dest_addr;
        msg_rma.rma_iov = &rma_iov;
        msg_rma.rma_iov_count = 1;
        msg_rma.context = NULL;
        msg_rma.data = 0;

        // Setup op_rma
        op_rma.ep = ep;
        op_rma.msg = msg_rma;
        op_rma.flags = FI_COMPLETION;  // No FI_CXI_CNTR_WB - we use atomic signal instead

        // Setup deferred work
        work.triggering_cntr = trigger_cntr;
        work.completion_cntr = completion_cntr;
        work.threshold = threshold;
        work.op_type = FI_OP_WRITE;
        work.op.rma = &op_rma;

        int ret = fi_control(&domain->fid, FI_QUEUE_WORK, &work);
        if (ret) {
            fprintf(stderr, "Rank %d: fi_control(FI_QUEUE_WORK/RMA) failed: %s (%d)\n",
                    rank, fi_strerror(-ret), ret);
            exit(1);
        }
    }

    // Queue an RMA read operation
    // Triggered by trigger_cntr reaching threshold
    // On completion, increments completion_cntr
    // local_buf is the LOCAL DESTINATION (read pulls remote data INTO it).
    void queue_rma_read(
        struct fid_domain* domain,
        struct fid_ep* ep,
        void* local_buf,
        void* desc,
        size_t size,
        fi_addr_t source_addr,
        uint64_t remote_addr,
        uint64_t remote_key,
        struct fid_cntr* trigger_cntr,
        struct fid_cntr* completion_cntr,
        uint64_t threshold)
    {
        // Setup iovec for local destination
        iov.iov_base = local_buf;
        iov.iov_len = size;

        // Setup remote address
        rma_iov.addr = remote_addr;
        rma_iov.len = size;
        rma_iov.key = remote_key;

        // Setup message
        // CRITICAL: desc is void**, must point to persistent storage
        stored_rma_desc = desc;
        msg_rma.msg_iov = &iov;
        msg_rma.desc = &stored_rma_desc;
        msg_rma.iov_count = 1;
        msg_rma.addr = source_addr;
        msg_rma.rma_iov = &rma_iov;
        msg_rma.rma_iov_count = 1;
        msg_rma.context = NULL;
        msg_rma.data = 0;

        // Setup op_rma
        op_rma.ep = ep;
        op_rma.msg = msg_rma;
        op_rma.flags = FI_COMPLETION;  // No FI_CXI_CNTR_WB - we use atomic signal instead

        // Setup deferred work
        work.triggering_cntr = trigger_cntr;
        work.completion_cntr = completion_cntr;
        work.threshold = threshold;
        work.op_type = FI_OP_READ;
        work.op.rma = &op_rma;

        int ret = fi_control(&domain->fid, FI_QUEUE_WORK, &work);
        if (ret) {
            fprintf(stderr, "Rank %d: fi_control(FI_QUEUE_WORK/RMA_READ) failed: %s (%d)\n",
                    rank, fi_strerror(-ret), ret);
            exit(1);
        }
    }

    // Queue an atomic signal operation
    // Triggered by trigger_cntr reaching threshold
    // Performs atomic add to signal GPU completion
    void queue_atomic_signal(
        struct fid_domain* domain,
        struct fid_ep* ep,
        void* operand_buf,
        void* operand_desc,
        void* result_buf,
        uint64_t result_key,
        uint64_t result_addr,  // Remote addr (may be same machine for self-atomic)
        fi_addr_t target_addr,
        struct fid_cntr* trigger_cntr,
        struct fid_cntr* completion_cntr,
        uint64_t threshold)
    {
        // Source operand (value to add)
        atomic_iov.addr = operand_buf;
        atomic_iov.count = 1;  // Number of elements

        // Destination (atomic_result on GPU)
        atomic_rma_iov.addr = result_addr;
        atomic_rma_iov.count = 1;  // Number of elements
        atomic_rma_iov.key = result_key;

        // Setup atomic message
        // CRITICAL: desc is void**, must point to persistent storage
        stored_atomic_desc = operand_desc;
        atomic_msg.msg_iov = &atomic_iov;
        atomic_msg.desc = &stored_atomic_desc;
        atomic_msg.iov_count = 1;
        atomic_msg.addr = target_addr;
        atomic_msg.rma_iov = &atomic_rma_iov;
        atomic_msg.rma_iov_count = 1;
        atomic_msg.datatype = FI_UINT64;
        atomic_msg.op = FI_SUM;
        atomic_msg.context = NULL;
        atomic_msg.data = 0;

        // Setup atomic operation
        op_atomic.ep = ep;
        op_atomic.msg = atomic_msg;
        op_atomic.flags = FI_COMPLETION;

        // Setup deferred work
        atomic_work.op_type = FI_OP_ATOMIC;
        atomic_work.op.atomic = &op_atomic;
        atomic_work.triggering_cntr = trigger_cntr;
        atomic_work.completion_cntr = completion_cntr;
        atomic_work.threshold = threshold;

        int ret = fi_control(&domain->fid, FI_QUEUE_WORK, &atomic_work);
        if (ret) {
            fprintf(stderr, "Rank %d: fi_control(FI_QUEUE_WORK/ATOMIC) failed: %s (%d)\n",
                    rank, fi_strerror(-ret), ret);
            exit(1);
        }
    }
};
