/*
 * pmi2.hpp
 *
 * PMI2 header file for the bootstrap library.
 */

#ifndef PMI2_HPP
#define PMI2_HPP

#include "common.hpp"
#include <pmi2.h>

#include <cstdlib>
#include <cstdio>
#include <string>

class PMI2 : public Bootstrap {
public:
    PMI2();
    ~PMI2();

    // Core initialization/finalization
    bool bootstrap_initialize() override;
    bool bootstrap_finalize() override;
    std::string get_bootstrap_name() const override;

    // Process information
    int get_rank() const override;
    int get_size() const override;

    // Node-local rank information
    int get_local_rank() const override;
    int get_local_size() const override;
    int get_node_id() const override;

    // Synchronization barrier (wraps PMI2_KVS_Fence)
    bool bootstrap_barrier() override;

    // Key-Value Store operations
    bool bootstrap_kvs_put(const char* key, const char* value) override;
    bool bootstrap_kvs_get(const char* key, char* value, int* value_len) override;

    // Address exchange helper (like exchange_addrs in gda-comp.cpp)
    bool bootstrap_exchange(const char* my_data, size_t data_len, char* all_data) override;

private:
    int rank_;
    int size_;
    int spawned_;
    int appnum_;

    // Node-local information
    int local_rank_;
    int local_size_;
    int node_id_;

    // Helper to query node attributes
    bool query_node_info();
};

#endif