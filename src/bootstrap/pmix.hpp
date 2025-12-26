/*
 * pmix.hpp
 *
 * PMIx header file for the bootstrap library.
 */

#ifndef PMIX_HPP
#define PMIX_HPP

#include <pmix.h>
#include "common.hpp"

#include <cstdlib>
#include <cstdio>

class PMIX : public Bootstrap {
public:
    PMIX();
    ~PMIX();

    // Core initialization/finalization
    bool bootstrap_initialize() override;
    bool bootstrap_finalize() override;
    std::string get_bootstrap_name() const override;

    // Process information
    int get_rank() const override;
    int get_size() const override;

    // Synchronization barrier
    bool bootstrap_barrier() override;

    // Key-Value Store operations
    bool bootstrap_kvs_put(const char* key, const char* value) override;
    bool bootstrap_kvs_get(const char* key, char* value, int* value_len) override;

    // Address exchange helper
    bool bootstrap_exchange(const char* my_data, size_t data_len, char* all_data) override;

private:
    int rank_;
    int size_;
    pmix_proc_t myproc_;
};

#endif
