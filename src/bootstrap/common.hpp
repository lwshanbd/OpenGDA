/**
 * common.hpp
 *
 * Common header file for the bootstrap library.
 */

#ifndef COMMON_HPP
#define COMMON_HPP

#include <string>
#include <memory>
#include <cstddef>

class Bootstrap {
public:
    Bootstrap();
    virtual ~Bootstrap() = default;

    // Core initialization/finalization
    virtual bool bootstrap_initialize() = 0;
    virtual bool bootstrap_finalize() = 0;
    virtual std::string get_bootstrap_name() const = 0;

    // Process information
    virtual int get_rank() const = 0;
    virtual int get_size() const = 0;

    // Node-local rank information (for same-node detection)
    virtual int get_local_rank() const = 0;      // Rank within the node (0, 1, 2, ...)
    virtual int get_local_size() const = 0;      // Number of processes on this node
    virtual int get_node_id() const = 0;         // Unique node identifier

    // Synchronization barrier
    virtual bool bootstrap_barrier() = 0;

    // Key-Value Store operations
    virtual bool bootstrap_kvs_put(const char* key, const char* value) = 0;
    virtual bool bootstrap_kvs_get(const char* key, char* value, int* value_len) = 0;

    // Address exchange helper
    virtual bool bootstrap_exchange(const char* my_data, size_t data_len, char* all_data) = 0;

    static std::unique_ptr<Bootstrap> create_bootstrap(const std::string& type);

    bool is_initialized() const { return bootstrap_initialized; }

protected:
    bool bootstrap_initialized = false;
};




#endif