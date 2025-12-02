#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_trigger.h>

#include <rdma/fi_cxi_ext.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unistd.h>

#ifdef USE_AMDGPU
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime.h>
#include <hwloc.h>
#endif

#include "../common/log.hpp"

#include <map>
#include <mutex>
#include <chrono>
#include <vector>

// OFI Error checking macro
#define OFI_CHECK(x, msg)                                                      \
  do {                                                                         \
    int ret = (x);                                                             \
    if (ret) {                                                                 \
      OPENGDA_Error("ofi", "%s failed: %s (%d)", msg, fi_strerror(-ret), ret); \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ============================================================================
// MRManager - Memory Region Management
// ============================================================================

/**
 * MRManager provides centralized management of registered memory regions.
 * Features:
 * - Track all registered MRs
 * - Query by address or key
 * - Prevent duplicate registrations
 * - Collect statistics
 * - Thread-safe operations
 */
class MRManager {
public:
    /**
     * Information stored for each registered memory region
     */
    struct MRInfo {
        struct fid_mr* mr;           // Libfabric MR handle
        void* addr;                  // Base address of memory region
        size_t size;                 // Size in bytes
        bool is_device_mem;          // True if GPU memory, false if host memory
        uint64_t key;                // Remote key for RDMA operations
        void* desc;                  // Local descriptor
        std::chrono::time_point<std::chrono::steady_clock> registered_time;
    };

    /**
     * Statistics about registered memory regions
     */
    struct MRStats {
        size_t total_count;          // Total number of registered MRs
        size_t host_count;           // Number of host memory MRs
        size_t device_count;         // Number of device memory MRs
        size_t total_host_bytes;     // Total host memory registered (bytes)
        size_t total_device_bytes;   // Total device memory registered (bytes)
    };

    MRManager();
    ~MRManager();

    // Disable copy and move
    MRManager(const MRManager&) = delete;
    MRManager& operator=(const MRManager&) = delete;

    // ========================================================================
    // Core MR Management Operations
    // ========================================================================

    /**
     * Add a newly registered MR to the manager
     * @param mr Libfabric MR handle
     * @param addr Base address of the memory region
     * @param size Size of the memory region in bytes
     * @param is_device_mem True if GPU memory, false if host memory
     * @return true on success, false if address already registered
     */
    bool add_mr(struct fid_mr* mr, void* addr, size_t size, bool is_device_mem);

    /**
     * Remove an MR from management (does not deregister, just removes from tracking)
     * @param mr MR handle to remove
     * @return true if found and removed, false otherwise
     */
    bool remove_mr(struct fid_mr* mr);

    /**
     * Remove an MR by its base address
     * @param addr Base address of the memory region
     * @return true if found and removed, false otherwise
     */
    bool remove_by_addr(void* addr);

    // ========================================================================
    // Query Operations
    // ========================================================================

    /**
     * Find MR by base address
     * @param addr Base address to search for
     * @return MR handle if found, nullptr otherwise
     */
    struct fid_mr* find_by_addr(void* addr);

    /**
     * Find MR by remote key
     * @param key Remote key to search for
     * @return MR handle if found, nullptr otherwise
     */
    struct fid_mr* find_by_key(uint64_t key);

    /**
     * Get full MR information by base address
     * @param addr Base address to search for
     * @return Pointer to MRInfo if found, nullptr otherwise (pointer valid until next modification)
     */
    const MRInfo* get_info_by_addr(void* addr) const;

    /**
     * Get full MR information by remote key
     * @param key Remote key to search for
     * @return Pointer to MRInfo if found, nullptr otherwise (pointer valid until next modification)
     */
    const MRInfo* get_info_by_key(uint64_t key) const;

    /**
     * Check if an address is already registered
     * @param addr Address to check
     * @return true if registered, false otherwise
     */
    bool is_registered(void* addr) const;

    /**
     * Get list of all registered addresses
     * @return Vector of all registered base addresses
     */
    std::vector<void*> get_all_addresses() const;

    // ========================================================================
    // Statistics
    // ========================================================================

    /**
     * Get current statistics
     * @return Structure containing current statistics
     */
    MRStats get_stats() const;

    /**
     * Print statistics to log (Info level)
     */
    void print_stats() const;

    // ========================================================================
    // Cleanup
    // ========================================================================

    /**
     * Remove all MRs from tracking (does not deregister them)
     */
    void clear_all();

    /**
     * Get number of currently tracked MRs
     * @return Number of registered MRs
     */
    size_t get_mr_count() const;

private:
    // Primary index: address -> MR info
    std::map<void*, MRInfo> mrs_by_addr_;

    // Secondary index: remote key -> address (for fast key lookups)
    std::map<uint64_t, void*> mrs_by_key_;

    // Mutex for thread-safe access
    mutable std::mutex mutex_;

    // Statistics counters (protected by mutex_)
    size_t host_count_;
    size_t device_count_;
    size_t total_host_bytes_;
    size_t total_device_bytes_;

    // Helper to update statistics when adding MR
    void update_stats_add(size_t size, bool is_device_mem);

    // Helper to update statistics when removing MR
    void update_stats_remove(size_t size, bool is_device_mem);
};

// ============================================================================
// OFI - OpenFabrics Interface
// ============================================================================

class OFI {
public:
  OFI(int rank);
  ~OFI();

  bool ofi_initialize();
  bool ofi_finalize();

  // ========================================================================
  // Memory Registration Functions
  // ========================================================================

  /**
   * Register memory region for RDMA operations
   * This function now automatically tracks the MR in MRManager
   * @param buf Pointer to memory buffer (host or device)
   * @param size Size of memory region in bytes
   * @param is_device_mem True if GPU memory, false if host memory
   * @return MR handle on success, nullptr on failure
   */
  struct fid_mr* register_memory(void* buf, size_t size, bool is_device_mem);

  /**
   * Deregister memory region
   * This function automatically removes the MR from MRManager
   * @param mr MR handle to deregister
   */
  void deregister_memory(struct fid_mr* mr);

  /**
   * Deregister memory region by address
   * @param addr Base address of the memory region
   * @return true if found and deregistered, false otherwise
   */
  bool deregister_memory_by_addr(void* addr);

  // ========================================================================
  // MR Query Functions
  // ========================================================================

  /**
   * Find registered MR by address
   * @param addr Base address to search for
   * @return MR handle if found, nullptr otherwise
   */
  struct fid_mr* find_mr_by_addr(void* addr);

  /**
   * Find registered MR by remote key
   * @param key Remote key to search for
   * @return MR handle if found, nullptr otherwise
   */
  struct fid_mr* find_mr_by_key(uint64_t key);

  /**
   * Check if address is registered
   * @param addr Address to check
   * @return true if registered, false otherwise
   */
  bool is_memory_registered(void* addr);

  /**
   * Get MR information
   * @param addr Base address of the memory region
   * @return Pointer to MRInfo if found, nullptr otherwise
   */
  const MRManager::MRInfo* get_mr_info(void* addr);

  // ========================================================================
  // MR Statistics
  // ========================================================================

  /**
   * Get MR statistics
   * @return Structure containing current MR statistics
   */
  MRManager::MRStats get_mr_stats();

  /**
   * Print MR statistics to log
   */
  void print_mr_stats();

  // ========================================================================
  // Accessors
  // ========================================================================

  struct fid_domain* get_domain() { return domain; }
  struct fid_ep* get_endpoint() { return ep; }
  struct fi_info* get_info() { return cxi_info; }
  MRManager* get_mr_manager() { return &mr_manager_; }

private:
    bool ofi_initialized = false;

    struct fi_info *hints;
    struct fi_info *info;
    struct fi_info *cxi_info;
    struct fid_fabric *fabric;
    struct fid_domain *domain;
    struct fid_av *av;
    struct fid_cq *cq;
    struct fid_ep *ep;

    int rank;
    int device_id;

    // MR management
    MRManager mr_manager_;
};