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
// CntrManager - Counter Management for Deferred Work Queue (DWQ)
// ============================================================================

/**
 * CntrManager provides centralized management of libfabric counters for DWQ.
 *
 * Features:
 * - Manages 32 counters (16 triggering + 16 completion)
 * - Obtains MMIO addresses for each counter
 * - Registers MMIO with HIP for GPU access (if USE_AMDGPU)
 * - Provides GPU device pointers for doorbell operations
 * - Thread-safe counter allocation and release
 */
class CntrManager {
public:
    // Number of counter pairs (trigger + completion)
    static constexpr int NUM_CNTR_PAIRS = 16;
    static constexpr int TOTAL_CNTRS = NUM_CNTR_PAIRS * 2;  // 32 total

    /**
     * Information for a single counter
     */
    struct CntrInfo {
        struct fid_cntr* cntr;              // Libfabric counter handle
        struct fi_cxi_cntr_ops* ops;        // CXI counter ops interface
        void* mmio_addr;                    // MMIO address (host accessible)
        size_t mmio_len;                    // MMIO region length

#ifdef USE_AMDGPU
        volatile uint64_t* dev_addr;        // GPU device pointer to MMIO
        bool hip_registered;                // Whether HIP registration succeeded
#endif

        bool allocated;                     // Whether this counter is in use
        int index;                          // Counter index (0-15 for trigger, 0-15 for completion)
        bool is_trigger;                    // true=trigger, false=completion
    };

    /**
     * Counter pair (trigger + completion)
     */
    struct CntrPair {
        int index;                          // Pair index (0-15)
        CntrInfo* trigger;                  // Triggering counter
        CntrInfo* completion;               // Completion counter
        bool allocated;                     // Whether this pair is in use
    };

    /**
     * Statistics about counter usage
     */
    struct CntrStats {
        int total_pairs;                    // Total number of pairs (16)
        int allocated_pairs;                // Number of allocated pairs
        int free_pairs;                     // Number of free pairs
        int total_trigger_cntrs;            // Total triggering counters
        int total_completion_cntrs;         // Total completion counters
    };

    CntrManager();
    ~CntrManager();

    // Disable copy and move
    CntrManager(const CntrManager&) = delete;
    CntrManager& operator=(const CntrManager&) = delete;

    // ========================================================================
    // Initialization
    // ========================================================================

    /**
     * Initialize all counters (must be called during OFI initialization)
     * @param domain Libfabric domain
     * @return true on success, false on failure
     */
    bool initialize(struct fid_domain* domain);

    /**
     * Cleanup all counters
     */
    void finalize();

    /**
     * Check if manager is initialized
     */
    bool is_initialized() const { return initialized_; }

    // ========================================================================
    // Counter Allocation
    // ========================================================================

    /**
     * Allocate a counter pair (trigger + completion)
     * @param pair_out Pointer to store allocated pair info
     * @return true on success, false if no pairs available
     */
    bool allocate_pair(CntrPair** pair_out);

    /**
     * Release a counter pair
     * @param pair Pair to release
     * @return true on success, false if pair was not allocated
     */
    bool release_pair(CntrPair* pair);

    /**
     * Release a counter pair by index
     * @param index Pair index (0-15)
     * @return true on success, false if not found or not allocated
     */
    bool release_pair_by_index(int index);

    // ========================================================================
    // Query Operations
    // ========================================================================

    /**
     * Get counter pair by index
     * @param index Pair index (0-15)
     * @return Pointer to pair if valid, nullptr otherwise
     */
    CntrPair* get_pair(int index);

    /**
     * Get triggering counter by pair index
     * @param pair_index Pair index (0-15)
     * @return Pointer to trigger counter info, nullptr if invalid
     */
    CntrInfo* get_trigger_cntr(int pair_index);

    /**
     * Get completion counter by pair index
     * @param pair_index Pair index (0-15)
     * @return Pointer to completion counter info, nullptr if invalid
     */
    CntrInfo* get_completion_cntr(int pair_index);

    /**
     * Find first available (free) counter pair
     * @return Index of free pair (0-15), or -1 if none available
     */
    int find_free_pair() const;

    /**
     * Get number of allocated pairs
     */
    int get_allocated_count() const;

    /**
     * Get number of free pairs
     */
    int get_free_count() const;

    // ========================================================================
    // Statistics
    // ========================================================================

    /**
     * Get current statistics
     * @return Structure containing current statistics
     */
    CntrStats get_stats() const;

    /**
     * Print statistics to log (Info level)
     */
    void print_stats() const;

private:
    bool initialized_;
    struct fid_domain* domain_;

    // Arrays for all counters
    CntrInfo trigger_cntrs_[NUM_CNTR_PAIRS];      // 16 triggering counters
    CntrInfo completion_cntrs_[NUM_CNTR_PAIRS];   // 16 completion counters
    CntrPair pairs_[NUM_CNTR_PAIRS];              // 16 counter pairs

    // Allocation tracking
    int allocated_count_;

    // Mutex for thread-safe access
    mutable std::mutex mutex_;

    // Helper functions
    bool create_counter(CntrInfo* info, int index, bool is_trigger);
    void destroy_counter(CntrInfo* info);
    bool setup_mmio(CntrInfo* info);
#ifdef USE_AMDGPU
    bool register_with_hip(CntrInfo* info);
    void unregister_from_hip(CntrInfo* info);
#endif
};

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
  // Counter Management Functions
  // ========================================================================

  /**
   * Allocate a counter pair for DWQ operations
   * @param pair_out Pointer to store allocated pair
   * @return true on success, false if no pairs available
   */
  bool allocate_cntr_pair(CntrManager::CntrPair** pair_out);

  /**
   * Release a counter pair
   * @param pair Pair to release
   * @return true on success
   */
  bool release_cntr_pair(CntrManager::CntrPair* pair);

  /**
   * Release counter pair by index
   * @param index Pair index (0-15)
   * @return true on success
   */
  bool release_cntr_pair_by_index(int index);

  /**
   * Get counter pair by index
   * @param index Pair index (0-15)
   * @return Pointer to pair, or nullptr if invalid
   */
  CntrManager::CntrPair* get_cntr_pair(int index);

  /**
   * Get counter statistics
   * @return Structure containing counter statistics
   */
  CntrManager::CntrStats get_cntr_stats();

  /**
   * Print counter statistics to log
   */
  void print_cntr_stats();

  // ========================================================================
  // Accessors
  // ========================================================================

  struct fid_domain* get_domain() { return domain; }
  struct fid_ep* get_endpoint() { return ep; }
  struct fi_info* get_info() { return cxi_info; }
  MRManager* get_mr_manager() { return &mr_manager_; }
  CntrManager* get_cntr_manager() { return &cntr_manager_; }

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

    // Counter management for DWQ
    CntrManager cntr_manager_;
};