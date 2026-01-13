#include "ofi.hpp"
#include "../bootstrap/common.hpp"
#include <map>
#include <string>
#include <algorithm>

// ============================================================================
// CntrManager Implementation
// ============================================================================

CntrManager::CntrManager()
    : initialized_(false)
    , domain_(nullptr)
    , allocated_count_(0) {

    // Initialize all counter info structures
    for (int i = 0; i < NUM_CNTR_PAIRS; i++) {
        trigger_cntrs_[i] = {};
        trigger_cntrs_[i].index = i;
        trigger_cntrs_[i].is_trigger = true;
        trigger_cntrs_[i].allocated = false;

        completion_cntrs_[i] = {};
        completion_cntrs_[i].index = i;
        completion_cntrs_[i].is_trigger = false;
        completion_cntrs_[i].allocated = false;

        pairs_[i].index = i;
        pairs_[i].trigger = &trigger_cntrs_[i];
        pairs_[i].completion = &completion_cntrs_[i];
        pairs_[i].allocated = false;
    }
}

CntrManager::~CntrManager() {
    if (initialized_) {
        finalize();
    }
}

bool CntrManager::initialize(struct fid_domain* domain) {
    if (initialized_) {
        OPENGDA_Warn("cntr_manager", "Already initialized");
        return false;
    }

    if (!domain) {
        OPENGDA_Error("cntr_manager", "Invalid domain");
        return false;
    }

    domain_ = domain;

    OPENGDA_Info("cntr_manager", "Initializing %d counter pairs (%d total counters)...",
                 NUM_CNTR_PAIRS, TOTAL_CNTRS);

    // Create all triggering counters
    for (int i = 0; i < NUM_CNTR_PAIRS; i++) {
        if (!create_counter(&trigger_cntrs_[i], i, true)) {
            OPENGDA_Error("cntr_manager", "Failed to create trigger counter %d", i);
            finalize();
            return false;
        }

        if (!setup_mmio(&trigger_cntrs_[i])) {
            OPENGDA_Error("cntr_manager", "Failed to setup MMIO for trigger counter %d", i);
            finalize();
            return false;
        }

#ifdef USE_AMDGPU
        if (!register_with_hip(&trigger_cntrs_[i])) {
            OPENGDA_Warn("cntr_manager", "Failed to register trigger counter %d with HIP", i);
            // Continue - may still work without GPU access
        }
#endif
#ifdef USE_NVGPU
        if (!register_with_cuda(&trigger_cntrs_[i])) {
            OPENGDA_Warn("cntr_manager", "Failed to register trigger counter %d with CUDA", i);
            // Continue - may still work without GPU access
        }
#endif
    }

    // Create all completion counters
    for (int i = 0; i < NUM_CNTR_PAIRS; i++) {
        if (!create_counter(&completion_cntrs_[i], i, false)) {
            OPENGDA_Error("cntr_manager", "Failed to create completion counter %d", i);
            finalize();
            return false;
        }

        if (!setup_mmio(&completion_cntrs_[i])) {
            OPENGDA_Error("cntr_manager", "Failed to setup MMIO for completion counter %d", i);
            finalize();
            return false;
        }

#ifdef USE_AMDGPU
        if (!register_with_hip(&completion_cntrs_[i])) {
            OPENGDA_Warn("cntr_manager", "Failed to register completion counter %d with HIP", i);
            // Continue - may still work without GPU access
        }
#endif
#ifdef USE_NVGPU
        if (!register_with_cuda(&completion_cntrs_[i])) {
            OPENGDA_Warn("cntr_manager", "Failed to register completion counter %d with CUDA", i);
            // Continue - may still work without GPU access
        }
#endif
    }

    initialized_ = true;
    OPENGDA_Info("cntr_manager", "Successfully initialized %d counter pairs", NUM_CNTR_PAIRS);
    return true;
}

void CntrManager::finalize() {
    if (!initialized_) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    OPENGDA_Info("cntr_manager", "Finalizing counter manager...");

    // Destroy all completion counters
    for (int i = 0; i < NUM_CNTR_PAIRS; i++) {
        destroy_counter(&completion_cntrs_[i]);
    }

    // Destroy all triggering counters
    for (int i = 0; i < NUM_CNTR_PAIRS; i++) {
        destroy_counter(&trigger_cntrs_[i]);
    }

    allocated_count_ = 0;
    initialized_ = false;
    domain_ = nullptr;

    OPENGDA_Info("cntr_manager", "Counter manager finalized");
}

bool CntrManager::create_counter(CntrInfo* info, int index, bool is_trigger) {
    if (!info || !domain_) {
        return false;
    }

    struct fi_cntr_attr cntr_attr = {};
    cntr_attr.events = FI_CNTR_EVENTS_COMP;
    cntr_attr.wait_obj = FI_WAIT_UNSPEC;

    int ret = fi_cntr_open(domain_, &cntr_attr, &info->cntr, NULL);
    if (ret) {
        OPENGDA_Error("cntr_manager", "fi_cntr_open failed for %s counter %d: %s (%d)",
                      is_trigger ? "trigger" : "completion", index,
                      fi_strerror(-ret), ret);
        return false;
    }

    // Get CXI counter ops
    ret = fi_open_ops(&info->cntr->fid, FI_CXI_COUNTER_OPS, 0,
                      (void**)&info->ops, NULL);
    if (ret) {
        OPENGDA_Error("cntr_manager", "fi_open_ops failed for %s counter %d: %s (%d)",
                      is_trigger ? "trigger" : "completion", index,
                      fi_strerror(-ret), ret);
        fi_close(&info->cntr->fid);
        info->cntr = nullptr;
        return false;
    }

    OPENGDA_Debug("cntr_manager", "Created %s counter %d",
                  is_trigger ? "trigger" : "completion", index);

    return true;
}

void CntrManager::destroy_counter(CntrInfo* info) {
    if (!info) {
        return;
    }

#ifdef USE_AMDGPU
    if (info->hip_registered) {
        unregister_from_hip(info);
    }
#endif
#ifdef USE_NVGPU
    if (info->cuda_registered) {
        unregister_from_cuda(info);
    }
#endif

    if (info->cntr) {
        fi_close(&info->cntr->fid);
        info->cntr = nullptr;
    }

    info->ops = nullptr;
    info->mmio_addr = nullptr;
    info->mmio_len = 0;
#ifdef USE_AMDGPU
    info->dev_addr = nullptr;
    info->hip_registered = false;
#endif
#ifdef USE_NVGPU
    info->dev_addr = nullptr;
    info->cuda_registered = false;
#endif
}

bool CntrManager::setup_mmio(CntrInfo* info) {
    if (!info || !info->ops) {
        return false;
    }

    int ret = info->ops->get_mmio_addr(&info->cntr->fid, &info->mmio_addr,
                                       &info->mmio_len);
    if (ret) {
        OPENGDA_Error("cntr_manager", "get_mmio_addr failed for %s counter %d: %s (%d)",
                      info->is_trigger ? "trigger" : "completion", info->index,
                      fi_strerror(-ret), ret);
        return false;
    }

    OPENGDA_Debug("cntr_manager", "Got MMIO for %s counter %d: addr=%p, len=%zu",
                  info->is_trigger ? "trigger" : "completion", info->index,
                  info->mmio_addr, info->mmio_len);

    return true;
}

#ifdef USE_AMDGPU
bool CntrManager::register_with_hip(CntrInfo* info) {
    if (!info || !info->mmio_addr) {
        return false;
    }

    // Register MMIO memory with HIP
    hipError_t hip_err = hipHostRegister(info->mmio_addr, info->mmio_len,
                                         hipHostRegisterMapped);
    if (hip_err != hipSuccess) {
        OPENGDA_Error("cntr_manager", "hipHostRegister failed for %s counter %d: %s (%d)",
                      info->is_trigger ? "trigger" : "completion", info->index,
                      hipGetErrorString(hip_err), hip_err);
        return false;
    }

    // Get GPU device pointer
    hip_err = hipHostGetDevicePointer((void**)&info->dev_addr, info->mmio_addr, 0);
    if (hip_err != hipSuccess) {
        OPENGDA_Error("cntr_manager", "hipHostGetDevicePointer failed for %s counter %d: %s (%d)",
                      info->is_trigger ? "trigger" : "completion", info->index,
                      hipGetErrorString(hip_err), hip_err);
        hipError_t unreg_err = hipHostUnregister(info->mmio_addr);
        (void)unreg_err;  // Suppress unused warning
        return false;
    }
    info->hip_registered = true;

    OPENGDA_Debug("cntr_manager", "Registered %s counter %d with HIP: dev_addr=%p",
                  info->is_trigger ? "trigger" : "completion", info->index,
                  (void*)info->dev_addr);

    return true;
}

void CntrManager::unregister_from_hip(CntrInfo* info) {
    if (!info || !info->hip_registered || !info->mmio_addr) {
        return;
    }

    hipError_t hip_err = hipHostUnregister(info->mmio_addr);
    if (hip_err != hipSuccess) {
        OPENGDA_Warn("cntr_manager", "hipHostUnregister failed for %s counter %d: %s (%d)",
                     info->is_trigger ? "trigger" : "completion", info->index,
                     hipGetErrorString(hip_err), hip_err);
    }

    info->dev_addr = nullptr;
    info->hip_registered = false;
}
#endif

#ifdef USE_NVGPU
bool CntrManager::register_with_cuda(CntrInfo* info) {
    if (!info || !info->mmio_addr) {
        return false;
    }

    // Register MMIO memory with CUDA
    cudaError_t cuda_err = cudaHostRegister(info->mmio_addr, info->mmio_len,
                                            cudaHostRegisterMapped);
    if (cuda_err != cudaSuccess) {
        OPENGDA_Error("cntr_manager", "cudaHostRegister failed for %s counter %d: %s (%d)",
                      info->is_trigger ? "trigger" : "completion", info->index,
                      cudaGetErrorString(cuda_err), cuda_err);
        return false;
    }

    // Get GPU device pointer
    cuda_err = cudaHostGetDevicePointer((void**)&info->dev_addr, info->mmio_addr, 0);
    if (cuda_err != cudaSuccess) {
        OPENGDA_Error("cntr_manager", "cudaHostGetDevicePointer failed for %s counter %d: %s (%d)",
                      info->is_trigger ? "trigger" : "completion", info->index,
                      cudaGetErrorString(cuda_err), cuda_err);
        cudaError_t unreg_err = cudaHostUnregister(info->mmio_addr);
        (void)unreg_err;  // Suppress unused warning
        return false;
    }
    info->cuda_registered = true;

    OPENGDA_Debug("cntr_manager", "Registered %s counter %d with CUDA: dev_addr=%p",
                  info->is_trigger ? "trigger" : "completion", info->index,
                  (void*)info->dev_addr);

    return true;
}

void CntrManager::unregister_from_cuda(CntrInfo* info) {
    if (!info || !info->cuda_registered || !info->mmio_addr) {
        return;
    }

    cudaError_t cuda_err = cudaHostUnregister(info->mmio_addr);
    if (cuda_err != cudaSuccess) {
        OPENGDA_Warn("cntr_manager", "cudaHostUnregister failed for %s counter %d: %s (%d)",
                     info->is_trigger ? "trigger" : "completion", info->index,
                     cudaGetErrorString(cuda_err), cuda_err);
    }

    info->dev_addr = nullptr;
    info->cuda_registered = false;
}
#endif

bool CntrManager::allocate_pair(CntrPair** pair_out) {
    if (!pair_out) {
        OPENGDA_Error("cntr_manager", "Invalid output parameter");
        return false;
    }

    if (!initialized_) {
        OPENGDA_Error("cntr_manager", "Manager not initialized");
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Find first free pair
    for (int i = 0; i < NUM_CNTR_PAIRS; i++) {
        if (!pairs_[i].allocated) {
            pairs_[i].allocated = true;
            pairs_[i].trigger->allocated = true;
            pairs_[i].completion->allocated = true;
            allocated_count_++;

            *pair_out = &pairs_[i];

            OPENGDA_Debug("cntr_manager", "Allocated counter pair %d", i);
            return true;
        }
    }

    OPENGDA_Warn("cntr_manager", "No free counter pairs available (%d/%d allocated)",
                 allocated_count_, NUM_CNTR_PAIRS);
    return false;
}

bool CntrManager::release_pair(CntrPair* pair) {
    if (!pair) {
        OPENGDA_Error("cntr_manager", "Invalid pair");
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    if (!pair->allocated) {
        OPENGDA_Warn("cntr_manager", "Pair %d not allocated", pair->index);
        return false;
    }

    pair->allocated = false;
    pair->trigger->allocated = false;
    pair->completion->allocated = false;

    if (allocated_count_ > 0) {
        allocated_count_--;
    }

    OPENGDA_Debug("cntr_manager", "Released counter pair %d", pair->index);
    return true;
}

bool CntrManager::release_pair_by_index(int index) {
    if (index < 0 || index >= NUM_CNTR_PAIRS) {
        OPENGDA_Error("cntr_manager", "Invalid pair index %d", index);
        return false;
    }

    return release_pair(&pairs_[index]);
}

CntrManager::CntrPair* CntrManager::get_pair(int index) {
    if (index < 0 || index >= NUM_CNTR_PAIRS) {
        return nullptr;
    }

    return &pairs_[index];
}

CntrManager::CntrInfo* CntrManager::get_trigger_cntr(int pair_index) {
    if (pair_index < 0 || pair_index >= NUM_CNTR_PAIRS) {
        return nullptr;
    }

    return &trigger_cntrs_[pair_index];
}

CntrManager::CntrInfo* CntrManager::get_completion_cntr(int pair_index) {
    if (pair_index < 0 || pair_index >= NUM_CNTR_PAIRS) {
        return nullptr;
    }

    return &completion_cntrs_[pair_index];
}

int CntrManager::find_free_pair() const {
    std::lock_guard<std::mutex> lock(mutex_);

    for (int i = 0; i < NUM_CNTR_PAIRS; i++) {
        if (!pairs_[i].allocated) {
            return i;
        }
    }

    return -1;  // No free pairs
}

int CntrManager::get_allocated_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocated_count_;
}

int CntrManager::get_free_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return NUM_CNTR_PAIRS - allocated_count_;
}

CntrManager::CntrStats CntrManager::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    CntrStats stats;
    stats.total_pairs = NUM_CNTR_PAIRS;
    stats.allocated_pairs = allocated_count_;
    stats.free_pairs = NUM_CNTR_PAIRS - allocated_count_;
    stats.total_trigger_cntrs = NUM_CNTR_PAIRS;
    stats.total_completion_cntrs = NUM_CNTR_PAIRS;

    return stats;
}

void CntrManager::print_stats() const {
    CntrStats stats = get_stats();

    OPENGDA_Info("cntr_manager", "=== Counter Statistics ===");
    OPENGDA_Info("cntr_manager", "Total pairs: %d", stats.total_pairs);
    OPENGDA_Info("cntr_manager", "  Allocated: %d", stats.allocated_pairs);
    OPENGDA_Info("cntr_manager", "  Free: %d", stats.free_pairs);
    OPENGDA_Info("cntr_manager", "Total counters: %d (%d trigger + %d completion)",
                 TOTAL_CNTRS, stats.total_trigger_cntrs, stats.total_completion_cntrs);

#ifdef USE_AMDGPU
    int hip_registered_count = 0;
    for (int i = 0; i < NUM_CNTR_PAIRS; i++) {
        if (trigger_cntrs_[i].hip_registered) hip_registered_count++;
        if (completion_cntrs_[i].hip_registered) hip_registered_count++;
    }
    OPENGDA_Info("cntr_manager", "HIP registered: %d/%d counters",
                 hip_registered_count, TOTAL_CNTRS);
#endif
#ifdef USE_NVGPU
    int cuda_registered_count = 0;
    for (int i = 0; i < NUM_CNTR_PAIRS; i++) {
        if (trigger_cntrs_[i].cuda_registered) cuda_registered_count++;
        if (completion_cntrs_[i].cuda_registered) cuda_registered_count++;
    }
    OPENGDA_Info("cntr_manager", "CUDA registered: %d/%d counters",
                 cuda_registered_count, TOTAL_CNTRS);
#endif
}

// ============================================================================
// MRManager Implementation
// ============================================================================

MRManager::MRManager()
    : host_count_(0)
    , device_count_(0)
    , total_host_bytes_(0)
    , total_device_bytes_(0) {
}

MRManager::~MRManager() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!mrs_by_addr_.empty()) {
        OPENGDA_Warn("mr_manager", "Destroying MRManager with %zu MRs still registered",
                     mrs_by_addr_.size());
    }
}

bool MRManager::add_mr(struct fid_mr* mr, void* addr, size_t size, bool is_device_mem) {
    if (!mr || !addr) {
        OPENGDA_Error("mr_manager", "Cannot add null MR or address");
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Check if address is already registered
    if (mrs_by_addr_.find(addr) != mrs_by_addr_.end()) {
        OPENGDA_Warn("mr_manager", "Address %p already registered", addr);
        return false;
    }

    // Get MR key and descriptor
    uint64_t key = fi_mr_key(mr);
    void* desc = fi_mr_desc(mr);

    // Create MR info
    MRInfo info;
    info.mr = mr;
    info.addr = addr;
    info.size = size;
    info.is_device_mem = is_device_mem;
    info.key = key;
    info.desc = desc;
    info.registered_time = std::chrono::steady_clock::now();

    // Add to indices
    mrs_by_addr_[addr] = info;
    mrs_by_key_[key] = addr;

    // Update statistics
    update_stats_add(size, is_device_mem);

    OPENGDA_Debug("mr_manager", "Added MR: addr=%p, size=%zu, key=0x%lx, %s",
                  addr, size, key, is_device_mem ? "GPU" : "Host");

    return true;
}

bool MRManager::remove_mr(struct fid_mr* mr) {
    if (!mr) {
        OPENGDA_Error("mr_manager", "Cannot remove null MR");
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Find the MR by searching through all entries
    for (auto it = mrs_by_addr_.begin(); it != mrs_by_addr_.end(); ++it) {
        if (it->second.mr == mr) {
            uint64_t key = it->second.key;
            size_t size = it->second.size;
            bool is_device_mem = it->second.is_device_mem;

            OPENGDA_Debug("mr_manager", "Removed MR: addr=%p, size=%zu, key=0x%lx",
                          it->second.addr, size, key);

            // Remove from indices
            mrs_by_key_.erase(key);
            mrs_by_addr_.erase(it);

            // Update statistics
            update_stats_remove(size, is_device_mem);

            return true;
        }
    }

    OPENGDA_Warn("mr_manager", "MR %p not found in manager", mr);
    return false;
}

bool MRManager::remove_by_addr(void* addr) {
    if (!addr) {
        OPENGDA_Error("mr_manager", "Cannot remove null address");
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = mrs_by_addr_.find(addr);
    if (it == mrs_by_addr_.end()) {
        OPENGDA_Warn("mr_manager", "Address %p not found in manager", addr);
        return false;
    }

    uint64_t key = it->second.key;
    size_t size = it->second.size;
    bool is_device_mem = it->second.is_device_mem;

    // Remove from indices
    mrs_by_key_.erase(key);
    mrs_by_addr_.erase(it);

    // Update statistics
    update_stats_remove(size, is_device_mem);

    OPENGDA_Debug("mr_manager", "Removed MR by address: addr=%p, size=%zu, key=0x%lx",
                  addr, size, key);

    return true;
}

struct fid_mr* MRManager::find_by_addr(void* addr) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = mrs_by_addr_.find(addr);
    if (it != mrs_by_addr_.end()) {
        return it->second.mr;
    }
    return nullptr;
}

struct fid_mr* MRManager::find_by_key(uint64_t key) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto key_it = mrs_by_key_.find(key);
    if (key_it != mrs_by_key_.end()) {
        void* addr = key_it->second;
        auto addr_it = mrs_by_addr_.find(addr);
        if (addr_it != mrs_by_addr_.end()) {
            return addr_it->second.mr;
        }
    }
    return nullptr;
}

const MRManager::MRInfo* MRManager::get_info_by_addr(void* addr) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = mrs_by_addr_.find(addr);
    if (it != mrs_by_addr_.end()) {
        return &(it->second);
    }
    return nullptr;
}

const MRManager::MRInfo* MRManager::find_containing(void* addr) const {
    std::lock_guard<std::mutex> lock(mutex_);

    uintptr_t target = reinterpret_cast<uintptr_t>(addr);

    for (const auto& pair : mrs_by_addr_) {
        uintptr_t base = reinterpret_cast<uintptr_t>(pair.first);
        size_t size = pair.second.size;
        if (target >= base && target < base + size) {
            return &(pair.second);
        }
    }
    return nullptr;
}

const MRManager::MRInfo* MRManager::get_info_by_key(uint64_t key) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto key_it = mrs_by_key_.find(key);
    if (key_it != mrs_by_key_.end()) {
        void* addr = key_it->second;
        auto addr_it = mrs_by_addr_.find(addr);
        if (addr_it != mrs_by_addr_.end()) {
            return &(addr_it->second);
        }
    }
    return nullptr;
}

bool MRManager::is_registered(void* addr) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return mrs_by_addr_.find(addr) != mrs_by_addr_.end();
}

std::vector<void*> MRManager::get_all_addresses() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<void*> addresses;
    addresses.reserve(mrs_by_addr_.size());

    for (const auto& pair : mrs_by_addr_) {
        addresses.push_back(pair.first);
    }

    return addresses;
}

MRManager::MRStats MRManager::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    MRStats stats;
    stats.total_count = mrs_by_addr_.size();
    stats.host_count = host_count_;
    stats.device_count = device_count_;
    stats.total_host_bytes = total_host_bytes_;
    stats.total_device_bytes = total_device_bytes_;

    return stats;
}

void MRManager::print_stats() const {
    MRStats stats = get_stats();

    OPENGDA_Info("mr_manager", "=== MR Statistics ===");
    OPENGDA_Info("mr_manager", "Total MRs: %zu", stats.total_count);
    OPENGDA_Info("mr_manager", "  Host MRs: %zu (%zu bytes, %.2f MB)",
                 stats.host_count, stats.total_host_bytes,
                 stats.total_host_bytes / (1024.0 * 1024.0));
    OPENGDA_Info("mr_manager", "  Device MRs: %zu (%zu bytes, %.2f MB)",
                 stats.device_count, stats.total_device_bytes,
                 stats.total_device_bytes / (1024.0 * 1024.0));
    OPENGDA_Info("mr_manager", "Total memory: %.2f MB",
                 (stats.total_host_bytes + stats.total_device_bytes) / (1024.0 * 1024.0));
}

void MRManager::clear_all() {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t count = mrs_by_addr_.size();
    mrs_by_addr_.clear();
    mrs_by_key_.clear();

    host_count_ = 0;
    device_count_ = 0;
    total_host_bytes_ = 0;
    total_device_bytes_ = 0;

    OPENGDA_Info("mr_manager", "Cleared all MRs (%zu removed)", count);
}

size_t MRManager::get_mr_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return mrs_by_addr_.size();
}

void MRManager::update_stats_add(size_t size, bool is_device_mem) {
    if (is_device_mem) {
        device_count_++;
        total_device_bytes_ += size;
    } else {
        host_count_++;
        total_host_bytes_ += size;
    }
}

void MRManager::update_stats_remove(size_t size, bool is_device_mem) {
    if (is_device_mem) {
        if (device_count_ > 0) device_count_--;
        if (total_device_bytes_ >= size) {
            total_device_bytes_ -= size;
        } else {
            total_device_bytes_ = 0;
        }
    } else {
        if (host_count_ > 0) host_count_--;
        if (total_host_bytes_ >= size) {
            total_host_bytes_ -= size;
        } else {
            total_host_bytes_ = 0;
        }
    }
}

// ============================================================================
// CompletionQueue Implementation
// ============================================================================

CompletionQueue::CompletionQueue()
    : next_seq_num_(1) {  // Start at 1, 0 is INVALID_SEQ
}

CompletionQueue::~CompletionQueue() {
    // Clear all tracking
    std::lock_guard<std::mutex> lock(mutex_);
    pending_ops_.clear();
    completed_seq_nums_.clear();
}

uint64_t CompletionQueue::allocate_seq_num() {
    return next_seq_num_.fetch_add(1, std::memory_order_relaxed);
}

void CompletionQueue::register_pending(uint64_t seq_num, DWQOperation* op) {
    if (seq_num == INVALID_SEQ || !op) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    pending_ops_[seq_num] = op;

    OPENGDA_Debug("cq", "Registered pending op: seq=%lu", seq_num);
}

void CompletionQueue::mark_completed(uint64_t seq_num) {
    if (seq_num == INVALID_SEQ) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Remove from pending
    auto it = pending_ops_.find(seq_num);
    if (it != pending_ops_.end()) {
        pending_ops_.erase(it);
    }

    // Add to completed
    completed_seq_nums_.insert(seq_num);

    OPENGDA_Debug("cq", "Marked completed: seq=%lu", seq_num);
}

void CompletionQueue::unregister(uint64_t seq_num) {
    if (seq_num == INVALID_SEQ) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Remove from pending
    pending_ops_.erase(seq_num);

    // Remove from completed
    completed_seq_nums_.erase(seq_num);

    OPENGDA_Debug("cq", "Unregistered op: seq=%lu", seq_num);
}

bool CompletionQueue::poll_one(uint64_t seq_num) {
    if (seq_num == INVALID_SEQ) {
        return false;
    }

    // First poll all pending operations to update completion status
    poll_pending_operations();

    std::lock_guard<std::mutex> lock(mutex_);

    // Check if in completed set
    return completed_seq_nums_.count(seq_num) > 0;
}

bool CompletionQueue::wait_one(uint64_t seq_num, int timeout_ms) {
    if (seq_num == INVALID_SEQ) {
        return false;
    }

    auto start = std::chrono::steady_clock::now();

    while (true) {
        if (poll_one(seq_num)) {
            return true;
        }

        // Check timeout
        if (timeout_ms >= 0) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - start).count();
            if (elapsed_ms >= timeout_ms) {
                return false;
            }
        }

        // Brief sleep to avoid spinning
        usleep(10);  // 10 microseconds
    }
}

std::vector<uint64_t> CompletionQueue::poll_any(size_t max_count) {
    // First poll all pending operations
    poll_pending_operations();

    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<uint64_t> result;
    result.reserve(std::min(max_count, completed_seq_nums_.size()));

    for (auto it = completed_seq_nums_.begin();
         it != completed_seq_nums_.end() && result.size() < max_count;
         ++it) {
        result.push_back(*it);
    }

    return result;
}

bool CompletionQueue::wait_batch(const std::vector<uint64_t>& seq_nums, int timeout_ms) {
    if (seq_nums.empty()) {
        return true;
    }

    auto start = std::chrono::steady_clock::now();

    while (true) {
        // Poll all pending
        poll_pending_operations();

        // Check if all in the batch are completed
        bool all_done = true;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (uint64_t seq : seq_nums) {
                if (seq != INVALID_SEQ && completed_seq_nums_.count(seq) == 0) {
                    all_done = false;
                    break;
                }
            }
        }

        if (all_done) {
            return true;
        }

        // Check timeout
        if (timeout_ms >= 0) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - start).count();
            if (elapsed_ms >= timeout_ms) {
                return false;
            }
        }

        usleep(10);
    }
}

bool CompletionQueue::wait_all(int timeout_ms) {
    auto start = std::chrono::steady_clock::now();

    while (true) {
        // Poll all pending
        poll_pending_operations();

        // Check if all pending are now completed
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (pending_ops_.empty()) {
                return true;
            }
        }

        // Check timeout
        if (timeout_ms >= 0) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - start).count();
            if (elapsed_ms >= timeout_ms) {
                return false;
            }
        }

        usleep(10);
    }
}

size_t CompletionQueue::get_pending_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pending_ops_.size();
}

size_t CompletionQueue::get_completed_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return completed_seq_nums_.size();
}

bool CompletionQueue::is_pending(uint64_t seq_num) const {
    if (seq_num == INVALID_SEQ) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    return pending_ops_.count(seq_num) > 0;
}

bool CompletionQueue::is_completed(uint64_t seq_num) const {
    if (seq_num == INVALID_SEQ) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    return completed_seq_nums_.count(seq_num) > 0;
}

void CompletionQueue::clear_completed() {
    std::lock_guard<std::mutex> lock(mutex_);
    completed_seq_nums_.clear();
}

std::vector<uint64_t> CompletionQueue::get_pending_seq_nums() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<uint64_t> result;
    result.reserve(pending_ops_.size());
    for (const auto& pair : pending_ops_) {
        result.push_back(pair.first);
    }
    return result;
}

void CompletionQueue::poll_pending_operations() {
    // Make a copy of pending ops to avoid holding lock while polling
    std::vector<std::pair<uint64_t, DWQOperation*>> ops_to_poll;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ops_to_poll.reserve(pending_ops_.size());
        for (const auto& pair : pending_ops_) {
            ops_to_poll.push_back(pair);
        }
    }

    // Poll each operation and mark completed ones
    for (const auto& pair : ops_to_poll) {
        uint64_t seq = pair.first;
        DWQOperation* op = pair.second;

        if (op && op->is_completed()) {
            mark_completed(seq);
        }
    }
}

// ============================================================================
// CompletionSignalPool Implementation
// ============================================================================

CompletionSignalPool::CompletionSignalPool()
    : base_addr_(nullptr)
    , mr_desc_(nullptr)
    , mr_key_(0)
    , pool_offset_in_mr_(0)
    , allocated_count_(0)
    , initialized_(false) {
    allocated_.reset();
}

CompletionSignalPool::~CompletionSignalPool() {
    finalize();
}

bool CompletionSignalPool::initialize(volatile uint64_t* base_addr, size_t pool_offset,
                                       void* mr_desc, uint64_t mr_key) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) {
        OPENGDA_Warn("signal_pool", "Already initialized");
        return false;
    }

    if (!base_addr) {
        OPENGDA_Error("signal_pool", "Invalid base address");
        return false;
    }

    base_addr_ = base_addr;
    pool_offset_in_mr_ = pool_offset;
    mr_desc_ = mr_desc;
    mr_key_ = mr_key;

    allocated_.reset();
    allocated_count_ = 0;
    initialized_ = true;

    OPENGDA_Info("signal_pool", "Initialized with %zu signals at offset %zu",
                 MAX_SIGNALS, pool_offset);
    return true;
}

void CompletionSignalPool::finalize() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        return;
    }

    if (allocated_count_ > 0) {
        OPENGDA_Warn("signal_pool", "%zu signals still allocated at finalize",
                    allocated_count_);
    }

    base_addr_ = nullptr;
    mr_desc_ = nullptr;
    mr_key_ = 0;
    pool_offset_in_mr_ = 0;
    allocated_.reset();
    allocated_count_ = 0;
    initialized_ = false;
}

bool CompletionSignalPool::allocate(volatile uint64_t** signal_out, size_t* offset_out) {
    if (!signal_out || !offset_out) {
        OPENGDA_Error("signal_pool", "Invalid output parameters");
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        OPENGDA_Error("signal_pool", "Pool not initialized");
        return false;
    }

    // Find first free slot
    for (size_t i = 0; i < MAX_SIGNALS; i++) {
        if (!allocated_.test(i)) {
            allocated_.set(i);
            allocated_count_++;

            // Each signal is SIGNAL_SIZE (128 bytes) apart for cache-line alignment
            *signal_out = (volatile uint64_t*)((char*)base_addr_ + i * SIGNAL_SIZE);
            *offset_out = pool_offset_in_mr_ + i * SIGNAL_SIZE;

            OPENGDA_Debug("signal_pool", "Allocated signal %zu, offset=%zu, total=%zu",
                         i, *offset_out, allocated_count_);
            return true;
        }
    }

    OPENGDA_Error("signal_pool", "No free signals available (%zu/%zu allocated)",
                 allocated_count_, MAX_SIGNALS);
    return false;
}

void CompletionSignalPool::release(volatile uint64_t* signal) {
    if (!signal) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        return;
    }

    // Calculate index from address (signals are SIGNAL_SIZE bytes apart)
    ptrdiff_t byte_diff = (char*)signal - (char*)base_addr_;
    if (byte_diff < 0 || (size_t)byte_diff >= POOL_SIZE) {
        OPENGDA_Warn("signal_pool", "Signal %p not in pool range", (void*)signal);
        return;
    }

    size_t index = (size_t)byte_diff / SIGNAL_SIZE;
    if (!allocated_.test(index)) {
        OPENGDA_Warn("signal_pool", "Signal %zu not allocated", index);
        return;
    }

    allocated_.reset(index);
    allocated_count_--;

    OPENGDA_Debug("signal_pool", "Released signal %zu, remaining=%zu",
                 index, allocated_count_);
}

size_t CompletionSignalPool::get_allocated_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocated_count_;
}

// ============================================================================
// OFI Implementation
// ============================================================================

#if defined(USE_AMDGPU) || defined(USE_NVGPU)
// Helper structure to store device affinity information
struct DeviceAffinity {
    std::string pci_id;           // PCI address like "0000:c1:00.0"
    hwloc_obj_type_t affinity_type;  // Type of affinity object (GROUP/PACKAGE/NUMANODE)
    int affinity_index;           // Logical index of affinity object
};

// Helper function to get device affinity using hwloc
// Returns true if affinity was found, false otherwise
static bool get_device_affinity(hwloc_topology_t topo, hwloc_obj_t osdev,
                                 DeviceAffinity& affinity) {
    // Get parent PCI device
    hwloc_obj_t pci_dev = osdev->parent;
    while (pci_dev && pci_dev->type != HWLOC_OBJ_PCI_DEVICE) {
        pci_dev = pci_dev->parent;
    }

    if (!pci_dev) return false;

    // Store PCI ID
    char pci_str[64];
    snprintf(pci_str, sizeof(pci_str), "%04x:%02x:%02x.%01x",
             pci_dev->attr->pcidev.domain,
             pci_dev->attr->pcidev.bus,
             pci_dev->attr->pcidev.dev,
             pci_dev->attr->pcidev.func);
    affinity.pci_id = pci_str;

    // Get CPU affinity
    hwloc_obj_t ancestor = hwloc_get_non_io_ancestor_obj(topo, pci_dev);
    if (!ancestor) return false;

    hwloc_bitmap_t cpuset = hwloc_bitmap_alloc();
    hwloc_bitmap_copy(cpuset, ancestor->cpuset);

    int first_cpu = hwloc_bitmap_first(cpuset);
    hwloc_bitmap_free(cpuset);

    if (first_cpu == -1) return false;

    // Find affinity object (Group/Package/NUMANODE)
    hwloc_obj_t pu = hwloc_get_pu_obj_by_os_index(topo, first_cpu);
    if (!pu) return false;

    hwloc_obj_t node_obj = pu->parent;
    while (node_obj) {
        if (node_obj->type == HWLOC_OBJ_GROUP ||
            node_obj->type == HWLOC_OBJ_PACKAGE ||
            node_obj->type == HWLOC_OBJ_NUMANODE) {
            affinity.affinity_type = node_obj->type;
            affinity.affinity_index = node_obj->logical_index;
            return true;
        }
        node_obj = node_obj->parent;
    }

    return false;
}
#endif

OFI::OFI(int rank, int size, Bootstrap* bootstrap) {
    this->rank_ = rank;
    this->size_ = size;
    this->bootstrap_ = bootstrap;
    this->device_id = 0; // Default to GPU 0
    this->local_ep_addr_ = nullptr;
    this->local_ep_addr_len_ = 0;
    this->local_fi_addr_ = FI_ADDR_NOTAVAIL;

    // Initialize default MR info structs
    memset(&default_host_mr_, 0, sizeof(default_host_mr_));
    memset(&default_gpu_mr_, 0, sizeof(default_gpu_mr_));

    // Initialize shared atomic operand (will be allocated in ofi_initialize)
    shared_atomic_operand_ = nullptr;
    shared_atomic_operand_mr_ = nullptr;

    // Initialize peer info array
    peers_.resize(size);
    for (int i = 0; i < size; i++) {
        peers_[i].rank = i;
        peers_[i].fi_addr = FI_ADDR_NOTAVAIL;
        peers_[i].valid = false;
        memset(&peers_[i].mr_info, 0, sizeof(PeerMRInfo));
        peers_[i].node_id = -1;
        peers_[i].same_node = false;
#ifdef USE_AMDGPU
        memset(&peers_[i].ipc_info, 0, sizeof(PeerIPCInfo));
        peers_[i].can_use_ipc = false;
#endif
    }

#ifdef USE_AMDGPU
    // Initialize IPC-related fields
    local_node_id_ = -1;
    memset(&local_ipc_handle_, 0, sizeof(local_ipc_handle_));
    local_ipc_handle_valid_ = false;
#endif

    // Initialize logging system
    opengda::LogContext::instance().init(rank);

    // Initialize default MR configuration (reads environment variables)
    init_default_mr_config();

    #ifdef USE_AMDGPU
    // Get PCI Bus ID of the first HIP-visible GPU
    char hip_pci_bus_id[32] = {0};
    int gpu_count = 0;
    bool gpu_found = false;

    hipError_t hip_err = hipGetDeviceCount(&gpu_count);
    if (hip_err == hipSuccess && gpu_count > 0) {
        // Use device 0 (first visible GPU from HIP's perspective)
        device_id = 0;
        hip_err = hipDeviceGetPCIBusId(hip_pci_bus_id, sizeof(hip_pci_bus_id), device_id);
        if (hip_err == hipSuccess) {
            OPENGDA_Info("ofi", "HIP reports GPU %d at PCI Bus ID: %s", device_id, hip_pci_bus_id);
        } else {
            OPENGDA_Warn("ofi", "hipDeviceGetPCIBusId failed: %s", hipGetErrorString(hip_err));
        }
    } else {
        OPENGDA_Warn("ofi", "No HIP devices found (count=%d, error=%s)",
                     gpu_count, hipGetErrorString(hip_err));
    }

    // Initialize hwloc topology for GPU-NIC affinity detection
    hwloc_topology_t topo;
    hwloc_topology_init(&topo);
    hwloc_topology_set_io_types_filter(topo, HWLOC_TYPE_FILTER_KEEP_ALL);
    hwloc_topology_load(topo);

    // Find the GPU matching the HIP PCI Bus ID in hwloc
    DeviceAffinity gpu_affinity;
    std::string gpu_name;

    if (hip_pci_bus_id[0] != '\0') {
        // Convert HIP PCI Bus ID format to hwloc format if needed
        // HIP format is usually "0000:d1:00.0", which matches hwloc format
        hwloc_obj_t osdev = NULL;
        while ((osdev = hwloc_get_next_osdev(topo, osdev)) != NULL) {
            // Look for AMD GPU (OSDEV_GPU or OSDEV_COPROC)
            if (osdev->attr->osdev.type == HWLOC_OBJ_OSDEV_GPU ||
                osdev->attr->osdev.type == HWLOC_OBJ_OSDEV_COPROC) {

                DeviceAffinity temp_affinity;
                if (get_device_affinity(topo, osdev, temp_affinity)) {
                    // Compare PCI Bus IDs (case-insensitive)
                    if (strcasecmp(temp_affinity.pci_id.c_str(), hip_pci_bus_id) == 0) {
                        gpu_affinity = temp_affinity;
                        gpu_found = true;
                        gpu_name = osdev->name ? osdev->name : "unknown";
                        OPENGDA_Info("ofi", "Found GPU '%s' at PCI %s (matched HIP device 0), affinity: %s L#%d",
                                     gpu_name.c_str(), gpu_affinity.pci_id.c_str(),
                                     hwloc_obj_type_string(gpu_affinity.affinity_type),
                                     gpu_affinity.affinity_index);
                        break;  // Found the matching GPU
                    }
                }
            }
        }

        if (!gpu_found) {
            OPENGDA_Warn("ofi", "Could not find GPU with PCI ID %s in hwloc", hip_pci_bus_id);
        }
    }

    // Build map of network devices and their affinities
    std::map<std::string, DeviceAffinity> nic_affinities;
    hwloc_obj_t osdev_nic = NULL;
    while ((osdev_nic = hwloc_get_next_osdev(topo, osdev_nic)) != NULL) {
        // Look for network devices (including CXI/OpenFabrics)
        if (osdev_nic->attr->osdev.type == HWLOC_OBJ_OSDEV_NETWORK ||
            osdev_nic->attr->osdev.type == HWLOC_OBJ_OSDEV_OPENFABRICS) {
            DeviceAffinity nic_affinity;
            if (get_device_affinity(topo, osdev_nic, nic_affinity)) {
                std::string nic_name = osdev_nic->name ? osdev_nic->name : "";
                if (!nic_name.empty()) {
                    nic_affinities[nic_name] = nic_affinity;
                    OPENGDA_Debug("ofi", "Found NIC '%s' at PCI %s, affinity: %s L#%d",
                                  nic_name.c_str(), nic_affinity.pci_id.c_str(),
                                  hwloc_obj_type_string(nic_affinity.affinity_type),
                                  nic_affinity.affinity_index);
                }
            }
        }
    }
    #endif

    #ifdef USE_NVGPU
    // Get PCI Bus ID of the first CUDA-visible GPU
    char cuda_pci_bus_id[32] = {0};
    int gpu_count = 0;
    bool gpu_found = false;

    cudaError_t cuda_err = cudaGetDeviceCount(&gpu_count);
    if (cuda_err == cudaSuccess && gpu_count > 0) {
        // Use device 0 (first visible GPU from CUDA's perspective)
        device_id = 0;
        cuda_err = cudaDeviceGetPCIBusId(cuda_pci_bus_id, sizeof(cuda_pci_bus_id), device_id);
        if (cuda_err == cudaSuccess) {
            OPENGDA_Info("ofi", "CUDA reports GPU %d at PCI Bus ID: %s", device_id, cuda_pci_bus_id);
        } else {
            OPENGDA_Warn("ofi", "cudaDeviceGetPCIBusId failed: %s", cudaGetErrorString(cuda_err));
        }
    } else {
        OPENGDA_Warn("ofi", "No CUDA devices found (count=%d, error=%s)",
                     gpu_count, cudaGetErrorString(cuda_err));
    }

    // Initialize hwloc topology for GPU-NIC affinity detection
    hwloc_topology_t topo;
    hwloc_topology_init(&topo);
    hwloc_topology_set_io_types_filter(topo, HWLOC_TYPE_FILTER_KEEP_ALL);
    hwloc_topology_load(topo);

    // Find the GPU matching the CUDA PCI Bus ID in hwloc
    DeviceAffinity gpu_affinity;
    std::string gpu_name;

    if (cuda_pci_bus_id[0] != '\0') {
        // Convert CUDA PCI Bus ID format to hwloc format if needed
        // CUDA format is usually "0000:d1:00.0", which matches hwloc format
        hwloc_obj_t osdev = NULL;
        while ((osdev = hwloc_get_next_osdev(topo, osdev)) != NULL) {
            // Look for NVIDIA GPU (OSDEV_GPU or OSDEV_COPROC)
            if (osdev->attr->osdev.type == HWLOC_OBJ_OSDEV_GPU ||
                osdev->attr->osdev.type == HWLOC_OBJ_OSDEV_COPROC) {

                DeviceAffinity temp_affinity;
                if (get_device_affinity(topo, osdev, temp_affinity)) {
                    // Compare PCI Bus IDs (case-insensitive)
                    if (strcasecmp(temp_affinity.pci_id.c_str(), cuda_pci_bus_id) == 0) {
                        gpu_affinity = temp_affinity;
                        gpu_found = true;
                        gpu_name = osdev->name ? osdev->name : "unknown";
                        OPENGDA_Info("ofi", "Found GPU '%s' at PCI %s (matched CUDA device 0), affinity: %s L#%d",
                                     gpu_name.c_str(), gpu_affinity.pci_id.c_str(),
                                     hwloc_obj_type_string(gpu_affinity.affinity_type),
                                     gpu_affinity.affinity_index);
                        break;  // Found the matching GPU
                    }
                }
            }
        }

        if (!gpu_found) {
            OPENGDA_Warn("ofi", "Could not find GPU with PCI ID %s in hwloc", cuda_pci_bus_id);
        }
    }

    // Build map of network devices and their affinities
    std::map<std::string, DeviceAffinity> nic_affinities;
    hwloc_obj_t osdev_nic = NULL;
    while ((osdev_nic = hwloc_get_next_osdev(topo, osdev_nic)) != NULL) {
        // Look for network devices (including CXI/OpenFabrics)
        if (osdev_nic->attr->osdev.type == HWLOC_OBJ_OSDEV_NETWORK ||
            osdev_nic->attr->osdev.type == HWLOC_OBJ_OSDEV_OPENFABRICS) {
            DeviceAffinity nic_affinity;
            if (get_device_affinity(topo, osdev_nic, nic_affinity)) {
                std::string nic_name = osdev_nic->name ? osdev_nic->name : "";
                if (!nic_name.empty()) {
                    nic_affinities[nic_name] = nic_affinity;
                    OPENGDA_Debug("ofi", "Found NIC '%s' at PCI %s, affinity: %s L#%d",
                                  nic_name.c_str(), nic_affinity.pci_id.c_str(),
                                  hwloc_obj_type_string(nic_affinity.affinity_type),
                                  nic_affinity.affinity_index);
                }
            }
        }
    }
    #endif

    // Initialize OFI
    hints = fi_allocinfo();
    hints->caps = FI_RMA | FI_MSG | FI_HMEM | FI_ATOMIC;
    hints->mode = FI_CONTEXT2; // DWQ requires FI_CONTEXT2
    hints->ep_attr->type = FI_EP_RDM;
    hints->domain_attr->mr_mode = FI_MR_VIRT_ADDR | FI_MR_ALLOCATED |
                                  FI_MR_PROV_KEY | FI_MR_LOCAL | FI_MR_ENDPOINT;
    hints->domain_attr->threading = FI_THREAD_SAFE;
    hints->domain_attr->control_progress = FI_PROGRESS_MANUAL;
    hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;

    info = NULL;
    OFI_CHECK(fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), NULL,
                         NULL, 0, hints, &info),
              "fi_getinfo");
    fi_freeinfo(hints);

    // Find CXI provider with GPU affinity awareness
    cxi_info = NULL;

    #ifdef USE_AMDGPU
    struct fi_info *best_cxi = NULL;

    if (gpu_found) {
        // Try to find CXI device with same affinity as GPU
        for (struct fi_info *cur = info; cur; cur = cur->next) {
            if (cur->fabric_attr && cur->fabric_attr->prov_name &&
                strcmp(cur->fabric_attr->prov_name, "cxi") == 0) {

                const char* cxi_domain_name = cur->domain_attr->name;

                // Extract CXI device ID (e.g., "cxi0" -> "0")
                int cxi_id = -1;
                if (sscanf(cxi_domain_name, "cxi%d", &cxi_id) != 1 || cxi_id < 0) {
                    continue;
                }

                // In hwloc, CXI devices appear as "hsi" instead of "cxi"
                // Map: cxi0 -> hsi0, cxi1 -> hsi1, etc.
                char hsi_name[32];
                snprintf(hsi_name, sizeof(hsi_name), "hsi%d", cxi_id);

                // Match with hwloc NIC names
                for (const auto& nic_pair : nic_affinities) {
                    const std::string& nic_name = nic_pair.first;
                    const DeviceAffinity& nic_affinity = nic_pair.second;

                    // Check if NIC name matches the mapped hsi name
                    if (nic_name == hsi_name) {
                        // Check if affinity matches GPU
                        if (nic_affinity.affinity_type == gpu_affinity.affinity_type &&
                            nic_affinity.affinity_index == gpu_affinity.affinity_index) {
                            best_cxi = cur;
                            OPENGDA_Info("ofi", "Selected CXI '%s' (hwloc: %s) with matching GPU affinity (%s L#%d)",
                                         cxi_domain_name, hsi_name,
                                         hwloc_obj_type_string(nic_affinity.affinity_type),
                                         nic_affinity.affinity_index);
                            break;
                        }
                    }
                }

                if (best_cxi) break;
            }
        }
    }

    // Fallback: use first available CXI if no affinity match found
    if (!best_cxi) {
        for (struct fi_info *cur = info; cur; cur = cur->next) {
            if (cur->fabric_attr && cur->fabric_attr->prov_name &&
                strcmp(cur->fabric_attr->prov_name, "cxi") == 0) {
                best_cxi = cur;
                OPENGDA_Warn("ofi", "Using fallback CXI device '%s' (no GPU affinity match)",
                             cur->domain_attr->name);
                break;
            }
        }
    }

    cxi_info = best_cxi;
    hwloc_topology_destroy(topo);

    #elif defined(USE_NVGPU)
    struct fi_info *best_cxi = NULL;

    if (gpu_found) {
        // Try to find CXI device with same affinity as GPU
        for (struct fi_info *cur = info; cur; cur = cur->next) {
            if (cur->fabric_attr && cur->fabric_attr->prov_name &&
                strcmp(cur->fabric_attr->prov_name, "cxi") == 0) {

                const char* cxi_domain_name = cur->domain_attr->name;

                // Extract CXI device ID (e.g., "cxi0" -> "0")
                int cxi_id = -1;
                if (sscanf(cxi_domain_name, "cxi%d", &cxi_id) != 1 || cxi_id < 0) {
                    continue;
                }

                // In hwloc, CXI devices appear as "hsi" instead of "cxi"
                // Map: cxi0 -> hsi0, cxi1 -> hsi1, etc.
                char hsi_name[32];
                snprintf(hsi_name, sizeof(hsi_name), "hsi%d", cxi_id);

                // Match with hwloc NIC names
                for (const auto& nic_pair : nic_affinities) {
                    const std::string& nic_name = nic_pair.first;
                    const DeviceAffinity& nic_affinity = nic_pair.second;

                    // Check if NIC name matches the mapped hsi name
                    if (nic_name == hsi_name) {
                        // Check if affinity matches GPU
                        if (nic_affinity.affinity_type == gpu_affinity.affinity_type &&
                            nic_affinity.affinity_index == gpu_affinity.affinity_index) {
                            best_cxi = cur;
                            OPENGDA_Info("ofi", "Selected CXI '%s' (hwloc: %s) with matching GPU affinity (%s L#%d)",
                                         cxi_domain_name, hsi_name,
                                         hwloc_obj_type_string(nic_affinity.affinity_type),
                                         nic_affinity.affinity_index);
                            break;
                        }
                    }
                }

                if (best_cxi) break;
            }
        }
    }

    // Fallback: use first available CXI if no affinity match found
    if (!best_cxi) {
        for (struct fi_info *cur = info; cur; cur = cur->next) {
            if (cur->fabric_attr && cur->fabric_attr->prov_name &&
                strcmp(cur->fabric_attr->prov_name, "cxi") == 0) {
                best_cxi = cur;
                OPENGDA_Warn("ofi", "Using fallback CXI device '%s' (no GPU affinity match)",
                             cur->domain_attr->name);
                break;
            }
        }
    }

    cxi_info = best_cxi;
    hwloc_topology_destroy(topo);

    #else
    // Non-GPU build: just use first CXI provider
    for (struct fi_info *cur = info; cur; cur = cur->next) {
        if (cur->fabric_attr && cur->fabric_attr->prov_name &&
            strcmp(cur->fabric_attr->prov_name, "cxi") == 0) {
            cxi_info = cur;
            break;
        }
    }
    #endif

    if (!cxi_info) {
        OPENGDA_Error("ofi", "No CXI provider found");
        exit(1);
    }

    OPENGDA_Info("ofi", "Using CXI provider: %s", cxi_info->domain_attr->name);
    OPENGDA_Debug("ofi", "  - mr_mode: 0x%lx", (unsigned long)cxi_info->domain_attr->mr_mode);
    OPENGDA_Debug("ofi", "  - inject_size: %zu bytes", cxi_info->tx_attr->inject_size);
    OPENGDA_Debug("ofi", "  - max_msg_size: %zu bytes", cxi_info->ep_attr->max_msg_size);

    // Create Fabric
    fabric = NULL;
    OFI_CHECK(fi_fabric(cxi_info->fabric_attr, &fabric, NULL), "fi_fabric");

    // Create Domain
    domain = NULL;
    OFI_CHECK(fi_domain(fabric, cxi_info, &domain, NULL), "fi_domain");

    // Create Address Vector
    av = NULL;
    struct fi_av_attr av_attr = {};
    av_attr.type = FI_AV_MAP;
    OFI_CHECK(fi_av_open(domain, &av_attr, &av, NULL), "fi_av_open");

    // Create Completion Queue
    cq = NULL;
    struct fi_cq_attr cq_attr = {};
    cq_attr.size = 128;
    cq_attr.format = FI_CQ_FORMAT_CONTEXT;
    OFI_CHECK(fi_cq_open(domain, &cq_attr, &cq, NULL), "fi_cq_open");

    // Create Endpoint
    ep = NULL;
    OFI_CHECK(fi_endpoint(domain, cxi_info, &ep, NULL), "fi_endpoint");
    OFI_CHECK(fi_ep_bind(ep, &av->fid, 0), "fi_ep_bind(av)");
    OFI_CHECK(fi_ep_bind(ep, &cq->fid, FI_TRANSMIT | FI_RECV), "fi_ep_bind(cq)");
    OFI_CHECK(fi_enable(ep), "fi_enable");

    ofi_initialized = true;

    // Initialize Counter Manager for DWQ
    if (!cntr_manager_.initialize(domain)) {
        OPENGDA_Error("ofi", "Failed to initialize counter manager");
        exit(1);
    }

    // Allocate and register default memory regions
    if (default_mr_config_.enable_host_mr) {
        if (!allocate_default_host_mr()) {
            OPENGDA_Error("ofi", "Failed to allocate default host MR");
            exit(1);
        }
    }

#ifdef USE_AMDGPU
    if (default_mr_config_.enable_gpu_mr) {
        if (!allocate_default_gpu_mr()) {
            OPENGDA_Error("ofi", "Failed to allocate default GPU MR");
            exit(1);
        }

        // Initialize completion signal pool at the end of GPU MR
        // Layout: [user data ... ] [signal pool (last POOL_SIZE bytes)]
        size_t pool_offset = default_gpu_mr_.size - CompletionSignalPool::POOL_SIZE;
        volatile uint64_t* pool_base = (volatile uint64_t*)((uint8_t*)default_gpu_mr_.buffer + pool_offset);

        // Clear the signal pool region
        hipMemset((void*)pool_base, 0, CompletionSignalPool::POOL_SIZE);
        hipDeviceSynchronize();

        if (!signal_pool_.initialize(pool_base, pool_offset,
                                      default_gpu_mr_.desc, default_gpu_mr_.key)) {
            OPENGDA_Error("ofi", "Failed to initialize completion signal pool");
            exit(1);
        }

        // Allocate and register shared atomic operand
        // This is a single uint64_t with value 1, shared by all DWQ operations
        hipError_t hip_ret = hipMalloc(&shared_atomic_operand_, sizeof(uint64_t));
        if (hip_ret != hipSuccess) {
            OPENGDA_Error("ofi", "Failed to allocate shared atomic operand");
            exit(1);
        }
        uint64_t operand_value = 1;
        hipMemcpy(shared_atomic_operand_, &operand_value, sizeof(uint64_t), hipMemcpyHostToDevice);
        hipDeviceSynchronize();

        shared_atomic_operand_mr_ = register_memory(shared_atomic_operand_, sizeof(uint64_t), true);
        if (!shared_atomic_operand_mr_) {
            OPENGDA_Error("ofi", "Failed to register shared atomic operand MR");
            exit(1);
        }
        OPENGDA_Info("ofi", "Shared atomic operand initialized at %p", shared_atomic_operand_);
    }
#endif

    // Exchange endpoint addresses and MR information with all peers
    if (!exchange_addresses()) {
        OPENGDA_Error("ofi", "Failed to exchange addresses with peers");
        exit(1);
    }

}

bool OFI::ofi_initialize() {
    if (ofi_initialized) {
        return true;
    }

    return false;
}

// ============================================================================
// Memory Registration
// ============================================================================

struct fid_mr* OFI::register_memory(void* buf, size_t size, bool is_device_mem) {
    if (!ofi_initialized) {
        OPENGDA_Error("ofi", "Cannot register memory: OFI not initialized");
        return nullptr;
    }

    // Check if already registered
    if (mr_manager_.is_registered(buf)) {
        OPENGDA_Warn("ofi", "Memory at %p is already registered", buf);
        return mr_manager_.find_by_addr(buf);
    }

    struct fi_mr_attr mr_attr = {};
    struct iovec iov;
    iov.iov_base = buf;
    iov.iov_len = size;

    mr_attr.mr_iov = &iov;
    mr_attr.iov_count = 1;
    mr_attr.access = FI_SEND | FI_RECV | FI_READ | FI_WRITE |
                     FI_REMOTE_READ | FI_REMOTE_WRITE;

    if (is_device_mem) {
        #ifdef USE_AMDGPU
        mr_attr.iface = FI_HMEM_ROCR;
        mr_attr.device.reserved = device_id;
        OPENGDA_Debug("ofi", "Registering AMD GPU memory: buf=%p, size=%zu, device=%d",
                      buf, size, device_id);
        #elif defined(USE_NVGPU)
        mr_attr.iface = FI_HMEM_CUDA;
        mr_attr.device.reserved = device_id;
        OPENGDA_Debug("ofi", "Registering NVIDIA GPU memory: buf=%p, size=%zu, device=%d",
                      buf, size, device_id);
        #else
        OPENGDA_Error("ofi", "GPU memory registration requested but no GPU support defined");
        return nullptr;
        #endif
    } else {
        mr_attr.iface = FI_HMEM_SYSTEM;
        OPENGDA_Debug("ofi", "Registering host memory: buf=%p, size=%zu", buf, size);
    }

    struct fid_mr *mr = nullptr;
    int ret = fi_mr_regattr(domain, &mr_attr, 0, &mr);
    if (ret) {
        OPENGDA_Error("ofi", "fi_mr_regattr failed: %s (%d)", fi_strerror(-ret), ret);
        return nullptr;
    }

    // If endpoint-level MR is required, bind and enable
    if (cxi_info->domain_attr->mr_mode & FI_MR_ENDPOINT) {
        ret = fi_mr_bind(mr, &ep->fid, 0);
        if (ret) {
            OPENGDA_Error("ofi", "fi_mr_bind failed: %s (%d)", fi_strerror(-ret), ret);
            fi_close(&mr->fid);
            return nullptr;
        }

        ret = fi_mr_enable(mr);
        if (ret) {
            OPENGDA_Error("ofi", "fi_mr_enable failed: %s (%d)", fi_strerror(-ret), ret);
            fi_close(&mr->fid);
            return nullptr;
        }
    }

    OPENGDA_Info("ofi", "Memory registered: buf=%p, size=%zu, key=0x%lx, %s",
                 buf, size, (unsigned long)fi_mr_key(mr),
                 is_device_mem ? "GPU" : "Host");

    // Add to MR manager
    if (!mr_manager_.add_mr(mr, buf, size, is_device_mem)) {
        OPENGDA_Error("ofi", "Failed to add MR to manager");
        fi_close(&mr->fid);
        return nullptr;
    }

    return mr;
}

void OFI::deregister_memory(struct fid_mr* mr) {
    if (!mr) {
        OPENGDA_Warn("ofi", "Attempted to deregister null MR");
        return;
    }

    OPENGDA_Debug("ofi", "Deregistering memory: MR=%p, key=0x%lx",
                  mr, (unsigned long)fi_mr_key(mr));

    // Remove from manager first
    mr_manager_.remove_mr(mr);

    // Close the MR
    int ret = fi_close(&mr->fid);
    if (ret) {
        OPENGDA_Error("ofi", "fi_close(mr) failed: %s (%d)", fi_strerror(-ret), ret);
    }
}

bool OFI::deregister_memory_by_addr(void* addr) {
    struct fid_mr* mr = mr_manager_.find_by_addr(addr);
    if (!mr) {
        OPENGDA_Warn("ofi", "No MR found for address %p", addr);
        return false;
    }

    deregister_memory(mr);
    return true;
}

struct fid_mr* OFI::find_mr_by_addr(void* addr) {
    return mr_manager_.find_by_addr(addr);
}

struct fid_mr* OFI::find_mr_by_key(uint64_t key) {
    return mr_manager_.find_by_key(key);
}

bool OFI::is_memory_registered(void* addr) {
    return mr_manager_.is_registered(addr);
}

const MRManager::MRInfo* OFI::get_mr_info(void* addr) {
    // First try exact match, then range lookup
    const MRManager::MRInfo* info = mr_manager_.get_info_by_addr(addr);
    if (!info) {
        info = mr_manager_.find_containing(addr);
    }
    return info;
}

MRManager::MRStats OFI::get_mr_stats() {
    return mr_manager_.get_stats();
}

void OFI::print_mr_stats() {
    mr_manager_.print_stats();
}

// Destructor - cleanup any remaining MRs
OFI::~OFI() {
#ifdef USE_AMDGPU
    // Clean up IPC mappings first
    cleanup_ipc_mappings();
#endif

    // Clean up default MRs first (they have their own memory allocations)
    cleanup_default_mrs();

    // Get all registered MRs and deregister them
    std::vector<void*> addrs = mr_manager_.get_all_addresses();

    if (!addrs.empty()) {
        OPENGDA_Warn("ofi", "Cleaning up %zu unreleased MRs", addrs.size());

        for (void* addr : addrs) {
            struct fid_mr* mr = mr_manager_.find_by_addr(addr);
            if (mr) {
                OPENGDA_Debug("ofi", "Auto-deregistering MR at %p", addr);
                fi_close(&mr->fid);
            }
        }

        mr_manager_.clear_all();
    }

    // Finalize Counter Manager (before closing domain)
    cntr_manager_.finalize();

    // Close OFI resources
    if (ofi_initialized) {
        if (ep) fi_close(&ep->fid);
        if (cq) fi_close(&cq->fid);
        if (av) fi_close(&av->fid);
        if (domain) fi_close(&domain->fid);
        if (fabric) fi_close(&fabric->fid);
        if (info) fi_freeinfo(info);
    }

    // Clean up local endpoint address
    if (local_ep_addr_) {
        free(local_ep_addr_);
        local_ep_addr_ = nullptr;
    }
}

// ============================================================================
// Counter Management Wrappers
// ============================================================================

bool OFI::allocate_cntr_pair(CntrManager::CntrPair** pair_out) {
    return cntr_manager_.allocate_pair(pair_out);
}

bool OFI::release_cntr_pair(CntrManager::CntrPair* pair) {
    return cntr_manager_.release_pair(pair);
}

bool OFI::release_cntr_pair_by_index(int index) {
    return cntr_manager_.release_pair_by_index(index);
}

CntrManager::CntrPair* OFI::get_cntr_pair(int index) {
    return cntr_manager_.get_pair(index);
}

CntrManager::CntrStats OFI::get_cntr_stats() {
    return cntr_manager_.get_stats();
}

void OFI::print_cntr_stats() {
    cntr_manager_.print_stats();
}

size_t OFI::get_dwq_depth() {
    if (!domain) {
        OPENGDA_Warn("ofi", "Domain not initialized, cannot query DWQ depth");
        return 0;
    }

    // Get CXI domain ops
    struct fi_cxi_dom_ops* dom_ops = nullptr;
    int ret = fi_open_ops(&domain->fid, FI_CXI_DOM_OPS_5, 0, (void**)&dom_ops, NULL);
    if (ret) {
        // Try older versions
        ret = fi_open_ops(&domain->fid, FI_CXI_DOM_OPS_4, 0, (void**)&dom_ops, NULL);
        if (ret) {
            ret = fi_open_ops(&domain->fid, FI_CXI_DOM_OPS_3, 0, (void**)&dom_ops, NULL);
            if (ret) {
                OPENGDA_Warn("ofi", "Failed to get CXI domain ops: %s (%d)",
                            fi_strerror(-ret), ret);
                return 0;
            }
        }
    }

    if (!dom_ops || !dom_ops->get_dwq_depth) {
        OPENGDA_Warn("ofi", "get_dwq_depth not available in CXI domain ops");
        return 0;
    }

    size_t depth = 0;
    ret = dom_ops->get_dwq_depth(&domain->fid, &depth);
    if (ret) {
        OPENGDA_Warn("ofi", "get_dwq_depth failed: %s (%d)",
                    fi_strerror(-ret), ret);
        return 0;
    }

    OPENGDA_Info("ofi", "DWQ depth: %zu triggered operations available", depth);
    return depth;
}

bool OFI::flush_work_queue() {
    if (!domain) {
        OPENGDA_Error("ofi", "Domain not initialized, cannot flush work queue");
        return false;
    }

    int ret = fi_control(&domain->fid, FI_FLUSH_WORK, NULL);
    if (ret) {
        OPENGDA_Error("ofi", "FI_FLUSH_WORK failed: %s (%d)",
                     fi_strerror(-ret), ret);
        return false;
    }

    OPENGDA_Debug("ofi", "Work queue flushed successfully");
    return true;
}

bool OFI::get_shared_atomic_operand(uint64_t** addr_out, struct fid_mr** mr_out) {
    if (!addr_out || !mr_out) {
        return false;
    }

    if (!shared_atomic_operand_ || !shared_atomic_operand_mr_) {
        OPENGDA_Error("ofi", "Shared atomic operand not initialized");
        return false;
    }

    *addr_out = shared_atomic_operand_;
    *mr_out = shared_atomic_operand_mr_;
    return true;
}

// ============================================================================
// Default MR Helper Functions
// ============================================================================

static size_t parse_size_env(const char* env_name, size_t default_value) {
    const char* env_val = getenv(env_name);
    if (!env_val) {
        return default_value;
    }

    char* endptr;
    size_t value = strtoull(env_val, &endptr, 10);

    // Handle suffixes (K, M, G, T)
    if (*endptr != '\0') {
        switch (*endptr) {
            case 'K': case 'k':
                value *= 1024ULL;
                break;
            case 'M': case 'm':
                value *= 1024ULL * 1024;
                break;
            case 'G': case 'g':
                value *= 1024ULL * 1024 * 1024;
                break;
            case 'T': case 't':
                value *= 1024ULL * 1024 * 1024 * 1024;
                break;
        }
    }

    return value;
}

void OFI::init_default_mr_config() {
    // Start with compile-time defaults
    default_mr_config_.host_mr_size = DEFAULT_HOST_MR_SIZE;
    default_mr_config_.gpu_mr_size = DEFAULT_GPU_MR_SIZE;
    default_mr_config_.enable_host_mr = true;
    default_mr_config_.enable_gpu_mr = true;

    // Override with environment variables if set
    default_mr_config_.host_mr_size = parse_size_env(ENV_HOST_MR_SIZE, DEFAULT_HOST_MR_SIZE);
    default_mr_config_.gpu_mr_size = parse_size_env(ENV_GPU_MR_SIZE, DEFAULT_GPU_MR_SIZE);

    // Check for disable flags (size = 0 means disabled)
    if (default_mr_config_.host_mr_size == 0) {
        default_mr_config_.enable_host_mr = false;
    }
    if (default_mr_config_.gpu_mr_size == 0) {
        default_mr_config_.enable_gpu_mr = false;
    }

    OPENGDA_Info("ofi", "Default MR config: host=%zu bytes (%s), GPU=%zu bytes (%s)",
                 default_mr_config_.host_mr_size,
                 default_mr_config_.enable_host_mr ? "enabled" : "disabled",
                 default_mr_config_.gpu_mr_size,
                 default_mr_config_.enable_gpu_mr ? "enabled" : "disabled");
}

bool OFI::allocate_default_host_mr() {
    size_t size = default_mr_config_.host_mr_size;

    // Allocate page-aligned host memory
    void* buf = nullptr;
    int ret = posix_memalign(&buf, 4096, size);
    if (ret != 0 || buf == nullptr) {
        OPENGDA_Error("ofi", "Failed to allocate %zu bytes for default host MR", size);
        return false;
    }

    OPENGDA_Info("ofi", "Allocated default host buffer: %zu bytes at %p", size, buf);

    // Register with OFI
    struct fid_mr* mr = register_memory(buf, size, false);
    if (!mr) {
        OPENGDA_Error("ofi", "Failed to register default host MR");
        free(buf);
        return false;
    }

    // Store info
    default_host_mr_.buffer = buf;
    default_host_mr_.size = size;
    default_host_mr_.mr = mr;
    default_host_mr_.key = fi_mr_key(mr);
    default_host_mr_.desc = fi_mr_desc(mr);
    default_host_mr_.is_device_mem = false;
    default_host_mr_.allocated = true;

    OPENGDA_Info("ofi", "Default host MR registered: size=%zu, key=0x%lx",
                 size, (unsigned long)default_host_mr_.key);

    return true;
}

bool OFI::allocate_default_gpu_mr() {
#ifdef USE_AMDGPU
    size_t size = default_mr_config_.gpu_mr_size;

    // Allocate GPU memory
    void* buf = nullptr;
    hipError_t hip_err = hipMalloc(&buf, size);
    if (hip_err != hipSuccess || buf == nullptr) {
        OPENGDA_Error("ofi", "hipMalloc failed for %zu bytes: %s",
                      size, hipGetErrorString(hip_err));
        return false;
    }

    OPENGDA_Info("ofi", "Allocated default GPU buffer: %zu bytes at %p (device %d)",
                 size, buf, device_id);

    // Register with OFI
    struct fid_mr* mr = register_memory(buf, size, true);
    if (!mr) {
        OPENGDA_Error("ofi", "Failed to register default GPU MR");
        hipFree(buf);
        return false;
    }

    // Store info
    default_gpu_mr_.buffer = buf;
    default_gpu_mr_.size = size;
    default_gpu_mr_.mr = mr;
    default_gpu_mr_.key = fi_mr_key(mr);
    default_gpu_mr_.desc = fi_mr_desc(mr);
    default_gpu_mr_.is_device_mem = true;
    default_gpu_mr_.allocated = true;

    OPENGDA_Info("ofi", "Default GPU MR registered: size=%zu, key=0x%lx",
                 size, (unsigned long)default_gpu_mr_.key);

    return true;
#else
    OPENGDA_Warn("ofi", "GPU MR requested but USE_AMDGPU not defined");
    return false;
#endif
}

void OFI::cleanup_default_mrs() {
    // Clean up signal pool first (before GPU MR)
    signal_pool_.finalize();

    // Clean up shared atomic operand
#ifdef USE_AMDGPU
    if (shared_atomic_operand_mr_) {
        deregister_memory(shared_atomic_operand_mr_);
        shared_atomic_operand_mr_ = nullptr;
    }
    if (shared_atomic_operand_) {
        hipFree(shared_atomic_operand_);
        shared_atomic_operand_ = nullptr;
    }
#endif

    // Clean up GPU MR
    if (default_gpu_mr_.allocated) {
        OPENGDA_Debug("ofi", "Cleaning up default GPU MR");

        if (default_gpu_mr_.mr) {
            deregister_memory(default_gpu_mr_.mr);
        }

#ifdef USE_AMDGPU
        if (default_gpu_mr_.buffer) {
            hipFree(default_gpu_mr_.buffer);
        }
#endif

        memset(&default_gpu_mr_, 0, sizeof(default_gpu_mr_));
    }

    // Clean up host MR
    if (default_host_mr_.allocated) {
        OPENGDA_Debug("ofi", "Cleaning up default host MR");

        if (default_host_mr_.mr) {
            deregister_memory(default_host_mr_.mr);
        }

        if (default_host_mr_.buffer) {
            free(default_host_mr_.buffer);
        }

        memset(&default_host_mr_, 0, sizeof(default_host_mr_));
    }
}

// ============================================================================
// Address Exchange Implementation
// ============================================================================

void OFI::bytes_to_hex(const uint8_t* bytes, size_t len, char* hex) {
    static const char hex_chars[] = "0123456789abcdef";
    for (size_t i = 0; i < len; i++) {
        hex[2 * i] = hex_chars[(bytes[i] >> 4) & 0xf];
        hex[2 * i + 1] = hex_chars[bytes[i] & 0xf];
    }
    hex[2 * len] = '\0';
}

int OFI::hex_to_bytes(const char* hex, uint8_t* bytes, size_t max_len) {
    size_t hex_len = strlen(hex);
    size_t byte_len = hex_len / 2;
    if (byte_len > max_len) {
        byte_len = max_len;
    }

    for (size_t i = 0; i < byte_len; i++) {
        unsigned int byte;
        if (sscanf(hex + 2 * i, "%2x", &byte) != 1) {
            return -1;
        }
        bytes[i] = (uint8_t)byte;
    }

    return (int)byte_len;
}

bool OFI::exchange_addresses() {
    if (!bootstrap_ || !bootstrap_->is_initialized()) {
        OPENGDA_Error("ofi", "Bootstrap not initialized for address exchange");
        return false;
    }

    OPENGDA_Info("ofi", "Starting address exchange with %d peers", size_);

    // Step 1: Get local endpoint address
    local_ep_addr_len_ = 0;
    fi_getname(&ep->fid, NULL, &local_ep_addr_len_);

    local_ep_addr_ = malloc(local_ep_addr_len_);
    if (!local_ep_addr_) {
        OPENGDA_Error("ofi", "Failed to allocate memory for local endpoint address");
        return false;
    }

    int ret = fi_getname(&ep->fid, local_ep_addr_, &local_ep_addr_len_);
    if (ret) {
        OPENGDA_Error("ofi", "fi_getname failed: %s", fi_strerror(-ret));
        return false;
    }

    OPENGDA_Debug("ofi", "Local endpoint address length: %zu bytes", local_ep_addr_len_);

    // Step 2: Prepare exchange data
    // Format: [ep_addr_len(4)][ep_addr(var)][mr_info(ExchangeData)]
    size_t exchange_size = sizeof(uint32_t) + local_ep_addr_len_ + sizeof(ExchangeData);
    char* my_data = (char*)malloc(exchange_size);
    if (!my_data) {
        OPENGDA_Error("ofi", "Failed to allocate exchange data buffer");
        return false;
    }

    // Fill in exchange data
    uint32_t addr_len = (uint32_t)local_ep_addr_len_;
    memcpy(my_data, &addr_len, sizeof(uint32_t));
    memcpy(my_data + sizeof(uint32_t), local_ep_addr_, local_ep_addr_len_);

    ExchangeData* my_mr_data = (ExchangeData*)(my_data + sizeof(uint32_t) + local_ep_addr_len_);
    memset(my_mr_data, 0, sizeof(ExchangeData));

    // Fill in node ID for same-node detection
    my_mr_data->node_id = bootstrap_->get_node_id();
#ifdef USE_AMDGPU
    local_node_id_ = my_mr_data->node_id;
#endif

    // Fill in MR info
    if (default_host_mr_.allocated) {
        my_mr_data->host_mr_addr = (uint64_t)default_host_mr_.buffer;
        my_mr_data->host_mr_key = default_host_mr_.key;
        my_mr_data->host_mr_size = default_host_mr_.size;
    }

    if (default_gpu_mr_.allocated) {
        my_mr_data->gpu_mr_addr = (uint64_t)default_gpu_mr_.buffer;
        my_mr_data->gpu_mr_key = default_gpu_mr_.key;
        my_mr_data->gpu_mr_size = default_gpu_mr_.size;

#ifdef USE_AMDGPU
        // Create IPC handle for our GPU buffer (for same-node peers)
        hipError_t hip_err = hipIpcGetMemHandle(&local_ipc_handle_, default_gpu_mr_.buffer);
        if (hip_err == hipSuccess) {
            local_ipc_handle_valid_ = true;
            memcpy(&my_mr_data->gpu_ipc_handle, &local_ipc_handle_, sizeof(hipIpcMemHandle_t));
            my_mr_data->gpu_device_id = device_id;
            my_mr_data->ipc_handle_valid = true;
            OPENGDA_Info("ofi", "Created IPC handle for GPU buffer on node %d", local_node_id_);
        } else {
            OPENGDA_Warn("ofi", "hipIpcGetMemHandle failed: %s - IPC disabled for this rank",
                        hipGetErrorString(hip_err));
            local_ipc_handle_valid_ = false;
            my_mr_data->ipc_handle_valid = false;
        }
#endif
    }

    // Convert to hex for exchange
    size_t hex_len = 2 * exchange_size;
    char* my_hex = (char*)malloc(hex_len + 1);
    bytes_to_hex((uint8_t*)my_data, exchange_size, my_hex);

    // Step 3: Exchange with all peers using bootstrap
    char* all_hex = (char*)malloc(size_ * (hex_len + 1));
    if (!all_hex) {
        OPENGDA_Error("ofi", "Failed to allocate buffer for all peer data");
        free(my_hex);
        free(my_data);
        return false;
    }

    if (!bootstrap_->bootstrap_exchange(my_hex, hex_len, all_hex)) {
        OPENGDA_Error("ofi", "bootstrap_exchange failed");
        free(all_hex);
        free(my_hex);
        free(my_data);
        return false;
    }

    OPENGDA_Debug("ofi", "Address exchange completed, processing peer data");

    // Step 4: Process received data and insert into address vector
    uint8_t* peer_data = (uint8_t*)malloc(exchange_size);

    for (int i = 0; i < size_; i++) {
        const char* peer_hex = all_hex + i * (hex_len + 1);

        int decoded = hex_to_bytes(peer_hex, peer_data, exchange_size);
        if (decoded <= 0) {
            OPENGDA_Error("ofi", "Failed to decode data from rank %d", i);
            continue;
        }

        // Extract endpoint address length and address
        uint32_t peer_addr_len;
        memcpy(&peer_addr_len, peer_data, sizeof(uint32_t));

        void* peer_ep_addr = peer_data + sizeof(uint32_t);

        // Insert into address vector
        fi_addr_t peer_fi_addr = FI_ADDR_NOTAVAIL;
        int inserted = fi_av_insert(av, peer_ep_addr, 1, &peer_fi_addr, 0, NULL);
        if (inserted != 1 || peer_fi_addr == FI_ADDR_NOTAVAIL) {
            OPENGDA_Error("ofi", "fi_av_insert failed for rank %d", i);
            continue;
        }

        // Extract MR info
        ExchangeData* peer_mr_data = (ExchangeData*)(peer_data + sizeof(uint32_t) + peer_addr_len);

        // Store peer info
        peers_[i].rank = i;
        peers_[i].fi_addr = peer_fi_addr;
        peers_[i].mr_info.host_mr_addr = peer_mr_data->host_mr_addr;
        peers_[i].mr_info.host_mr_key = peer_mr_data->host_mr_key;
        peers_[i].mr_info.host_mr_size = peer_mr_data->host_mr_size;
        peers_[i].mr_info.gpu_mr_addr = peer_mr_data->gpu_mr_addr;
        peers_[i].mr_info.gpu_mr_key = peer_mr_data->gpu_mr_key;
        peers_[i].mr_info.gpu_mr_size = peer_mr_data->gpu_mr_size;
        peers_[i].valid = true;

        // Store node info for same-node detection
        peers_[i].node_id = peer_mr_data->node_id;
#ifdef USE_AMDGPU
        peers_[i].same_node = (peer_mr_data->node_id == local_node_id_);

        // Store IPC handle if peer is on same node and IPC is available
        if (peers_[i].same_node && peer_mr_data->ipc_handle_valid && i != rank_) {
            memcpy(&peers_[i].ipc_info.ipc_handle, &peer_mr_data->gpu_ipc_handle, sizeof(hipIpcMemHandle_t));
            peers_[i].ipc_info.mapped_size = peer_mr_data->gpu_mr_size;
            peers_[i].ipc_info.peer_device_id = peer_mr_data->gpu_device_id;
            peers_[i].ipc_info.mapped = false;  // Will be mapped in setup_peer_ipc_mappings
            peers_[i].can_use_ipc = true;

            OPENGDA_Debug("ofi", "Peer %d is same-node (node_id=%d), IPC available",
                         i, peer_mr_data->node_id);
        } else {
            peers_[i].can_use_ipc = false;
        }
#else
        peers_[i].same_node = false;
#endif

        // Store local fi_addr
        if (i == rank_) {
            local_fi_addr_ = peer_fi_addr;
        }

        OPENGDA_Debug("ofi", "Peer %d: fi_addr=%lu, host_mr_key=0x%lx, gpu_mr_key=0x%lx, node_id=%d, same_node=%d",
                      i, (unsigned long)peer_fi_addr,
                      (unsigned long)peers_[i].mr_info.host_mr_key,
                      (unsigned long)peers_[i].mr_info.gpu_mr_key,
                      peers_[i].node_id, peers_[i].same_node);
    }

    // Cleanup
    free(peer_data);
    free(all_hex);
    free(my_hex);
    free(my_data);

    // Final barrier to ensure all peers are ready
    if (!bootstrap_->bootstrap_barrier()) {
        OPENGDA_Error("ofi", "Final barrier after address exchange failed");
        return false;
    }

#ifdef USE_AMDGPU
    // Setup IPC mappings for same-node peers
    if (!setup_peer_ipc_mappings()) {
        OPENGDA_Warn("ofi", "Failed to setup some IPC mappings - continuing with OFI fallback");
        // Not fatal - we can still use OFI for same-node communication
    }
#endif

    OPENGDA_Info("ofi", "Address exchange completed successfully");
    return true;
}

// ============================================================================
// Peer Information Accessors
// ============================================================================

const PeerInfo* OFI::get_peer_info(int rank) const {
    if (rank < 0 || rank >= size_) {
        return nullptr;
    }
    return peers_[rank].valid ? &peers_[rank] : nullptr;
}

fi_addr_t OFI::get_peer_fi_addr(int rank) const {
    if (rank < 0 || rank >= size_) {
        return FI_ADDR_NOTAVAIL;
    }
    return peers_[rank].valid ? peers_[rank].fi_addr : FI_ADDR_NOTAVAIL;
}

// ============================================================================
// DWQ Operation Support
// ============================================================================

std::unique_ptr<DWQOperation> OFI::create_dwq_operation() {
    auto op = std::make_unique<DWQOperation>();
    if (!op->initialize(this)) {
        OPENGDA_Error("ofi", "Failed to initialize DWQ operation");
        return nullptr;
    }
    return op;
}

// ============================================================================
// DWQOperation Implementation
// ============================================================================

DWQOperation::DWQOperation()
    : state_(State::UNINITIALIZED)
    , op_type_(OpType::WRITE)
    , ofi_(nullptr)
    , seq_num_(CompletionQueue::INVALID_SEQ)
    , completion_queue_(nullptr)
    , cntr_pair_(nullptr)
    , completion_signal_(nullptr)
    , completion_signal_mr_(nullptr)
    , completion_signal_offset_(0)
    , uses_signal_pool_(false)
    , atomic_operand_(nullptr)
    , atomic_operand_mr_(nullptr)
    , uses_shared_operand_(false)
    , threshold_(1)
    , completion_threshold_(0)
    , op_rma_(nullptr)
    , msg_rma_(nullptr)
    , iov_(nullptr)
    , rma_iov_(nullptr)
    , op_atomic_(nullptr)
    , atomic_msg_(nullptr)
    , atomic_iov_(nullptr)
    , atomic_rma_iov_(nullptr)
    , local_desc_(nullptr)
{
    memset(&rma_work_, 0, sizeof(rma_work_));
    memset(&atomic_work_, 0, sizeof(atomic_work_));
}

DWQOperation::~DWQOperation() {
    cleanup();
}

void DWQOperation::cleanup() {
    // Unregister from completion queue
    if (completion_queue_ && seq_num_ != CompletionQueue::INVALID_SEQ) {
        completion_queue_->unregister(seq_num_);
        seq_num_ = CompletionQueue::INVALID_SEQ;
    }
    completion_queue_ = nullptr;

    // Free RMA work structures
    if (op_rma_) { free(op_rma_); op_rma_ = nullptr; }
    if (msg_rma_) { free(msg_rma_); msg_rma_ = nullptr; }
    if (iov_) { free(iov_); iov_ = nullptr; }
    if (rma_iov_) { free(rma_iov_); rma_iov_ = nullptr; }

    // Free atomic work structures
    if (op_atomic_) { free(op_atomic_); op_atomic_ = nullptr; }
    if (atomic_msg_) { free(atomic_msg_); atomic_msg_ = nullptr; }
    if (atomic_iov_) { free(atomic_iov_); atomic_iov_ = nullptr; }
    if (atomic_rma_iov_) { free(atomic_rma_iov_); atomic_rma_iov_ = nullptr; }

    // Release completion signal
    if (completion_signal_) {
        if (uses_signal_pool_ && ofi_) {
            // Release back to pool
            ofi_->get_signal_pool()->release(completion_signal_);
        } else if (completion_signal_mr_ && ofi_) {
            // Deregister and free separately allocated memory
            ofi_->deregister_memory(completion_signal_mr_);
#ifdef USE_AMDGPU
            hipFree((void*)completion_signal_);
#endif
#ifdef USE_NVGPU
            cudaFree((void*)completion_signal_);
#endif
        }
        completion_signal_ = nullptr;
        completion_signal_mr_ = nullptr;
    }

    // Release atomic operand (only if not using shared)
    if (atomic_operand_ && !uses_shared_operand_) {
        if (atomic_operand_mr_ && ofi_) {
            ofi_->deregister_memory(atomic_operand_mr_);
        }
#ifdef USE_AMDGPU
        hipFree(atomic_operand_);
#endif
#ifdef USE_NVGPU
        cudaFree(atomic_operand_);
#endif
    }
    atomic_operand_ = nullptr;
    atomic_operand_mr_ = nullptr;
    uses_signal_pool_ = false;
    uses_shared_operand_ = false;

    // Release counter pair back to OFI
    if (ofi_ && cntr_pair_) {
        ofi_->release_cntr_pair(cntr_pair_);
        cntr_pair_ = nullptr;
    }

    state_ = State::UNINITIALIZED;
    ofi_ = nullptr;
}

bool DWQOperation::initialize(OFI* ofi) {
    if (!ofi) {
        OPENGDA_Error("dwq", "OFI instance is null");
        return false;
    }

    if (state_ != State::UNINITIALIZED) {
        OPENGDA_Error("dwq", "DWQOperation already initialized");
        return false;
    }

    ofi_ = ofi;

    // Allocate counter pair for RMA trigger/completion
    if (!ofi_->allocate_cntr_pair(&cntr_pair_)) {
        OPENGDA_Error("dwq", "Failed to allocate counter pair");
        cleanup();
        return false;
    }

    // Try to use signal pool and shared atomic operand (more efficient)
    CompletionSignalPool* pool = ofi_->get_signal_pool();
    if (pool && pool->is_initialized()) {
        // Allocate completion signal from pool
        if (!pool->allocate(&completion_signal_, &completion_signal_offset_)) {
            OPENGDA_Error("dwq", "Failed to allocate from signal pool");
            cleanup();
            return false;
        }
        uses_signal_pool_ = true;
        completion_signal_mr_ = nullptr;  // Use pool's MR

        // Clear the signal
#ifdef USE_AMDGPU
        hipMemset((void*)completion_signal_, 0, sizeof(uint64_t));
        hipDeviceSynchronize();
#endif
#ifdef USE_NVGPU
        cudaMemset((void*)completion_signal_, 0, sizeof(uint64_t));
        cudaDeviceSynchronize();
#endif

        // Use shared atomic operand
        if (!ofi_->get_shared_atomic_operand(&atomic_operand_, &atomic_operand_mr_)) {
            OPENGDA_Error("dwq", "Failed to get shared atomic operand");
            cleanup();
            return false;
        }
        uses_shared_operand_ = true;

        OPENGDA_Debug("dwq", "Using signal pool (offset=%zu) and shared operand",
                     completion_signal_offset_);
    } else {
        // Fallback: Allocate GPU memory separately (less efficient)
        OPENGDA_Warn("dwq", "Signal pool not available, allocating separately");
        uses_signal_pool_ = false;
        uses_shared_operand_ = false;

#ifdef USE_AMDGPU
        uint64_t* signal_ptr = nullptr;
        hipError_t hip_ret = hipMalloc(&signal_ptr, sizeof(uint64_t));
        if (hip_ret != hipSuccess) {
            OPENGDA_Error("dwq", "hipMalloc(completion_signal) failed: %s",
                          hipGetErrorString(hip_ret));
            cleanup();
            return false;
        }
        completion_signal_ = signal_ptr;
        hip_ret = hipMemset((void*)completion_signal_, 0, sizeof(uint64_t));
        if (hip_ret != hipSuccess) {
            OPENGDA_Error("dwq", "hipMemset(completion_signal) failed");
            cleanup();
            return false;
        }

        hip_ret = hipMalloc(&atomic_operand_, sizeof(uint64_t));
        if (hip_ret != hipSuccess) {
            OPENGDA_Error("dwq", "hipMalloc(atomic_operand) failed: %s",
                          hipGetErrorString(hip_ret));
            cleanup();
            return false;
        }
        uint64_t operand_value = 1;
        hip_ret = hipMemcpy(atomic_operand_, &operand_value, sizeof(uint64_t),
                            hipMemcpyHostToDevice);
        if (hip_ret != hipSuccess) {
            OPENGDA_Error("dwq", "hipMemcpy(atomic_operand) failed");
            cleanup();
            return false;
        }
        hipDeviceSynchronize();
#endif

#ifdef USE_NVGPU
        uint64_t* signal_ptr = nullptr;
        cudaError_t cuda_ret = cudaMalloc(&signal_ptr, sizeof(uint64_t));
        if (cuda_ret != cudaSuccess) {
            OPENGDA_Error("dwq", "cudaMalloc(completion_signal) failed: %s",
                          cudaGetErrorString(cuda_ret));
            cleanup();
            return false;
        }
        completion_signal_ = signal_ptr;
        cuda_ret = cudaMemset((void*)completion_signal_, 0, sizeof(uint64_t));
        if (cuda_ret != cudaSuccess) {
            OPENGDA_Error("dwq", "cudaMemset(completion_signal) failed");
            cleanup();
            return false;
        }

        cuda_ret = cudaMalloc(&atomic_operand_, sizeof(uint64_t));
        if (cuda_ret != cudaSuccess) {
            OPENGDA_Error("dwq", "cudaMalloc(atomic_operand) failed: %s",
                          cudaGetErrorString(cuda_ret));
            cleanup();
            return false;
        }
        uint64_t operand_value = 1;
        cuda_ret = cudaMemcpy(atomic_operand_, &operand_value, sizeof(uint64_t),
                              cudaMemcpyHostToDevice);
        if (cuda_ret != cudaSuccess) {
            OPENGDA_Error("dwq", "cudaMemcpy(atomic_operand) failed");
            cleanup();
            return false;
        }
        cudaDeviceSynchronize();
#endif

        // Register completion signal and atomic operand as MRs
        completion_signal_mr_ = ofi_->register_memory((void*)completion_signal_,
                                                       sizeof(uint64_t), true);
        if (!completion_signal_mr_) {
            OPENGDA_Error("dwq", "Failed to register completion_signal MR");
            cleanup();
            return false;
        }

        atomic_operand_mr_ = ofi_->register_memory(atomic_operand_,
                                                    sizeof(uint64_t), true);
        if (!atomic_operand_mr_) {
            OPENGDA_Error("dwq", "Failed to register atomic_operand MR");
            cleanup();
            return false;
        }
    }

    // Allocate work structures (must remain valid until completion)
    op_rma_ = (struct fi_op_rma*)malloc(sizeof(struct fi_op_rma));
    msg_rma_ = (struct fi_msg_rma*)malloc(sizeof(struct fi_msg_rma));
    iov_ = (struct iovec*)malloc(sizeof(struct iovec));
    rma_iov_ = (struct fi_rma_iov*)malloc(sizeof(struct fi_rma_iov));

    op_atomic_ = (struct fi_op_atomic*)malloc(sizeof(struct fi_op_atomic));
    atomic_msg_ = (struct fi_msg_atomic*)malloc(sizeof(struct fi_msg_atomic));
    atomic_iov_ = (struct fi_ioc*)malloc(sizeof(struct fi_ioc));
    atomic_rma_iov_ = (struct fi_rma_ioc*)malloc(sizeof(struct fi_rma_ioc));

    if (!op_rma_ || !msg_rma_ || !iov_ || !rma_iov_ ||
        !op_atomic_ || !atomic_msg_ || !atomic_iov_ || !atomic_rma_iov_) {
        OPENGDA_Error("dwq", "Failed to allocate work structures");
        cleanup();
        return false;
    }

    state_ = State::INITIALIZED;
    OPENGDA_Debug("dwq", "DWQOperation initialized successfully");
    return true;
}

bool DWQOperation::prepare_write(void* local_buf, size_t size, int target_rank,
                                  uint64_t remote_offset) {
    if (!ofi_) {
        OPENGDA_Error("dwq", "OFI not set");
        return false;
    }

    // Get peer info for target
    const PeerInfo* peer = ofi_->get_peer_info(target_rank);
    if (!peer || !peer->valid) {
        OPENGDA_Error("dwq", "Invalid target rank: %d", target_rank);
        return false;
    }

    // Use peer's GPU MR for remote address
    uint64_t remote_addr = peer->mr_info.gpu_mr_addr + remote_offset;
    uint64_t remote_key = peer->mr_info.gpu_mr_key;

    return prepare_write_explicit(local_buf, size, peer->fi_addr,
                                  remote_addr, remote_key);
}

bool DWQOperation::prepare_write_explicit(void* local_buf, size_t size,
                                           fi_addr_t target_addr,
                                           uint64_t remote_addr,
                                           uint64_t remote_key) {
    if (state_ != State::INITIALIZED) {
        OPENGDA_Error("dwq", "Operation not in INITIALIZED state");
        return false;
    }

    if (!ofi_ || !cntr_pair_ || !cntr_pair_->trigger || !cntr_pair_->completion) {
        OPENGDA_Error("dwq", "Invalid OFI or counter state");
        return false;
    }

    op_type_ = OpType::WRITE;

    // Get local MR info for the buffer
    const MRManager::MRInfo* mr_info = ofi_->get_mr_info(local_buf);
    if (!mr_info) {
        OPENGDA_Error("dwq", "Local buffer not registered: %p", local_buf);
        return false;
    }
    local_desc_ = mr_info->desc;

    // Reset counters
    fi_cntr_set(cntr_pair_->trigger->cntr, 0);
    fi_cntr_set(cntr_pair_->completion->cntr, 0);

    // Increment completion threshold (monotonic counter pattern)
    // NIC will atomic SUM +1 to completion_signal_, GPU waits for >= threshold
    // This avoids hipMemset in hot path which causes jitter
    ++completion_threshold_;

    // Determine remote address mode
    // For CXI (no FI_MR_VIRT_ADDR): remote_addr is treated as offset from MR base
    // For providers with FI_MR_VIRT_ADDR: remote_addr is absolute virtual address
    uint64_t remote_addr_for_rma = remote_addr;
    // Note: CXI uses offset mode, so remote_addr should be passed as offset directly

    // Setup RMA work (WORK 1)
    iov_->iov_base = local_buf;
    iov_->iov_len = size;

    rma_iov_->addr = remote_addr_for_rma;
    rma_iov_->len = size;
    rma_iov_->key = remote_key;

    memset(msg_rma_, 0, sizeof(*msg_rma_));
    msg_rma_->msg_iov = iov_;
    msg_rma_->desc = &local_desc_;
    msg_rma_->iov_count = 1;
    msg_rma_->addr = target_addr;
    msg_rma_->rma_iov = rma_iov_;
    msg_rma_->rma_iov_count = 1;
    msg_rma_->context = NULL;
    msg_rma_->data = 0;

    memset(op_rma_, 0, sizeof(*op_rma_));
    op_rma_->ep = ofi_->get_endpoint();
    op_rma_->msg = *msg_rma_;
    op_rma_->flags = FI_COMPLETION;

    memset(&rma_work_, 0, sizeof(rma_work_));
    rma_work_.triggering_cntr = cntr_pair_->trigger->cntr;
    rma_work_.completion_cntr = cntr_pair_->completion->cntr;
    rma_work_.threshold = threshold_;
    rma_work_.op_type = FI_OP_WRITE;
    rma_work_.op.rma = op_rma_;

    // Queue RMA work
    struct fid_domain* domain = ofi_->get_domain();
    int ret = fi_control(&domain->fid, FI_QUEUE_WORK, &rma_work_);
    if (ret) {
        OPENGDA_Error("dwq", "fi_control(FI_QUEUE_WORK, rma) failed: %s (%d)",
                      fi_strerror(-ret), ret);
        return false;
    }

    // Setup atomic work (WORK 2) - NIC signals GPU when RMA completes
    void* desc_atomic_operand = fi_mr_desc(atomic_operand_mr_);

    // Source operand (value to add)
    atomic_iov_->addr = atomic_operand_;
    atomic_iov_->count = 1;

    // Destination (completion_signal on GPU) - local atomic to self
    struct fi_info* fi_info = ofi_->get_info();
    if (uses_signal_pool_) {
        // Using signal pool - use pool's MR key and signal offset
        CompletionSignalPool* pool = ofi_->get_signal_pool();
        if (fi_info && (fi_info->domain_attr->mr_mode & FI_MR_VIRT_ADDR)) {
            atomic_rma_iov_->addr = (uint64_t)completion_signal_;
        } else {
            atomic_rma_iov_->addr = completion_signal_offset_;
        }
        atomic_rma_iov_->key = pool->get_mr_key();
    } else {
        // Using separately registered MR
        if (fi_info && (fi_info->domain_attr->mr_mode & FI_MR_VIRT_ADDR)) {
            atomic_rma_iov_->addr = (uint64_t)completion_signal_;
        } else {
            atomic_rma_iov_->addr = 0;
        }
        atomic_rma_iov_->key = fi_mr_key(completion_signal_mr_);
    }
    atomic_rma_iov_->count = 1;

    // Setup atomic message
    memset(atomic_msg_, 0, sizeof(*atomic_msg_));
    atomic_msg_->msg_iov = atomic_iov_;
    atomic_msg_->desc = &desc_atomic_operand;
    atomic_msg_->iov_count = 1;
    atomic_msg_->addr = ofi_->get_local_fi_addr();  // Target is local
    atomic_msg_->rma_iov = atomic_rma_iov_;
    atomic_msg_->rma_iov_count = 1;
    atomic_msg_->datatype = FI_UINT64;
    atomic_msg_->op = FI_SUM;
    atomic_msg_->context = NULL;
    atomic_msg_->data = 0;

    memset(op_atomic_, 0, sizeof(*op_atomic_));
    op_atomic_->ep = ofi_->get_endpoint();
    op_atomic_->msg = *atomic_msg_;
    op_atomic_->flags = FI_COMPLETION;

    memset(&atomic_work_, 0, sizeof(atomic_work_));
    atomic_work_.op_type = FI_OP_ATOMIC;
    atomic_work_.op.atomic = op_atomic_;
    // Triggered when RMA completion counter reaches threshold
    atomic_work_.triggering_cntr = cntr_pair_->completion->cntr;
    atomic_work_.completion_cntr = nullptr;  // GPU polls completion_signal_ directly
    atomic_work_.threshold = threshold_;

    // Queue atomic work
    ret = fi_control(&domain->fid, FI_QUEUE_WORK, &atomic_work_);
    if (ret) {
        OPENGDA_Error("dwq", "fi_control(FI_QUEUE_WORK, atomic) failed: %s (%d)",
                      fi_strerror(-ret), ret);
        return false;
    }

    // Assign sequence number and register with completion queue
    completion_queue_ = ofi_->get_completion_queue();
    seq_num_ = completion_queue_->allocate_seq_num();
    completion_queue_->register_pending(seq_num_, this);

    state_ = State::PREPARED;
    OPENGDA_Debug("dwq", "DWQ write prepared: seq=%lu, %zu bytes to fi_addr %lu",
                  seq_num_, size, target_addr);
    return true;
}

bool DWQOperation::prepare_read(void* local_buf, size_t size, int source_rank,
                                 uint64_t remote_offset) {
    if (!ofi_) {
        OPENGDA_Error("dwq", "OFI not set");
        return false;
    }

    // Get peer info for source
    const PeerInfo* peer = ofi_->get_peer_info(source_rank);
    if (!peer || !peer->valid) {
        OPENGDA_Error("dwq", "Invalid source rank: %d", source_rank);
        return false;
    }

    // Use peer's GPU MR for remote address
    uint64_t remote_addr = peer->mr_info.gpu_mr_addr + remote_offset;
    uint64_t remote_key = peer->mr_info.gpu_mr_key;

    return prepare_read_explicit(local_buf, size, peer->fi_addr,
                                 remote_addr, remote_key);
}

bool DWQOperation::prepare_read_explicit(void* local_buf, size_t size,
                                          fi_addr_t source_addr,
                                          uint64_t remote_addr,
                                          uint64_t remote_key) {
    if (state_ != State::INITIALIZED) {
        OPENGDA_Error("dwq", "Operation not in INITIALIZED state");
        return false;
    }

    if (!ofi_ || !cntr_pair_ || !cntr_pair_->trigger || !cntr_pair_->completion) {
        OPENGDA_Error("dwq", "Invalid OFI or counter state");
        return false;
    }

    op_type_ = OpType::READ;

    // Get local MR info for the buffer
    const MRManager::MRInfo* mr_info = ofi_->get_mr_info(local_buf);
    if (!mr_info) {
        OPENGDA_Error("dwq", "Local buffer not registered: %p", local_buf);
        return false;
    }
    local_desc_ = mr_info->desc;

    // Reset counters
    fi_cntr_set(cntr_pair_->trigger->cntr, 0);
    fi_cntr_set(cntr_pair_->completion->cntr, 0);

    // Increment completion threshold (monotonic counter pattern)
    // NIC will atomic SUM +1 to completion_signal_, GPU waits for >= threshold
    // This avoids hipMemset in hot path which causes jitter
    ++completion_threshold_;

    // Determine remote address mode
    // For CXI (no FI_MR_VIRT_ADDR): remote_addr is treated as offset from MR base
    // For providers with FI_MR_VIRT_ADDR: remote_addr is absolute virtual address
    uint64_t remote_addr_for_rma = remote_addr;
    // Note: CXI uses offset mode, so remote_addr should be passed as offset directly

    // Setup RMA read work (WORK 1)
    // For read: local buffer receives data FROM remote
    iov_->iov_base = local_buf;
    iov_->iov_len = size;

    rma_iov_->addr = remote_addr_for_rma;
    rma_iov_->len = size;
    rma_iov_->key = remote_key;

    memset(msg_rma_, 0, sizeof(*msg_rma_));
    msg_rma_->msg_iov = iov_;
    msg_rma_->desc = &local_desc_;
    msg_rma_->iov_count = 1;
    msg_rma_->addr = source_addr;
    msg_rma_->rma_iov = rma_iov_;
    msg_rma_->rma_iov_count = 1;
    msg_rma_->context = NULL;
    msg_rma_->data = 0;

    memset(op_rma_, 0, sizeof(*op_rma_));
    op_rma_->ep = ofi_->get_endpoint();
    op_rma_->msg = *msg_rma_;
    op_rma_->flags = FI_COMPLETION;

    memset(&rma_work_, 0, sizeof(rma_work_));
    rma_work_.triggering_cntr = cntr_pair_->trigger->cntr;
    rma_work_.completion_cntr = cntr_pair_->completion->cntr;
    rma_work_.threshold = threshold_;
    rma_work_.op_type = FI_OP_READ;  // Read operation instead of write
    rma_work_.op.rma = op_rma_;

    // Queue RMA work
    struct fid_domain* domain = ofi_->get_domain();
    int ret = fi_control(&domain->fid, FI_QUEUE_WORK, &rma_work_);
    if (ret) {
        OPENGDA_Error("dwq", "fi_control(FI_QUEUE_WORK, rma read) failed: %s (%d)",
                      fi_strerror(-ret), ret);
        return false;
    }

    // Setup atomic work (WORK 2) - NIC signals GPU when RMA completes
    void* desc_atomic_operand = fi_mr_desc(atomic_operand_mr_);

    // Source operand (value to add)
    atomic_iov_->addr = atomic_operand_;
    atomic_iov_->count = 1;

    // Destination (completion_signal on GPU) - local atomic to self
    struct fi_info* fi_info = ofi_->get_info();
    if (uses_signal_pool_) {
        // Using signal pool - use pool's MR key and signal offset
        CompletionSignalPool* pool = ofi_->get_signal_pool();
        if (fi_info && (fi_info->domain_attr->mr_mode & FI_MR_VIRT_ADDR)) {
            atomic_rma_iov_->addr = (uint64_t)completion_signal_;
        } else {
            atomic_rma_iov_->addr = completion_signal_offset_;
        }
        atomic_rma_iov_->key = pool->get_mr_key();
    } else {
        // Using separately registered MR
        if (fi_info && (fi_info->domain_attr->mr_mode & FI_MR_VIRT_ADDR)) {
            atomic_rma_iov_->addr = (uint64_t)completion_signal_;
        } else {
            atomic_rma_iov_->addr = 0;
        }
        atomic_rma_iov_->key = fi_mr_key(completion_signal_mr_);
    }
    atomic_rma_iov_->count = 1;

    // Setup atomic message
    memset(atomic_msg_, 0, sizeof(*atomic_msg_));
    atomic_msg_->msg_iov = atomic_iov_;
    atomic_msg_->desc = &desc_atomic_operand;
    atomic_msg_->iov_count = 1;
    atomic_msg_->addr = ofi_->get_local_fi_addr();  // Target is local
    atomic_msg_->rma_iov = atomic_rma_iov_;
    atomic_msg_->rma_iov_count = 1;
    atomic_msg_->datatype = FI_UINT64;
    atomic_msg_->op = FI_SUM;
    atomic_msg_->context = NULL;
    atomic_msg_->data = 0;

    memset(op_atomic_, 0, sizeof(*op_atomic_));
    op_atomic_->ep = ofi_->get_endpoint();
    op_atomic_->msg = *atomic_msg_;
    op_atomic_->flags = FI_COMPLETION;

    memset(&atomic_work_, 0, sizeof(atomic_work_));
    atomic_work_.op_type = FI_OP_ATOMIC;
    atomic_work_.op.atomic = op_atomic_;
    // Triggered when RMA completion counter reaches threshold
    atomic_work_.triggering_cntr = cntr_pair_->completion->cntr;
    atomic_work_.completion_cntr = nullptr;  // GPU polls completion_signal_ directly
    atomic_work_.threshold = threshold_;

    // Queue atomic work
    ret = fi_control(&domain->fid, FI_QUEUE_WORK, &atomic_work_);
    if (ret) {
        OPENGDA_Error("dwq", "fi_control(FI_QUEUE_WORK, atomic) failed: %s (%d)",
                      fi_strerror(-ret), ret);
        return false;
    }

    // Assign sequence number and register with completion queue
    completion_queue_ = ofi_->get_completion_queue();
    seq_num_ = completion_queue_->allocate_seq_num();
    completion_queue_->register_pending(seq_num_, this);

    state_ = State::PREPARED;
    OPENGDA_Debug("dwq", "DWQ read prepared: seq=%lu, %zu bytes from fi_addr %lu",
                  seq_num_, size, source_addr);
    return true;
}

volatile uint64_t* DWQOperation::get_trigger_addr() const {
    if (state_ != State::PREPARED) {
        return nullptr;
    }
    if (!cntr_pair_ || !cntr_pair_->trigger) {
        return nullptr;
    }
#if defined(USE_AMDGPU) || defined(USE_NVGPU)
    return cntr_pair_->trigger->dev_addr;
#else
    return nullptr;
#endif
}

volatile uint64_t* DWQOperation::get_completion_signal() const {
    if (state_ != State::PREPARED) {
        return nullptr;
    }
    return completion_signal_;
}

bool DWQOperation::reset() {
    if (state_ == State::UNINITIALIZED) {
        return false;
    }

    // Unregister from completion queue
    if (completion_queue_ && seq_num_ != CompletionQueue::INVALID_SEQ) {
        completion_queue_->unregister(seq_num_);
        seq_num_ = CompletionQueue::INVALID_SEQ;
    }
    completion_queue_ = nullptr;

    // Reset counters
    if (cntr_pair_ && cntr_pair_->trigger && cntr_pair_->completion) {
        fi_cntr_set(cntr_pair_->trigger->cntr, 0);
        fi_cntr_set(cntr_pair_->completion->cntr, 0);
    }

    // NOTE: completion_signal_ is NOT cleared here!
    // Using monotonic counter pattern (same as proxy barrier's d_slot_done):
    // - NIC atomically adds +1 to completion_signal_ after each operation
    // - Each prepare() increments completion_threshold_
    // - GPU waits for completion_signal_ >= completion_threshold_
    // This avoids hipMemset/cudaMemset in hot path which causes jitter.

    state_ = State::INITIALIZED;
    return true;
}

bool DWQOperation::is_completed() const {
    if (state_ != State::PREPARED) {
        return state_ == State::COMPLETED;
    }

    // Check completion signal (written by DWQ atomic operation)
    if (completion_signal_) {
        return *completion_signal_ >= threshold_;
    }
    return false;
}

bool DWQOperation::wait_completion(int timeout_ms) {
    if (state_ != State::PREPARED) {
        return state_ == State::COMPLETED;
    }

    if (!completion_signal_) {
        return false;
    }

    // Poll completion signal with timeout
    auto start = std::chrono::steady_clock::now();
    while (*completion_signal_ < threshold_) {
        if (timeout_ms >= 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();
            if (elapsed >= timeout_ms) {
                return false;  // Timeout
            }
        }
        // Brief pause to avoid spinning too hard
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }

    state_ = State::COMPLETED;
    return true;
}

// ============================================================================
// IPC (Same-Node GPU Communication) Implementation
// ============================================================================

#ifdef USE_AMDGPU
bool OFI::setup_local_ipc_handle() {
    if (!default_gpu_mr_.allocated || !default_gpu_mr_.buffer) {
        OPENGDA_Debug("ofi", "No GPU buffer available for IPC");
        return false;
    }

    hipError_t hip_err = hipIpcGetMemHandle(&local_ipc_handle_, default_gpu_mr_.buffer);
    if (hip_err != hipSuccess) {
        OPENGDA_Warn("ofi", "hipIpcGetMemHandle failed: %s", hipGetErrorString(hip_err));
        local_ipc_handle_valid_ = false;
        return false;
    }

    local_ipc_handle_valid_ = true;
    OPENGDA_Info("ofi", "Created local IPC handle for GPU buffer");
    return true;
}

bool OFI::setup_peer_ipc_mappings() {
    int ipc_mapped_count = 0;
    int ipc_failed_count = 0;

    for (int i = 0; i < size_; i++) {
        if (i == rank_) {
            continue;  // Skip self
        }

        if (!peers_[i].can_use_ipc) {
            continue;  // Skip peers without IPC support
        }

        // Open IPC handle from peer
        void* mapped_ptr = nullptr;
        hipError_t hip_err = hipIpcOpenMemHandle(&mapped_ptr,
                                                  peers_[i].ipc_info.ipc_handle,
                                                  hipIpcMemLazyEnablePeerAccess);
        if (hip_err != hipSuccess) {
            OPENGDA_Warn("ofi", "hipIpcOpenMemHandle failed for peer %d: %s",
                        i, hipGetErrorString(hip_err));
            peers_[i].can_use_ipc = false;
            peers_[i].ipc_info.mapped = false;
            ipc_failed_count++;
            continue;
        }

        peers_[i].ipc_info.mapped_ptr = mapped_ptr;
        peers_[i].ipc_info.mapped = true;
        ipc_mapped_count++;

        OPENGDA_Debug("ofi", "IPC mapped peer %d GPU buffer at %p (size=%zu)",
                     i, mapped_ptr, peers_[i].ipc_info.mapped_size);
    }

    OPENGDA_Info("ofi", "IPC setup complete: %d mappings successful, %d failed",
                 ipc_mapped_count, ipc_failed_count);

    return (ipc_failed_count == 0);
}

void OFI::cleanup_ipc_mappings() {
    for (int i = 0; i < size_; i++) {
        if (i == rank_) {
            continue;
        }

        if (peers_[i].ipc_info.mapped && peers_[i].ipc_info.mapped_ptr) {
            hipError_t hip_err = hipIpcCloseMemHandle(peers_[i].ipc_info.mapped_ptr);
            if (hip_err != hipSuccess) {
                OPENGDA_Warn("ofi", "hipIpcCloseMemHandle failed for peer %d: %s",
                            i, hipGetErrorString(hip_err));
            }
            peers_[i].ipc_info.mapped_ptr = nullptr;
            peers_[i].ipc_info.mapped = false;
        }
    }

    OPENGDA_Debug("ofi", "IPC mappings cleaned up");
}

bool OFI::is_ipc_available(int rank) const {
    if (rank < 0 || rank >= size_ || rank == rank_) {
        return false;
    }
    return peers_[rank].can_use_ipc && peers_[rank].ipc_info.mapped;
}

void* OFI::get_ipc_ptr(int rank) const {
    if (!is_ipc_available(rank)) {
        return nullptr;
    }
    return peers_[rank].ipc_info.mapped_ptr;
}

size_t OFI::get_ipc_size(int rank) const {
    if (!is_ipc_available(rank)) {
        return 0;
    }
    return peers_[rank].ipc_info.mapped_size;
}
#endif // USE_AMDGPU

// ============================================================================
// ProxyManager Implementation
// ============================================================================

ProxyManager::ProxyManager()
    : initialized_(false)
    , ofi_(nullptr) {
}

ProxyManager::~ProxyManager() {
    if (initialized_) {
        finalize();
    }
}

bool ProxyManager::initialize(OFI* ofi) {
    if (initialized_) {
        OPENGDA_Warn("proxy_manager", "Already initialized");
        return false;
    }

    if (!ofi) {
        OPENGDA_Error("proxy_manager", "Invalid OFI instance");
        return false;
    }

    ofi_ = ofi;
    initialized_ = true;

    OPENGDA_Info("proxy_manager", "Initialized (window_size_max=%d, ring_size=%d)",
                 MAX_WINDOW_SIZE, RING_BUFFER_SIZE);
    return true;
}

void ProxyManager::finalize() {
    if (!initialized_) {
        return;
    }

    // Stop and destroy all active contexts
    std::lock_guard<std::mutex> lock(contexts_mutex_);
    for (auto ctx : active_contexts_) {
        if (ctx->running.load()) {
            stop_proxy(ctx);
        }
        free_barrier_resources(ctx);
        delete ctx;
    }
    active_contexts_.clear();

    initialized_ = false;
    OPENGDA_Debug("proxy_manager", "Finalized");
}

ProxyManager::ProxyBarrierContext* ProxyManager::create_barrier(int window_size) {
    if (!initialized_) {
        OPENGDA_Error("proxy_manager", "Not initialized");
        return nullptr;
    }

    // Validate and adjust window size
    if (window_size <= 0) {
        window_size = DEFAULT_WINDOW_SIZE;
    }
    if (window_size > MAX_WINDOW_SIZE) {
        OPENGDA_Warn("proxy_manager", "Window size %d exceeds max %d, clamping",
                     window_size, MAX_WINDOW_SIZE);
        window_size = MAX_WINDOW_SIZE;
    }

    // Create context
    ProxyBarrierContext* ctx = new ProxyBarrierContext();
    ctx->window_size = window_size;
    ctx->mype = ofi_->get_rank();
    ctx->npes = ofi_->get_size();
    ctx->ofi = ofi_;
    ctx->initialized = false;
    ctx->running.store(false);
    ctx->stop_requested.store(false);
    ctx->total_rearms.store(0);
    ctx->ring_polls.store(0);
    ctx->idle_polls.store(0);
    ctx->cq_events_drained.store(0);

    // Calculate number of phases (log2(npes), rounded up)
    ctx->num_phases = 0;
    for (int k = 1; k < ctx->npes; k <<= 1) {
        ctx->num_phases++;
    }
    if (ctx->num_phases > GDA_BARRIER_MAX_PHASES) {
        OPENGDA_Error("proxy_manager", "Too many phases (%d > %d) for %d ranks",
                      ctx->num_phases, GDA_BARRIER_MAX_PHASES, ctx->npes);
        delete ctx;
        return nullptr;
    }

    OPENGDA_Info("proxy_manager", "Creating proxy barrier: window=%d, npes=%d, phases=%d",
                 window_size, ctx->npes, ctx->num_phases);

    // Allocate resources
    if (!allocate_barrier_resources(ctx, window_size)) {
        OPENGDA_Error("proxy_manager", "Failed to allocate barrier resources");
        delete ctx;
        return nullptr;
    }

    // Setup slots and initial DWQ operations
    if (!setup_barrier_slots(ctx)) {
        OPENGDA_Error("proxy_manager", "Failed to setup barrier slots");
        free_barrier_resources(ctx);
        delete ctx;
        return nullptr;
    }

    // Arm all slots initially
    if (!arm_all_slots(ctx)) {
        OPENGDA_Error("proxy_manager", "Failed to arm initial slots");
        free_barrier_resources(ctx);
        delete ctx;
        return nullptr;
    }

    ctx->initialized = true;

    // Track active context
    {
        std::lock_guard<std::mutex> lock(contexts_mutex_);
        active_contexts_.push_back(ctx);
    }

    OPENGDA_Info("proxy_manager", "Proxy barrier created successfully");
    return ctx;
}

void ProxyManager::destroy_barrier(ProxyBarrierContext* ctx) {
    if (!ctx) return;

    // Stop proxy if running
    if (ctx->running.load()) {
        stop_proxy(ctx);
    }

    // Remove from active contexts
    {
        std::lock_guard<std::mutex> lock(contexts_mutex_);
        auto it = std::find(active_contexts_.begin(), active_contexts_.end(), ctx);
        if (it != active_contexts_.end()) {
            active_contexts_.erase(it);
        }
    }

    free_barrier_resources(ctx);
    delete ctx;
}

bool ProxyManager::allocate_barrier_resources(ProxyBarrierContext* ctx, int window_size) {
    int total_slots = window_size * ctx->num_phases;

    // Allocate GPU-accessible memory for state arrays
#ifdef USE_AMDGPU
    hipError_t hip_err;

    // armed_epoch array [window_size]
    hip_err = hipHostMalloc((void**)&ctx->armed_epoch,
                            sizeof(uint64_t) * window_size,
                            hipHostMallocMapped);
    if (hip_err != hipSuccess) {
        OPENGDA_Error("proxy_manager", "hipHostMalloc(armed_epoch) failed: %s",
                      hipGetErrorString(hip_err));
        return false;
    }

    // slot_state array [window_size]
    hip_err = hipHostMalloc((void**)&ctx->slot_state,
                            sizeof(int) * window_size,
                            hipHostMallocMapped);
    if (hip_err != hipSuccess) {
        OPENGDA_Error("proxy_manager", "hipHostMalloc(slot_state) failed");
        hipHostFree((void*)ctx->armed_epoch);
        return false;
    }

    // sync_arr [npes] - allocated from GPU MR buffer for RDMA compatibility
    // (will be set later after getting GPU MR info)
    ctx->sync_arr = nullptr;

    // sync_counter
    hip_err = hipHostMalloc((void**)&ctx->sync_counter,
                            sizeof(uint64_t),
                            hipHostMallocMapped);
    if (hip_err != hipSuccess) {
        OPENGDA_Error("proxy_manager", "hipHostMalloc(sync_counter) failed");
        hipHostFree((void*)ctx->armed_epoch);
        hipHostFree((void*)ctx->slot_state);
        // sync_arr will be set from GPU MR, not freed here
        return false;
    }

    // Ring buffer for GPU->CPU communication
    ctx->ring_size = RING_BUFFER_SIZE;
    hip_err = hipHostMalloc((void**)&ctx->free_ring,
                            sizeof(uint64_t) * ctx->ring_size,
                            hipHostMallocMapped);
    if (hip_err != hipSuccess) {
        OPENGDA_Error("proxy_manager", "hipHostMalloc(free_ring) failed");
        hipHostFree((void*)ctx->armed_epoch);
        hipHostFree((void*)ctx->slot_state);
        // sync_arr points into GPU MR, not freed separately
        hipHostFree((void*)ctx->sync_counter);
        return false;
    }

    hip_err = hipHostMalloc((void**)&ctx->free_ring_head,
                            sizeof(uint64_t),
                            hipHostMallocMapped);
    if (hip_err != hipSuccess) {
        OPENGDA_Error("proxy_manager", "hipHostMalloc(free_ring_head) failed");
        hipHostFree((void*)ctx->armed_epoch);
        hipHostFree((void*)ctx->slot_state);
        hipHostFree((void*)ctx->sync_counter);
        hipHostFree((void*)ctx->free_ring);
        return false;
    }

    hip_err = hipHostMalloc((void**)&ctx->free_ring_tail,
                            sizeof(uint64_t),
                            hipHostMallocMapped);
    if (hip_err != hipSuccess) {
        OPENGDA_Error("proxy_manager", "hipHostMalloc(free_ring_tail) failed");
        hipHostFree((void*)ctx->armed_epoch);
        hipHostFree((void*)ctx->slot_state);
        hipHostFree((void*)ctx->sync_counter);
        hipHostFree((void*)ctx->free_ring);
        hipHostFree((void*)ctx->free_ring_head);
        return false;
    }

    // GPU handles array
    hip_err = hipHostMalloc((void**)&ctx->gpu_handles,
                            sizeof(gda_gpu_handle_t) * total_slots,
                            hipHostMallocMapped);
    if (hip_err != hipSuccess) {
        OPENGDA_Error("proxy_manager", "hipHostMalloc(gpu_handles) failed");
        // Cleanup all previous allocations
        hipHostFree((void*)ctx->armed_epoch);
        hipHostFree((void*)ctx->slot_state);
        hipHostFree((void*)ctx->sync_counter);
        hipHostFree((void*)ctx->free_ring);
        hipHostFree((void*)ctx->free_ring_head);
        hipHostFree((void*)ctx->free_ring_tail);
        return false;
    }
#else
    // Non-GPU fallback (allocate regular memory)
    ctx->armed_epoch = (volatile uint64_t*)malloc(sizeof(uint64_t) * window_size);
    ctx->slot_state = (volatile int*)malloc(sizeof(int) * window_size);
    ctx->sync_arr = (volatile uint64_t*)malloc(sizeof(uint64_t) * ctx->npes);
    ctx->sync_counter = (volatile uint64_t*)malloc(sizeof(uint64_t));
    ctx->ring_size = RING_BUFFER_SIZE;
    ctx->free_ring = (volatile uint64_t*)malloc(sizeof(uint64_t) * ctx->ring_size);
    ctx->free_ring_head = (volatile uint64_t*)malloc(sizeof(uint64_t));
    ctx->free_ring_tail = (volatile uint64_t*)malloc(sizeof(uint64_t));
    ctx->gpu_handles = (gda_gpu_handle_t*)malloc(sizeof(gda_gpu_handle_t) * total_slots);

    if (!ctx->armed_epoch || !ctx->slot_state || !ctx->sync_arr ||
        !ctx->sync_counter || !ctx->free_ring || !ctx->free_ring_head ||
        !ctx->free_ring_tail || !ctx->gpu_handles) {
        OPENGDA_Error("proxy_manager", "Memory allocation failed");
        // Cleanup
        free((void*)ctx->armed_epoch);
        free((void*)ctx->slot_state);
        free((void*)ctx->sync_arr);
        free((void*)ctx->sync_counter);
        free((void*)ctx->free_ring);
        free((void*)ctx->free_ring_head);
        free((void*)ctx->free_ring_tail);
        free(ctx->gpu_handles);
        return false;
    }
#endif

    // Initialize state arrays
    for (int i = 0; i < window_size; i++) {
        ctx->armed_epoch[i] = 0;
        ctx->slot_state[i] = GDA_PROXY_SLOT_NEED_REARM;
    }
#ifndef USE_AMDGPU
    // For non-GPU builds, sync_arr is allocated here; initialize it
    for (int i = 0; i < ctx->npes; i++) {
        ctx->sync_arr[i] = 0;
    }
#endif
    // For USE_AMDGPU, sync_arr is set from GPU MR later and initialized with hipMemset
    *ctx->sync_counter = 0;
    *ctx->free_ring_head = 0;
    *ctx->free_ring_tail = 0;

    // Allocate slots vector
    ctx->slots.resize(total_slots);
    for (int i = 0; i < total_slots; i++) {
        ctx->slots[i].index = i;
        ctx->slots[i].cntr_pair = nullptr;
        ctx->slots[i].completion_signal = nullptr;
        ctx->slots[i].initialized = false;
    }

    // Calculate phase targets and sources (dissemination algorithm)
    for (int phase = 0; phase < ctx->num_phases; phase++) {
        int distance = 1 << phase;
        ctx->phase_targets[phase] = (ctx->mype + distance) % ctx->npes;
        ctx->phase_sources[phase] = (ctx->mype - distance + ctx->npes) % ctx->npes;
    }

    // Get peer info for phase targets
    for (int phase = 0; phase < ctx->num_phases; phase++) {
        int target = ctx->phase_targets[phase];
        const PeerInfo* peer = ofi_->get_peer_info(target);
        if (!peer || !peer->valid) {
            OPENGDA_Error("proxy_manager", "Invalid peer info for target rank %d", target);
            return false;
        }
        ctx->phase_fi_addrs[phase] = peer->fi_addr;
        ctx->phase_remote_addrs[phase] = peer->mr_info.gpu_mr_addr;
        ctx->phase_remote_keys[phase] = peer->mr_info.gpu_mr_key;
    }

    // Use the GPU MR buffer for sync_arr (for RDMA compatibility)
    // This ensures remote RDMA writes go to the same buffer GPU reads from
    const DefaultMRInfo* gpu_mr = ofi_->get_default_gpu_mr();
    if (!gpu_mr || !gpu_mr->allocated) {
        OPENGDA_Error("proxy_manager", "No GPU MR available");
        return false;
    }

    // Set sync_arr to point into the beginning of GPU MR buffer
    size_t sync_arr_size = sizeof(uint64_t) * ctx->npes;
    ctx->sync_arr_offset = 0;
    ctx->sync_arr = (volatile uint64_t*)gpu_mr->buffer;

    // Use the GPU MR's registration info for RDMA operations
    ctx->sync_arr_mr = nullptr;  // Not separately registered
    ctx->sync_arr_key = gpu_mr->key;
    ctx->sync_arr_desc = gpu_mr->desc;

    // Initialize sync_arr in GPU memory
    hipError_t hip_init_err = hipMemset((void*)ctx->sync_arr, 0, sync_arr_size);
    if (hip_init_err != hipSuccess) {
        OPENGDA_Error("proxy_manager", "Failed to initialize sync_arr: %s",
                      hipGetErrorString(hip_init_err));
        return false;
    }
    hipDeviceSynchronize();

    OPENGDA_Debug("proxy_manager", "Set sync_arr in GPU MR: addr=%p, key=0x%lx, size=%zu",
                  (void*)ctx->sync_arr, (unsigned long)ctx->sync_arr_key, sync_arr_size);

    OPENGDA_Debug("proxy_manager", "Allocated barrier resources: %d slots",
                  total_slots);
    return true;
}

void ProxyManager::free_barrier_resources(ProxyBarrierContext* ctx) {
    if (!ctx) return;

    // Note: sync_arr_mr is nullptr since we use GPU MR directly
    // No need to deregister

    // Release counter pairs
    for (auto& slot : ctx->slots) {
        if (slot.cntr_pair) {
            ofi_->release_cntr_pair(slot.cntr_pair);
            slot.cntr_pair = nullptr;
        }
        // Note: completion signals are from the pool, not owned by slot
    }
    ctx->slots.clear();

#ifdef USE_AMDGPU
    if (ctx->armed_epoch) hipHostFree((void*)ctx->armed_epoch);
    if (ctx->slot_state) hipHostFree((void*)ctx->slot_state);
    // sync_arr points into GPU MR buffer, don't free it
    if (ctx->sync_counter) hipHostFree((void*)ctx->sync_counter);
    if (ctx->free_ring) hipHostFree((void*)ctx->free_ring);
    if (ctx->free_ring_head) hipHostFree((void*)ctx->free_ring_head);
    if (ctx->free_ring_tail) hipHostFree((void*)ctx->free_ring_tail);
    if (ctx->gpu_handles) hipHostFree(ctx->gpu_handles);
#else
    free((void*)ctx->armed_epoch);
    free((void*)ctx->slot_state);
    free((void*)ctx->sync_arr);
    free((void*)ctx->sync_counter);
    free((void*)ctx->free_ring);
    free((void*)ctx->free_ring_head);
    free((void*)ctx->free_ring_tail);
    free(ctx->gpu_handles);
#endif

    ctx->armed_epoch = nullptr;
    ctx->slot_state = nullptr;
    ctx->sync_arr = nullptr;
    ctx->sync_counter = nullptr;
    ctx->free_ring = nullptr;
    ctx->free_ring_head = nullptr;
    ctx->free_ring_tail = nullptr;
    ctx->gpu_handles = nullptr;
}

bool ProxyManager::setup_barrier_slots(ProxyBarrierContext* ctx) {
    CompletionSignalPool* signal_pool = ofi_->get_signal_pool();
    if (!signal_pool || !signal_pool->is_initialized()) {
        OPENGDA_Error("proxy_manager", "Signal pool not available");
        return false;
    }

    const DefaultMRInfo* gpu_mr = ofi_->get_default_gpu_mr();
    if (!gpu_mr) {
        OPENGDA_Error("proxy_manager", "No GPU MR available");
        return false;
    }

    // Allocate counter pairs and completion signals for each slot
    for (int slot_idx = 0; slot_idx < (int)ctx->slots.size(); slot_idx++) {
        ProxySlot& slot = ctx->slots[slot_idx];

        // Allocate counter pair
        if (!ofi_->allocate_cntr_pair(&slot.cntr_pair)) {
            OPENGDA_Error("proxy_manager", "Failed to allocate counter pair for slot %d",
                          slot_idx);
            return false;
        }

        // Allocate completion signal from pool
        if (!signal_pool->allocate(&slot.completion_signal, &slot.completion_signal_offset)) {
            OPENGDA_Error("proxy_manager", "Failed to allocate completion signal for slot %d",
                          slot_idx);
            return false;
        }

        // Initialize completion signal to 0
        *slot.completion_signal = 0;

        // Setup GPU handle for this slot
        gda_gpu_handle_t& gpu_handle = ctx->gpu_handles[slot_idx];
        gpu_handle.trigger_addr = slot.cntr_pair->trigger->dev_addr;
        gpu_handle.completion_addr = slot.completion_signal;
        gpu_handle.trigger_threshold = 1;
        gpu_handle.is_ipc = 0;
        gpu_handle.ipc_dest_addr = nullptr;
        gpu_handle.ipc_src_addr = nullptr;
        gpu_handle.ipc_size = 0;

        slot.initialized = true;
    }

    OPENGDA_Debug("proxy_manager", "Setup %zu barrier slots", ctx->slots.size());
    return true;
}

bool ProxyManager::arm_slot(ProxyBarrierContext* ctx, int slot_idx, uint64_t epoch) {
    if (slot_idx < 0 || slot_idx >= (int)ctx->slots.size()) {
        OPENGDA_Error("proxy_manager", "Invalid slot index %d", slot_idx);
        return false;
    }

    ProxySlot& slot = ctx->slots[slot_idx];
    if (!slot.initialized || !slot.cntr_pair) {
        OPENGDA_Error("proxy_manager", "Slot %d not initialized", slot_idx);
        return false;
    }

    // Calculate which window slot and phase this is
    int window_slot = slot_idx / ctx->num_phases;
    int phase = slot_idx % ctx->num_phases;

    // Reset counters
    fi_cntr_set(slot.cntr_pair->trigger->cntr, 0);
    fi_cntr_set(slot.cntr_pair->completion->cntr, 0);

    // Reset completion signal
    *slot.completion_signal = 0;

    // Get remote info for this phase's target
    int target = ctx->phase_targets[phase];
    fi_addr_t target_fi_addr = ctx->phase_fi_addrs[phase];
    uint64_t remote_addr = ctx->phase_remote_addrs[phase];
    uint64_t remote_key = ctx->phase_remote_keys[phase];

    // Get local buffer info (sync_arr slot for our rank)
    // Use the registered sync_arr MR descriptor for RDMA
    void* local_desc = ctx->sync_arr_desc;

    // We write our sync_arr[mype] to remote's sync_arr[mype]
    // Local buffer: address of our sync_arr[mype]
    // Remote buffer: offset mype * sizeof(uint64_t) in remote's sync_arr

    void* local_buf = (void*)&ctx->sync_arr[ctx->mype];
    size_t xfer_size = sizeof(uint64_t);
    uint64_t remote_offset = ctx->mype * sizeof(uint64_t);  // Our slot in remote's sync_arr

    // Setup RDMA write work
    memset(&slot.iov, 0, sizeof(slot.iov));
    slot.iov.iov_base = local_buf;
    slot.iov.iov_len = xfer_size;

    memset(&slot.rma_iov, 0, sizeof(slot.rma_iov));
    slot.rma_iov.addr = remote_addr + remote_offset;
    slot.rma_iov.len = xfer_size;
    slot.rma_iov.key = remote_key;

    memset(&slot.msg_rma, 0, sizeof(slot.msg_rma));
    slot.msg_rma.msg_iov = &slot.iov;
    slot.msg_rma.desc = &local_desc;
    slot.msg_rma.iov_count = 1;
    slot.msg_rma.addr = target_fi_addr;
    slot.msg_rma.rma_iov = &slot.rma_iov;
    slot.msg_rma.rma_iov_count = 1;

    memset(&slot.op_rma, 0, sizeof(slot.op_rma));
    slot.op_rma.ep = ofi_->get_endpoint();
    slot.op_rma.msg = slot.msg_rma;
    slot.op_rma.flags = FI_COMPLETION;

    memset(&slot.rma_work, 0, sizeof(slot.rma_work));
    slot.rma_work.triggering_cntr = slot.cntr_pair->trigger->cntr;
    slot.rma_work.completion_cntr = slot.cntr_pair->completion->cntr;
    slot.rma_work.threshold = 1;
    slot.rma_work.op_type = FI_OP_WRITE;
    slot.rma_work.op.rma = &slot.op_rma;

    // Queue RMA work
    int ret = fi_control(&ofi_->get_domain()->fid, FI_QUEUE_WORK, &slot.rma_work);
    if (ret) {
        OPENGDA_Error("proxy_manager", "fi_control(QUEUE_WORK) failed for slot %d: %s",
                      slot_idx, fi_strerror(-ret));
        return false;
    }

    // Setup atomic work (for completion notification)
    uint64_t* atomic_operand = nullptr;
    struct fid_mr* atomic_operand_mr = nullptr;
    if (!ofi_->get_shared_atomic_operand(&atomic_operand, &atomic_operand_mr)) {
        OPENGDA_Error("proxy_manager", "Failed to get atomic operand");
        return false;
    }

    memset(&slot.atomic_iov, 0, sizeof(slot.atomic_iov));
    slot.atomic_iov.addr = atomic_operand;
    slot.atomic_iov.count = 1;

    // Atomic writes to local completion signal
    CompletionSignalPool* signal_pool = ofi_->get_signal_pool();
    uint64_t signal_addr;
    if (ofi_->get_info()->domain_attr->mr_mode & FI_MR_VIRT_ADDR) {
        signal_addr = (uint64_t)slot.completion_signal;
    } else {
        signal_addr = slot.completion_signal_offset;
    }

    memset(&slot.atomic_rma_iov, 0, sizeof(slot.atomic_rma_iov));
    slot.atomic_rma_iov.addr = signal_addr;
    slot.atomic_rma_iov.count = 1;
    slot.atomic_rma_iov.key = signal_pool->get_mr_key();

    memset(&slot.atomic_msg, 0, sizeof(slot.atomic_msg));
    slot.atomic_msg.msg_iov = &slot.atomic_iov;
    void* atomic_desc = fi_mr_desc(atomic_operand_mr);
    slot.atomic_msg.desc = &atomic_desc;
    slot.atomic_msg.iov_count = 1;
    slot.atomic_msg.addr = ofi_->get_local_fi_addr();  // Atomic to self
    slot.atomic_msg.rma_iov = &slot.atomic_rma_iov;
    slot.atomic_msg.rma_iov_count = 1;
    slot.atomic_msg.datatype = FI_UINT64;
    slot.atomic_msg.op = FI_SUM;

    memset(&slot.op_atomic, 0, sizeof(slot.op_atomic));
    slot.op_atomic.ep = ofi_->get_endpoint();
    slot.op_atomic.msg = slot.atomic_msg;
    slot.op_atomic.flags = FI_COMPLETION;

    memset(&slot.atomic_work, 0, sizeof(slot.atomic_work));
    slot.atomic_work.triggering_cntr = slot.cntr_pair->completion->cntr;
    slot.atomic_work.completion_cntr = nullptr;  // No completion counter for atomic
    slot.atomic_work.threshold = 1;
    slot.atomic_work.op_type = FI_OP_ATOMIC;
    slot.atomic_work.op.atomic = &slot.op_atomic;

    // Queue atomic work
    ret = fi_control(&ofi_->get_domain()->fid, FI_QUEUE_WORK, &slot.atomic_work);
    if (ret) {
        OPENGDA_Error("proxy_manager", "fi_control(QUEUE_WORK atomic) failed for slot %d: %s",
                      slot_idx, fi_strerror(-ret));
        return false;
    }

    return true;
}

bool ProxyManager::arm_all_slots(ProxyBarrierContext* ctx) {
    OPENGDA_Debug("proxy_manager", "Arming all %zu slots", ctx->slots.size());

    for (int slot_idx = 0; slot_idx < (int)ctx->slots.size(); slot_idx++) {
        if (!arm_slot(ctx, slot_idx, 0)) {
            OPENGDA_Error("proxy_manager", "Failed to arm slot %d", slot_idx);
            return false;
        }
    }

    // Mark all window slots as armed for epoch 0
    for (int w = 0; w < ctx->window_size; w++) {
        ctx->armed_epoch[w] = 0;
        ctx->slot_state[w] = GDA_PROXY_SLOT_ARMED;
    }

    __sync_synchronize();  // Memory barrier

    OPENGDA_Debug("proxy_manager", "All slots armed");
    return true;
}

bool ProxyManager::start_proxy(ProxyBarrierContext* ctx) {
    if (!ctx || !ctx->initialized) {
        OPENGDA_Error("proxy_manager", "Invalid or uninitialized context");
        return false;
    }

    if (ctx->running.load()) {
        OPENGDA_Warn("proxy_manager", "Proxy already running");
        return true;
    }

    ctx->stop_requested.store(false);
    ctx->running.store(true);

    // Start proxy thread
    ctx->proxy_thread = std::thread(proxy_thread_func, ctx);

    OPENGDA_Info("proxy_manager", "Proxy thread started");
    return true;
}

bool ProxyManager::stop_proxy(ProxyBarrierContext* ctx) {
    if (!ctx) {
        return false;
    }

    if (!ctx->running.load()) {
        return true;  // Already stopped
    }

    ctx->stop_requested.store(true);

    if (ctx->proxy_thread.joinable()) {
        ctx->proxy_thread.join();
    }

    ctx->running.store(false);

    OPENGDA_Info("proxy_manager", "Proxy thread stopped (rearms=%lu, polls=%lu, idle=%lu, cq=%lu)",
                 ctx->total_rearms.load(),
                 ctx->ring_polls.load(),
                 ctx->idle_polls.load(),
                 ctx->cq_events_drained.load());

    return true;
}

ProxyManager::ProxyStats ProxyManager::get_stats(ProxyBarrierContext* ctx) {
    ProxyStats stats = {};
    if (ctx) {
        stats.total_rearms = ctx->total_rearms.load();
        stats.ring_polls = ctx->ring_polls.load();
        stats.idle_polls = ctx->idle_polls.load();
        stats.cq_events_drained = ctx->cq_events_drained.load();
    }
    return stats;
}

// Internal arm_slot that doesn't log as much (forward declaration for use in process_ring_buffer)
static bool arm_slot_internal(ProxyManager::ProxyBarrierContext* ctx, int slot_idx, uint64_t epoch) {
    if (slot_idx < 0 || slot_idx >= (int)ctx->slots.size()) {
        return false;
    }

    ProxyManager::ProxySlot& slot = ctx->slots[slot_idx];
    if (!slot.initialized || !slot.cntr_pair) {
        return false;
    }

    OFI* ofi = ctx->ofi;

    // Reset counters
    fi_cntr_set(slot.cntr_pair->trigger->cntr, 0);
    fi_cntr_set(slot.cntr_pair->completion->cntr, 0);

    // Reset completion signal
    *slot.completion_signal = 0;

    // Queue RMA work (structures already setup from initial arm)
    int ret = fi_control(&ofi->get_domain()->fid, FI_QUEUE_WORK, &slot.rma_work);
    if (ret) {
        return false;
    }

    // Queue atomic work
    ret = fi_control(&ofi->get_domain()->fid, FI_QUEUE_WORK, &slot.atomic_work);
    if (ret) {
        return false;
    }

    return true;
}

void ProxyManager::proxy_thread_func(ProxyBarrierContext* ctx) {
    OPENGDA_Debug("proxy_manager", "Proxy thread starting");

    while (!ctx->stop_requested.load()) {
        bool did_work = false;

        // Process ring buffer entries
        process_ring_buffer(ctx);
        ctx->ring_polls.fetch_add(1);

        // Drain CQ to prevent overflow
        drain_cq(ctx);

        if (!did_work) {
            ctx->idle_polls.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::microseconds(PROXY_POLL_INTERVAL_US));
        }
    }

    OPENGDA_Debug("proxy_manager", "Proxy thread exiting");
}

void ProxyManager::process_ring_buffer(ProxyBarrierContext* ctx) {
    volatile uint64_t* ring = ctx->free_ring;
    volatile uint64_t* head = ctx->free_ring_head;
    volatile uint64_t* tail = ctx->free_ring_tail;
    int ring_size = ctx->ring_size;

    // Process all available entries
    while (*tail != *head) {
        uint64_t entry = ring[*tail];
        uint64_t epoch = entry >> 8;
        int slot = (int)(entry & 0xFF);

        // Validate slot
        if (slot < 0 || slot >= ctx->window_size) {
            OPENGDA_Warn("proxy_manager", "Invalid slot %d in ring buffer", slot);
            *tail = (*tail + 1) % ring_size;
            continue;
        }

        // Calculate next epoch for this slot
        uint64_t next_epoch = epoch + ctx->window_size;

        // Rearm all phases for this window slot
        int base_slot_idx = slot * ctx->num_phases;
        bool arm_success = true;
        for (int phase = 0; phase < ctx->num_phases; phase++) {
            int slot_idx = base_slot_idx + phase;
            if (!arm_slot_internal(ctx, slot_idx, next_epoch)) {
                arm_success = false;
                break;
            }
        }

        if (arm_success) {
            // Update armed_epoch for this window slot
            ctx->armed_epoch[slot] = next_epoch;
            ctx->slot_state[slot] = GDA_PROXY_SLOT_ARMED;
            __sync_synchronize();

            ctx->total_rearms.fetch_add(1);
        }

        // Advance tail
        *tail = (*tail + 1) % ring_size;
    }
}

void ProxyManager::drain_cq(ProxyBarrierContext* ctx) {
    struct fid_cq* cq = ctx->ofi->get_cq();
    struct fi_cq_entry entries[32];

    while (true) {
        int ret = fi_cq_read(cq, entries, 32);
        if (ret > 0) {
            ctx->cq_events_drained.fetch_add(ret);
        } else if (ret == -FI_EAGAIN) {
            break;  // No more entries
        } else if (ret < 0) {
            // Error - could log but continue
            break;
        }
    }
}