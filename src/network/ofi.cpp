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
// OFI Implementation
// ============================================================================

#ifdef USE_AMDGPU
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

    // Initialize peer info array
    peers_.resize(size);
    for (int i = 0; i < size; i++) {
        peers_[i].rank = i;
        peers_[i].fi_addr = FI_ADDR_NOTAVAIL;
        peers_[i].valid = false;
        memset(&peers_[i].mr_info, 0, sizeof(PeerMRInfo));
    }

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

    // Initialize OFI
    hints = fi_allocinfo();
    hints->caps = FI_RMA | FI_MSG | FI_HMEM;
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
        OPENGDA_Debug("ofi", "Registering GPU memory: buf=%p, size=%zu, device=%d",
                      buf, size, device_id);
        #else
        OPENGDA_Error("ofi", "GPU memory registration requested but USE_AMDGPU not defined");
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
    return mr_manager_.get_info_by_addr(addr);
}

MRManager::MRStats OFI::get_mr_stats() {
    return mr_manager_.get_stats();
}

void OFI::print_mr_stats() {
    mr_manager_.print_stats();
}

// Destructor - cleanup any remaining MRs
OFI::~OFI() {
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

        // Store local fi_addr
        if (i == rank_) {
            local_fi_addr_ = peer_fi_addr;
        }

        OPENGDA_Debug("ofi", "Peer %d: fi_addr=%lu, host_mr_key=0x%lx, gpu_mr_key=0x%lx",
                      i, (unsigned long)peer_fi_addr,
                      (unsigned long)peers_[i].mr_info.host_mr_key,
                      (unsigned long)peers_[i].mr_info.gpu_mr_key);
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