#include "ofi.hpp"
#include <map>
#include <string>
#include <algorithm>

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

OFI::OFI(int rank) {
    this->rank = rank;
    this->device_id = 0; // Default to GPU 0

    // Initialize logging system
    opengda::LogContext::instance().init(rank);

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

    // Close OFI resources
    if (ofi_initialized) {
        if (ep) fi_close(&ep->fid);
        if (cq) fi_close(&cq->fid);
        if (av) fi_close(&av->fid);
        if (domain) fi_close(&domain->fid);
        if (fabric) fi_close(&fabric->fid);
        if (info) fi_freeinfo(info);
    }
}