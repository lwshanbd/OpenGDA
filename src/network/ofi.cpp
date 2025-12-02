#include "ofi.hpp"
#include <map>
#include <string>

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
        hip_err = hipDeviceGetPCIBusId(hip_pci_bus_id, sizeof(hip_pci_bus_id), 0);
        if (hip_err == hipSuccess) {
            OPENGDA_Info("ofi", "HIP reports GPU 0 at PCI Bus ID: %s", hip_pci_bus_id);
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