/*
 * device_affinity.hpp - hwloc-based GPU-NIC affinity detection
 *
 * Uses hwloc to detect NUMA affinity and select GPU-NIC pairs that share
 * the same NUMA/Package/Group for optimal PCIe locality.
 */
#pragma once

#include <hip/hip_runtime.h>
#include <hwloc.h>
#include <rdma/fabric.h>

#include <cstdio>
#include <cstring>
#include <map>
#include <string>

struct DeviceAffinity {
    std::string pci_id;              // PCI address like "0000:c1:00.0"
    hwloc_obj_type_t affinity_type;  // Type: GROUP/PACKAGE/NUMANODE
    int affinity_index;              // Logical index of affinity object

    DeviceAffinity() : affinity_type(HWLOC_OBJ_TYPE_MAX), affinity_index(-1) {}
};

class DeviceAffinityDetector {
public:
    int selected_gpu_id;             // HIP device ID to use
    std::string selected_cxi_domain; // CXI domain name (e.g., "cxi0")
    bool affinity_matched;           // True if GPU-NIC affinity was found

    // Auto-detect best GPU-NIC pair
    DeviceAffinityDetector() :
        selected_gpu_id(0),
        affinity_matched(false),
        topo_(nullptr)
    {
        // Initialize hwloc topology
        hwloc_topology_init(&topo_);
        hwloc_topology_set_io_types_filter(topo_, HWLOC_TYPE_FILTER_KEEP_ALL);
        hwloc_topology_load(topo_);

        detect_affinity();
    }

    // Use specified GPU and find matching CXI
    explicit DeviceAffinityDetector(int gpu_id) :
        selected_gpu_id(gpu_id),
        affinity_matched(false),
        topo_(nullptr)
    {
        // Initialize hwloc topology
        hwloc_topology_init(&topo_);
        hwloc_topology_set_io_types_filter(topo_, HWLOC_TYPE_FILTER_KEEP_ALL);
        hwloc_topology_load(topo_);

        detect_affinity_for_gpu(gpu_id);
    }

    ~DeviceAffinityDetector() {
        if (topo_) {
            hwloc_topology_destroy(topo_);
        }
    }

    // No copy/move
    DeviceAffinityDetector(const DeviceAffinityDetector&) = delete;
    DeviceAffinityDetector& operator=(const DeviceAffinityDetector&) = delete;

    // Find the best CXI provider from fi_info list based on detected affinity
    struct fi_info* select_cxi_provider(struct fi_info* info_list) {
        if (!affinity_matched || selected_cxi_domain.empty()) {
            // No affinity detected, return first CXI provider
            for (struct fi_info* cur = info_list; cur; cur = cur->next) {
                if (cur->fabric_attr && cur->fabric_attr->prov_name &&
                    strcmp(cur->fabric_attr->prov_name, "cxi") == 0) {
                    return cur;
                }
            }
            return nullptr;
        }

        // Find CXI provider matching our selected domain
        for (struct fi_info* cur = info_list; cur; cur = cur->next) {
            if (cur->fabric_attr && cur->fabric_attr->prov_name &&
                strcmp(cur->fabric_attr->prov_name, "cxi") == 0) {
                if (cur->domain_attr && cur->domain_attr->name &&
                    strcmp(cur->domain_attr->name, selected_cxi_domain.c_str()) == 0) {
                    return cur;
                }
            }
        }

        // Fallback to first CXI if exact match not found
        for (struct fi_info* cur = info_list; cur; cur = cur->next) {
            if (cur->fabric_attr && cur->fabric_attr->prov_name &&
                strcmp(cur->fabric_attr->prov_name, "cxi") == 0) {
                return cur;
            }
        }
        return nullptr;
    }

private:
    hwloc_topology_t topo_;
    DeviceAffinity gpu_affinity_;
    std::map<std::string, DeviceAffinity> nic_affinities_;

    // Get device affinity using hwloc
    bool get_device_affinity(hwloc_obj_t osdev, DeviceAffinity& affinity) {
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
        hwloc_obj_t ancestor = hwloc_get_non_io_ancestor_obj(topo_, pci_dev);
        if (!ancestor) return false;

        hwloc_bitmap_t cpuset = hwloc_bitmap_alloc();
        hwloc_bitmap_copy(cpuset, ancestor->cpuset);

        int first_cpu = hwloc_bitmap_first(cpuset);
        hwloc_bitmap_free(cpuset);

        if (first_cpu == -1) return false;

        // Find affinity object (Group/Package/NUMANODE)
        hwloc_obj_t pu = hwloc_get_pu_obj_by_os_index(topo_, first_cpu);
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

    void detect_affinity() {
        // Get all available GPUs and their affinities
        int gpu_count = 0;
        hipError_t hip_err = hipGetDeviceCount(&gpu_count);
        if (hip_err != hipSuccess || gpu_count == 0) {
            fprintf(stderr, "Warning: No HIP devices found\n");
            return;
        }

        // Build map of GPU device ID -> affinity
        std::map<int, DeviceAffinity> gpu_affinities;
        std::map<int, std::string> gpu_pci_ids;

        for (int gpu_id = 0; gpu_id < gpu_count; gpu_id++) {
            char pci_bus_id[32] = {0};
            hip_err = hipDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), gpu_id);
            if (hip_err != hipSuccess) {
                continue;
            }
            gpu_pci_ids[gpu_id] = pci_bus_id;
        }

        // Find GPU affinities in hwloc
        hwloc_obj_t osdev = nullptr;
        while ((osdev = hwloc_get_next_osdev(topo_, osdev)) != nullptr) {
            if (osdev->attr->osdev.type == HWLOC_OBJ_OSDEV_GPU ||
                osdev->attr->osdev.type == HWLOC_OBJ_OSDEV_COPROC) {

                DeviceAffinity temp_affinity;
                if (get_device_affinity(osdev, temp_affinity)) {
                    // Match with HIP GPU by PCI ID
                    for (const auto& gpu_pair : gpu_pci_ids) {
                        if (strcasecmp(temp_affinity.pci_id.c_str(),
                                       gpu_pair.second.c_str()) == 0) {
                            gpu_affinities[gpu_pair.first] = temp_affinity;
                            break;
                        }
                    }
                }
            }
        }

        // Build map of NIC names -> affinities
        hwloc_obj_t osdev_nic = nullptr;
        while ((osdev_nic = hwloc_get_next_osdev(topo_, osdev_nic)) != nullptr) {
            if (osdev_nic->attr->osdev.type == HWLOC_OBJ_OSDEV_NETWORK ||
                osdev_nic->attr->osdev.type == HWLOC_OBJ_OSDEV_OPENFABRICS) {
                DeviceAffinity nic_affinity;
                if (get_device_affinity(osdev_nic, nic_affinity)) {
                    std::string nic_name = osdev_nic->name ? osdev_nic->name : "";
                    if (!nic_name.empty()) {
                        nic_affinities_[nic_name] = nic_affinity;
                    }
                }
            }
        }

        // Find best GPU-NIC pair with matching affinity
        for (const auto& gpu_pair : gpu_affinities) {
            int gpu_id = gpu_pair.first;
            const DeviceAffinity& gpu_aff = gpu_pair.second;

            // Look for a matching NIC (CXI devices appear as "hsi" in hwloc)
            for (int cxi_id = 0; cxi_id < 8; cxi_id++) {  // Check cxi0-cxi7
                char hsi_name[32];
                snprintf(hsi_name, sizeof(hsi_name), "hsi%d", cxi_id);

                auto it = nic_affinities_.find(hsi_name);
                if (it != nic_affinities_.end()) {
                    const DeviceAffinity& nic_aff = it->second;

                    if (nic_aff.affinity_type == gpu_aff.affinity_type &&
                        nic_aff.affinity_index == gpu_aff.affinity_index) {
                        // Found a matching pair!
                        selected_gpu_id = gpu_id;
                        char cxi_domain[32];
                        snprintf(cxi_domain, sizeof(cxi_domain), "cxi%d", cxi_id);
                        selected_cxi_domain = cxi_domain;
                        affinity_matched = true;
                        gpu_affinity_ = gpu_aff;

                        // printf("DeviceAffinity: Selected GPU %d (PCI %s) with CXI %s "
                        //        "(affinity: %s L#%d)\n",
                        //        selected_gpu_id, gpu_aff.pci_id.c_str(),
                        //        selected_cxi_domain.c_str(),
                        //        hwloc_obj_type_string(gpu_aff.affinity_type),
                        //        gpu_aff.affinity_index);
                        return;
                    }
                }
            }
        }

        // No affinity match found - use GPU 0 and let libfabric choose CXI
        selected_gpu_id = 0;
        affinity_matched = false;
        fprintf(stderr, "Warning: No GPU-NIC affinity match found, using GPU 0\n");
    }

    // Detect affinity for a specific GPU
    void detect_affinity_for_gpu(int target_gpu_id) {
        // Get GPU PCI ID
        char pci_bus_id[32] = {0};
        hipError_t hip_err = hipDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), target_gpu_id);
        if (hip_err != hipSuccess) {
            fprintf(stderr, "Warning: Cannot get PCI ID for GPU %d\n", target_gpu_id);
            return;
        }

        // Find GPU affinity in hwloc
        DeviceAffinity gpu_aff;
        bool found_gpu = false;

        hwloc_obj_t osdev = nullptr;
        while ((osdev = hwloc_get_next_osdev(topo_, osdev)) != nullptr) {
            if (osdev->attr->osdev.type == HWLOC_OBJ_OSDEV_GPU ||
                osdev->attr->osdev.type == HWLOC_OBJ_OSDEV_COPROC) {

                DeviceAffinity temp_affinity;
                if (get_device_affinity(osdev, temp_affinity)) {
                    if (strcasecmp(temp_affinity.pci_id.c_str(), pci_bus_id) == 0) {
                        gpu_aff = temp_affinity;
                        found_gpu = true;
                        break;
                    }
                }
            }
        }

        if (!found_gpu) {
            fprintf(stderr, "Warning: Cannot find hwloc affinity for GPU %d\n", target_gpu_id);
            return;
        }

        // Build NIC affinities map
        hwloc_obj_t osdev_nic = nullptr;
        while ((osdev_nic = hwloc_get_next_osdev(topo_, osdev_nic)) != nullptr) {
            if (osdev_nic->attr->osdev.type == HWLOC_OBJ_OSDEV_NETWORK ||
                osdev_nic->attr->osdev.type == HWLOC_OBJ_OSDEV_OPENFABRICS) {
                DeviceAffinity nic_affinity;
                if (get_device_affinity(osdev_nic, nic_affinity)) {
                    std::string nic_name = osdev_nic->name ? osdev_nic->name : "";
                    if (!nic_name.empty()) {
                        nic_affinities_[nic_name] = nic_affinity;
                    }
                }
            }
        }

        // Find matching CXI for this GPU
        for (int cxi_id = 0; cxi_id < 8; cxi_id++) {
            char hsi_name[32];
            snprintf(hsi_name, sizeof(hsi_name), "hsi%d", cxi_id);

            auto it = nic_affinities_.find(hsi_name);
            if (it != nic_affinities_.end()) {
                const DeviceAffinity& nic_aff = it->second;

                if (nic_aff.affinity_type == gpu_aff.affinity_type &&
                    nic_aff.affinity_index == gpu_aff.affinity_index) {
                    char cxi_domain[32];
                    snprintf(cxi_domain, sizeof(cxi_domain), "cxi%d", cxi_id);
                    selected_cxi_domain = cxi_domain;
                    affinity_matched = true;
                    gpu_affinity_ = gpu_aff;

                    // printf("DeviceAffinity: GPU %d (PCI %s) -> CXI %s "
                    //        "(affinity: %s L#%d)\n",
                    //        target_gpu_id, gpu_aff.pci_id.c_str(),
                    //        selected_cxi_domain.c_str(),
                    //        hwloc_obj_type_string(gpu_aff.affinity_type),
                    //        gpu_aff.affinity_index);
                    return;
                }
            }
        }

        fprintf(stderr, "Warning: No matching CXI for GPU %d, using default\n", target_gpu_id);
    }
};
