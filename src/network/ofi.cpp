#include "ofi.hpp"
#include <fstream>
#include <climits>
#include <cmath>
#include <hwloc.h>


// Helper function to get NUMA node of a GPU device using hwloc
#ifdef USE_AMDGPU
static int get_gpu_numa_node(int device_id) {
    hwloc_topology_t topology;
    hwloc_topology_init(&topology);

    // Load I/O devices
    hwloc_topology_set_io_types_filter(topology, HWLOC_TYPE_FILTER_KEEP_ALL);
    hwloc_topology_load(topology);

    // Find GPU by device_id using ROCm/HSA interface
    // GPUs are typically exposed as OS devices with name like "renderD128", "renderD129", etc.
    int gpu_count = 0;
    int numa_node = -1;

    hwloc_obj_t obj = NULL;
    while ((obj = hwloc_get_next_osdev(topology, obj)) != NULL) {
        if (obj->attr->osdev.type == HWLOC_OBJ_OSDEV_GPU) {
            if (gpu_count == device_id) {
                // Get the NUMA node of this GPU
                hwloc_obj_t ancestor = hwloc_get_non_io_ancestor_obj(topology, obj);
                if (ancestor) {
                    // Find the NUMA node ancestor
                    hwloc_obj_t numa_obj = hwloc_get_ancestor_obj_by_type(topology, HWLOC_OBJ_NUMANODE, ancestor);
                    if (numa_obj) {
                        numa_node = numa_obj->os_index;
                    }
                }
                break;
            }
            gpu_count++;
        }
    }

    hwloc_topology_destroy(topology);
    return numa_node;
}
#endif
// Helper function to get NUMA node of a CXI device from its name using hwloc
// CXI device names are typically like "cxi0", "cxi1", etc.
static int get_cxi_numa_node(const char* cxi_name) {
    if (!cxi_name) return -1;

    hwloc_topology_t topology;
    hwloc_topology_init(&topology);

    // Load I/O devices
    hwloc_topology_set_io_types_filter(topology, HWLOC_TYPE_FILTER_KEEP_ALL);
    hwloc_topology_load(topology);

    int numa_node = -1;

    // Method 1: Try to find by PCI device name in hwloc
    hwloc_obj_t obj = NULL;
    while ((obj = hwloc_get_next_osdev(topology, obj)) != NULL) {
        if (obj->name && strcmp(obj->name, cxi_name) == 0) {
            // Found the CXI device, get its NUMA node
            hwloc_obj_t ancestor = hwloc_get_non_io_ancestor_obj(topology, obj);
            if (ancestor) {
                hwloc_obj_t numa_obj = hwloc_get_ancestor_obj_by_type(topology, HWLOC_OBJ_NUMANODE, ancestor);
                if (numa_obj) {
                    numa_node = numa_obj->os_index;
                }
            }
            break;
        }
    }

    // Method 2: If not found by name, try PCI devices (CXI devices are on PCIe)
    if (numa_node == -1) {
        obj = NULL;
        while ((obj = hwloc_get_next_pcidev(topology, obj)) != NULL) {
            // Check if this PCI device name matches (some systems expose CXI as PCI device name)
            const char* pci_name = hwloc_obj_get_info_by_name(obj, "Device");
            if (pci_name && strcmp(pci_name, cxi_name) == 0) {
                hwloc_obj_t ancestor = hwloc_get_non_io_ancestor_obj(topology, obj);
                if (ancestor) {
                    hwloc_obj_t numa_obj = hwloc_get_ancestor_obj_by_type(topology, HWLOC_OBJ_NUMANODE, ancestor);
                    if (numa_obj) {
                        numa_node = numa_obj->os_index;
                    }
                }
                break;
            }
        }
    }

    // Method 3: Fallback to sysfs if hwloc doesn't find it
    if (numa_node == -1) {
        int cxi_num = -1;
        if (sscanf(cxi_name, "cxi%d", &cxi_num) == 1) {
            char sysfs_path[256];
            snprintf(sysfs_path, sizeof(sysfs_path),
                     "/sys/class/cxi/cxi%d/device/numa_node", cxi_num);

            std::ifstream numa_file(sysfs_path);
            if (numa_file.is_open()) {
                numa_file >> numa_node;
                numa_file.close();
            }
        }
    }

    hwloc_topology_destroy(topology);
    return numa_node;
}

OFI::OFI() {
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
  
    // Find CXI provider
    cxi_info = NULL;
    
    #ifdef USE_AMDGPU
    // GPU-aware CXI selection: find CXI device on same NUMA node as GPU
    int rank = 0;  // Assuming rank is available in your context
    int device_id = 0;  // Assuming device_id is available in your context

    // Get GPU NUMA node
    int gpu_numa_node = get_gpu_numa_node(device_id);
    OFI_DEBUG("Rank %d: GPU %d is on NUMA node %d\n", rank, device_id, gpu_numa_node);

    // Find all CXI providers and their NUMA nodes
    struct fi_info *best_cxi = NULL;
    int best_numa_distance = INT_MAX;

    for (struct fi_info *cur = info; cur; cur = cur->next) {
        if (cur->fabric_attr && cur->fabric_attr->prov_name &&
            strcmp(cur->fabric_attr->prov_name, "cxi") == 0) {

            // Get CXI device name from domain_attr->name (e.g., "cxi0")
            const char* cxi_name = cur->domain_attr->name;
            int cxi_numa_node = get_cxi_numa_node(cxi_name);

            OFI_DEBUG("Rank %d: Found CXI device %s on NUMA node %d\n",
                     rank, cxi_name, cxi_numa_node);

            // Calculate NUMA distance (simple heuristic)
            int numa_distance;
            if (gpu_numa_node == cxi_numa_node && gpu_numa_node >= 0) {
                // Same NUMA node - best case (local PCIe)
                numa_distance = 0;
            } else if (gpu_numa_node >= 0 && cxi_numa_node >= 0) {
                // Different NUMA nodes - approximate distance
                numa_distance = abs(gpu_numa_node - cxi_numa_node) * 10;
            } else {
                // NUMA info not available
                numa_distance = 100;
            }

            // Select CXI with smallest NUMA distance
            if (numa_distance < best_numa_distance) {
                best_numa_distance = numa_distance;
                best_cxi = cur;
            }
        }
    }

    if (best_cxi) {
        cxi_info = best_cxi;
        OFI_DEBUG("Rank %d: Selected CXI device %s (NUMA distance: %d)\n",
                 rank, cxi_info->domain_attr->name, best_numa_distance);
    } else {
        // Fallback: use first available CXI
        for (struct fi_info *cur = info; cur; cur = cur->next) {
            if (cur->fabric_attr && cur->fabric_attr->prov_name &&
                strcmp(cur->fabric_attr->prov_name, "cxi") == 0) {
                cxi_info = cur;
                OFI_DEBUG("Rank %d: Warning - using fallback CXI device %s\n",
                         rank, cxi_info->domain_attr->name);
                break;
            }
        }
    }
    #else
    for (struct fi_info *cur = info; cur; cur = cur->next) {
      if (cur->fabric_attr && cur->fabric_attr->prov_name &&
          strcmp(cur->fabric_attr->prov_name, "cxi") == 0) {
        cxi_info = cur;
        break;
      }
    }
    #endif
    OFI_DEBUG("Using CXI provider: %s", cxi_info->domain_attr->name);
    OFI_DEBUG("  - mr_mode: 0x%lx", (unsigned long)cxi_info->domain_attr->mr_mode);
    OFI_DEBUG("  - inject_size: %zu bytes", cxi_info->tx_attr->inject_size);
    OFI_DEBUG("  - max_msg_size: %zu bytes", cxi_info->ep_attr->max_msg_size);

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

    

}