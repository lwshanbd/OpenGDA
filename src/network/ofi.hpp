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

// OFI Error checking macro
#define OFI_CHECK(x, msg)                                                      \
  do {                                                                         \
    int ret = (x);                                                             \
    if (ret) {                                                                 \
      OPENGDA_Error("ofi", "%s failed: %s (%d)", msg, fi_strerror(-ret), ret); \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

class OFI {
public:
  OFI(int rank);
  ~OFI() = default;

  bool ofi_initialize();
  bool ofi_finalize();

  // Memory registration functions
  struct fid_mr* register_memory(void* buf, size_t size, bool is_device_mem);
  void deregister_memory(struct fid_mr* mr);

  // Accessors
  struct fid_domain* get_domain() { return domain; }
  struct fid_ep* get_endpoint() { return ep; }
  struct fi_info* get_info() { return cxi_info; }

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

};