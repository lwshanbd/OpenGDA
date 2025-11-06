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

#define OFI_CHECK(x, msg)                                                      \
  do {                                                                         \
    int ret = (x);                                                             \
    if (ret) {                                                                 \
      fprintf(stderr, "%s failed: %s (%d)\n", msg, fi_strerror(-ret), ret);    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define OFI_DEBUG(fmt, ...) fprintf(stderr, fmt, __VA_ARGS__);

class OFI {
public:
  OFI();
  ~OFI();

private:
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