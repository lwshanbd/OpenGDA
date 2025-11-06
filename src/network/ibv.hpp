/*
 * ibv.hpp - InfiniBand Verbs network interface
 */

#ifndef IBV_HPP
#define IBV_HPP

#include <infiniband/verbs.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define IBV_CHECK(x, msg)                                                      \
  do {                                                                         \
    int ret = (x);                                                             \
    if (ret) {                                                                 \
      fprintf(stderr, "%s failed: %d\n", msg, ret);                            \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define IBV_DEBUG(fmt, ...) fprintf(stderr, fmt, __VA_ARGS__)

class IBV {
public:
  IBV();
  ~IBV();

private:
    struct ibv_context *context;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
};

#endif
