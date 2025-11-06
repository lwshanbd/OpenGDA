#include "ibv.hpp"

IBV::IBV() {
    // TODO: Initialize InfiniBand Verbs

    // Get device list
    struct ibv_device **dev_list;
    int num_devices;

    dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list) {
        fprintf(stderr, "Failed to get IB devices list\n");
        exit(1);
    }

    if (num_devices == 0) {
        fprintf(stderr, "No IB devices found\n");
        exit(1);
    }

    // Open first device
    context = ibv_open_device(dev_list[0]);
    if (!context) {
        fprintf(stderr, "Failed to open IB device\n");
        exit(1);
    }

    IBV_DEBUG("Opened IB device: %s\n", ibv_get_device_name(dev_list[0]));

    // Allocate Protection Domain
    pd = ibv_alloc_pd(context);
    if (!pd) {
        fprintf(stderr, "Failed to allocate PD\n");
        exit(1);
    }

    // Create Completion Queue
    cq = ibv_create_cq(context, 128, NULL, NULL, 0);
    if (!cq) {
        fprintf(stderr, "Failed to create CQ\n");
        exit(1);
    }

    // Free device list
    ibv_free_device_list(dev_list);

    IBV_DEBUG("IBV initialized successfully\n%s", "");
}

IBV::~IBV() {
    // TODO: Cleanup InfiniBand Verbs
    if (qp) ibv_destroy_qp(qp);
    if (cq) ibv_destroy_cq(cq);
    if (pd) ibv_dealloc_pd(pd);
    if (context) ibv_close_device(context);
}
