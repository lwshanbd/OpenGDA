#include "pmi2.hpp"

PMI2::PMI2() : Bootstrap() {
    int rc = PMI2_Init(&spawned, &size, &rank, &appnum);
    if (rc != PMI2_SUCCESS) {
        fprintf(stderr, "PMI2_Init failed (%d)\n", rc);
        exit(1);
    }

}

PMI2::~PMI2() {
    int rc = PMI2_Finalize();
    if (rc != PMI2_SUCCESS) {
        fprintf(stderr, "PMI2_Finalize failed (%d)\n", rc);
        exit(1);
    }
}