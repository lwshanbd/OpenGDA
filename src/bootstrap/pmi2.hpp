/*
 * pmi2.hpp
 *
 * PMI2 header file for the bootstrap library.
 */

#ifndef PMI2_HPP
#define PMI2_HPP

#include <pmi2.h>
#include "common.hpp"

#include <cstdlib>
#include <cstdio>
#include <string>

class PMI2 : public Bootstrap {
public:
    PMI2();
    ~PMI2();

    bool bootstrap_initialize() override;
    bool bootstrap_finalize() override;
    std::string get_bootstrap_name() const override;

private:
    int rank;
    int size;
    int device_id;
    int spawned;
    int appnum;

};

#endif