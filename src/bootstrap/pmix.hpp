/*
 * pmix.hpp
 *
 * PMIx header file for the bootstrap library.
 */

#ifndef PMIX_HPP
#define PMIX_HPP

#include <pmix.h>
#include "common.hpp"

#include <cstdlib>
#include <cstdio>

class PMIX : public Bootstrap {
public:
    PMIX();
    ~PMIX();

private:
    int rank;
    int size;
    int device_id;
    pmix_proc_t myproc;
};

#endif
