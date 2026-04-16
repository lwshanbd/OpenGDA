// src/gicc/bootstrap/bootstrap.hpp
//
// Unified process-launch / metadata-exchange abstraction for GICC.
// At build time, exactly one of GICC_BOOTSTRAP_MPI or GICC_BOOTSTRAP_PMI2
// is defined (see GICC_BOOTSTRAP in the top-level CMakeLists). That decision
// selects the implementation class aliased here as gicc::Bootstrap.
//
// User code should include only this header. No MPI_* or PMI_* symbols
// ever need to appear outside the detail headers.
#pragma once

#if defined(GICC_BOOTSTRAP_MPI) && defined(GICC_BOOTSTRAP_PMI2)
    #error "Define exactly one of GICC_BOOTSTRAP_MPI / GICC_BOOTSTRAP_PMI2, not both"
#endif

#if defined(GICC_BOOTSTRAP_MPI)
    #include "gicc/bootstrap/detail/bootstrap_mpi.hpp"
    namespace gicc { using Bootstrap = detail::BootstrapMPI; }
#elif defined(GICC_BOOTSTRAP_PMI2)
    #include "gicc/bootstrap/detail/bootstrap_pmi2.hpp"
    namespace gicc { using Bootstrap = detail::BootstrapPMI2; }
#else
    #error "Define GICC_BOOTSTRAP_MPI or GICC_BOOTSTRAP_PMI2 " \
           "(set via -DGICC_BOOTSTRAP=mpi|pmi2 in CMake)"
#endif
