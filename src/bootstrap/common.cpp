#include "common.hpp"

#ifdef BOOTSTRAP_PMI2
#include "pmi2.hpp"
#endif

#ifdef BOOTSTRAP_PMIX
#include "pmix.hpp"
#endif

Bootstrap::Bootstrap() {
    // Base class constructor
}

std::unique_ptr<Bootstrap> Bootstrap::create_bootstrap(const std::string &type) {
    if (type == "pmi2") {
        #ifdef BOOTSTRAP_PMI2
        return std::make_unique<PMI2>();
        #else
        return nullptr;
        #endif
    } else if (type == "pmix") {
        #ifdef BOOTSTRAP_PMIX
        return std::make_unique<PMIX>();
        #else
        return nullptr;
        #endif
    }
    return nullptr;
}