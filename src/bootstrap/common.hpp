/**
 * common.hpp
 *
 * Common header file for the bootstrap library.
 */

#ifndef COMMON_HPP
#define COMMON_HPP

#include <string>
#include <memory>

class Bootstrap {
public:
    Bootstrap();
    virtual ~Bootstrap() = default;

    virtual bool bootstrap_initialize() = 0;
    virtual bool bootstrap_finalize() = 0;
    virtual std::string get_bootstrap_name() const = 0;

    static std::unique_ptr<Bootstrap> create_bootstrap(const std::string& type);


};




#endif