/**
 * simple_test.cpp - Simple test program for OpenGDA library
 *
 * This program demonstrates basic usage of the OpenGDA library.
 */

#include <gda.h>
#include <iostream>

int main(int argc, char** argv) {
    // unsetenv("ROCR_VISIBLE_DEVICES");
    // setenv("ROCR_VISIBLE_DEVICES", "5", 1);
    std::cout << "OpenGDA Simple Test" << std::endl;
    std::cout << "Library Version: " << gda_get_version() << std::endl;

    // Initialize OpenGDA
    int ret = gda_init();
    if (ret != 0) {
        std::cerr << "Failed to initialize OpenGDA: " << ret << std::endl;
        return 1;
    }

    std::cout << "OpenGDA initialized successfully!" << std::endl;

    // Your application code here...

    // Finalize OpenGDA
    ret = gda_finalize();
    if (ret != 0) {
        std::cerr << "Failed to finalize OpenGDA: " << ret << std::endl;
        return 1;
    }

    std::cout << "OpenGDA finalized successfully!" << std::endl;
    return 0;
}

// Compile (from examples/ directory):
// g++ -I../build/include -L../build/src simple_test.cpp -lopengda -Wl,-rpath,$(realpath ../build/src) -o simple_test
