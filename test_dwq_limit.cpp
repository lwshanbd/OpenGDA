#include <iostream>
#include <vector>
#include "gda.h"

int main() {
    if (gda_init() != 0) {
        std::cerr << "Failed to init" << std::endl;
        return 1;
    }

    int mype = gda_rank();
    int npes = gda_size();

    void* gpu_buf = gda_gpu_buf();
    size_t gpu_buf_size = gda_gpu_buf_size();

    std::vector<gda_handle_t*> handles;
    int target = (mype + 1) % npes;

    // Try to create as many handles as possible
    for (int i = 0; i < 1000; i++) {
        gda_handle_t* h = gda_put(gpu_buf, sizeof(uint64_t), target, i * sizeof(uint64_t));
        if (!h) {
            std::cout << "Rank " << mype << ": Failed at handle " << i << std::endl;
            break;
        }
        handles.push_back(h);
        if (i % 10 == 0) {
            std::cout << "Rank " << mype << ": Created handle " << i << std::endl;
        }
    }

    std::cout << "Rank " << mype << ": Total handles created: " << handles.size() << std::endl;

    // Cleanup
    for (auto h : handles) {
        gda_free(h);
    }

    gda_finalize();
    return 0;
}
