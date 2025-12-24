#pragma once
#include <vector>
#include "block_palette.hpp"

namespace block_palette {
#ifdef USE_OPENCL
    // Attempt to map colors using OpenCL; returns true on success
    bool opencl_map(const std::vector<Point3>& pts, std::vector<int>& out_indices);
#else
    inline bool opencl_map(const std::vector<Point3>&, std::vector<int>&) { return false; }
#endif
}
