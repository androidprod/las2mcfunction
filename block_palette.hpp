// block_palette.hpp
#pragma once
#include <array>
#include <vector>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>

namespace block_palette
{
    using Point3 = std::array<double, 3>;
    // forward-declare OpenCL mapping entry (implemented in opencl_map.cpp when USE_OPENCL=1)
    bool opencl_map(const std::vector<Point3> &pts, std::vector<int> &out_indices);
    struct RGB
    {
        int r, g, b;
    };
    struct Block
    {
        const char *name;
        RGB c;
    };
    static const Block COLOR_MAP[] = {
        {"white_concrete", {207, 213, 214}}, {"orange_concrete", {224, 97, 0}}, {"magenta_concrete", {169, 48, 159}}, {"light_blue_concrete", {36, 137, 199}}, {"yellow_concrete", {241, 175, 21}}, {"lime_concrete", {94, 168, 24}}, {"pink_concrete", {214, 101, 143}}, {"gray_concrete", {54, 57, 61}}, {"light_gray_concrete", {125, 125, 115}}, {"cyan_concrete", {21, 119, 136}}, {"purple_concrete", {100, 32, 156}}, {"blue_concrete", {45, 47, 143}}, {"brown_concrete", {96, 60, 32}}, {"green_concrete", {73, 91, 36}}, {"red_concrete", {142, 33, 33}}, {"black_concrete", {8, 10, 15}}, {"white_wool", {233, 236, 236}}, {"orange_wool", {240, 118, 19}}, {"magenta_wool", {190, 68, 201}}, {"light_blue_wool", {58, 175, 217}}, {"yellow_wool", {249, 199, 35}}, {"lime_wool", {112, 185, 25}}, {"pink_wool", {237, 141, 172}}, {"gray_wool", {62, 68, 71}}, {"light_gray_wool", {142, 142, 134}}, {"cyan_wool", {21, 137, 145}}, {"purple_wool", {122, 42, 172}}, {"blue_wool", {53, 57, 157}}, {"brown_wool", {114, 71, 40}}, {"green_wool", {85, 110, 27}}, {"red_wool", {162, 34, 35}}, {"black_wool", {21, 21, 26}}, {"white_terracotta", {209, 178, 161}}, {"orange_terracotta", {161, 83, 37}}, {"magenta_terracotta", {150, 88, 109}}, {"light_blue_terracotta", {113, 108, 137}}, {"yellow_terracotta", {186, 133, 35}}, {"lime_terracotta", {103, 117, 52}}, {"pink_terracotta", {160, 77, 78}}, {"gray_terracotta", {57, 42, 35}}, {"light_gray_terracotta", {135, 107, 98}}, {"cyan_terracotta", {87, 91, 91}}, {"purple_terracotta", {118, 70, 86}}, {"blue_terracotta", {74, 59, 91}}, {"brown_terracotta", {77, 51, 36}}, {"green_terracotta", {76, 82, 42}}, {"red_terracotta", {143, 61, 47}}, {"black_terracotta", {37, 23, 16}}};
    static const int N_BLOCKS = sizeof(COLOR_MAP) / sizeof(COLOR_MAP[0]);
    inline int color_distance2(const RGB &a, int r, int g, int b)
    {
        int dr = a.r - r;
        int dg = a.g - g;
        int db = a.b - b;
        return dr * dr + dg * dg + db * db;
    }
    inline int nearest_block_cpu(int r, int g, int b)
    {
        int best = 0;
        int best_dist = std::numeric_limits<int>::max();
        for (int i = 0; i < N_BLOCKS; ++i)
        {
            int d = color_distance2(COLOR_MAP[i].c, r, g, b);
            if (d < best_dist)
            {
                best_dist = d;
                best = i;
            }
        }
        return best;
    }
    inline void cpu_map(const std::vector<Point3> &pts, std::vector<int> &out_indices)
    {
        out_indices.resize(pts.size());
        for (size_t i = 0; i < pts.size(); ++i)
        {
            int r = static_cast<int>(std::fmod(std::abs(pts[i][0]), 256.0));
            int g = static_cast<int>(std::fmod(std::abs(pts[i][1]), 256.0));
            int b = static_cast<int>(std::fmod(std::abs(pts[i][2]), 256.0));
            out_indices[i] = nearest_block_cpu(r, g, b);
        }
    }

    // Multithreaded CPU mapper: safe partitioning using std::thread
    inline void cpu_map_mt(const std::vector<Point3> &pts, std::vector<int> &out_indices, unsigned threads = 0)
    {
        size_t n = pts.size();
        out_indices.resize(n);
        if (n == 0) return;
        if (threads == 0) threads = std::max<unsigned>(1, std::thread::hardware_concurrency());
        size_t base = n / threads; size_t rem = n % threads;
        std::vector<std::thread> ths; ths.reserve(threads);
        size_t start = 0;
        for (unsigned t = 0; t < threads; ++t)
        {
            size_t cnt = base + (t < rem ? 1 : 0);
            size_t s = start;
            if (cnt == 0) break;
            ths.emplace_back([s, cnt, &pts, &out_indices]() {
                for (size_t i = 0; i < cnt; ++i)
                {
                    size_t idx = s + i;
                    int r = static_cast<int>(std::fmod(std::abs(pts[idx][0]), 256.0));
                    int g = static_cast<int>(std::fmod(std::abs(pts[idx][1]), 256.0));
                    int b = static_cast<int>(std::fmod(std::abs(pts[idx][2]), 256.0));
                    out_indices[idx] = nearest_block_cpu(r, g, b);
                }
            });
            start += cnt;
        }
        for (auto &th : ths) if (th.joinable()) th.join();
    }
#ifdef ENABLE_CUDA
    // CUDA mapping implementation (defined in cuda_kernel.cu) uses C linkage
    extern "C" bool cuda_map_colors(const int *R, const int *G, const int *B, int n, int *out_block_index);
    inline bool cuda_map(const std::vector<Point3> &pts, std::vector<int> &out_indices)
    {
        if (pts.empty())
        {
            out_indices.clear();
            return true;
        }
        std::vector<int> R(pts.size()), G(pts.size()), B(pts.size());
        for (size_t i = 0; i < pts.size(); ++i)
        {
            R[i] = static_cast<int>(std::fmod(std::abs(pts[i][0]), 256.0));
            G[i] = static_cast<int>(std::fmod(std::abs(pts[i][1]), 256.0));
            B[i] = static_cast<int>(std::fmod(std::abs(pts[i][2]), 256.0));
        }
        out_indices.resize(pts.size());
        return cuda_map_colors(R.data(), G.data(), B.data(), static_cast<int>(pts.size()), out_indices.data());
    }
#endif
    // palette data accessor (C++ linkage, callable as block_palette::get_palette_data())
    inline const unsigned char *get_palette_data()
    {
        static std::vector<unsigned char> palette_data;
        if (palette_data.empty())
        {
            palette_data.reserve(N_BLOCKS * 3);
            for (int i = 0; i < N_BLOCKS; ++i)
            {
                palette_data.push_back(static_cast<unsigned char>(COLOR_MAP[i].c.r));
                palette_data.push_back(static_cast<unsigned char>(COLOR_MAP[i].c.g));
                palette_data.push_back(static_cast<unsigned char>(COLOR_MAP[i].c.b));
            }
        }
        return palette_data.data();
    }
    inline void map_blocks(const std::vector<Point3> &pts, std::vector<int> &out_indices, bool prefer_cuda)
    {
#ifdef ENABLE_CUDA
        if (prefer_cuda)
        {
            if (cuda_map(pts, out_indices))
                return;
        }
#else
    (void)prefer_cuda;
#endif
    // Try OpenCL first if available (and CUDA not used/failed)
#ifdef USE_OPENCL
    if (opencl_map(pts, out_indices)) return;
#endif
    // use multithreaded CPU mapping for better performance
    unsigned hw = std::max<unsigned>(1, std::thread::hardware_concurrency());
    cpu_map_mt(pts, out_indices, hw);
    }
    inline const char *block_name(int index)
    {
        if (index < 0 || index >= N_BLOCKS)
            return "minecraft:stone";
        return COLOR_MAP[index].name;
    }
}
