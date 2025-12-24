#ifdef USE_OPENCL
#include <CL/cl.h>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include "opencl_map.hpp"

namespace block_palette {

static std::once_flag cl_init_flag;
static cl_context cl_ctx = nullptr;
static cl_command_queue cl_q = nullptr;
static cl_program cl_prog = nullptr;
static cl_kernel cl_kernel_closest = nullptr;
static cl_device_id cl_dev = nullptr;

static const char *kernel_src = R"CLC(
__kernel void closest(__global const int* R,__global const int* G,__global const int* B,__global const int* colors,__global int* out,const int nblocks,const int n){
    int i = get_global_id(0);
    if(i>=n) return;
    int r=R[i], g=G[i], b=B[i]; int best=0; int bestD=0x7fffffff;
    for(int j=0;j<nblocks;j++){
        int dr=r-colors[j*3+0]; int dg=g-colors[j*3+1]; int db=b-colors[j*3+2]; int d=dr*dr+dg*dg+db*db; if(d<bestD){bestD=d;best=j;} }
    out[i]=best;
}
)CLC";

static bool build_program()
{
    cl_int err = CL_SUCCESS;
    cl_uint num_platforms = 0;
    if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS || num_platforms == 0) return false;
    std::vector<cl_platform_id> plats(num_platforms);
    clGetPlatformIDs(num_platforms, plats.data(), nullptr);
    for (cl_uint pi = 0; pi < num_platforms; ++pi) {
        cl_platform_id pid = plats[pi];
        cl_uint nd = 0;
        if (clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL, 0, nullptr, &nd) != CL_SUCCESS || nd == 0) continue;
        std::vector<cl_device_id> dids(nd);
        clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL, nd, dids.data(), nullptr);
        // pick first device
        cl_dev = dids[0];
        cl_ctx = clCreateContext(nullptr, 1, &cl_dev, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) { cl_ctx = nullptr; continue; }
        cl_q = clCreateCommandQueueWithProperties(cl_ctx, cl_dev, 0, &err);
        if (err != CL_SUCCESS) { clReleaseContext(cl_ctx); cl_ctx = nullptr; continue; }
        const char* src = kernel_src;
        cl_prog = clCreateProgramWithSource(cl_ctx, 1, &src, nullptr, &err);
        if (err != CL_SUCCESS) { clReleaseCommandQueue(cl_q); clReleaseContext(cl_ctx); cl_q = nullptr; cl_ctx = nullptr; continue; }
        err = clBuildProgram(cl_prog, 1, &cl_dev, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t len = 0; clGetProgramBuildInfo(cl_prog, cl_dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
            std::vector<char> log(len + 1);
            clGetProgramBuildInfo(cl_prog, cl_dev, CL_PROGRAM_BUILD_LOG, len, log.data(), nullptr);
            // build failed; cleanup and try next device
            clReleaseProgram(cl_prog); cl_prog = nullptr; clReleaseCommandQueue(cl_q); clReleaseContext(cl_ctx); cl_q = nullptr; cl_ctx = nullptr; continue;
        }
        cl_kernel_closest = clCreateKernel(cl_prog, "closest", &err);
        if (err != CL_SUCCESS) { clReleaseProgram(cl_prog); cl_prog = nullptr; clReleaseCommandQueue(cl_q); clReleaseContext(cl_ctx); cl_q = nullptr; cl_ctx = nullptr; continue; }
        return true;
    }
    return false;
}

bool opencl_map(const std::vector<Point3>& pts, std::vector<int>& out_indices)
{
    if (pts.empty()) { out_indices.clear(); return true; }
    std::call_once(cl_init_flag, []() { build_program(); });
    if (!cl_ctx || !cl_q || !cl_prog || !cl_kernel_closest) return false;

    size_t n = pts.size();
    std::vector<int> R(n), G(n), B(n);
    for (size_t i = 0; i < n; ++i) {
        R[i] = static_cast<int>(std::fmod(std::abs(pts[i][0]), 256.0));
        G[i] = static_cast<int>(std::fmod(std::abs(pts[i][1]), 256.0));
        B[i] = static_cast<int>(std::fmod(std::abs(pts[i][2]), 256.0));
    }
    cl_int err = CL_SUCCESS;
    cl_mem mR = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * n, R.data(), &err); if (err != CL_SUCCESS) return false;
    cl_mem mG = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * n, G.data(), &err); if (err != CL_SUCCESS) { clReleaseMemObject(mR); return false; }
    cl_mem mB = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * n, B.data(), &err); if (err != CL_SUCCESS) { clReleaseMemObject(mR); clReleaseMemObject(mG); return false; }
    std::vector<int> colors; colors.reserve(N_BLOCKS * 3);
    for (int i = 0; i < N_BLOCKS; ++i) { colors.push_back(COLOR_MAP[i].c.r); colors.push_back(COLOR_MAP[i].c.g); colors.push_back(COLOR_MAP[i].c.b); }
    cl_mem mC = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * colors.size(), colors.data(), &err); if (err != CL_SUCCESS) { clReleaseMemObject(mR); clReleaseMemObject(mG); clReleaseMemObject(mB); return false; }
    cl_mem mOut = clCreateBuffer(cl_ctx, CL_MEM_WRITE_ONLY, sizeof(int) * n, nullptr, &err); if (err != CL_SUCCESS) { clReleaseMemObject(mR); clReleaseMemObject(mG); clReleaseMemObject(mB); clReleaseMemObject(mC); return false; }

    err  = clSetKernelArg(cl_kernel_closest, 0, sizeof(cl_mem), &mR);
    err |= clSetKernelArg(cl_kernel_closest, 1, sizeof(cl_mem), &mG);
    err |= clSetKernelArg(cl_kernel_closest, 2, sizeof(cl_mem), &mB);
    err |= clSetKernelArg(cl_kernel_closest, 3, sizeof(cl_mem), &mC);
    err |= clSetKernelArg(cl_kernel_closest, 4, sizeof(cl_mem), &mOut);
    int nb = N_BLOCKS; err |= clSetKernelArg(cl_kernel_closest, 5, sizeof(int), &nb);
    int nn = (int)n; err |= clSetKernelArg(cl_kernel_closest, 6, sizeof(int), &nn);
    if (err != CL_SUCCESS) { clReleaseMemObject(mR); clReleaseMemObject(mG); clReleaseMemObject(mB); clReleaseMemObject(mC); clReleaseMemObject(mOut); return false; }

    size_t gws = n;
    err = clEnqueueNDRangeKernel(cl_q, cl_kernel_closest, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { clReleaseMemObject(mR); clReleaseMemObject(mG); clReleaseMemObject(mB); clReleaseMemObject(mC); clReleaseMemObject(mOut); return false; }
    clFinish(cl_q);
    out_indices.resize(n);
    err = clEnqueueReadBuffer(cl_q, mOut, CL_TRUE, 0, sizeof(int) * n, out_indices.data(), 0, nullptr, nullptr);
    clReleaseMemObject(mR); clReleaseMemObject(mG); clReleaseMemObject(mB); clReleaseMemObject(mC); clReleaseMemObject(mOut);
    return err == CL_SUCCESS;
}

} // namespace block_palette

#endif // USE_OPENCL
