// cuda_kernel.cu
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include "block_palette.hpp"
extern "C" bool cuda_map_colors(const int* R,const int* G,const int* B,int n,int* out);
__device__ __constant__ unsigned char d_palette[block_palette::N_BLOCKS * 3];
__global__ void nearest_kernel(const int* R,const int* G,const int* B,int n,int* out){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=n) return;
    int r=R[i], g=G[i], b=B[i];
    int best=0;
    int bestD=INT_MAX;
    for(int j=0;j<block_palette::N_BLOCKS;++j){
        int pr=d_palette[j*3+0], pg=d_palette[j*3+1], pb=d_palette[j*3+2];
        int dr=r-pr, dg=g-pg, db=b-pb;
        int d = dr*dr + dg*dg + db*db;
        if(d<bestD){ bestD=d; best=j; }
    }
    out[i]=best;
}
extern "C" bool cuda_map_colors(const int* R,const int* G,const int* B,int n,int* out){
    if(!R||!G||!B||!out||n<=0) return false;
    if(cudaMemcpyToSymbol(d_palette, block_palette::get_palette_data(), static_cast<size_t>(block_palette::N_BLOCKS)*3*sizeof(unsigned char), 0, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    int *dR=nullptr,*dG=nullptr,*dB=nullptr,*dOut=nullptr;
    if(cudaMalloc(&dR,sizeof(int)*n)!=cudaSuccess) return false;
    if(cudaMalloc(&dG,sizeof(int)*n)!=cudaSuccess){ cudaFree(dR); return false; }
    if(cudaMalloc(&dB,sizeof(int)*n)!=cudaSuccess){ cudaFree(dR); cudaFree(dG); return false; }
    if(cudaMalloc(&dOut,sizeof(int)*n)!=cudaSuccess){ cudaFree(dR); cudaFree(dG); cudaFree(dB); return false; }
    if(cudaMemcpy(dR,R,sizeof(int)*n,cudaMemcpyHostToDevice)!=cudaSuccess){ cudaFree(dR); cudaFree(dG); cudaFree(dB); cudaFree(dOut); return false; }
    if(cudaMemcpy(dG,G,sizeof(int)*n,cudaMemcpyHostToDevice)!=cudaSuccess){ cudaFree(dR); cudaFree(dG); cudaFree(dB); cudaFree(dOut); return false; }
    if(cudaMemcpy(dB,B,sizeof(int)*n,cudaMemcpyHostToDevice)!=cudaSuccess){ cudaFree(dR); cudaFree(dG); cudaFree(dB); cudaFree(dOut); return false; }
    int block = 256;
    int grid = (n + block - 1) / block;
    nearest_kernel<<<grid,block>>>(dR,dG,dB,n,dOut);
    if(cudaDeviceSynchronize() != cudaSuccess){ cudaFree(dR); cudaFree(dG); cudaFree(dB); cudaFree(dOut); return false; }
    if(cudaMemcpy(out,dOut,sizeof(int)*n,cudaMemcpyDeviceToHost)!=cudaSuccess){ cudaFree(dR); cudaFree(dG); cudaFree(dB); cudaFree(dOut); return false; }
    cudaFree(dR); cudaFree(dG); cudaFree(dB); cudaFree(dOut);
    return true;
}
