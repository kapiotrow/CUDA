#include "reduction.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <stdio.h>

namespace cg = cooperative_groups;

__global__ void reductionKernelBasic(int *sum, int *input, int width)
{
    __shared__ int tile[BLOCK_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > width) return;

    // load shared memory
    if (i < width) tile[threadIdx.x] = input[i];
    else tile[threadIdx.x] = 0;

    __syncthreads();

    // perform reductio
    for (int idx = 1; (1 << (idx - 1)) < BLOCK_SIZE; idx++)
    {
        if (threadIdx.x % (1 << idx) == 0)
            atomicAdd(&tile[threadIdx.x], tile[threadIdx.x + (1 << idx - 1)]);
        __syncthreads();
    }

    if (threadIdx.x == 0) atomicAdd(sum, tile[0]);
}

__global__ void reductionKernelOptimized(int *sum, int *input, int width)
{
    __shared__ int tile[BLOCK_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > width) return;

    // load shared memory
    if (i < width) tile[threadIdx.x] = input[i];
    else tile[threadIdx.x] = 0;

    __syncthreads();

    // perform reductio
    for (int idx = 7; idx >= 0; idx--)
    {
        if (threadIdx.x < (1 << i))
            atomicAdd(&tile[threadIdx.x], tile[threadIdx.x + (1 << idx)]);
        __syncthreads();
    }

    if (threadIdx.x == 0) atomicAdd(sum, tile[0]);
}

__device__ int blockReduceCG(cg::thread_group g, int *shared, int value)
{
    // thread id in group
    unsigned int tid = g.thread_rank();
    unsigned int gsize = g.size();

    // load shared memory
    shared[tid] = value;

    g.sync();

    // perform reduction
    for (unsigned int stride = gsize >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared[tid] += shared[tid + stride];
        }

        g.sync();
    }

    return shared[0];
}

__global__ void reductionKernelCooperativeGroups(int *sum, const int *input, int width)
{
    __shared__ int tile[BLOCK_SIZE];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    int inVal = (gid < width) ? input[gid] : 0;

    cg::thread_block tb = cg::this_thread_block();

    int blockSum = blockReduceCG(tb, tile, inVal);

    if (threadIdx.x == 0)
    {
        atomicAdd(sum, blockSum);
    }
}


int reductionOnDevice(const std::vector<int> &data, ReductionMethod method)
{
    int sum = 0;

    int *d_data = nullptr;
    int *d_sum = nullptr;

    cudaMalloc((void **)&d_data, data.size() * sizeof(int));
    cudaMalloc((void **)&d_sum, sizeof(int));

    cudaMemcpy(d_data, data.data(), data.size() * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = BLOCK_SIZE;
    int numBlocks = (data.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (numBlocks < 1) numBlocks = 1;

    switch (method)
    {
    case ReductionMethod::Basic:
    {
        reductionKernelBasic<<<numBlocks, blockSize>>>(d_sum, d_data, data.size());
        break;
    }

    case ReductionMethod::Optimized:
    {
        reductionKernelOptimized<<<numBlocks, blockSize>>>(d_sum, d_data, data.size());
        break;
    }

    case ReductionMethod::CooperativeGroups:
    {
        reductionKernelCooperativeGroups<<<numBlocks, blockSize>>>(d_sum, d_data, data.size());
        break;
    }
    
    default:
    {
        cudaFree(d_data);
        throw std::runtime_error("Incorrect multiplication method, choose one of the followowing: Basic, Optimized, CooperativeGroup.");
        break;
    } 
    }

    cudaDeviceSynchronize();
    cudaMemcpy(&sum, d_sum, sizeof(sum), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_sum);

    return sum;
}

int reductionOnHost(const std::vector<int> &data)
{
    int sum = 0;
    for (const auto &val : data)
    {
        sum += val;
    }
    return sum;
}
