#include "scan.h"

__global__ void kernelScan(int *out, const int *in, size_t n)
{
    __shared__ int tile[BLOCK_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int buf = 0;
    if (i > n) return;

    // load shared memory
    if (i < n) tile[threadIdx.x] = in[i];
    else tile[threadIdx.x] = 0;

    __syncthreads();

    // perform reduction
    for (int idx = 1; (1 << (idx - 1)) < BLOCK_SIZE; idx++)
    {
        if (threadIdx.x % (1 << idx) == 0)
        {
            atomicAdd(&tile[threadIdx.x], tile[threadIdx.x - (1 << idx - 1)]);
            out[threadIdx.x] = tile[threadIdx.x];
        }
        __syncthreads();
    }

    if (threadIdx.x == BLOCK_SIZE)
    {
        atomicAdd(&out[BLOCK_SIZE], tile[BLOCK_SIZE]);
    }
}

__global__ void kernelAddSums(int *out, const int *sums, size_t n)
{

}

std::vector<int> scanOnDevice(const std::vector<int> &in, ScanMethod method)
{
    std::vector<int> out(in.size());

    int *d_in = nullptr;
    int *d_out = nullptr;

    cudaMalloc((void **)&d_in, in.size() * sizeof(int));
    cudaMalloc((void **)&d_out, out.size() * sizeof(int));

    cudaMemcpy(d_in, in.data(), in.size() * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = BLOCK_SIZE;
    int numBlocks = (in.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (numBlocks < 1) numBlocks = 1;

    kernelScan<<<numBlocks, blockSize>>>(d_out, d_in, in.size());

    cudaDeviceSynchronize();
    cudaMemcpy(&out, d_out, out.size() * sizeof(out), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

    return out;
}

std::vector<int> scanOnHost(const std::vector<int> &in)
{
    std::vector<int> out(in.size());
    if (in.size() == 0)
    {
        return out;
    }

    out[0] = in[0];
    for (size_t i = 1; i < in.size(); ++i)
    {
        out[i] = out[i - 1] + in[i];
    }

    return out;
}
