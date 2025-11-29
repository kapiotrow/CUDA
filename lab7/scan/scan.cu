#include "scan.h"

__global__ void kernelScan(int *out, const int *in, int *blockSums, size_t n)
{
    __shared__ int tile[BLOCK_SIZE];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // załaduj dane lub 0
    tile[threadIdx.x] = (gid < n ? in[gid] : 0);
    __syncthreads();

    // Kogge–Stone
    for (int offset = 1; offset < blockDim.x; offset <<= 1)
    {
        int val = 0;
        if (threadIdx.x >= offset)
            val = tile[threadIdx.x - offset];

        __syncthreads();
        tile[threadIdx.x] += val;
        __syncthreads();
    }

    // zapis wyników
    if (gid < n)
        out[gid] = tile[threadIdx.x];

    // zapis sumy bloku (ostatni wątek)
    if (threadIdx.x == blockDim.x - 1)
        blockSums[blockIdx.x] = tile[threadIdx.x];
}


__global__ void kernelAddSums(int *out, const int *sums, size_t n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    int add = sums[blockIdx.x];

    out[gid] += add;
}


std::vector<int> scanOnDevice(const std::vector<int> &in, ScanMethod method)
{
    size_t n = in.size();
    std::vector<int> out(n);

    if (n == 0) return out;

    int *d_in, *d_out, *d_sums;

    cudaMalloc((void**)&d_in,  n * sizeof(int));
    cudaMalloc((void**)&d_out, n * sizeof(int));

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMalloc((void**)&d_sums, numBlocks * sizeof(int));

    cudaMemcpy(d_in, in.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    // skan bloków
    kernelScan<<<numBlocks, BLOCK_SIZE>>>(d_out, d_in, d_sums, n);
    cudaDeviceSynchronize();

    // skan tablicy sum bloków
    std::vector<int> h_sums(numBlocks);
    cudaMemcpy(h_sums.data(), d_sums, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 1; i < numBlocks; i++)
        h_sums[i] += h_sums[i - 1];

    // przesuniecie
    h_sums.insert(h_sums.begin(), 0);  
    h_sums.pop_back();

    cudaMemcpy(d_sums, h_sums.data(), numBlocks * sizeof(int), cudaMemcpyHostToDevice);

    // dodanie sum blokowych do każdego bloku
    kernelAddSums<<<numBlocks, BLOCK_SIZE>>>(d_out, d_sums, n);
    cudaDeviceSynchronize();

    // kopiowanie wyników
    cudaMemcpy(out.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_sums);

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
