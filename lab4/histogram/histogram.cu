#include "histogram.h"

#define PART_SIZE (4)
#define N_LETTERS (26)

// Histogram - basic parallel implementation
__global__ void histogram_1(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;
    
    int binWidth = (N_LETTERS + nBins - 1) / nBins;
    
    long chunkSize = (size + totalThreads - 1) / totalThreads;
    long start = threadId * chunkSize;
    long end = min(start + chunkSize, size);
    
    for (long i = start; i < end; i++)
    {
        unsigned char ch = buffer[i];
        if (ch >= 'a' && ch <= 'z') {
            int alphabetPosition = ch - 'a';
            int binIndex = alphabetPosition / binWidth;
            if (binIndex < nBins) {
                atomicAdd(&histogram[binIndex], 1);
            }
        }
    }

}

// Histogram - interleaved partitioning
__global__ void histogram_2(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;
    
    int binWidth = (N_LETTERS + nBins - 1) / nBins;
    
    for (long i = tid; i < size; i += totalThreads)
    {
        unsigned char ch = buffer[i];
        if (ch >= 'a' && ch <= 'z') {
            int alphabetPosition = ch - 'a';
            int binIndex = alphabetPosition / binWidth;
            if (binIndex < nBins) {
                atomicAdd(&histogram[binIndex], 1);
            }
        }
    }
}

// Histogram - interleaved partitioning + privatisation
__global__ void histogram_3(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
    extern __shared__ unsigned int shared_hist[];
    
    int tid = threadIdx.x;
    int blockId = blockIdx.x;
    int blockDimX = blockDim.x;
    
    int binWidth = (N_LETTERS + nBins - 1) / nBins;
    
    for (int i = tid; i < nBins; i += blockDimX)
    {
        shared_hist[i] = 0;
    }
    __syncthreads();
    
    int totalThreads = blockDimX * gridDim.x;
    long startIdx = blockId * blockDimX + tid;
    
    for (long i = startIdx; i < size; i += totalThreads)
    {
        unsigned char ch = buffer[i];
        if (ch >= 'a' && ch <= 'z') {
            int alphabetPosition = ch - 'a';
            int binIndex = alphabetPosition / binWidth;
            if (binIndex < nBins) {
                atomicAdd(&shared_hist[binIndex], 1);
            }
        }
    }
    __syncthreads();
    
    for (int i = tid; i < nBins; i += blockDimX)
    {
        if (shared_hist[i] > 0)
        {
            atomicAdd(&histogram[i], shared_hist[i]);
        }
    }
}


std::vector<unsigned int> computeHistogramOnDevice(const std::vector<unsigned char> &data, int nBins, HistMethod method)
{
    std::vector<unsigned int> histogram(nBins, 0);

    // allocate input and output data in the device
    unsigned char *d_data = nullptr;
    unsigned int *d_histogram = nullptr;

    size_t data_size = data.size() * sizeof(unsigned char);
    size_t histogram_size = nBins * sizeof(unsigned int);

    cudaMalloc((void **)&d_data, data_size);
    cudaMalloc((void **)&d_histogram, histogram_size);

    cudaMemset(d_histogram, 0, histogram_size);

    // copy data to the device
    cudaMemcpy(d_data, data.data(), data_size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (data.size() + blockSize - 1) / blockSize;

    switch (method)
    {
        case HistMethod::Block:
            {
                numBlocks = (data.size() + blockSize * PART_SIZE - 1) / (blockSize * PART_SIZE);
                numBlocks = min(numBlocks, 60);
                if (numBlocks < 1) numBlocks = 1;
                histogram_1<<<numBlocks, blockSize>>>(d_data, data.size(), d_histogram, nBins);
                break;
            }

        case HistMethod::Interleaved:
            {
                numBlocks = min(numBlocks, 60);
                if (numBlocks < 1) numBlocks = 1;
                histogram_2<<<numBlocks, blockSize>>>(d_data, data.size(), d_histogram, nBins);
                break;
            }

        case HistMethod::Privatised:
            {
                numBlocks = min(numBlocks, 60);
                if (numBlocks < 1) numBlocks = 1;
                size_t sharedMemSize = nBins * sizeof(unsigned int);
                histogram_3<<<numBlocks, blockSize, sharedMemSize>>>(d_data, data.size(), d_histogram, nBins);
                break;
            }

        default:
            {
                cudaFree(d_data);
                cudaFree(d_histogram);
                throw std::runtime_error("Incorrect multiplication method, choose one of the followowing: Block, Interleaved, Privatised.");
        
            }
    }

    cudaDeviceSynchronize();
    cudaMemcpy(histogram.data(), d_histogram, histogram_size, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_histogram);

    return histogram;

}

std::vector<unsigned int> computeHistogramOnHost(const std::vector<unsigned char> &data, int nBins)
{
    std::vector<unsigned int> histogram(nBins, 0);
    int binWidth = (N_LETTERS + nBins - 1) / nBins; // ceiling division

    for (const auto &ch : data)
    {
        int alphabetPosition = ch - 'a';
        if (alphabetPosition >= 0 && alphabetPosition < N_LETTERS)
        {
            histogram[alphabetPosition / binWidth]++;
        }
    }

    return histogram;
}
