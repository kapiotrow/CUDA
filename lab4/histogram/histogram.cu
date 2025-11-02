#include "histogram.h"

#define PART_SIZE 256

// Histogram - basic parallel implementation
__global__ void histogram_1(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint text_idx = 0u;
    unsigned char letter = 0;

    for (int i = 0; i < PART_SIZE; i++)
    {
        text_idx = idx * PART_SIZE + i;
        if (idx < size)
        {
            letter = buffer[text_idx];
            atomicAdd(&histogram[min(letter - 'a', 26)], 1);
        }
    }

}

// Histogram - interleaved partitioning
__global__ void histogram_2(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
}

// Histogram - interleaved partitioning + privatisation
__global__ void histogram_3(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
}


std::vector<unsigned int> computeHistogramOnDevice(const std::vector<unsigned char> &data, int nBins, HistMethod method)
{
    std::vector<unsigned int> histogram(nBins, 0);
    int binWidth = (N_LETTERS + nBins - 1) / nBins; // ceiling division

    // allocate input and output data in the device
    float *d_data = nullptr;
    float *d_histogram = nullptr;

    size_t data_size = data.size() * sizeof(unsigned char);
    size_t histogram_size = histogram.size() * sizeof(unsigned int);

    cudaMalloc((void **)&d_data, data_size);
    cudaMalloc((void **)&d_histogram, histogram_size);

    // copy data to the device
    cudaMemcpy(d_data, data.getDataConstPtr(), data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_histogram, histogram.getDataConstPtr(), histogram_size, cudaMemcpyHostToDevice);

    // kernel configuration
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid(
        (B.getCols() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (A.getRows() + threadsPerBlock.y - 1) / threadsPerBlock.y
    ); 


    switch (method)
    {
        case HistMethod::Block:
            // TODO
            break;

        case HistMethod::Interleaved:
            // TODO
            break;

        case HistMethod::Privatised:
            // TODO
            break;

        default:
            cudaFree(d_data);
            cudaFree(d_histogram);
            throw std::runtime_error("Incorrect multiplication method, choose one of the followowing: Block, Interleaved, Privatised.");
    }

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
