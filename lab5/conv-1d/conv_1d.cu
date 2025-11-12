#include "conv_1d.h"

__constant__ float c_mask[MAX_MASK_WIDTH];

#define BLOCK_SIZE 32u

__global__ void conv1dBasicKernel(float *output, const float *signal, const float *mask, const int width, const int maskWidth)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // output index
    float sum = 0;
    int idx = 0;

    for (int j = 0; j < maskWidth; j++)
    {
        idx = (i - (maskWidth / 2) + j);

        if (idx >= 0 && idx < width)
        {
            sum += signal[idx] * mask[j];
        }
    }

    if (i < width) output[i] = sum;
}

__global__ void conv1dTiledKernel(float *output, const float *signal, const int width, const int maskWidth)
{
    __shared__ float tileSignal[BLOCK_SIZE + MAX_MASK_WIDTH - 1];

    int tx = threadIdx.x;
    int i = blockIdx.x * BLOCK_SIZE + tx;   // output index
    int halo = maskWidth / 2;

    // Compute start index of the tile in the signal
    int tileStart = blockIdx.x * BLOCK_SIZE - halo;

    // Load data into shared memory (with zero padding)
    if (tileStart + tx < 0 || tileStart + tx >= width)
        tileSignal[tx] = 0.0f;
    else
        tileSignal[tx] = signal[tileStart + tx];

    // Each block may need to load additional halo elements at the end
    if (tx < maskWidth - 1) {
        int extraIndex = tileStart + BLOCK_SIZE + tx;
        if (extraIndex < 0 || extraIndex >= width)
            tileSignal[BLOCK_SIZE + tx] = 0.0f;
        else
            tileSignal[BLOCK_SIZE + tx] = signal[extraIndex];
    }

    __syncthreads();

    // Perform convolution
    if (i < width) {
        float sum = 0.0f;
        for (int j = 0; j < maskWidth; j++) {
            sum += tileSignal[tx + j] * c_mask[j];
        }
        output[i] = sum;
    }
}

std::vector<float> convolutionOnDevice(const std::vector<float> &signal, const std::vector<float> &mask, ConvMethod method)
{
    int signalWidth = static_cast<int>(signal.size());
    int maskWidth = static_cast<int>(mask.size());
    int outputWidth = signalWidth;

    std::vector<float> output(outputWidth, 0.0f);

    float *d_signal = nullptr;
    float *d_mask = nullptr;
    float *d_output = nullptr;

    cudaMalloc((void **)&d_signal, signal.size() * sizeof(float));
    cudaMalloc((void **)&d_mask, mask.size() * sizeof(float));
    cudaMalloc((void **)&d_output, output.size() * sizeof(float));

    cudaMemcpy(d_signal, signal.data(), signal.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask.data(), mask.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_mask, mask.data(), mask.size() * sizeof(mask[0]));

    int blockSize = BLOCK_SIZE;
    int numBlocks = (signal.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;

    switch (method)
    {
        case ConvMethod::Basic:
        {
            conv1dBasicKernel<<<numBlocks, blockSize>>>(d_output, d_signal, d_mask, signal.size(), mask.size());
            break;
        }
        
        case ConvMethod::Tiled:
        {
            size_t tile_size = blockSize+mask.size();
            conv1dTiledKernel<<<numBlocks, blockSize>>>(d_output, d_signal, signal.size(), mask.size());
            break;
        }

        default:
        {
            cudaFree(d_signal);
            cudaFree(d_mask);
            throw std::runtime_error("Incorrect multiplication method, choose one of the followowing: Basic, Tiled.");
        }
    }

    cudaDeviceSynchronize();
    cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_signal);
    cudaFree(d_mask);
    cudaFree(d_output);

    return output;
}

std::vector<float> convolutionOnHost(const std::vector<float> &signal, const std::vector<float> &mask)
{
    int signalWidth = static_cast<int>(signal.size());
    int maskWidth = static_cast<int>(mask.size());
    int outputWidth = signalWidth;

    std::vector<float> output(outputWidth, 0.0f);

    // Convolution with zero padding
    int n = maskWidth / 2;
    for (int idxP = 0; idxP < outputWidth; ++idxP)
    {
        float convAccum = 0.0f;
        for (int i = idxP - n; i <= idxP + n; ++i)
        {
            if (i >= 0 && i < signalWidth)
            {
                convAccum += signal[i] * mask[i - (idxP - n)];
            }
        }
        output[idxP] = convAccum;
    }

    return output;
}
