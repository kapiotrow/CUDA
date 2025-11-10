#include "conv_1d.h"

__constant__ float c_mask[MAX_MASK_WIDTH];

__global__ void conv1dBasicKernel(float *output, const float *signal, const float *mask, const int width, const int maskWidth)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // output index
    float sum = 0;
    int idx = 0;

    for (int j = 0; j < maskWidth; j++)
    {
        idx = (i - (maskWidth / 2) + j);

        if (idx >= 0 || idx < width)
        {
            sum += signal[idx] * mask[j];
        }
    }

    output[i] = sum;
}

__global__ void conv1dTiledKernel(float *output, const float *signal, const int width, const int maskWidth)
{
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

    int blockSize = 256;
    int numBlocks = (signal.size() + blockSize - 1) / blockSize;

    switch (method)
    {
        case ConvMethod::Basic:
        {
            conv1dBasicKernel<<<numBlocks, blockSize>>>(d_output, d_signal, d_mask, signal.size(), mask.size());
            break;
        }
        
        case ConvMethod::Tiled:
        {
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
