#include "conv_2d.h"

__constant__ float c_mask[MAX_MASK_SIZE * MAX_MASK_SIZE];
#define BLOCK_SIZE 16u

__global__ void basicConvolution2D(float *output, const float *input, const int width, const int height, const int maskSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // column
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // row

    if (x >= width || y >= height)
        return;

    int maskCenter = maskSize / 2;
    float sum = 0.0f;

    for (int j = 0; j < maskSize; j++) {
        for (int i = 0; i < maskSize; i++) {
            int in_x = x + (i - maskCenter);
            int in_y = y + (j - maskCenter);

            // boundary check
            if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                float pixel = input[in_y * width + in_x];
                float coeff = c_mask[j * maskSize + i];
                sum += pixel * coeff;
            }
        }
    }

    output[y * width + x] = sum;
}

__global__ void tiledConvolution2D(float *output, const float *input, const int width, const int height, const int maskSize)
{
}

Image convolutionOnDevice(const Image &img, const std::vector<float> &mask, ConvMethod method)
{
    // Get image dimensions
    int height = img.getHeight();
    int width = img.getWidth();
    int maskSize = static_cast<int>(std::sqrt(mask.size()));

    // Create output image
    Image output(width, height, img.isGray());

    float *d_img = nullptr;
    float *d_output = nullptr;

    cudaMalloc((void **)&d_img, height * width * sizeof(float));
    cudaMalloc((void **)&d_output, height * width * sizeof(float));

    cudaMemcpy(d_img, img.getDataConstPtr(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_mask, mask.data(), maskSize * maskSize * sizeof(mask[0]));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
          (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    switch (method)
    {
        case ConvMethod::Basic:
        {
            basicConvolution2D<<<grid, block>>>(d_output, d_img, width, height, maskSize);
            break;
        }
        
        case ConvMethod::Tiled:
        {
            // size_t tile_size = blockSize+mask.size();
            // tiledConvolution2D<<<numBlocks, blockSize>>>(d_output, d_img, width, height, maskSize);
            break;
        }

        default:
        {
            cudaFree(d_img);
            throw std::runtime_error("Incorrect multiplication method, choose one of the followowing: Basic, Tiled.");
        }
    }

    cudaDeviceSynchronize();
    cudaMemcpy(output.getDataPtr(), d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_img);
    cudaFree(d_output);

    return output;
}

Image convolutionOnHost(const Image &img, const std::vector<float> &mask)
{
    // Get image dimensions
    int height = img.getHeight();
    int width = img.getWidth();
    int maskSize = static_cast<int>(std::sqrt(mask.size()));
    int r = maskSize / 2;

    // Create output image
    Image output(width, height, img.isGray());

    // Perform convolution
    for (int row = 0; row < height; ++row)
    {
        for (int col = 0; col < width; ++col)
        {
            float pixelValue = 0.0f;
            for (int mRow = -r; mRow <= r; ++mRow)
            {
                for (int mCol = -r; mCol <= r; ++mCol)
                {
                    int imgRow = row + mRow;
                    int imgCol = col + mCol;
                    float imgValue = 0.0f;

                    if (imgRow >= 0 && imgRow < height && imgCol >= 0 && imgCol < width)
                    {
                        imgValue = img.getDataConstPtr()[imgRow * width + imgCol];
                    }

                    float maskValue = mask[(mRow + r) * maskSize + (mCol + r)];
                    pixelValue += imgValue * maskValue;
                }
            }
            output.getDataPtr()[row * width + col] = pixelValue;
        }
    }

    return output;
}
