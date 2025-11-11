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
    __shared__ float tile[BLOCK_SIZE + 2 * MAX_MASK_SIZE][BLOCK_SIZE + 2 * MAX_MASK_SIZE];

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    int maskRadius = maskSize / 2;

    int sharedX = threadIdx.x + maskRadius;
    int sharedY = threadIdx.y + maskRadius;

    int inputX = x - maskRadius;
    int inputY = y - maskRadius;

    if (inputX >= 0 && inputX < width && inputY >= 0 && inputY < height)
        tile[sharedY][sharedX] = input[inputY * width + inputX];
    else
        tile[sharedY][sharedX] = 0.0f;


    if (threadIdx.x < maskRadius) {
        int leftX = inputX - maskRadius;
        int rightX = inputX + BLOCK_SIZE;

        int leftY = inputY;
        int rightY = inputY;

        // left halo
        if (leftX >= 0 && leftY >= 0 && leftY < height)
            tile[sharedY][sharedX - maskRadius] = input[leftY * width + leftX];
        else
            tile[sharedY][sharedX - maskRadius] = 0.0f;

        // right halo
        if (rightX < width && rightY >= 0 && rightY < height)
            tile[sharedY][sharedX + BLOCK_SIZE] = input[rightY * width + rightX];
        else
            tile[sharedY][sharedX + BLOCK_SIZE] = 0.0f;
    }

    if (threadIdx.y < maskRadius) {
        int topY = inputY - maskRadius;
        int bottomY = inputY + BLOCK_SIZE;

        int topX = inputX;
        int bottomX = inputX;

        // top halo
        if (topY >= 0 && topX >= 0 && topX < width)
            tile[sharedY - maskRadius][sharedX] = input[topY * width + topX];
        else
            tile[sharedY - maskRadius][sharedX] = 0.0f;

        // bottom halo
        if (bottomY < height && bottomX >= 0 && bottomX < width)
            tile[sharedY + BLOCK_SIZE][sharedX] = input[bottomY * width + bottomX];
        else
            tile[sharedY + BLOCK_SIZE][sharedX] = 0.0f;
    }

    if (threadIdx.x < maskRadius && threadIdx.y < maskRadius) {
        // top-left
        int tx = inputX - maskRadius;
        int ty = inputY - maskRadius;
        tile[sharedY - maskRadius][sharedX - maskRadius] =
            (tx >= 0 && ty >= 0) ? input[ty * width + tx] : 0.0f;

        // top-right
        tx = inputX + BLOCK_SIZE;
        ty = inputY - maskRadius;
        tile[sharedY - maskRadius][sharedX + BLOCK_SIZE] =
            (tx < width && ty >= 0) ? input[ty * width + tx] : 0.0f;

        // bottom-left
        tx = inputX - maskRadius;
        ty = inputY + BLOCK_SIZE;
        tile[sharedY + BLOCK_SIZE][sharedX - maskRadius] =
            (tx >= 0 && ty < height) ? input[ty * width + tx] : 0.0f;

        // bottom-right
        tx = inputX + BLOCK_SIZE;
        ty = inputY + BLOCK_SIZE;
        tile[sharedY + BLOCK_SIZE][sharedX + BLOCK_SIZE] =
            (tx < width && ty < height) ? input[ty * width + tx] : 0.0f;
    }

    __syncthreads();


    if (x < width && y < height) {
        float sum = 0.0f;

        for (int j = -maskRadius; j <= maskRadius; j++) {
            for (int i = -maskRadius; i <= maskRadius; i++) {
                sum += tile[sharedY + j][sharedX + i] *
                       c_mask[(j + maskRadius) * maskSize + (i + maskRadius)];
            }
        }

        output[y * width + x] = sum;
    }
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
            tiledConvolution2D<<<grid, block>>>(d_output, d_img, width, height, maskSize);
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
