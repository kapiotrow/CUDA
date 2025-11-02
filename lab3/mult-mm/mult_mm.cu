#include "mult_mm.h"

#define TILE_WIDTH  16

__global__ void matrixMulKernel(const float *A, const float *B, float *C,
                                int A_rows, int A_cols, int B_cols)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < A_rows && col < B_cols)
    {
        float Pvalue = 0.0f;

        for (int k = 0; k < A_cols; k++) 
        {
            Pvalue += A[row * A_cols + k] * B[k * B_cols + col];
        }

        C[row * B_cols + col] = Pvalue;
    }

}

__global__ void matrixMulTiledKernel(const float *A, const float *B, float *C,
                                     int A_rows, int A_cols, int B_cols)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];
    
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float Pvalue = 0.0f;

    for (int k = 0; k < (A_cols + TILE_WIDTH - 1) / TILE_WIDTH; k++)
    {
        int a_col = k * TILE_WIDTH + threadIdx.x;
        int b_row = k * TILE_WIDTH + threadIdx.y;

        if (row < A_rows && a_col < A_cols)
            tileA[threadIdx.y][threadIdx.x] = A[row * A_cols + a_col];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        
        if (b_row < A_cols && col < B_cols)
            tileB[threadIdx.y][threadIdx.x] = B[b_row * B_cols + col];
        else 
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++)
        {
            Pvalue += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < A_rows && col < B_cols) C[row * B_cols + col] = Pvalue;
}

__global__ void matrixMulGranularKernel(const float *A, const float *B, float *C,
                                        int A_rows, int A_cols, int B_cols)
{
}

Matrix multMatrixMatrixOnDevice(const Matrix &A, const Matrix &B, MultMethod method)
{
    if (A.getCols() != B.getRows())
    {
        throw std::runtime_error("Incompatible matrix dimensions for multiplication");
    }

    Matrix C(A.getRows(), B.getCols());

    // allocate input and output data in the device
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    size_t A_size = A.getRows() * A.getCols() * sizeof(float);
    size_t B_size = B.getRows() * B.getCols() * sizeof(float);
    size_t C_size = C.getRows() * C.getCols() * sizeof(float);

    cudaMalloc((void **)&d_A, A_size);
    cudaMalloc((void **)&d_B, B_size);
    cudaMalloc((void **)&d_C, C_size);

    // copy data to the device
    cudaMemcpy(d_A, A.getDataConstPtr(), A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.getDataConstPtr(), B_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, B.getDataConstPtr(), C_size, cudaMemcpyHostToDevice);

    // kernel configuration
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid(
        (B.getCols() + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (A.getRows() + threadsPerBlock.y - 1) / threadsPerBlock.y
    ); 

    switch (method)
    {
        case MultMethod::Standard:
            matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, A.getRows(), A.getCols(), B.getCols());
            break;

        case MultMethod::Tiled:
            matrixMulTiledKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, A.getRows(), A.getCols(), B.getCols());
            break;

        case MultMethod::Granular:
            // TODO
            break;

        default:
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            throw std::runtime_error("Incorrect multiplication method, choose one of the followowing: Standard, Tiled, Granular.");
    }

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw std::runtime_error(cudaGetErrorString(err));
    }

    cudaMemcpy(C.getDataPtr(), d_C, C_size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}

Matrix multMatrixMatrixOnHost(const Matrix &A, const Matrix &B)
{
    if (A.getCols() != B.getRows())
    {
        throw std::runtime_error("Incompatible matrix dimensions for multiplication");
    }

    Matrix C(A.getRows(), B.getCols());
    for (unsigned int i = 0; i < A.getRows(); ++i)
    {
        for (unsigned int j = 0; j < B.getCols(); ++j)
        {
            for (unsigned int k = 0; k < A.getCols(); ++k)
            {
                C.getDataPtr()[i * C.getCols() + j] +=
                    A.getDataConstPtr()[i * A.getCols() + k] *
                    B.getDataConstPtr()[k * B.getCols() + j];
            }
        }
    }
    return C;
}
