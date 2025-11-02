#include "add_mm.h"

__global__ void addMatricesByElements(const float *A, const float *B, float *C, int ncols, int nrows)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= ncols || y >= nrows) return;

    int index = (y * ncols) + x;
    C[index] = A[index] + B[index];
}

__global__ void addMatricesByRows(const float *A, const float *B, float *C, int ncols, int nrows)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= nrows) return;

    for (int x = 0; x < ncols; x++)
    {
        int index = (y* ncols) + x;
        C[index] = A[index] + B[index];
    }
}

__global__ void addMatricesByColumns(const float *A, const float *B, float *C, int ncols, int nrows)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= ncols) return;

    for (int y = 0; y < nrows; y++)
    {
        int index = (y* ncols) + x;
        C[index] = A[index] + B[index];
    }
}

Matrix addMatricesOnDevice(const Matrix &A, const Matrix &B, AddMethod method)
{
    if (A.getCols() != B.getCols() || A.getRows() != B.getRows())
    {
        throw std::runtime_error("Matrix dimensions do not match for addition.");
    }

    Matrix C(A.getRows(), A.getCols());

    // allocate input and output data in the device
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    size_t size = A.getRows() * A.getCols() * sizeof(float);

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // copy data to device
    cudaMemcpy(d_A, A.getDataConstPtr(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.getDataConstPtr(), size, cudaMemcpyHostToDevice);

    // kernel configuration
    dim3 threadsPerBlock(1, 1, 1);
    dim3 blocksPerGrid(1, 1, 1); 
    
    int ncols = A.getCols();
    int nrows = A.getRows();

    switch (method) 
    {
        case AddMethod::ByElements:
            threadsPerBlock.x = min(max(ncols, 1), 32);
            threadsPerBlock.y = min(max(nrows, 1), 32);

            blocksPerGrid.x = (ncols + threadsPerBlock.x - 1) / threadsPerBlock.x;
            blocksPerGrid.y = (nrows + threadsPerBlock.y - 1) / threadsPerBlock.y;
            
            addMatricesByElements<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, ncols, nrows);

            break;
        case AddMethod::ByRows:
            threadsPerBlock.y = min(max(nrows, 1), 256);
            blocksPerGrid.y = (nrows + threadsPerBlock.y - 1) / threadsPerBlock.y;
            
            addMatricesByRows<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, ncols, nrows);

            break;

        case AddMethod::ByColumns:
            threadsPerBlock.x = min(max(ncols, 1), 256);
            blocksPerGrid.x = (ncols + threadsPerBlock.x - 1) / threadsPerBlock.x;
            
            addMatricesByColumns<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, ncols, nrows);

            break;

        default:
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            throw std::runtime_error("Incorrect Method");
    }

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw std::runtime_error(cudaGetErrorString(err));
    }

    cudaMemcpy(C.getDataPtr(), d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return C;

}

Matrix addMatricesOnHost(const Matrix &A, const Matrix &B)
{
    if (A.getRows() != B.getRows() || A.getCols() != B.getCols())
    {
        throw std::invalid_argument("Matrices must have the same dimensions for addition.");
    }

    Matrix C(A.getRows(), A.getCols());
    for (unsigned int i = 0; i < A.getRows(); ++i)
    {
        for (unsigned int j = 0; j < A.getCols(); ++j)
        {
            C.getDataPtr()[i * A.getCols() + j] = A.getDataConstPtr()[i * A.getCols() + j] + B.getDataConstPtr()[i * A.getCols() + j];
        }
    }
    return C;
}
