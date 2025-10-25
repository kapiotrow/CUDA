#include "mult_mv.h"


#define GET_IDX_ROW_MAJOR(width, x, y) (((y) * (width)) + (x))

__global__ void multMatrixVector(float *b, float *A, float *x, unsigned int nrows, unsigned int ncols)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= nrows) return;

    float b_accum = 0.0f;

    for (unsigned int col = 0; col < ncols; ++col)
    {
        b_accum += A[GET_IDX_ROW_MAJOR(ncols, col, row)] * x[col];
    }

    b[row] = b_accum;
}

Matrix multMatrixVectorOnDevice(const Matrix &A, const Matrix &x)
{
}

Matrix multMatrixVectorOnHost(const Matrix &A, const Matrix &x)
{
    if (A.getCols() != x.getRows())
    {
        throw std::runtime_error("Matrix and vector dimensions do not match for multiplication.");
    }

    Matrix b(A.getRows(), 1);
    for (unsigned int i = 0; i < A.getRows(); ++i)
    {
        float sum = 0.0f;
        for (unsigned int j = 0; j < A.getCols(); ++j)
        {
            sum += A.getDataConstPtr()[i * A.getCols() + j] * x.getDataConstPtr()[j];
        }
        b.getDataPtr()[i] = sum;
    }
    return b;
}
