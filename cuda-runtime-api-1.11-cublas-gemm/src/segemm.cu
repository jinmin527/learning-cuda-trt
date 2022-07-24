
#include <cuda_runtime.h>

__global__ void gemm_kernel_0(const float* A, const float* B, float* C, int m, int n, int k)
{
    // A -> m x n
    // B -> n x k
    // C -> m x k
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= m || col >= k)
        return;

    float csum = 0;
    for (int i = 0; i < n; ++i){
        csum += A[row * n + i] * B[i * k + col];
    }
    C[row * k + col] = csum;
}

void gemm_0(const float* A, const float* B, float* C, int m, int n, int k, cudaStream_t stream){

    dim3 block(16, 16, 1);
    dim3 grid((k + block.x - 1) / block.x, (m + block.y - 1) / block.y, 1);
    gemm_kernel_0<<<grid, block, 0, stream>>>(A, B, C, m, n, k);
}


// 核心思想是矩阵分块计算后累加，在高数里面利用矩阵的分块原理
// 并且利用到了共享内存技术，降低访问耗时
// Compute C = A * B
template<int BLOCK_SIZE>
__global__ void gemm_kernel_1(const float *A, const float *B, float *C, int m, int n, int k) {
    
    __shared__ float sharedM[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedN[BLOCK_SIZE][BLOCK_SIZE];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by*BLOCK_SIZE + ty;
    int col = bx*BLOCK_SIZE + tx;
    float v = 0.0;

    for (int i = 0; i < (int)(ceil((float)n/BLOCK_SIZE)); i++)
    {
        if (i*BLOCK_SIZE + tx < n && row < m)
            sharedM[ty][tx] = A[row*n + i*BLOCK_SIZE + tx];
        else
            sharedM[ty][tx] = 0.0;

        if (i*BLOCK_SIZE + ty < n && col < k)
            sharedN[ty][tx] = B[(i*BLOCK_SIZE + ty)*k + col];
        else
            sharedN[ty][tx] = 0.0;
        __syncthreads();

        for(int j = 0; j < BLOCK_SIZE; j++)
            v += sharedM[ty][j] * sharedN[j][tx];
        __syncthreads();
    }

    if (row < m && col < k)
        C[row*k + col] = v;
}

void gemm_1(const float* A, const float* B, float* C, int m, int n, int k, cudaStream_t stream){

    dim3 block(16, 16, 1);
    dim3 grid((k + block.x - 1) / block.x, (m + block.y - 1) / block.y, 1);
    gemm_kernel_1<16><<<grid, block, 0, stream>>>(A, B, C, m, n, k);
}