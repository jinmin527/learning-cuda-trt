#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int ndata){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= ndata) return;
    /*    dims                 indexs
        gridDim.z            blockIdx.z
        gridDim.y            blockIdx.y
        gridDim.x            blockIdx.x
        blockDim.z           threadIdx.z
        blockDim.y           threadIdx.y
        blockDim.x           threadIdx.x

        Pseudo code:
        position = 0
        for i in 6:
            position *= dims[i]
            position += indexs[i]
    */
    c[idx] = a[idx] + b[idx];
}

void vector_add(const float* a, const float* b, float* c, int ndata){

    const int nthreads = 512;
    int block_size = ndata < nthreads ? ndata : nthreads;  // 如果ndata < nthreads 那block_size = ndata就够了
    int grid_size = (ndata + block_size - 1) / block_size; // 其含义是我需要多少个blocks可以处理完所有的任务
    printf("block_size = %d, grid_size = %d\n", block_size, grid_size);
    vector_add_kernel<<<grid_size, block_size, 0, nullptr>>>(a, b, c, ndata);

    // 在核函数执行结束后，通过cudaPeekAtLastError获取得到的代码，来知道是否出现错误
    // cudaPeekAtLastError和cudaGetLastError都可以获取得到错误代码
    // cudaGetLastError是获取错误代码并清除掉，也就是再一次执行cudaGetLastError获取的会是success
    // 而cudaPeekAtLastError是获取当前错误，但是再一次执行cudaPeekAtLastError或者cudaGetLastErro拿到的还是那个错
    cudaError_t code = cudaPeekAtLastError();
    if(code != cudaSuccess){    
        const char* err_name    = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        printf("kernel error %s:%d  test_print_kernel failed. \n  code = %s, message = %s\n", __FILE__, __LINE__, err_name, err_message);   
    }
}