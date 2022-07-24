#include <cuda_runtime.h>
#include <stdio.h>

__global__ void launch_keep_item_kernel(float* input_array, int input_size, float threshold, float* output_array, int output_capacity){
    /* 
    该函数做的事情如图：
        figure/2.launch_keep_item_kernel.png
     */


    int input_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(input_array[input_index] < threshold)
        return;

    int output_index = atomicAdd(output_array, 1); // figure/atomicAdd.png 图解

    if(output_index >= output_capacity)            // 如果计的数 超出 容量
        return;

    float* output_item = output_array + 1 + output_index * 2;
    output_item[0] = input_array[input_index];
    output_item[1] = input_index;
}

void launch_keep_item(float* input_array, int input_size, float threshold, float* output_array, int output_capacity){

    const int nthreads = 512;
    int block_size = input_size < nthreads ? input_size : nthreads;
    int grid_size = (input_size + block_size - 1) / block_size;
    launch_keep_item_kernel<<<grid_size, block_size, block_size * sizeof(float), nullptr>>>(input_array, input_size, threshold, output_array, output_capacity);
}