#include <cuda_runtime.h>
#include <stdio.h>

__global__ void demo_kernel(){

    if(blockIdx.x == 0 && threadIdx.x == 0)
        printf("Run kernel. blockIdx = %d,%d,%d  threadIdx = %d,%d,%d\n",
            blockIdx.x, blockIdx.y, blockIdx.z,
            threadIdx.x, threadIdx.y, threadIdx.z
        );
}

void launch(int* grids, int* blocks){

    dim3 grid_dims(grids[0], grids[1], grids[2]);
    dim3 block_dims(blocks[0], blocks[1], blocks[2]);
    demo_kernel<<<grid_dims, block_dims, 0, nullptr>>>();
}