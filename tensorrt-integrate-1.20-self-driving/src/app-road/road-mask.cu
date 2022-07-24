

#include <common/cuda_tools.hpp>

namespace Road{

    static __device__ unsigned char cast(float value){
        return value >= 255 ? 255 : (value < 0 ? 0 : (unsigned char)value);
    }

    static __global__ void decode_to_mask_kernel(const float* input, unsigned char* output, int edge){  

        int position = blockDim.x * blockIdx.x + threadIdx.x;
		if (position >= edge) return;

        const float* p = input + position * 4;
        output[position * 3 + 0] = cast(p[0] * 70 + p[1] * 255 + p[2] * 0 + p[3] * 0);
        output[position * 3 + 1] = cast(p[0] * 70 + p[1] * 0 + p[2] * 255 + p[3] * 0);
        output[position * 3 + 2] = cast(p[0] * 70 + p[1] * 0 + p[2] * 0 + p[3] * 255);
    }

    void decode_to_mask_invoker(const float* input, unsigned char* output, int edge, cudaStream_t stream){
        
        auto grid = CUDATools::grid_dims(edge);
        auto block = CUDATools::block_dims(edge);
        checkCudaKernel(decode_to_mask_kernel<<<grid, block, 0, stream>>>(input, output, edge));
    }
};