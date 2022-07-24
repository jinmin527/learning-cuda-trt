
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define min(a, b)  ((a) < (b) ? (a) : (b))
#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
    if(code != cudaSuccess){    
        const char* err_name = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
        return false;
    }
    return true;
}

void launch_reduce_sum(float* input_array, int input_size, float* output);

int main(){
    const int n = 101;
    float* input_host = new float[n];
    float* input_device = nullptr;
    float ground_truth = 0;
    for(int i = 0; i < n; ++i){
        input_host[i] = i;
        ground_truth += i;
    }

    checkRuntime(cudaMalloc(&input_device, n * sizeof(float)));
    checkRuntime(cudaMemcpy(input_device, input_host, n * sizeof(float), cudaMemcpyHostToDevice));

    float output_host = 0;
    float* output_device = nullptr;
    checkRuntime(cudaMalloc(&output_device, sizeof(float)));
    checkRuntime(cudaMemset(output_device, 0, sizeof(float)));

    launch_reduce_sum(input_device, n, output_device);
    checkRuntime(cudaPeekAtLastError());

    checkRuntime(cudaMemcpy(&output_host, output_device, sizeof(float), cudaMemcpyDeviceToHost));
    checkRuntime(cudaDeviceSynchronize());

    printf("output_host = %f, ground_truth = %f\n", output_host, ground_truth);
    if(fabs(output_host - ground_truth) <= __FLT_EPSILON__){ // fabs 求绝对值
        printf("结果正确.\n");
    }else{
        printf("结果错误.\n");
    }

    cudaFree(input_device);
    cudaFree(output_device);

    delete [] input_host;
    printf("done\n");
    return 0;
}