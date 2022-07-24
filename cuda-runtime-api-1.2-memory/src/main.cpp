// CUDA运行时头文件
#include <cuda_runtime.h>

#include <stdio.h>
#include <string.h>

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

int main(){

    int device_id = 0;
    checkRuntime(cudaSetDevice(device_id));

    float* memory_device = nullptr;
    checkRuntime(cudaMalloc(&memory_device, 100 * sizeof(float))); // pointer to device

    float* memory_host = new float[100];
    memory_host[2] = 520.25;
    checkRuntime(cudaMemcpy(memory_device, memory_host, sizeof(float) * 100, cudaMemcpyHostToDevice)); // 返回的地址是开辟的device地址，存放在memory_device

    float* memory_page_locked = nullptr;
    checkRuntime(cudaMallocHost(&memory_page_locked, 100 * sizeof(float))); // 返回的地址是被开辟的pin memory的地址，存放在memory_page_locked
    checkRuntime(cudaMemcpy(memory_page_locked, memory_device, sizeof(float) * 100, cudaMemcpyDeviceToHost)); // 

    printf("%f\n", memory_page_locked[2]);
    checkRuntime(cudaFreeHost(memory_page_locked));
    delete [] memory_host;
    checkRuntime(cudaFree(memory_device)); 

    return 0;
}