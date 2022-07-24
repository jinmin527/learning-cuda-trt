
#include <cuda_runtime.h>
#include <stdio.h>

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

void vector_add(const float* a, const float* b, float* c, int ndata);

int main(){

    const int size = 3;
    float vector_a[size] = {2, 3, 2};
    float vector_b[size] = {5, 3, 3};
    float vector_c[size] = {0};

    float* vector_a_device = nullptr;
    float* vector_b_device = nullptr;
    float* vector_c_device = nullptr;
    checkRuntime(cudaMalloc(&vector_a_device, size * sizeof(float)));
    checkRuntime(cudaMalloc(&vector_b_device, size * sizeof(float)));
    checkRuntime(cudaMalloc(&vector_c_device, size * sizeof(float)));

    checkRuntime(cudaMemcpy(vector_a_device, vector_a, size * sizeof(float), cudaMemcpyHostToDevice));
    checkRuntime(cudaMemcpy(vector_b_device, vector_b, size * sizeof(float), cudaMemcpyHostToDevice));

    vector_add(vector_a_device, vector_b_device, vector_c_device, size);
    checkRuntime(cudaMemcpy(vector_c, vector_c_device, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    for(int i = 0; i < size; ++i){
        printf("vector_c[%d] = %f\n", i, vector_c[i]);
    }
    checkRuntime(cudaFree(vector_a_device));
    checkRuntime(cudaFree(vector_b_device));
    checkRuntime(cudaFree(vector_c_device));
    return 0;
}