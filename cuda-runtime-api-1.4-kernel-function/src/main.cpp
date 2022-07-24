
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

void test_print(const float* pdata, int ndata);

int main(){
    float* parray_host = nullptr;
    float* parray_device = nullptr;
    int narray = 10;
    int array_bytes = sizeof(float) * narray;

    parray_host = new float[narray];
    checkRuntime(cudaMalloc(&parray_device, array_bytes));

    for(int i = 0; i < narray; ++i)
        parray_host[i] = i;
    
    checkRuntime(cudaMemcpy(parray_device, parray_host, array_bytes, cudaMemcpyHostToDevice));
    test_print(parray_device, narray);
    checkRuntime(cudaDeviceSynchronize());

    checkRuntime(cudaFree(parray_device));
    delete[] parray_host;
    return 0;
}