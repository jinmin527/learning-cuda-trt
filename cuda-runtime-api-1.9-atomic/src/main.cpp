
#include <cuda_runtime.h>
#include <stdio.h>

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

// 保留元素中，大于特定值的元素，储存其值和索引
// input_array表示为需要判断的数组
// input_size为pdata长度
// threshold为过滤掉阈值，当input_array中的元素大于threshold时会被保留
// output_array为储存的结果数组，格式是：size, [value, index], [value, index]
// output_capacity为output_array中最多可以储存的元素数量
void launch_keep_item(float* input_array, int input_size, float threshold, float* output_array, int output_capacity);

int main(){

    const int n = 100000;
    float* input_host = new float[n];
    float* input_device = nullptr;
    for(int i = 0; i < n; ++i){
        input_host[i] = i % 100; // input_host对应地址上放的数据为 0 1 2 3 ... 99
    }

    checkRuntime(cudaMalloc(&input_device, n * sizeof(float)));
    checkRuntime(cudaMemcpy(input_device, input_host, n * sizeof(float), cudaMemcpyHostToDevice));

    int output_capacity = 20;
    float* output_host = new float[1 + output_capacity * 2]; // 1 + output_capacity * 2 = 41
    float* output_device = nullptr;
    checkRuntime(cudaMalloc(&output_device, (1 + output_capacity * 2) * sizeof(float)));
    checkRuntime(cudaMemset(output_device, 0, sizeof(float)));

    float threshold = 99;
    launch_keep_item(input_device, n, threshold, output_device, output_capacity);
    checkRuntime(cudaPeekAtLastError());

    checkRuntime(cudaMemcpy(output_host, output_device, (1 + output_capacity * 2) * sizeof(float), cudaMemcpyDeviceToHost));
    checkRuntime(cudaDeviceSynchronize());

    printf("output_size = %f\n", output_host[0]);
    int output_size = min(output_host[0], output_capacity);
    for(int i = 0; i < output_size; ++i){
        float* output_item = output_host + 1 + i * 2;
        float value = output_item[0];
        int index   = output_item[1];

        printf("output_host[%d] = %f, %d\n", i, value, index);
    }

    cudaFree(input_device);
    cudaFree(output_device);

    delete [] input_host;
    delete [] output_host;
    printf("done\n");
    return 0;
}