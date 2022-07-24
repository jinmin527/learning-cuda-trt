
// CUDA运行时头文件
#include <cuda_runtime.h>

#include <chrono>
#include <stdio.h>
#include <string.h>

using namespace std;

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

__global__ void add_vector(const float* a, const float* b, float* c, int count){

    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index >= count) return;
    c[index] = a[index] + b[index];
}

__global__ void mul_vector(const float* a, const float* b, float* c, int count){

    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index >= count) return;
    c[index] = a[index] * b[index];
}

cudaStream_t stream1, stream2;
float *a, *b, *c1, *c2;
const int num_element = 100000;
const size_t bytes = sizeof(float) * num_element;
const int blocks = 512;
const int grids = (num_element + blocks - 1) / blocks;
const int ntry = 1000;

// 多个流异步
void async(){

    cudaEvent_t event_start1, event_stop1;
    cudaEvent_t event_start2, event_stop2;
    checkRuntime(cudaEventCreate(&event_start1));
    checkRuntime(cudaEventCreate(&event_stop1));
    checkRuntime(cudaEventCreate(&event_start2));
    checkRuntime(cudaEventCreate(&event_stop2));

    auto tic = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
    checkRuntime(cudaEventRecord(event_start1, stream1));
    for(int i = 0; i < ntry; ++i)
        add_vector<<<grids, blocks, 0, stream1>>>(a, b, c1, num_element);
    checkRuntime(cudaEventRecord(event_stop1, stream1));
    
    checkRuntime(cudaEventRecord(event_start2, stream2));
    for(int i = 0; i < ntry; ++i)
        add_vector<<<grids, blocks, 0, stream2>>>(a, b, c2, num_element);
    checkRuntime(cudaEventRecord(event_stop2, stream2));

    checkRuntime(cudaStreamSynchronize(stream1));
    checkRuntime(cudaStreamSynchronize(stream2));
    auto toc = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / 1000.0;

    float time1, time2;
    checkRuntime(cudaEventElapsedTime(&time1, event_start1, event_stop1));
    checkRuntime(cudaEventElapsedTime(&time2, event_start2, event_stop2));
    printf("async: time1 = %.2f ms, time2 = %.2f ms, count = %.2f ms\n", time1, time2, toc - tic);
}

// 单个流串行
void sync(){

    cudaEvent_t event_start1, event_stop1;
    checkRuntime(cudaEventCreate(&event_start1));
    checkRuntime(cudaEventCreate(&event_stop1));

    auto tic = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
    checkRuntime(cudaEventRecord(event_start1, stream1));
    for(int i = 0; i < ntry; ++i)
        add_vector<<<grids, blocks, 0, stream1>>>(a, b, c1, num_element);
    
    for(int i = 0; i < ntry; ++i)
        add_vector<<<grids, blocks, 0, stream1>>>(a, b, c2, num_element);
    checkRuntime(cudaEventRecord(event_stop1, stream1));

    checkRuntime(cudaStreamSynchronize(stream1));
    auto toc = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / 1000.0;

    float time1;
    checkRuntime(cudaEventElapsedTime(&time1, event_start1, event_stop1));
    printf("sync: time1 = %.2f ms, count = %.2f ms\n", time1, toc - tic);
}

// 多个流之间并行
void multi_stream_async(){

    // 这个案例主要实现多个流之间互相等待，使用event控制实现
    // 存在step1  ->  step2 \ 
    //                      ->  step3   ->  step4
    //               stepa / 
    //
    // 这个案例中，存在流程1：step1 -> step2的流程
    //           存在流程2：stepa
    //           存在流程3：step3 -> step4，step3要求step2与stepa作为输入
    // 此时，可以让流程1使用stream1，流程2使用stream2，而流程3继续使用stream1，仅仅在stream1中加入等待（event的等待）

    // step1 = add_vector
    // step2 = mul_vector
    // step3 = add_vector
    // step4 = mul_vector
    // stepa = add_vector
    #define step1 add_vector
    #define step2 mul_vector
    #define step3 add_vector
    #define step4 mul_vector
    #define stepa add_vector

    cudaEvent_t event_async;
    checkRuntime(cudaEventCreate(&event_async));

    // stream1的执行流程
    step1<<<grids, blocks, 0, stream1>>>(a, b, c1, num_element);
    step2<<<grids, blocks, 0, stream1>>>(a, b, c1, num_element);

    // 等待event_async有事件
    checkRuntime(cudaStreamWaitEvent(stream1, event_async));
    step3<<<grids, blocks, 0, stream1>>>(a, b, c2, num_element);
    step4<<<grids, blocks, 0, stream1>>>(a, b, c2, num_element);

    // stream2的执行流程
    stepa<<<grids, blocks, 0, stream2>>>(a, b, c2, num_element);
    
    // 为event_async触发事件，通知cudaStreamWaitEvent函数可以继续了
    checkRuntime(cudaEventRecord(event_async, stream2));
    checkRuntime(cudaStreamSynchronize(stream1));

    printf("multi_stream_async done.\n");
}

int main(){

    // 本程序实现两个核函数的并行，通过多个流实现
    
    checkRuntime(cudaStreamCreate(&stream1));
    checkRuntime(cudaStreamCreate(&stream2));

    checkRuntime(cudaMalloc(&a, bytes));
    checkRuntime(cudaMalloc(&b, bytes));
    checkRuntime(cudaMalloc(&c1, bytes));
    checkRuntime(cudaMalloc(&c2, bytes));

    // 演示多流之间的异步执行
    async();

    // 演示单个流内的同步执行
    sync();

    // 演示多个流之间互相等待的操作
    multi_stream_async();
    return 0;
}