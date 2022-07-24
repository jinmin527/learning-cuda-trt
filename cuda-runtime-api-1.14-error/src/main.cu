
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
using namespace std;

__global__ void func(float* ptr){

    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if(pos == 999){
        ptr[999] = 5;
    }
}

int main(){

    float* ptr = nullptr;

    // 因为核函数是异步的，因此不会立即检查到他是否存在异常
    func<<<100, 10>>>(ptr);
    //func<<<100, 1050>>>(ptr);
    auto code1 = cudaPeekAtLastError();
    cout << cudaGetErrorString(code1) << endl;

    // 对当前设备的核函数进行同步，等待执行完毕，可以发现过程是否存在异常
    auto code2 = cudaDeviceSynchronize();
    cout << cudaGetErrorString(code2) << endl;

    // 异常会一直存在，以至于后续的函数都会失败
    float* new_ptr = nullptr;
    auto code3 = cudaMalloc(&new_ptr, 100);
    cout << cudaGetErrorString(code3) << endl;
    return 0;
}