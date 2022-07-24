
// CUDA驱动头文件cuda.h
#include <cuda.h>

#include <stdio.h>
#include <string.h>

// 使用有参宏定义检查cuda driver是否被正常初始化, 并定位程序出错的文件名、行数和错误信息
// 宏定义中带do...while循环可保证程序的正确性
#define checkDriver(op)    \
    do{                    \
        auto code = (op);  \
        if(code != CUresult::CUDA_SUCCESS){     \
            const char* err_name = nullptr;     \
            const char* err_message = nullptr;  \
            cuGetErrorName(code, &err_name);    \
            cuGetErrorString(code, &err_message);   \
            printf("%s:%d  %s failed. \n  code = %s, message = %s\n", __FILE__, __LINE__, #op, err_name, err_message);   \
            return -1;   \
        }                \
    }while(0)

int main(){

    //检查cuda driver的初始化。虽然不初始化或错误初始化某些API不会报错（不信你试试），但安全起见调用任何API前务必检查cuda driver初始化
    cuInit(2); // 正确的初始化应该给flag = 0
    checkDriver(cuInit(0));

    // 测试获取当前cuda驱动的版本
    int driver_version = 0;
    checkDriver(cuDriverGetVersion(&driver_version));
    printf("Driver version is %d\n", driver_version);

    // 测试获取当前设备信息
    char device_name[100];
    CUdevice device = 0;
    checkDriver(cuDeviceGetName(device_name, sizeof(device_name), device));
    printf("Device %d name is %s\n", device, device_name);
    return 0;
}