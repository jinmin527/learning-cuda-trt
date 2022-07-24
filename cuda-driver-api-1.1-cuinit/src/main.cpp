
// CUDA驱动头文件cuda.h
#include <cuda.h>

#include <stdio.h> // 因为要使用printf
#include <string.h>
int main(){

    /* 
    cuInit(int flags), 这里的flags目前必须给0;
        对于cuda的所有函数，必须先调用cuInit，否则其他API都会返回CUDA_ERROR_NOT_INITIALIZED
        https://docs.nvidia.com/cuda/archive/11.2.0/cuda-driver-api/group__CUDA__INITIALIZE.html
     */
    CUresult code=cuInit(0);  //CUresult 类型：用于接收一些可能的错误代码
    if(code != CUresult::CUDA_SUCCESS){
        const char* err_message = nullptr;
        cuGetErrorString(code, &err_message);    // 获取错误代码的字符串描述
        // cuGetErrorName (code, &err_message);  // 也可以直接获取错误代码的字符串
        printf("Initialize failed. code = %d, message = %s\n", code, err_message);
        return -1;
    }
    
    /* 
    测试获取当前cuda驱动的版本
    显卡、CUDA、CUDA Toolkit

        1. 显卡驱动版本，比如：Driver Version: 460.84
        2. CUDA驱动版本：比如：CUDA Version: 11.2
        3. CUDA Toolkit版本：比如自行下载时选择的10.2、11.2等；这与前两个不是一回事, CUDA Toolkit的每个版本都需要最低版本的CUDA驱动程序
        
        三者版本之间有依赖关系, 可参照https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
        nvidia-smi显示的是显卡驱动版本和此驱动最高支持的CUDA驱动版本
        
     */

    
    int driver_version = 0;
    code = cuDriverGetVersion(&driver_version);  // 获取驱动版本
    printf("CUDA Driver version is %d\n", driver_version); // 若driver_version为11020指的是11.2

    // 测试获取当前设备信息
    char device_name[100]; // char 数组
    CUdevice device = 0;
    code = cuDeviceGetName(device_name, sizeof(device_name), device);  // 获取设备名称、型号如：Tesla V100-SXM2-32GB // 数组名device_name当作指针
    printf("Device %d name is %s\n", device, device_name);
    return 0;
}