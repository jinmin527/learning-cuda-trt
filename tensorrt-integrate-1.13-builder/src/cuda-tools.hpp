#ifndef CUDA_TOOLS_HPP
#define CUDA_TOOLS_HPP

#include <cuda_runtime.h>
#include <string>

#define checkRuntime(call) CUDATools::check_runtime(call, #call, __LINE__, __FILE__)

#define checkKernel(...)                                                                             \
    __VA_ARGS__;                                                                                     \
    do{cudaError_t cudaStatus = cudaPeekAtLastError();                                               \
    if (cudaStatus != cudaSuccess){                                                                  \
        INFOE("launch failed: %s", cudaGetErrorString(cudaStatus));                                  \
    }} while(0);

namespace CUDATools{
    
    bool check_runtime(cudaError_t e, const char* call, int iLine, const char *szFile);
    bool check_device_id(int device_id);
    int current_device_id();
    std::string device_description();

    // 自动切换当前的deviceid，并在析构的时候切换回去
    class AutoDevice{
    public:
        AutoDevice(int device_id = 0);
        virtual ~AutoDevice();
    
    private:
        int old_ = -1;
    };
}


#endif // CUDA_TOOLS_HPP