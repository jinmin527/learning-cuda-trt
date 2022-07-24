# cuda-driver-api 1.1
1. CUDA Driver与CUDA Runtime相比更偏底层，就意味着Driver API有着更灵活的控制，也伴随着更复杂的编程
2. 因此CUDA driver需要做显式的初始化`cuInit(0)`，否则其他API都会返回`CUDA_ERROR_NOT_INITIALIZED`
3. 经过初始化后驱动和显卡的信息可以轻松获取：
    - 驱动版本管理 https://docs.nvidia.com/cuda/archive/11.2.0/cuda-driver-api/group__CUDA__VERSION.html#group__CUDA__VERSION
    - 设备信息管理 https://docs.nvidia.com/cuda/archive/11.2.0/cuda-driver-api/group__CUDA__DEVICE.html


# 写在文末
### CUDA的在线文档地址：
1. https://developer.nvidia.com/cuda-toolkit-archive
2. https://docs.nvidia.com/cuda/archive/11.2.0/

### Startup
1. `make run`

### 如果报错，提示nvcc错误
1. 对于gcc版本大于等于8的不支持，需要修改Makefile中的g++为g++7或者更低