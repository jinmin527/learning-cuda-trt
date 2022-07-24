# cuda-driver-api 1.2
1. CUDA driver需要做显式的初始化`cuInit(0)`，否则其他API都会返回`CUDA_ERROR_NOT_INITIALIZED`;
2. 采用宏定义可以在每次调用API前都检查初始化;
3. 采用封装带参宏定义使代码更清晰、好调试，养成一种良好的编码习惯也是很重要滴！

# 写在文末
### CUDA的在线文档地址：
1. https://developer.nvidia.com/cuda-toolkit-archive
2. https://docs.nvidia.com/cuda/archive/11.2.0/

### Startup
1. `make run`

### 如果报错，提示nvcc错误
1. 对于gcc版本大于等于8的不支持，需要修改Makefile中的g++为g++7或者更低