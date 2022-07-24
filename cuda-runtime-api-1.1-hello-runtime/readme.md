# CUDA的在线文档地址：
1. https://developer.nvidia.com/cuda-toolkit-archive
2. https://docs.nvidia.com/cuda/archive/11.2.0/

# Startup
1. `make run`

# 知识点
1. CUDA Runtime是封装了CUDA Driver的高级别更友好的API
2. cudaruntime需要引入cudart这个so文件
3. 上下文管理：
    - 3.1. 使用cuDevicePrimaryCtxRetain为每个设备设置context，不再手工管理context，并且不提供直接管理context的API 
    - 3.2. 任何依赖CUcontext的API被调用时，会触发CUcontext的创建和对设备的绑定
      - 此后任何API调用时，会以设备id为基准，调取绑定好的CUcontext
      - 因此被称为懒加载模式，避免了手动维护CUcontext的麻烦
4. cuda的状态返回值，都是cudaError_t类型，通过check宏捕获状态并处理是一种通用方式
    - 官方案例采用宏，而非这里的函数加宏
    - 函数加宏具有更加好的便利性 