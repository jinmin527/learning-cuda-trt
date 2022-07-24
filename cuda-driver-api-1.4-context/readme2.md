# cuda-driver-api 1.4
1. context 上下文：设备与特定进程相关连的所有状态。比如，你写的一段kernel code对GPU的使用会造成不同状态（内存映射、分配、加载的code），Context则保存着所有的管理数据来控制和使用设备。
2. 上下文管理可以干的事儿：
   1. 持有分配的内存列表
   2. 持有加载进该设备的kernel code
   3. cup与gpu之间的unified memory
   4. ...
3. 如何管理上下文：
   1. 在cuda driver同样需要显示管理上下文
        - 开始时`cuCtxCreate()`创建上下文，结束时`cuCtxDestroy`销毁上下文。像文件管理一样须手动开关。
        - 用`cuDevicePrimaryCtxRetain()`创建上下文更好！
        - `cuCtxGetCurrent()`获取当前上下文
        - 可以使用堆栈管理多个上下文`cuCtxPushCurrent()`压入，`cuCtxPopCurrent()`推出
   2. cuda runtime可以自动创建，是基于`cuDevicePrimaryCtxRetain()`创建的。
  

# 写在文末
### CUDA的在线文档地址：
1. https://developer.nvidia.com/cuda-toolkit-archive
2. https://docs.nvidia.com/cuda/archive/11.2.0/

### Startup
1. `make run`

### 如果报错，提示nvcc错误
1. 对于gcc版本大于等于8的不支持，需要修改Makefile中的g++为g++7或者更低