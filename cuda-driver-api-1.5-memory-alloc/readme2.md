# cuda-driver-api 1.5
1. 内存分配：
   - 1.1. 分配线性内存`cuMemAlloc()`:
      - 1.1.1. 线性内存：线性内存被组织在单个连续的地址空间中，可以直接以及线性地访问这些内存位置。
      - 1.1.2. 内存分配空间以字节为大小，并返回所分配的内存地址
      
    - 1.2. 分配主机锁页内存`cuMemAllocHost()`:
       - 2.1. 锁页内存：
            - 2.1.1. 定义：页面不允许被调入调出的叫锁页内存，反之叫可分页内存。
            - 2.1.2. 有啥好处：快。
                - a. 设备可以直接访问内存，与可分页内存相比，它的读写带宽要高得多
                - b. 驱动程序会跟踪使用`cuMemAllocHost()`分配的虚拟内存范围，并自动加速对cuMemcpy（）等函数的调用。
            - 2.1.3. 使用注意：分配过多锁业内存会减少系统可用于分页的内存量，可能会降低系统性能。因此，在主机和设备之间为数据交换分配临时区域时，最好少用此功能。
       - 2.2. 这里是从主机分配内存，因此不是输入device prt的地址，而是一个主机的二级地址。
2. 内存的初始化`cuMemsetD32(CUdeviceptr dstDevice, unsigned int  ui, size_t N)`, 将N个32位值的内存范围设置为指定的值ui
3. 内存的释放`cuMemFreeHost()`: 有借有还 再借不难~

# 写在文末
### CUDA的在线文档地址：
1. https://developer.nvidia.com/cuda-toolkit-archive
2. https://docs.nvidia.com/cuda/archive/11.2.0/

### Startup
1. `make run`

### 如果报错，提示nvcc错误
1. 对于gcc版本大于等于8的不支持，需要修改Makefile中的g++为g++7或者更低