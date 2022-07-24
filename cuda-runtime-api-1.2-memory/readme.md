# 项目目标：了解内存模型，学会操作数据复制
## 操作内存分配和数据复制
- 参考 1.transfer_data.jpg
  - main.cpp 做了的事情 
  1. 在gpu上开辟一块空间，并把地址记录在mem_device上
  2. 在cpu上开辟一块空间，并把地址记录在mem_host上，并修改了该地址所指区域的第二个值
  3. 把mem_host所指区域的数据都复制到mem_device的所指区域
  4. 在cpu上开辟一块空间,并把地址记录在mem_page_locked上
  5. 最后把mem_device所指区域的数据又复制回cpu上的mem_page_locked区域


## 内存模型及其知识点
<details> <!-- -->
<summary> 详情 </summary>

1. 关于内存模型，请参照 https://www.bilibili.com/video/BV1jX4y1w7Um
    - 内存大局上分为
        - 主机内存：Host Memory，也就是CPU内存，内存
        - 设备内存：Device Memory，也就是GPU内存，显存
            - 设备内存又分为：
                - 全局内存(3)：Global Memory
                - 寄存器内存(1)：Register Memory
                - 纹理内存(2)：Texture Memory
                - 共享内存(2)：Shared Memory
                - 常量内存(2)：Constant Memory
                - 本地内存(3)：Local Memory
            - 只需要知道，谁距离计算芯片近，谁速度就越快，空间越小，价格越贵
                - 清单的括号数字表示到计算芯片的距离
2. 通过cudaMalloc分配GPU内存，分配到setDevice指定的当前设备上
3. 通过cudaMallocHost分配page locked memory，即pinned memory，页锁定内存
    - 页锁定内存是主机内存，CPU可以直接访问
    - 页锁定内存也可以被GPU直接访问，使用DMA（Direct Memory Access）技术
        - 注意这么做的性能会比较差，因为主机内存距离GPU太远，隔着PCIE等，不适合大量数据传输
    - 页锁定内存是物理内存，过度使用会导致系统性能低下（导致虚拟内存等一系列技术变慢）
4. cudaMemcpy
    - 如果host不是页锁定内存，则：
        - Device To Host的过程，等价于
            - pinned = cudaMallocHost
            - copy Device to pinned
            - copy pinned to Host
            - free pinned
        - Host To Device的过程，等价于
            - pinned = cudaMallocHost
            - copy Host to pinned
            - copy pinned to Device
            - free pinned
    - 如果host是页锁定内存，则：
        - Device To Host的过程，等价于
            - copy Device to Host
        - Host To Device的过程，等价于
            - copy Host to Device

## Tips
<details> <!-- Tips -->
<summary> 详情 </summary>

- 建议先分配先释放
  - ```c++
    checkRuntime(cudaFreeHost(memory_page_locked));
    delete [] memory_host;
    checkRuntime(cudaFree(memory_device)); 
    ```
    使用cuda API来分配内存的一般都有自己对应的释放内存方法；而使用new来分配的使用delete来释放

</details> <!--Tips -->


