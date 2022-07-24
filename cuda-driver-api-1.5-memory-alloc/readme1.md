# 项目目标：了解并学会使用cuda高效的内存分配

<!-- vscode-markdown-toc -->
* 1. [运行](#)
* 2. [内存分配：](#-1)
* 3. [写在文末](#-1)
	* 3.1. [CUDA文档：](#CUDA)
	* 3.2. [如果报错，提示nvcc错误](#nvcc)
* 4. [补充计算机知识(供小白参考)](#-1)
	* 4.1. [RAM 和 Hard disk](#RAMHarddisk)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->



##  1. <a name=''></a>运行
`make run`


##  2. <a name='-1'></a>内存分配：
<details> <!-- 内存分配：-->
   <summary> 详情 </summary>
   
   - 1. 统一内存(Unified addressing) 参考 1.single-memory-space.jpg
   - 2. 分配线性内存`cuMemAlloc()`:
      - 2.1. 线性内存：线性内存被组织在单个连续的地址空间中，可以直接以及线性地访问这些内存位置。
      - 2.2. 内存分配空间以字节为大小，并返回所分配的内存地址
         
   - 3. 分配主机锁页内存`cuMemAllocHost()`:
     - 图示：3.data-transfer.jpg 我们选择右边的内存分配模型
     - 强力推荐：https://leimao.github.io/blog/Page-Locked-Host-Memory-Data-Transfer/
     - 强力推荐：https://www.youtube.com/watch?v=p9yZNLeOj4s
       - 参考 4.paged-memory.jpg
     - 3.1. 锁页内存：
         - 定义：页面不允许被调入调出的叫锁页内存，反之叫可分页内存。
         - 有啥好处：快。
         - a. 设备可以直接访问内存，与可分页内存相比，它的读写带宽要高得多
         - b. 驱动程序会跟踪使用`cuMemAllocHost()`分配的虚拟内存范围，并自动加速对cuMemcpy（）等函数的调用。
         - 使用注意：分配过多锁业内存会减少系统可用于分页的内存量，可能会降低系统性能。因此，在主机和设备之间为数据交换分配临时区域时，最好少用此功能。
     - 3.2. 这里是从主机分配内存，因此不是输入device prt的地址，而是一个主机的二级地址。

   3. 内存的初始化`cuMemsetD32(CUdeviceptr dstDevice, unsigned int  ui, size_t N)`, 将N个32位值的内存范围设置为指定的值ui
   4. 内存的释放`cuMemFreeHost()`: 有借有还 再借不难~
   </details> <!-- 内存分配：-->
   
<br>
<br>


##  3. <a name='-1'></a>写在文末
<details> <!-- 写在文末-->
<summary> 详情 </summary>

###  3.1. <a name='CUDA'></a>CUDA文档：
1. https://developer.nvidia.com/cuda-toolkit-archive
2. https://docs.nvidia.com/cuda/archive/11.2.0/
3. unified addressing
   - https://developer.nvidia.com/blog/unified-memory-cuda-beginners/

###  3.2. <a name='nvcc'></a>如果报错，提示nvcc错误
- 对于gcc版本大于等于8的不支持，需要修改Makefile中的g++为g++7或者更低


</details> <!-- 写在文末-->


##  4. <a name='-1'></a>补充计算机知识(供小白参考)
<details> <!-- 补充计算机知识-->
<summary> 详情 </summary>

###  4.1. <a name='RAMHarddisk'></a>RAM 和 Hard disk
- ref: [Difference between RAM and HDD](https://www.tutorialspoint.com/
- RAM存需要实时运行的东西，HDD存操作系统文件等

</details> <!-- 补充计算机知识-->

