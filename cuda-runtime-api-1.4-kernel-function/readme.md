# 知识点
1. cu文件一般写cuda的核函数
2. 在.vscode/settings.json中配置*.cu : cuda-cpp，可以使得代码被正确解析
3. Makefile中，cu交给nvcc进行编译
4. cu文件可以当做正常cpp写即可，他是cpp的超集，兼容支持cpp的所有特性
5. cu文件中引入了一些新的符号和语法：
    - `__global__`标记，核函数标记
        - 调用方必须是host
        - 返回值必须是void
        - 例如：`__global__ void kernel(const float* pdata, int ndata)`
        - 必须以`kernel<<<gridDim, blockDim, bytesSharedMemorySize, stream>>>(pdata, ndata)`的方式启动
            - 其参数类型是：`<<<dim3 gridDim, dim3 blockDim, size_t bytesSharedMemorySize, cudaStream_t stream>>>`
                - dim3有构造函数dim3(int x, int y=1, int z=1)
                    - 因此当直接赋值为int时，实则定义了dim.x = value, dim.y = 1, dim.z = 1
            - 其中gridDim, blockDim, bytesSharedMemory, stream是线程layout参数
                - 如果指定了stream，则把核函数加入到stream中异步执行
            - pdata和ndata则是核函数的函数调用参数
            - 函数调用参数必须传值，不能传引用等。参数可以是类类型等
        - 核函数的执行无论stream是否为nullptr，都将是异步执行
            - 因此在核函数中进行printf操作，你必须进行等待，例如cudaDeviceSynchronize、或者cudaStreamSynchronize，否则你将无法看到打印的信息
    - `__device__`标记，设备调用的函数
        - 调用方必须是device
    - `__host__`标记，主机调用函数
        - 调用方必须是主机
    - 也可以`__device__ __host__`两个标记同时有，表明该函数可以设备也可以主机
    - `__constant__`标记，定义常量内存
    - `__shared__`标记，定义共享内存
6. 通过cudaPeekAtLastError/cudaGetLastError函数，捕获核函数是否出现错误或者异常
7. 内存索引的计算公式
    ```python
    position = 0
    for i in range(6):
        position *= dims[i]
        position += indexs[i]
    ```
8. buildin变量，即内置变量，通过ctrl+鼠标左键点进去查看定义位置
    - 所有核函数都可以访问，其取值由执行器维护和改变
    - gridDim[x, y, z]：网格维度，线程布局的大小，是核函数启动时指定的
    - blockDim[x, y, z]：块维度，线程布局的大小，是核函数启动时指定的
    - blockIdx[x, y, z]：块索引，对应最大值是gridDim，由执行器根据当前执行的线程进行赋值，核函数内访问时已经被配置好
    - threadIdx[x, y, z]：线程索引，对应最大值是blockDim，由执行器根据当前执行的线程进行赋值，核函数内访问时已经被配置好
    - Dim是固定的，启动后不会改变，并且是Idx的最大值
    - 每个都具有x、y、z三个维度，分别以z、y、x为高低顺序

9. 关于概念thread, grid, block 和 threadIdx
   <details> <!--thread, grid, block -->
   <summary> 详情 </summary>

   - 首先，先不严谨地认为，GPU相当于一个立方体，这个立方体有很多小方块如图 [3.organization-of-threads.jpg](figure/3.organization-of-threads.jpg)
   - 每个小块都是一个thread，为了讨论方便，我们只考虑2D的，如图 [1.block-and-grid.jpg](figure/1.block-and-grid.jpg)
   - 我们关心的是某一个thread的位置（比如说[1.block-and-grid.jpg](figure/1.block-and-grid.jpg) 中的黄色方块）
     - 它在2D的位置是(blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y) =(1, 0, 1, 1)
     - 如果将这个2D展开成1D，这个黄色thread 的1D位置是 13
     - 计算方式 如图：[2.example.jpg](figure/2.example.jpg)
     - 但是一般情况，为了简化问题，我们只需要用到 threadIdx.x，blockIdx.x，blockDim.x 这三个量即可，所以计算idx 的公式如下：
       - int idx = threadIdx.x + blockIdx.x * blockDim.x; 其表示的含义是要求thread的1D idx，先得知道在第几个block里，再知道在这个block里的第几个thread
   
   </details> <!--thread, grid, block -->

