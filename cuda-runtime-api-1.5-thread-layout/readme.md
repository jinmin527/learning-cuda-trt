# 知识点
1. 在.vscode/settings.json中配置"*.cu": "cuda-cpp"可以实现对cuda的语法解析
   - 如图 [4.syntax-parser.jpg](figure/4.syntax-parser.jpg)
2. layout是设置核函数执行的线程数，要明白最大值、block最大线程数、warpsize取值
    - maxGridSize对应gridDim的取值最大值
    - maxThreadsDim对应blockDim的取值最大值
    - warpSize对应线程束中的线程数量
    - maxThreadsPerBlock对应blockDim元素乘积最大值
3. layout的4个主要变量的关系
    - gridDim是layout维度，其对应的索引是blockIdx
        - blockIdx的最大值是0到gridDim-1
    - blockDim是layout维度，其对应的索引是threadIdx
        - threadIdx的最大值是0到blockDim-1
        - blockDim维度乘积必须小于等于maxThreadsPerBlock
    - 所以称gridDim、blockDim为维度，启动核函数后是固定的
    - 所以称blockIdx、threadIdx为索引，启动核函数后，枚举每一个维度值，不同线程取值不同
    - 关于线程束带概念这里不讲，可以自行查询
4. 核函数启动时，<<<>>>的参数分别为：<<<gridDim, blockDim, shraed_memory_size, cudaStream_t>>>
    - shared_memory_size请看后面关于shared memory的讲解，配置动态的shared memory大小