# 知识点
1. nthreads的取值，不能大于block能取值的最大值。一般可以直接给512、256，性能就是比较不错的
    - (input_size + block_size - 1) / block_size;是向上取整
2. 对于一维数组时，采用只定义layout的x维度，若处理的是二维，则可以考虑定义x、y维度，例如处理的是图像
3. 关于把数据视作一维时，索引的计算
    - 以下是通用的计算公式
    ```python
    Pseudo code:
    position = 0
    for i in range(6):
        position *= dims[i]
        position += indexs[i]
    ```
    - 例如当只使用x维度时，实际上dims = [1, 1, gd, 1, 1, bd]，indexs = [0, 0, bi, 0, 0, ti]
        - 因为0和1的存在，上面的循环则可以简化为：idx = threadIdx.x + blockIdx.x * blockDim.x
        - 即：idx = ti + bi * bd