# 知识点
1. reduce思路，规约求和
    1. 将数组划分为n个块block。每个块大小为b，b设置为2的幂次方
    2. 分配b个大小的共享内存，记为float cache[b];
    3. 对每个块的数据进行规约求和：
        1. 定义plus = b / 2
        2. value = array[position], tx = threadIdx.x
        3. 将当前块内的每一个线程的value载入到cache中，即cache[tx] = value
        4. 对与tx < plus的线程，计算value += array[tx + plus]
        5. 定义plus = plus / 2，循环到3步骤
    4. 将每一个块的求和结果累加到output上（只需要考虑tx=0的线程就是总和的结果了）
2. __syncthreads，同步所有block内的线程，即block内的所有线程都执行到这一行后再并行往下执行
3. atomicAdd，原子加法，返回的是旧值

