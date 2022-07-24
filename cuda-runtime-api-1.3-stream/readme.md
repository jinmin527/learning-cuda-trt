# 知识点
1. stream是一个流句柄，可以当做是一个队列
    - cuda执行器从stream中一条条的读取并执行指令
    - 例如cudaMemcpyAsync函数等同于向stream这个队列中加入一个cudaMemcpy指令并排队
    - 使用到了stream的函数，便立即向stream中加入指令后立即返回，并不会等待指令执行结束
    - 通过cudaStreamSynchronize函数，等待stream中所有指令执行完毕，也就是队列为空
2. 当使用stream时，要注意
    - 由于异步函数会立即返回，因此传递进入的参数要考虑其生命周期，应确认函数调用结束后再做释放
3. 还可以向stream中加入Event，用以监控是否到达了某个检查点
    - `cudaEventCreate`，创建事件
    - `cudaEventRecord`，记录事件，即在stream中加入某个事件，当队列执行到该事件后，修改其状态
    - `cudaEventQuery`，查询事件当前状态
    - `cudaEventElapsedTime`，计算两个事件之间经历的时间间隔，若要统计某些核函数执行时间，请使用这个函数，能够得到最准确的统计
    - `cudaEventSynchronize`，同步某个事件，等待事件到达
    - `cudaStreamWaitEvent`，等待流中的某个事件
4. 默认流，对于cudaMemcpy等同步函数，其等价于执行了
    - cudaMemcpyAsync(... 默认流)   加入队列
    - cudaStreamSynchronize(默认流) 等待执行完成
    - 默认流与当前设备上下文类似，是与当前设备进行的关联
    - 因此，如果大量使用默认流，会导致性能低下