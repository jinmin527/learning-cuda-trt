# 知识点
1. 使用智能指针，对tensorrt返回值做封装，使得内存安全不会泄露
    ```c++
    template<typename _T>
    shared_ptr<_T> make_nvshared(_T* ptr){
        return shared_ptr<_T>(ptr, [](_T* p){p->destroy();});
    }
    // [](_T* p){p->destroy();} 这里用lambda 表达式的形式来表示 destroy 的方式
    ```
    - 因为他常常需要destroy进行释放
2. 使用cudaMallocHost对输入的host进行分配，使得主机内存复制到设备效率更高
3. 注意推理时的预处理，指定了rgb与bgr对调
4. 如果需要多个图像推理，需要：
    1. 在编译时，指定maxbatchsize为多个图
    2. 在推理时，指定输入的bindings shape的batch维度为使用的图像数，要求小于等于maxbatchsize
    3. 在收取结果的时候，tensor的shape是input指定的batch大小，按照batch处理即可
    