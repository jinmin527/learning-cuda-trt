# 知识点
0. [抖音辅助讲解视频合集](https://www.douyin.com/video/7074804712570113311)，建议以1.5倍速配合代码注解的提示观看

1. 仿射变换+双线性插值，在CV场景下，解决图像预处理是非常合适的。[若有疑问，可点击抖音短视频辅助讲解(建议1.5倍速观看)](https://v.douyin.com/NhM6QnU/)
    - 例如Yolo的letterbox，实则是边缘填充
    - 例如CenterNet的居中对齐
2. 该代码仿射变换对象是CV8UC3的图像，经过简单修改后，可以做到端到端，把结果输入到tensorRT中
    - 可以直接实现WarpAffine + Normalize + SwapRB
        - 参考这里：https://github.com/shouxieai/tensorRT_Pro/blob/main/src/tensorRT/common/preprocess_kernel.cu
    - 这样的话性能会非常好
3. 在仿射核函数里面，我们循环的次数，是dst.width * dst.height，以dst为参照集
    - 也因此，无论src多大，dst固定的话，计算量也是固定的
    - 另外，核函数里面得到的是dst.x、dst.y，我们需要获取对应在src上的坐标
        - 所以需要src.x, src.y = project(matrix, dst.x, dst.y)
        - 因此这里的project是dst -> src的变换
        - AffineMatrix的compute函数计算的是src -> dst的变换矩阵i2d
        - 所以才需要invertAffineTransform得到逆矩阵，即dst -> src，d2i