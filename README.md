# learning-cuda-trt
- A large number of cuda/tensorrt cases
- 在这个project中，提供大量的cuda和tensorrt学习案例
- cuda驱动api
- cuda运行时api
- tensorRT基础入门
    - 基本的tensorRT学习
    - 插件、onnx解析器
- tensorRT高阶应用
    - 导出onnx，前后处理
    - 具体项目为案例，掌握如何处理复杂情况

# 这是一个学习cuda、tensorrt的源代码案例项目
1. 大量案例，从基础的cuda驱动api、运行时api到tensorrt的基础入门、tensorrt的高级进阶
2. 模型的导出、模型的前后处理等等，多线程的封装等等
3. 希望能够帮助你进一步掌握tensorRT

# 使用方法-自行配置环境
1. 案例均使用makefile作为编译工具
    - 在其中以`${@CUDA_HOME}`此类带有@符号表示为特殊变量
    - 替换此类特殊变量为你系统真实环境，即可顺利使用
2. 大部分时候，配置完毕后，可以通过`make run`实现编译运行

# 使用方法-自动配置环境
1. 要求linux-ubuntu16.04以上系统，并配有GPU和显卡驱动大于495最佳
2. 安装python包，`pip install trtpy -U -i https://pypi.org/simple`
3. 配置快捷方式，`echo alias trtpy=\"python -m trtpy\" >> ~/.bashrc`
4. 应用快捷方式：`source ~/.bashrc`
5. 配置key：`trtpy set-key sxaikiwik`
6. 获取并配置环境：`trtpy get-env --cuda=11`
    - 目前仅支持10和11，如果驱动版本不适配，会提示找不到适配的版本
7. 自动改变配置变量：`trtpy prep-vars .`，把当前目录下的所有变量都自行替换
8. 即可运行`make run`

# Reference
- TensorRT的B站视频讲解：https://www.bilibili.com/video/BV1Xw411f7FW
- 官方的视频讲解：https://www.bilibili.com/video/BV15Y4y1W73E
- trtpy前期介绍文档：https://zhuanlan.zhihu.com/p/462980738
- 本源代码对应配套的视频教程讲解（腾讯课堂）：https://ke.qq.com/course/4993141