# 项目目标：了解cuda和显卡等基本概念 及 初始化cuda
<details> <!--如何运行 -->
    <summary> 1.如何运行 </summary>

- `make run`

</details> <!--如何运行 -->

<details> <!-- 显卡，显卡驱动 -->
    <summary> 2.显卡，显卡驱动,nvcc, cuda driver,cudatoolkit,cudnn到底是什么？ </summary>

- 关于显卡驱动与cuda驱动的版本匹配
  - [Table 1. CUDA 11.6 Update 1 Component Versions](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
  - 结论：尽量将显卡驱动升级到新的，因为显卡驱动向下兼容cuda驱动
- <details> <!-- 简单了解概念 -->
    <summary> 简单了解显卡相关概念 </summary>
    
  - 显卡：GPU
  - 显卡驱动：驱动软件，类比声卡驱动，摄像头驱动
  - GPU架构：gpu架构指的是硬件的设计方式，例如是否有L1 or L2缓存
  - CUDA: 其中一种理解是它是一种编程语言（像c++,python等，只不过它是专门用来操控GPU的）
  - cudnn: 这个其实就是一个专门为深度学习计算设计的软件库，里面提供了很多专门的计算函数
  - CUDAToolkit：这是我们真正需要首先安装的工具包，所谓的装cuda首先指的是它
  - 它里面包含了许多库，例如：cudart, cublas等
  - 其他涉及到的知识有nvcc与nvidia-smi, 多个 cuda 版本之间进行切换, cuda的安装等
  - 详细请参考: https://zhuanlan.zhihu.com/p/91334380
</details> <!-- 简单了解概念 -->

</details> <!-- 显卡，显卡驱动 -->


<details> <!--cuda-driver-api 与 cuda-runtime-api -->
    <summary> 3.cuda-driver-api 与 cuda-runtime-api </summary>

- CUDA Driver与CUDA Runtime相比更偏底层，就意味着Driver API有着更灵活的控制，也伴随着更复杂的编程
- 因此CUDA driver需要做显式的初始化`cuInit(0)`，否则其他API都会返回`CUDA_ERROR_NOT_INITIALIZED`
- 经过初始化后驱动和显卡的信息可以轻松获取：
    - 驱动版本管理 https://docs.nvidia.com/cuda/archive/11.2.0/cuda-driver-api/group__CUDA__VERSION.html#group__CUDA__VERSION
    - 设备信息管理 https://docs.nvidia.com/cuda/archive/11.2.0/cuda-driver-api/group__CUDA__DEVICE.html

</details> <!--cuda-driver-api 与 cuda-runtime-api -->

<details> <!--写在文末 -->
    <summary> 4.写在文末 </summary>

- CUDA的在线文档地址
  1. https://developer.nvidia.com/cuda-toolkit-archive
  2. https://docs.nvidia.com/cuda/archive/11.2.0/

- 报错
  - 提示nvcc错误。对于gcc版本大于等于8的不支持，需要修改Makefile中的g++为g++7或者更低  

</details> <!--写在文末 -->


<details> <!-- 5. (面向小白)C++基础 -->
<summary> 5.C++基础（供C++小白参考） </summary>
<details> <!-- 5.1 编译基础知识 -->
<summary> 5.1 编译基础知识 </summary>

- "c++脚本程序写完之后,并不能直接运行,需要进行编译,转成.o文件,再链接才能运行"
<details> <!-- 5.1.1 单文件或少文件编译 -->
<summary> 5.1.1 单文件或少文件编译 </summary>

- 源文件[.c/cpp] -> Object文件[.o]
```makefile
g++ -c [.c/cpp][.c/cpp]... -o [.o][.o]... -I[.h/hpp]
g++是编译命令 -c,-o,-I是选项 -c接源脚本文件 -o接目标文件 -I接头文件
(-c c++ /-o object/ -I include)
```
</details> <!-- 5.1.1 单文件或少文件编译 -->


<details> <!-- 5.1.2 多文件编译 -->
<summary> 5.1.2 多文件编译（使用Makefile 和 CMake） </summary>

- cmake比Makefile高级,但是两者的功能都是快速地进行批量的编译(因为当你有很多的c++源文件的时候,一个一个地去用g++去编译是很麻烦的）

- makefile详情参考以下两份资料
  - （1）https://zhuanlan.zhihu.com/p/396448133
  - （2）makefile-tutorial.md

