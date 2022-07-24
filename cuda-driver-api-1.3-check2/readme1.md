# 项目目标：了解带参宏定义在cuda编程中是如何使用的
<!-- vscode-markdown-toc -->
* 1. [<span style='color:red'>**重点:**</span>](#spanstylecolor:red:span)
* 2. [运行](#)
* 3. [CUDA文档：](#CUDA)
* 4. [如果报错，提示nvcc错误](#nvcc)
* 5. [C++基础](#C)
	* 5.1. [`#`的作用与宏](#-1)
	* 5.2. [函数的声明和定义](#-1)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->


##  1. <a name='spanstylecolor:red:span'></a><span style='color:red'>**重点:**</span>

- CUDA driver需要做显式的初始化`cuInit(0)`，否则其他API都会返回`CUDA_ERROR_NOT_INITIALIZED`;
- 采用宏定义可以在每次调用API前都检查初始化;
- 采用封装带参宏定义使代码更清晰、好调试，养成一种良好的编码习惯也是很重要滴！

##  2. <a name=''></a>运行
`make run`

##  3. <a name='CUDA'></a>CUDA文档：
1. https://developer.nvidia.com/cuda-toolkit-archive
2. https://docs.nvidia.com/cuda/archive/11.2.0/

##  4. <a name='nvcc'></a>如果报错，提示nvcc错误
- 对于gcc版本大于等于8的不支持，需要修改Makefile中的g++为g++7或者更低


##  5. <a name='C'></a>C++基础（供C++小白参考）
<details> <!--C++基础 -->
<summary> 详情 </summary>

###  5.1. <a name='-1'></a>`#`的作用与宏
[参见src/main.cpp](src/main.cpp) 9,10行:
<details> <!-- hong -->
<summary>详情</summary>   

宏定义将一个`标识符`定义为一个`字符串`
- 简单的例子
    1. #define <宏名> <宏体> <br>
	例子🌰🌰🌰：<br>
	#define PI 3.14 <br>
	当出现PI时，默认其为常量，且值为3.14 <br>
	
	2. #define <宏名>（<参数表>） <宏体> <br>
	例子🌰🌰🌰：<br>
	#define F(a,b) a+b <br>
	当出现 F(x,y)时，系统默认执行 x+y <br> 

- 本程序的例子：
  - `#define checkDriver(op)  __check_cuda_driver((op), #op, __FILE__, __LINE__)`    
  - 即每次我执行`checkDriver(my_func)`的时候，我真正执行的是__check_cuda_driver 这个函数，这函数传入的参数有my_func输出的结果（op），它的名字，当前所编译的文件和行数（方便之后报错时可以定位错误源）
  - `#op`中的`#`作用是把参数（变量）变成字符串

</details> <!-- hong -->

<br>

###  5.2. <a name='-1'></a>函数的声明和定义
[参见src/main.cpp](src/main.cpp) 9~23行:

<details> <!-- declaration -->
<summary>详情</summary>

- C++的脚本中所有的函数都是先声明后定义的
1. 声明是为了告诉**编译器**即<strong>将定义的函数名</strong>和<strong>返回值类型</strong>是什么  
2. 定义是告诉编译器函数的功能是什么    
3. 不声明则无法调用    
4. 声明叫做函数原型，函数定义叫做函数实现      
5. 声明并未给函数分配内存，只有定义的时候才给函数分配内存    
6. **函数的声明和定义不必一定在一个脚本中进行,可以将要定义的函数在.h头文件中进行统一的声明**<br>
    ```c++
    举个例子🌰🌰🌰：    
    int Add(int, int);       => 声明
    int Add(int a, int b){   => 定义     
    a = a + 2;      
    return (a + b)      
    }
    ```
</details> <!-- declaration -->




</details> <!--C++基础 -->



