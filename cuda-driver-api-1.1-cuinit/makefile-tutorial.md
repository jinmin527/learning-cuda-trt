# **Makefile**

GNU Make官方网站：https://www.gnu.org/software/make/        
GNU Make官方文档下载地址：https://www.gnu.org/software/make/manual/
# 1 基本格式
```
targets : prerequisties
    command
```
- target：目标文件，可以是OjectFile，也可以是执行文件，还可以是一个标签（Label），对于标签这种特性，在后续的“伪目标”章节中会有叙述。
- prerequisite：要生成那个target所需要的文件或是目标。
- command：是make需要执行的命令，


&emsp;
# 2 Makefile规则
（1）make会在当前目录下找到一个名字叫“Makefile”或“makefile”的文件。       
（2）如果找到，它会找文件中第一个目标文件（target），并把这个文件作为最终的目标文件。     
（3）如果target文件不存在，或是target文件依赖的.o文件(prerequities)的文件修改时间要比target这个文件新，就会执行后面所定义的命令command来生成target这个文件      
（4）如果target依赖的.o文件（prerequisties）也存在，make会在当前文件中找到target为.o文件的依赖性，如果找到，再根据那个规则生成.o文件。


&emsp;
# 3 Makefile的变量
&emsp;&emsp;变量在声明时需要给予初值，而在使用时，需要给在变量名前加上“$”符号，并用小括号“（）”把变量给包括起来。

&emsp;
## 3.1 变量的定义和使用
- 定义
```
include_paths := /usr/local/cuda-10.1/include \
                 /datav/lean/opencv4.5.1/include/opencv4 \
                 /datav/lean/tensorRT6.0.1.5_cuda10.1_cudnn7.6.0.3/include/ \
                 src \
                 src/tensorRT \
                 src/tensorRT/common \
```
- 使用
```
include_paths := $(foreach item,$(include_path),-I(item))
```

&emsp;
## 3.2 Makefile常用的预定义变量
- $@&emsp;&emsp;目标(target)的完整名称。
- $<&emsp;&emsp;第一个依赖文件（prerequisties）的名称。
- $^&emsp;&emsp;所有的依赖文件（prerequisties），以空格分开，不包含重复的依赖文件。

&emsp;
# 4 Makefile的常用运算符

&emsp;
## 4.1 赋值:=
&emsp;&emsp;用于变量的定义、赋值
```makefile
link_library := cudart opencv_core opencv_imgcodecs opencv_imgproc \
                 gomp nvinfer protobuf cudnn pthread \
                 cublas nvcaffe_parser nvinfer_plugin python3.8
```

## 4.2 累加+=
```makefile
cpp_compile_flags := -m64 -fPIC -g -O0 -std=c++11 -w -fopenmp

cpp_compile_flags += $(include_paths)
```

&emsp;
# 5 Makefile的常用函数 
&emsp;&emsp;函数调用，很像变量的使用，也是以“$”来标识的，其语法如下：
```
$(<function> <arguments>)
```
- \<function>就是函数名，make 支持的函数不多。
- \<arguments>是函数的参数，参数间以逗号“,”分隔，而函数名和参数之间以“空格”分隔。

&emsp;
## 5.1 shell
```
$(shell <command> <arguments>)
```
- 名称：shell命令函数——subst。
- 功能：调用shell命令command
- 返回：函数返回shell命令command的执行结果


- 示例：
```
cpp_srcs := $(shell find src -name "*.cpp") 

```



&emsp;
## 5.2 patsubst
```
$(patsubst <pattern>,<replacement>,<text>)
```
- 名称：模式字符串替换函数——patsubst。
- 功能：查看<text>中的单词是否符合模式<pattern>，如果匹配的话，则以<replacement>替换。这里，<pattern>可以包括通配符“%”，表示任意长度的字串。如果<replacement>中也包含“%”，那么，<replacement>中的这个“%”将是<pattern>中的那个“%”所代表的字串。
- 返回：函数返回被替换过后的字符串。
- 示例：
```
cpp_srcs := $(shell find src -name "*.cpp") #shell指令，src文件夹下找到.cpp文件
cpp_objs := $(patsubst %.cpp,%.o,$(cpp_srcs)) #cpp_srcs变量下cpp文件替换成 .o文件

```


&emsp;
## 5.3 subst
```
$(subst <from>,<to>,<text>)
```
- 名称：字符串替换函数——subst。
- 功能：把字串<text>中的<from>字符串替换成<to>。
- 返回：函数返回被替换过后的字符串。
- 示例：
```
cpp_srcs := $(shell find src -name "*.cpp") #shell指令，src文件夹下找到.cpp文件
cpp_objs := $(patsubst %.cpp,%.o,$(cpp_srcs)) #cpp_srcs变量下cpp文件替换成 .o文件
cpp_objs := $(subst src/,objs/,$(cpp_objs)) #cpp_objs 变量下替换后的 .o文件 从src文件夹移植到objs文件夹

```

&emsp;
## 5.4 foreach
```
$(foreach <var>,<list>,<text>)
```
- 名称：循环函数——subst。
- 功能：把字串\<list>中的元素逐一取出来，执行\<text>包含的表达式
- 返回：\<text>所返回的每个字符串所组成的整个字符串（以空格分隔）
- 示例：
```
library_paths := /datav/shared/100_du/03.08/lean/protobuf-3.11.4/lib \
                 /usr/local/cuda-10.1/lib64 \

library_paths := $(foreach item,$(library_paths),-L$(item))

```

## 5.5 dir
```
$(dir <names...>)
```
- 名称：取目录函数——dir。
- 功能：从文件名序列<names>中取出目录部分。目录部分是指最后一个反斜杠（“/”）之前
的部分。如果没有反斜杠，那么返回“./”。
- 返回：返回文件名序列<names>的目录部分。
- 示例： 
```
$(dir src/foo.c hacks)    # 返回值是“src/ ./”。
```


# 6 伪目标
&emsp;&emsp;“伪目标”不是一个文件，只是一个标签，。我们要显示地指明这个“目标”才能让其生效。“伪目标”的取名不能和文件名重名，不然其就失去了“伪目标”的意义了。

&emsp;&emsp;为了避免和文件重名的这种情况，我们可以使用一个特殊的标记“.PHONY”来显示地指明一个目标是“伪目标”，向 make 说明，不管是否有这个文件，这个目标就是“伪目标”。
```
.PHONY : clean
```

&emsp;&emsp;只要有这个声明，不管是否有“clean”文件，要运行“clean”这个目标，只有“make clean”。

&emsp;
# 7 编译过程
## 7.1 总览
```
源文件[.c/cpp] -> 预处理成[.i/.ii] -> 编译成[.s] -> 汇编成[.o] -> 链接成可执行文件
```

## 7.2 包含静态库的编译过程
> ### 步骤：     
>> （1）源文件[.c/cpp] -> Object文件[.o]
```
g++ -c [.c/cpp][.c/cpp]... -o [.o][.o]... -I[.h/hpp] -g
```
>> （2）Object文件[.o] -> 静态库文件[lib库名.a]
```
ar -r [lib库名.a] [.o][.o]...
```
>> （3）main文件[.c/cpp] -> Object文件[.o]
```
g++ -c [main.c/cpp] -o [.o] -I[.h/hpp] -g
```
>> （4）链接 main的Object文件 与 静态库文件[lib库名.a]
```
g++ [main.o] -o [可执行文件] -l[库名] -L[库路径]
```

## 7.3 包含动态库（共享库）的编译过程
>### 步骤：     
>> 源文件[.c/cpp] -> Object文件[.o]
```
g++ -c [.c/cpp][.c/cpp]... -o [.o][.o]... -I[.h/hpp] -g -fpic
```
>> Object文件[.o] -> 静态库文件[lib库名.a]
```
g++ -shared [.o][.o]... -o [lib库名.so] 
```
>> main文件[.c/cpp] -> Object文件[.o]
```
g++ -c [main.c/cpp] -o [.o] -I[.h/hpp] -g
```
>> 链接 main的Object文件 与 静态库文件[lib库名.a]
```
g++ [main.o] -o [可执行文件] -l[库名] -L[库路径] -Wl,-rpath=[库路径]
```
