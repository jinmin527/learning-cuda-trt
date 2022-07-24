# 项目目标：使用宏定义来检查是否有做cuda初始化

<span style='color:red'>重点：</span>用宏定义do while来写一个类似函数的东西来检查cuInit是否有执行（如下）
    `checkDriver(cuInit(0));`

<details> <!-- run -->
<summary> 运行 </summary>

`make run`
</details> <!-- run -->


<details> <!--详情-->
<summary> 详情 </summary>

1. CUDA driver需要做显式的初始化`cuInit(0)`，否则其他API都会返回`CUDA_ERROR_NOT_INITIALIZED`;
2. 采用宏定义可以在每次调用API前都检查初始化;
3. 宏定义中的`do...while(0)`使用：\
   虽然`do...while(0)`与顺序执行`...`效果一样，但前者可保证程序的正确性，例如：
   ```cpp
   #define swap(a, b){a = a+b; b = a-b;}

    int main()
    {
        int a = 3, b = 2;
        if(1)
            swap(a,b);
        else
            a = b = 0;
        return 0;
    }
   ```
   替换后看起来似乎没有问题，但会多一个`;`,从而使得编译错误：
    ```cpp
    int main()
    {
        int a = 3, b = 2;
        if(1)
        {
            a = a+b; 
            b = a-b;
        };
        else
            a = b = 0;
        return 0;
    }
    ```

    虽然可以去掉`swap(a,b);`中的`;`可以解决问题，但不太符合编码习惯
    因此宏的编写者统一使用`do...while(0)`解决这个问题
    但是你们不觉得在宏定义中写函数很丑嘛！要不断用`\`来换行
    下一节介绍一种优雅的写法

</details> <!--详情-->


<details> <!--写在文末 -->
<summary> 写在文末 </summary>

- CUDA的在线文档地址：
  1. https://developer.nvidia.com/cuda-toolkit-archive
  2. https://docs.nvidia.com/cuda/archive/11.2.0/

- 如果报错，提示nvcc错误
  - 对于gcc版本大于等于8的不支持，需要修改Makefile中的g++为g++7或者更低
</details> <!--写在文末 -->



