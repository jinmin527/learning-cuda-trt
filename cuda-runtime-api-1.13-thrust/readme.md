# 知识点
1. thrust是cuda开发的，基于cuda的stl库，便于使用
2. 因为通常没用到thrust，所以对这块儿也不做过多解释
3. 对于thrust中的lambda表达式，需要增加__device__标记表明函数可以被核函数调用，此时需要在makefile中增加--extended-lambda标记
4. 由于使用到了device vector，因此编译环境需要修改为nvcc编译，因此main.cpp改成了main.cu
5. 内存的复制和分配，被cuda封装