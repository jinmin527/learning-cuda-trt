
// CUDA驱动头文件cuda.h
#include <cuda.h>   // include <> 和 "" 的区别    
#include <stdio.h>  // include <> : 标准库文件 
#include <string.h> // include "" : 自定义文件  详细情况请查看 readme.md -> 5

#define checkDriver(op)  __check_cuda_driver((op), #op, __FILE__, __LINE__)

bool __check_cuda_driver(CUresult code, const char* op, const char* file, int line){
    if(code != CUresult::CUDA_SUCCESS){    // 如果 成功获取CUDA情况下的返回值 与我们给定的值(0)不相等， 即条件成立， 返回值为flase
        const char* err_name = nullptr;    // 定义了一个字符串常量的空指针
        const char* err_message = nullptr;  
        cuGetErrorName(code, &err_name);    
        cuGetErrorString(code, &err_message);   
        printf("%s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message); //打印错误信息
        return false;
    }
    return true;
}

int main(){

    // 检查cuda driver的初始化
    checkDriver(cuInit(0));

    // 为设备创建上下文
    CUcontext ctxA = nullptr;                                   // CUcontext 其实是 struct CUctx_st*（是一个指向结构体CUctx_st的指针）
    CUcontext ctxB = nullptr;
    CUdevice device = 0;
    checkDriver(cuCtxCreate(&ctxA, CU_CTX_SCHED_AUTO, device)); // 这一步相当于告知要某一块设备上的某块地方创建 ctxA 管理数据。输入参数 参考 https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/group__CUDA__CTX_g65dc0012348bc84810e2103a40d8e2cf.html
    checkDriver(cuCtxCreate(&ctxB, CU_CTX_SCHED_AUTO, device)); // 参考 1.ctx-stack.jpg
    printf("ctxA = %p\n", ctxA);
    printf("ctxB = %p\n", ctxB);
    /* 
        contexts 栈：
            ctxB -- top <--- current_context
            ctxA 
            ...
     */

    // 获取当前上下文信息
    CUcontext current_context = nullptr;
    checkDriver(cuCtxGetCurrent(&current_context));             // 这个时候current_context 就是上面创建的context
    printf("current_context = %p\n", current_context);

    // 可以使用上下文堆栈对设备管理多个上下文
    // 压入当前context
    checkDriver(cuCtxPushCurrent(ctxA));                        // 将这个 ctxA 压入CPU调用的thread上。专门用一个thread以栈的方式来管理多个contexts的切换
    checkDriver(cuCtxGetCurrent(&current_context));             // 获取current_context (即栈顶的context)
    printf("after pushing, current_context = %p\n", current_context);
    /* 
        contexts 栈：
            ctxA -- top <--- current_context
            ctxB
            ...
    */
    

    // 弹出当前context
    CUcontext popped_ctx = nullptr;
    checkDriver(cuCtxPopCurrent(&popped_ctx));                   // 将当前的context pop掉，并用popped_ctx承接它pop出来的context
    checkDriver(cuCtxGetCurrent(&current_context));              // 获取current_context(栈顶的)
    printf("after poping, popped_ctx = %p\n", popped_ctx);       // 弹出的是ctxA
    printf("after poping, current_context = %p\n", current_context); // current_context是ctxB

    checkDriver(cuCtxDestroy(ctxA));
    checkDriver(cuCtxDestroy(ctxB));

    // 更推荐使用cuDevicePrimaryCtxRetain获取与设备关联的context
    // 注意这个重点，以后的runtime也是基于此, 自动为设备只关联一个context
    checkDriver(cuDevicePrimaryCtxRetain(&ctxA, device));       // 在 device 上指定一个新地址对ctxA进行管理
    printf("ctxA = %p\n", ctxA);
    checkDriver(cuDevicePrimaryCtxRelease(device));
    return 0;
}