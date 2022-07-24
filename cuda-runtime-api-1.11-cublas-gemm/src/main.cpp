
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <chrono>

using namespace std;
using namespace cv;

#define min(a, b)  ((a) < (b) ? (a) : (b))
#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

void gemm_0(const float* A, const float* B, float* C, int m, int n, int k, cudaStream_t stream);
void gemm_1(const float* A, const float* B, float* C, int m, int n, int k, cudaStream_t stream);

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
    if(code != cudaSuccess){
        const char* err_name = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
        return false;
    }
    return true;
}

void verify(const cv::Mat& a, const cv::Mat& b, float eps = 1e-5){

    auto diff = cv::Mat(a - b);
    float* p = diff.ptr<float>(0);
    int error_count = 0;
    float max_diff = *p;
    float min_diff = *p;
    for(int i = 0; i < diff.rows*diff.cols; ++i, ++p){
        if(fabs(*p) >= eps){
            if(error_count < 10)
                printf("Error value: %f, %d\n", *p, i);

            error_count += 1;
        }

        max_diff = max(max_diff, *p);
        min_diff = min(min_diff, *p);
    }

    if(error_count > 0){
        printf("... error count = %d. max = %f, min = %f\n", error_count, max_diff, min_diff);
    }else{
        printf("Done no error. max = %f, min = %f\n", max_diff, min_diff);
    }
}

int main(){

    cv::Mat A, B, C;
    int m = 2000, n = 3000, k = 5000;
    A.create(m, n, CV_32F);
    B.create(n, k, CV_32F);
    C.create(m, k, CV_32F);
    cv::randn(A, 0, 1);
    cv::randn(B, 0, 1);

    cudaEvent_t tc0, tc1, tc2, tc3;
    cudaStream_t stream = nullptr;
    cublasHandle_t cublas_h = nullptr;
    cudaStreamCreate(&stream);
    cublasCreate(&cublas_h);
    cublasSetStream(cublas_h, stream);
    cudaEventCreate(&tc0);
    cudaEventCreate(&tc1);
    cudaEventCreate(&tc2);
    cudaEventCreate(&tc3);

    int Abytes = m * n * sizeof(float);
    int Bbytes = n * k * sizeof(float);
    int Cbytes = m * k * sizeof(float);

    cudaHostRegister(A.data, Abytes, cudaHostRegisterDefault);
    cudaHostRegister(B.data, Bbytes, cudaHostRegisterDefault);
    cudaHostRegister(C.data, Cbytes, cudaHostRegisterDefault);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, Abytes);
    cudaMalloc(&dB, Bbytes);
    cudaMalloc(&dC, Cbytes);

    cudaMemcpyAsync(dA, A.data, Abytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dB, B.data, Bbytes, cudaMemcpyHostToDevice, stream);

    // warm up
    for(int i = 0; i < 10; ++i)
        gemm_0(dA, dB, dC, m, n, k, stream);

    cudaEventRecord(tc0, stream);
    gemm_0(dA, dB, dC, m, n, k, stream);

    cudaEventRecord(tc1, stream);
    gemm_1(dA, dB, dC, m, n, k, stream);

    cudaEventRecord(tc2, stream);
    float alpha = 1.0f;
    float beta  = 0.0f;
    int lda = n;
    int ldb = k;
    int ldc = m;

    // 注意他的结果是列优先的，需要转置才能做对比
    // cv::Mat tmp(C.cols, C.rows, CV_32F, C.data);
    // verify(tmp.t(), tC, 1e-4);
    // C = OP(A) * OP(B) * alpha + beta
    // 注意这里的内存序列是列主序列的
    // A -> m x n
    // B -> n x k
    // C -> m x k
    
    cublasSgemm(cublas_h,
        cublasOperation_t::CUBLAS_OP_T,   // 提供的数据是行排列，因此需要转置
        cublasOperation_t::CUBLAS_OP_T,   // 提供的数据是行排列，因此需要转置
        m,           // op(A)的行数
        k,           // op(B)的列数
        n,           // op(A)的列数
        &alpha,      // 乘数
        dA,          // A的地址
        lda,         // op(A)的列数
        dB,          // B的地址
        ldb,         // op(B)的列数
        &beta,       // 加数
        dC,          // C的地址
        ldc          // C = op(A) * op(B)，C的行数，为啥，因为结果C是列排列的，所以C的结果是k x m的，需要转置到m x k
    );
    cudaEventRecord(tc3, stream);
    cudaMemcpyAsync(C.data, dC, Cbytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    float time_kernel_0 = 0;
    float time_kernel_1 = 0;
    float time_kernel_cublas = 0;
    cudaEventElapsedTime(&time_kernel_0, tc0, tc1);
    cudaEventElapsedTime(&time_kernel_1, tc1, tc2);
    cudaEventElapsedTime(&time_kernel_cublas, tc2, tc3);
    printf(
        "kernel_0 = %.5f ms, kernel_1 = %.5f ms, kernel_cublas = %.5f ms\n", 
        time_kernel_0, time_kernel_1, time_kernel_cublas
    );

    auto t0 = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
    auto tC = cv::Mat(A * B);
    auto t1 = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / 1000.0;

    printf("CPU times: %.5f ms\n", t1 - t0);

    // 这个转换只应用于cublas，因为cublas输出结果是列优先的
    bool cublas_result = true;
    if(cublas_result){
        cv::Mat tmp(C.cols, C.rows, CV_32F, C.data);
        tmp = tmp.t();

        verify(tmp, tC, 1e-3);
    }else{
        verify(C, tC, 1e-3);
    }
    return 0;
}