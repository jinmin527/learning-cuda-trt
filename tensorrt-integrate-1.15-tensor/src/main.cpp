
// tensorRT include
// 编译用的头文件
#include <NvInfer.h>

// onnx解析器的头文件
#include <onnx-tensorrt/NvOnnxParser.h>

// 推理用的运行时头文件
#include <NvInferRuntime.h>

// cuda include
#include <cuda_runtime.h>

// system include
#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <unistd.h>
#include <opencv2/opencv.hpp>

#include "cuda-tools.hpp"
#include "trt-builder.hpp"
#include "simple-logger.hpp"
#include "trt-tensor.hpp"

using namespace std;

class TRTLogger : public nvinfer1::ILogger{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
        if(severity <= Severity::kWARNING){
            if(severity == Severity::kWARNING){
                INFOW("%s", msg);
            }
            else if(severity <= Severity::kERROR){
                INFOE("%s", msg);
            }
            else{
                INFO("%s", msg);
            }
        }
    }
} logger;

// 通过智能指针管理nv返回的指针参数
// 内存自动释放，避免泄漏
template<typename _T>
shared_ptr<_T> make_nvshared(_T* ptr){
    return shared_ptr<_T>(ptr, [](_T* p){p->destroy();});
}

bool exists(const string& path){

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}

// 上一节的代码
bool build_model(){

    if(exists("engine.trtmodel")){
        printf("Engine.trtmodel has exists.\n");
        return true;
    }

    //SimpleLogger::set_log_level(SimpleLogger::LogLevel::Verbose);
    TRT::compile(
        TRT::Mode::FP32,
        10,
        "classifier.onnx",
        "engine.trtmodel",
        1 << 28
    );
    INFO("Done.");
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

vector<unsigned char> load_file(const string& file){
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0){
        in.seekg(0, ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}

vector<string> load_labels(const char* file){
    vector<string> lines;

    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open()){
        printf("open %d failed.\n", file);
        return lines;
    }
    
    string line;
    while(getline(in, line)){
        lines.push_back(line);
    }
    in.close();
    return lines;
}

void inference(){

    TRTLogger logger;
    auto engine_data = load_file("engine.trtmodel");
    auto runtime   = make_nvshared(nvinfer1::createInferRuntime(logger));
    auto engine = make_nvshared(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return;
    }

    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));
    auto execution_context = make_nvshared(engine->createExecutionContext());

    int input_batch   = 1;
    int input_channel = 3;
    int input_height  = 224;
    int input_width   = 224;
    int input_numel   = input_batch * input_channel * input_height * input_width;

    // tensor的建立并不会立即分配内存，而是在第一次需要使用的时候进行分配
    TRT::Tensor input_data({input_batch, input_channel, input_height, input_width}, TRT::DataType::Float);

    // 为input关联stream，使得在同一个pipeline中执行复制操作
    input_data.set_stream(stream);

    ///////////////////////////////////////////////////
    // image to float
    auto image = cv::imread("dog.jpg");
    float mean[] = {0.406, 0.456, 0.485};
    float std[]  = {0.225, 0.224, 0.229};

    // 对应于pytorch的代码部分
    cv::resize(image, image, cv::Size(input_width, input_height));
    image.convertTo(image, CV_32F);

    // 利用opencv mat的内存地址引用，实现input与mat的关联，然后利用split函数一次性完成mat到input的复制
    cv::Mat channel_based[3];
    for(int i = 0; i < 3; ++i)
        // 注意这里 2 - i是实现bgr -> rgb的方式
        // 这里cpu提供的参数0是表示batch的索引是0，第二个参数表示通道的索引，因此获取的是0, 2-i通道的地址
        // 而tensor最大的好处就是帮忙计算索引，否则手动计算就得写很多代码
        channel_based[i] = cv::Mat(input_height, input_width, CV_32F, input_data.cpu<float>(0, 2-i));

    cv::split(image, channel_based);

    // 利用opencv的mat操作加速减去均值和除以标准差
    for(int i = 0; i < 3; ++i)
        channel_based[i] = (channel_based[i] / 255.0f - mean[i]) / std[i];
    
    // 如果不写，input_data.gpu获取gpu地址时会自动进行复制
    // 目的就是把内存复制变为隐式进行
    input_data.to_gpu();

    // 3x3输入，对应3x3输出
    const int num_classes = 1000;
    TRT::Tensor output_data({input_batch, num_classes}, TRT::DataType::Float);
    output_data.set_stream(stream);

    // 明确当前推理时，使用的数据输入大小
    auto input_dims = execution_context->getBindingDimensions(0);
    input_dims.d[0] = input_batch;

    execution_context->setBindingDimensions(0, input_dims);
    float* bindings[] = {input_data.gpu<float>(), output_data.gpu<float>()};
    bool success      = execution_context->enqueueV2((void**)bindings, stream, nullptr);
    checkRuntime(cudaStreamSynchronize(stream));

    // 当获取cpu地址的时候，如果数据最新的在gpu上，就进行数据复制，然后再返回cpu地址
    float* prob = output_data.cpu<float>();
    int predict_label = std::max_element(prob, prob + num_classes) - prob;
    auto labels = load_labels("labels.imagenet.txt");
    auto predict_name = labels[predict_label];
    float confidence  = prob[predict_label];
    printf("Predict: %s, confidence = %f, label = %d\n", predict_name.c_str(), confidence, predict_label);

    checkRuntime(cudaStreamDestroy(stream));
}

int main(){
    
    if(!build_model()){
        return -1;
    }
    inference();
    return 0;
}