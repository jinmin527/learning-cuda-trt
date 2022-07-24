
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

#include "simple-logger.hpp"
#include "cuda-tools.hpp"
#include "trt-builder.hpp"
#include "trt-tensor.hpp"
#include "trt-infer.hpp"

using namespace std;

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

    auto engine = TRT::load_infer("engine.trtmodel");
    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        return;
    }

    engine->print();

    auto input       = engine->input();
    auto output      = engine->output();
    int input_width  = input->width();
    int input_height = input->height();

    ///////////////////////////////////////////////////
    // image to float
    auto image = cv::imread("dog.jpg");
    float mean[] = {0.406, 0.456, 0.485};
    float std[]  = {0.225, 0.224, 0.229};

    // 对应于pytorch的代码部分
    cv::resize(image, image, cv::Size(input_width, input_height));
    image.convertTo(image, CV_32F);

    cv::Mat channel_based[3];
    for(int i = 0; i < 3; ++i)
        channel_based[i] = cv::Mat(input_height, input_width, CV_32F, input->cpu<float>(0, 2-i));

    cv::split(image, channel_based);
    for(int i = 0; i < 3; ++i)
        channel_based[i] = (channel_based[i] / 255.0f - mean[i]) / std[i];

    engine->forward(true);    

    int num_classes   = output->size(1);
    float* prob       = output->cpu<float>();
    int predict_label = std::max_element(prob, prob + num_classes) - prob;
    auto labels       = load_labels("labels.imagenet.txt");
    auto predict_name = labels[predict_label];
    float confidence  = prob[predict_label];
    printf("Predict: %s, confidence = %f, label = %d\n", predict_name.c_str(), confidence, predict_label);
}

int main(){

    if(!build_model()){
        return -1;
    }
    inference();
    return 0;
}